import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import ray
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from unstable.utils.logging import setup_logger


@ray.remote
class VLLMActor:
    def __init__(self, cfg: Dict[str, Any], tracker, name: str):
        # Disable console logging inside Ray actor to avoid duplicated logs across processes.
        self.logger = setup_logger(
            f"actor-{name}", ray.get(tracker.get_log_dir.remote()), to_console=False
        )
        # Let Ray handle GPU isolation. Do NOT override CUDA_VISIBLE_DEVICES here.
        # Overriding with ray.get_gpu_ids() can remap to physical GPU 0 for all actors.
        self.gpu_ids = ray.get_gpu_ids()
        try:
            self.logger.info(
                f"Actor {name} assigned GPU ids (ray.get_gpu_ids): {self.gpu_ids}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        except Exception:
            pass
        
        # --- START: REVISED ENGINEARGS INITIALIZATION ---

        # 1. Create a copy of the config to safely modify it
        engine_args_dict = cfg.copy()
        
        # 2. Handle renamed/legacy parameters for backward compatibility
        if "max_parallel_seq" in engine_args_dict and "max_num_seqs" not in engine_args_dict:
            self.logger.info(f"Mapping legacy parameter 'max_parallel_seq' ({engine_args_dict['max_parallel_seq']}) to 'max_num_seqs'.")
            engine_args_dict["max_num_seqs"] = engine_args_dict.pop("max_parallel_seq")
        # vLLM EngineArgs expects 'model', some configs use 'model_name'
        if "model_name" in engine_args_dict and "model" not in engine_args_dict:
            self.logger.info(f"Mapping legacy parameter 'model_name' ({engine_args_dict['model_name']}) to 'model'.")
            engine_args_dict["model"] = engine_args_dict.pop("model_name")
        # Clamp max_num_seqs to a safer bound to reduce init-time memory pressure
        if "max_num_seqs" in engine_args_dict:
            original = int(engine_args_dict["max_num_seqs"])
            clamped = max(1, min(original, 32))
            if clamped != original:
                self.logger.info(f"Clamping max_num_seqs from {original} to {clamped} to reduce startup memory.")
                engine_args_dict["max_num_seqs"] = clamped
        else:
            # Provide a conservative default if not set
            engine_args_dict["max_num_seqs"] = 16

        # Provide safe defaults for stability unless explicitly configured
        engine_args_dict.setdefault("tensor_parallel_size", 1)
        engine_args_dict.setdefault("gpu_memory_utilization", 0.85)
        engine_args_dict.setdefault("trust_remote_code", True)
        # Prefer bf16 on modern GPUs; fall back to auto if it errors at runtime
        engine_args_dict.setdefault("dtype", "bfloat16")
        
        # 3. Extract parameters that are NOT part of EngineArgs
        # These are used for SamplingParams later, not engine creation.
        sampling_temp = engine_args_dict.pop("temperature", 0.7)
        sampling_top_p = engine_args_dict.pop("top_p", 0.95)
        sampling_max_tokens = engine_args_dict.pop("max_tokens", 4096)
        sampling_logprobs = bool(engine_args_dict.pop("enable_logprobs", False))

        # We also need to handle the nested lora_config dict and explicit enable flag
        enable_lora_flag = bool(engine_args_dict.pop("enable_lora", False))
        lora_config = engine_args_dict.pop("lora_config", {})
        lora_rank = int(lora_config.get("lora_rank", 32)) # Default if not found

        # 4. Initialize EngineArgs by unpacking the cleaned dictionary
        # This will automatically pass gpu_memory_utilization, max_num_batched_tokens, etc.
        self.logger.info(f"Initializing VLLM Engine with args: {engine_args_dict} | enable_lora={enable_lora_flag}, max_lora_rank={lora_rank}")
        # Only pass LoRA-related args when enabled (avoids extra overhead / incompat)
        if enable_lora_flag:
            engine_args = EngineArgs(
                enable_lora=True,
                max_lora_rank=lora_rank,
                **engine_args_dict
            )
        else:
            engine_args = EngineArgs(
                **engine_args_dict
            )

        try:
            self.engine = LLMEngine.from_engine_args(engine_args)
            self.logger.info("VLLM engine initialized successfully")
        except Exception as e:
            self.logger.error(f"VLLM engine initialization failed: {e}")
            raise

        # --- END: REVISED ENGINEARGS INITIALIZATION ---

        self.logger.info(f"vLLM model path or name: {engine_args.model}")
        self.logger.info(f"Model architecture: {self.engine.model_config.__dict__}")
        self._lora_enabled = enable_lora_flag
            
        # Use the parameters we extracted earlier
        self.sampling_params = SamplingParams(
            temperature=sampling_temp,
            top_p=sampling_top_p,
            max_tokens=sampling_max_tokens,
            logprobs=(1 if sampling_logprobs else None),
        )

        # The rest of your __init__ method remains the same...
        self._queue = deque()
        self._futures = {}
        self._next_id = 0
        self._req2lora = {}
        self._prev_tok_cnt = defaultdict(int)

        self.tracker = tracker
        self.name = name

        self._queued = 0
        self._running = 0
        self._tok_hist = deque()
        self._batch_task = asyncio.create_task(self._batch_loop())
        self._report_task = asyncio.create_task(self._report_loop())
        self._lora_ids = {"base": 0}
        self._next_lora_id = 1
        self._last_step_time = time.monotonic()
        # Track simple logprob proxy for diagnostics when enabled
        self._logp_hist = deque()

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None, max_tokens_override: Optional[int] = None) -> str:
        if lora_path is not None and not isinstance(lora_path, str): lora_path = str(lora_path)
        fut = asyncio.Future()
        self._queued += 1
        # queue includes optional per-request max token override
        self._queue.append((prompt, lora_path, max_tokens_override, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            try:
                await asyncio.sleep(0.02)
                if time.monotonic() - self._last_step_time > 30: 
                    self.logger.error(f"Potential deadlock detected - no engine steps for {time.monotonic() - self._last_step_time:.1f} seconds\nRunning requests: {dict(self._running)}\nQueue size: {len(self._queue)}") # 30 second deadlock detection
                while self._queue:
                    prompt, path, max_override, fut = self._queue.popleft()
                    lora = path or "base"
                    req_id = str(self._next_id); self._next_id += 1
                    self._futures[req_id] = fut
                    self._req2lora[req_id] = lora
                    self._queued -= 1
                    self._running += 1

                    if path and self._lora_enabled:
                        if path not in self._lora_ids:
                            self._lora_ids[path] = self._next_lora_id
                            self._next_lora_id += 1
                        lora_req = LoRARequest(path, self._lora_ids[path], path)
                    else:
                        lora_req = None
                        if path and not self._lora_enabled:
                            self.logger.warning(f"LoRA path provided ({path}) but LoRA is disabled; proceeding without LoRA.")
                    # Apply explicit per-call override if provided; else use default sampling params
                    sp = self.sampling_params if (max_override is None) else SamplingParams(
                        temperature=self.sampling_params.temperature,
                        top_p=self.sampling_params.top_p,
                        max_tokens=int(max_override),
                        logprobs=self.sampling_params.logprobs,
                    )
                    try: self.engine.add_request(req_id, prompt, sp, lora_request=lora_req)
                    except Exception as e:
                        self.logger.error(f"Failed to add request {req_id}: {e}")
                        self._running -= 1
                        self._req2lora.pop(req_id, None)
                        fut.set_exception(e)
                        continue
                try:
                    step_start = time.monotonic()
                    outs = self.engine.step()
                    step_duration = time.monotonic() - step_start
                    self._last_step_time = time.monotonic()
                    if step_duration > 5.0: self.logger.warning(f"Slow engine step: {step_duration:.1f}s") # Log slow steps
                except Exception as exc:   
                    self.logger.exception(f"engine.step() failed - running: {dict(self._running)}"); await asyncio.sleep(1.0)  # Brief pause before retry
                    continue

                for out in outs:
                    req_id = out.request_id
                    lora = self._req2lora.get(req_id, "base")
                    segment = out.outputs[-1]

                    tok_ids = getattr(segment, "token_ids", None) or []
                    prev = self._prev_tok_cnt[req_id]
                    new_tok = max(0, len(tok_ids) - prev)
                    self._prev_tok_cnt[req_id] = len(tok_ids)

                    now = time.monotonic()
                    for _ in range(new_tok): 
                        self._tok_hist.append(now)
                    if segment.finish_reason is not None:
                        # If vLLM provided logprobs, record a simple average of chosen token logprobs
                        try:
                            lp_list = getattr(segment, "logprobs", None)
                            if lp_list:
                                chosen = []
                                for d in lp_list:
                                    if isinstance(d, dict) and d:
                                        # with logprobs=1 we expect the chosen token
                                        chosen.append(max(d.values()))
                                if chosen:
                                    self._logp_hist.append(sum(chosen) / max(1, len(chosen)))
                        except Exception:
                            pass
                        fut = self._futures.pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(segment.text)
                        self._running -= 1
                        self._req2lora.pop(req_id, None)
                        self._prev_tok_cnt.pop(req_id, None)
            except Exception as e: self.logger.exception(f"Critical error in batch loop: {e}"); await asyncio.sleep(1.0)  # Prevent tight error loop

    async def _report_loop(self):
        self.logger.info("Starting _report_loop")
        while True:
            await asyncio.sleep(5.0) # only send every 5 sec
            # Aggregate recent chosen-token logprobs if available
            recent_lp = None
            try:
                if self._logp_hist:
                    # Average over last ~window by count (bounded by history length)
                    k = min(len(self._logp_hist), 256)
                    if k > 0:
                        recent = list(self._logp_hist)[-k:]
                        recent_lp = sum(recent) / k
            except Exception:
                recent_lp = None
            stats = {"queued": self._queued, "running": self._running, "tok_s": self._tok_rate()}
            if recent_lp is not None:
                stats["avg_chosen_logp"] = float(recent_lp)
            self.logger.info(f"inside while loop _report_loop stats: {stats}")
            try: ray.get(self.tracker.log_inference.remote(actor=self.name, gpu_ids=self.gpu_ids, stats=stats))
            except Exception as e: self.logger.warning(f"tracker logging failed: {e}")

    def _tok_rate(self, window: float = 2.0) -> float:
        now  = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window