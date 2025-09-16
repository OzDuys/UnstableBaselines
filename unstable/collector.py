import re, random, logging, itertools
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Protocol, Optional

import ray
from ray.exceptions import RayActorError, RayTaskError

import textarena as ta
assert ta.__version__ >= "0.6.16", f"TextArena package version is too old: {ta.__version__}. Required version is at least 0.6.16."

# local imports
from unstable.actor import VLLMActor
from unstable._types import GameSpec, GameInformation, PlayerTrajectory, TaskMeta
from unstable.reward_transformations import ComposeStepRewardTransforms, ComposeFinalRewardTransforms
from unstable.utils.logging import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING



class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str|Path, obs_fmt_fn: Callable[[str],str], extract_fn: Callable[[str], Tuple[str, Dict[str, Any]]], conversation_max_tokens: Optional[int] = None) -> None:
        self._actor, self._lora, self._fmt, self._extract = actor, lora_path, obs_fmt_fn, extract_fn
        self._conv_max = conversation_max_tokens

    def __call__(self, observation: str) -> str: 
        _, extracted, _, _ = self.act_full(observation)
        return extracted

    def act_full(self, observation: str) -> Tuple[str, str, str, dict]:
        prompt = self._fmt(observation=observation)
        # If this is a conversation turn and a cap is configured, apply per-call override.
        lower = (observation or "").lower()
        is_convo = ("converse freely" in lower) or ("free-chat" in lower) or ("free chat" in lower)
        max_override = self._conv_max if (self._conv_max is not None and is_convo) else None
        # Submit to shared vLLM actor; allow queuing/batching on the actor side
        raw = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora, max_tokens_override=max_override))
        extracted, format_feedback = self._extract(raw_action=raw)
        return raw, extracted, prompt, format_feedback

@ray.remote(num_cpus=0)
def run_game(game_spec: GameSpec, actor: VLLMActor):
    """Run a single game and return aggregated information.

    Change: Apply action extraction to OpenRouter opponents too so their
    outputs (e.g. Gemini) are parsed for [action] or \boxed{action} formats.
    """
    game_information = GameInformation(game_idx=game_spec.game_idx, eval_model_pid=game_spec.eval_model_pid, eval_opponent_name=game_spec.eval_opponent_name)

    # Extract optional per-env overrides and caps for use in actor wrapper
    _env_kwargs = dict(getattr(game_spec, "env_kwargs", {}))
    conversation_max_tokens = _env_kwargs.get("conversation_max_tokens")
    opponent_prompt_template = _env_kwargs.get("opponent_prompt_template")

    # Build a pid->AgentSpec mapping for reliable lookups
    _pid2spec = {spec.pid: spec for spec in game_spec.agent_specs}
    agents = {}
    for agent_spec in game_spec.agent_specs:
        is_openrouter = agent_spec.openrouter_name is not None
        model_obj = (
            CallableActorWrapper(
                actor=actor,
                lora_path=agent_spec.lora_path,
                obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template],
                extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn],
                conversation_max_tokens=conversation_max_tokens,
            ) if not is_openrouter else ta.agents.OpenRouterAgent(agent_spec.openrouter_name, system_prompt="")  # override STANDARD_GAME_PROMPT
        )
        agents[agent_spec.pid] = {
            "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None,
            "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
            "model": model_obj,
            "extract_fn": ACTION_EXTRACTION[agent_spec.action_extraction_fn],
            "is_openrouter": is_openrouter,
        }

    # Instantiate environment with any propagated constructor kwargs (e.g. communication_turns override)
    _ek = dict(getattr(game_spec, "env_kwargs", {}))
    _ek.pop("conversation_max_tokens", None)  # strip our custom key
    env = ta.make(game_spec.env_id, **_ek)
    env.reset(num_players=len(agents), seed=game_spec.seed)
    env.state.error_allowance = 0
    turn = 0

    while True:
        pid, obs = env.get_observation()
        agent_entry = agents[pid]
        if agent_entry["is_openrouter"]:  # opponent via OpenRouter API; allow template override
            # Prefer explicit opponent template from env kwargs; else use the agent's prompt_template; else default-auto if available
            tpl_key = opponent_prompt_template or getattr(_pid2spec.get(pid, None), "prompt_template", None) or ("default-auto" if "default-auto" in OBSERVATION_FORMATTING else "default")
            fmt_fn = OBSERVATION_FORMATTING.get(tpl_key, OBSERVATION_FORMATTING.get("default-auto", OBSERVATION_FORMATTING["default"]))
            prompt = fmt_fn(observation=obs)
            raw = agent_entry["model"](prompt)
            extracted, format_feedback = agent_entry["extract_fn"](raw)
        else:  # local checkpointed model (already does extraction internally)
            raw, extracted, prompt, format_feedback = agent_entry["model"].act_full(obs)

        done, step_info = env.step(extracted)
        turn += 1

        # General tracking
        game_information.pid.append(pid)
        game_information.obs.append(obs)
        game_information.full_actions.append(raw)
        game_information.extracted_actions.append(extracted)
        game_information.step_infos.append(step_info)
        game_information.names[pid] = agent_entry["name"]
        game_information.prompts.append(prompt)

        # Player specific tracking (only for learning / data-collecting agents)
        if agent_entry["traj"] is not None:
            traj = agent_entry["traj"]
            traj.obs.append(obs)
            traj.prompts.append(prompt)
            traj.actions.append(raw)
            traj.extracted_actions.append(extracted)
            format_feedback["invalid_move"] = False
            traj.format_feedbacks.append(format_feedback)
            traj.step_infos.append(step_info)

        if done:
            break

    final_rewards, game_info = env.close()
    for pid, agent_entry in agents.items():
        if agent_entry["traj"] is not None:
            traj = agent_entry["traj"]
            traj.final_reward = final_rewards[pid]
            traj.game_info = game_info[pid]
            traj.num_turns = turn
            if game_info[pid]["invalid_move"]:
                traj.format_feedbacks[-1]["invalid_move"] = True

    game_information.final_rewards = final_rewards
    game_information.num_turns = turn
    game_information.game_info = game_info
    # Populate rich per-step reward diagnostics for CSV output by asking the buffer to compute them
    try:
        # Access buffer for accurate transformed/shaped rewards
        # Note: buffer is not directly available here; we rely on runtime wiring: Collector has self.buffer
        # We will compute per-pid mapping using player trajectories that were actually collecting data
        pid2traj = {traj.pid: traj for traj in [agents[p]["traj"] for p in agents if agents[p]["traj"] is not None]}
        # For training tasks, Collector._post_train will have both buffer and these trajectories; compute here for consistency
        # We call a remote helper on the buffer to compute rewards for each trajectory, avoiding duplication of transform logic
        transformed_final_map = {}
        step_raw_final = []
        step_env_final = []
        step_shaped = []
        # We need self.buffer here; we're inside run_game (a task function) without self. So we can only attach raw replicates now.
        # The accurate computation will be done in _post_train when the buffer is available.
        step_raw_final = [final_rewards.get(p, None) for p in game_information.pid]
        step_env_final = step_raw_final[:]
        step_shaped = [None]*len(step_raw_final)
        transformed_final_map = dict(final_rewards)
        game_information.step_rewards = step_env_final
        game_information.step_rewards_raw_final = step_raw_final
        game_information.step_rewards_env = step_env_final
        game_information.step_rewards_shaped = step_shaped
        game_information.final_rewards_transformed = transformed_final_map
    except Exception:
        n = len(game_information.pid)
        game_information.step_rewards = [final_rewards.get(p, None) for p in game_information.pid]
        game_information.step_rewards_raw_final = [final_rewards.get(p, None) for p in game_information.pid]
        game_information.step_rewards_env = [final_rewards.get(p, None) for p in game_information.pid]
        game_information.step_rewards_shaped = [None]*n
        game_information.final_rewards_transformed = dict(final_rewards)
    return game_information, [agents[pid]["traj"] for pid in agents if agents[pid]["traj"] is not None]


@ray.remote
class Collector:
    def __init__(self, vllm_config, tracker, buffer, game_scheduler):
        # Disable console logs from Collector actor to avoid duplication across Ray workers
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()), to_console=False)
        self.tracker, self.buffer, self.game_scheduler = tracker, buffer, game_scheduler
        # Create a single vLLM actor that spans tensor_parallel_size GPUs to avoid multi-engine contention
        try:
            tp = int(vllm_config.get("tensor_parallel_size", 1))
        except Exception:
            tp = 1
        # Reserve exactly `tp` GPUs for the engine; run a single shared actor for all games
        self.actors = [VLLMActor.options(num_gpus=tp, max_concurrency=256).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-TP{tp}")]
        self._actor_iter = itertools.cycle(self.actors)

        # Warm up vLLM with a tiny prompt to ensure engine is responsive
        try:
            _ = ray.get(self.actors[0].submit_prompt.remote(prompt="Hello", lora_path=None, max_tokens_override=4), timeout=60)
            try:
                self.tracker.log_collector.remote({"vllm_warmup_ok": 1})
            except Exception:
                pass
        except Exception as e:
            try:
                self.tracker.log_collector.remote({"vllm_warmup_ok": 0, "vllm_warmup_error": str(e)})
            except Exception:
                pass

        # thead keeping
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info("Collector initialized")
        # runtime counters
        self._games_started = 0
        self._games_completed = 0
        self._games_failed = 0
        self._train_completed = 0
        self._eval_completed = 0
        self._actor_crashes = 0
        self._task_errors = 0
        self._launch_exceptions = 0
        self._loop_iter = 0
        # log throttling
        self._train_log_every = 50  # log every N train specs
        self._eval_log_every = 20   # log every N eval specs
        self._train_log_count = 0
        self._eval_log_count = 0
    
    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        while self._num_running("train") < max_train: # submit new train game
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_train_job.remote()) # sample game spec
                self._train_log_count += 1
                if self._train_log_count == 1 or (self._train_log_count % self._train_log_every == 0):
                    self.logger.info(f"received train game_spec (count={self._train_log_count}): {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("train", game_spec.env_id)
                self._games_started += 1
                # log start
                try:
                    self.tracker.log_collector.remote({
                        "games_started": self._games_started,
                        "games_in_flight": len(self.flight),
                        "running_train": self._num_running("train"),
                        "running_eval": self._num_running("eval"),
                    })
                except Exception:
                    pass
            except Exception as exc:
                self.logger.info(f"Exception in train game {game_spec}: {exc}")
                self._launch_exceptions += 1
                try:
                    self.tracker.log_collector.remote({"launch_exceptions": self._launch_exceptions})
                except Exception:
                    pass

        while max_eval!=None and self._num_running("eval") < max_eval:
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_eval_job.remote())
                self._eval_log_count += 1
                if self._eval_log_count == 1 or (self._eval_log_count % self._eval_log_every == 0):
                    self.logger.info(f"received eval game_spec (count={self._eval_log_count}): {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("eval", game_spec.env_id)
                self._games_started += 1
                try:
                    self.tracker.log_collector.remote({
                        "games_started": self._games_started,
                        "games_in_flight": len(self.flight),
                        "running_train": self._num_running("train"),
                        "running_eval": self._num_running("eval"),
                    })
                except Exception:
                    pass
            except Exception as exc:
                self.logger.info(f"Exception in eval game {game_spec}: {exc}")
                self._launch_exceptions += 1
                try:
                    self.tracker.log_collector.remote({"launch_exceptions": self._launch_exceptions})
                except Exception:
                    pass

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        try:
            game_information, player_trajs = ray.get(ref)
        except (RayTaskError, RayActorError) as err:
            self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True)
            self._games_failed += 1
            if isinstance(err, RayActorError):
                self._actor_crashes += 1
            else:
                self._task_errors += 1
            try:
                self.tracker.log_collector.remote({
                    "games_failed": self._games_failed,
                    "actor_crashes": self._actor_crashes,
                    "task_errors": self._task_errors,
                    "games_in_flight": len(self.flight),
                    "running_train": self._num_running("train"),
                    "running_eval": self._num_running("eval"),
                })
            except Exception:
                pass
            return
        self._post_train(meta, game_information, player_trajs) if meta.type=="train" else self._post_eval(meta, game_information)
        # success path
        self._games_completed += 1
        if meta.type == "train":
            self._train_completed += 1
        else:
            self._eval_completed += 1
        try:
            self.tracker.log_collector.remote({
                "games_completed": self._games_completed,
                "train_completed": self._train_completed,
                "eval_completed": self._eval_completed,
                "games_in_flight": len(self.flight),
                "running_train": self._num_running("train"),
                "running_eval": self._num_running("eval"),
            })
        except Exception:
            pass
    
    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        # Add trajectories to buffer and tracker
        for traj in player_trajs:
            self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id)
            self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
        # Now that buffer has the transforms, compute accurate shaped/env rewards for CSV diagnostic columns
        try:
            pid2traj = {traj.pid: traj for traj in player_trajs}
            # Compute per-pid reward maps via buffer helper
            rewards_map = {}
            for pid, traj in pid2traj.items():
                rewards_map[pid] = ray.get(self.buffer.compute_rewards_for_logging.remote(traj, meta.env_id))
            # Build per-turn lists aligned to acting pid
            step_raw_final = [rewards_map.get(p, {}).get("raw_final") if p in rewards_map else game_information.final_rewards.get(p, None) for p in game_information.pid]
            step_env_final = [rewards_map.get(p, {}).get("env_final") if p in rewards_map else game_information.final_rewards.get(p, None) for p in game_information.pid]
            step_shaped = []
            # For shaped, index by per-turn step index
            turn_counts = {pid: 0 for pid in pid2traj.keys()}
            for p in game_information.pid:
                if p in rewards_map:
                    idx = turn_counts[p]
                    shaped_list = rewards_map[p]["shaped_per_step"]
                    step_shaped.append(shaped_list[idx] if idx < len(shaped_list) else None)
                    turn_counts[p] += 1
                else:
                    step_shaped.append(None)
            # Final rewards transformed per pid
            final_transformed = {pid: rewards_map.get(pid, {}).get("env_final", game_information.final_rewards.get(pid, None)) for pid in game_information.final_rewards.keys()}
            # Write into GameInformation
            game_information.step_rewards = step_env_final
            game_information.step_rewards_raw_final = step_raw_final
            game_information.step_rewards_env = step_env_final
            game_information.step_rewards_shaped = step_shaped
            game_information.final_rewards_transformed = final_transformed
        except Exception as exc:
            self.logger.warning(f"Failed to compute shaped/env rewards for CSV: {exc}")
        # write per-game CSV for training
        self.tracker.add_train_game_information.remote(game_information=game_information, env_id=meta.env_id)
        self.game_scheduler.update.remote(game_info=game_information)

    def _post_eval(self, meta: TaskMeta, game_information: GameInformation):
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id)
    
    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("entered collect func")
        while ray.get(self.buffer.continue_collection.remote()):
            # Rate-limit noisy loop log
            if self._loop_iter == 0 or (self._loop_iter % 50 == 0):
                self.logger.info("collector loop heartbeat")
            self._loop_iter += 1
            try:
                self.tracker.log_collector.remote({
                    "loop_iter": self._loop_iter,
                    "actors": len(self.actors),
                    "games_in_flight": len(self.flight),
                    "running_train": self._num_running("train"),
                    "running_eval": self._num_running("eval"),
                })
            except Exception:
                pass
            self._launch_jobs(num_train_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished_job(done_ref[0])
