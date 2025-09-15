import ray, torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from unstable.learners.base import BaseLearner

@ray.remote
class REINFORCELearner(BaseLearner):
    def __init__(self, *args, kl_beta: float = 0.0, kl_ref_model_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # KL regularization config
        self.kl_beta: float = float(kl_beta or 0.0)
        self.kl_ref_model_name: Optional[str] = kl_ref_model_name
        self.ref_model = None
        self.ref_tokenizer = None
        if self.kl_beta > 0.0:
            try:
                name = self.kl_ref_model_name or self.model_name
                # Load a frozen reference policy on the same device (optional: could be moved to CPU, but slower)
                self.ref_model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
                for p in self.ref_model.parameters():
                    p.requires_grad_(False)
                self.ref_model.eval()
                self.ref_tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                if getattr(self.ref_tokenizer, "pad_token", None) is None:
                    self.ref_tokenizer.pad_token = self.ref_tokenizer.eos_token
                self.logger.info(f"KL regularization enabled (beta={self.kl_beta}) vs reference model: {name}")
            except Exception as exc:
                self.logger.warning(f"Failed to load KL reference model; disabling KL penalty. Error: {exc}")
                self.kl_beta = 0.0
                self.ref_model, self.ref_tokenizer = None, None

    def initialize_algorithm(self, max_train_len: int, max_generation_len: int):
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len # for Dr. GRPO trick

    def _prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device) # Tokenize with truncation
        return enc, advs, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, advs, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        out = self.policy_model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
        for i, o in enumerate(obs): mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling

        # Optional KL penalty vs frozen reference model (on response tokens only)
        kl_mean = None
        if self.kl_beta > 0.0 and self.ref_model is not None:
            with torch.no_grad():
                ref_out = self.ref_model(**enc)
                ref_logp = torch.nn.functional.log_softmax(ref_out.logits, dim=-1)
                ref_tok_logp = ref_logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
            # Per-token empirical KL estimate E_pi[log pi - log pref]
            kl_tokens = (tok_logp - ref_tok_logp) * mask
            # Normalize like seq_logp
            kl_per_seq = kl_tokens.sum(1) / self.max_generation_len
            kl_mean = kl_per_seq.mean()
            loss = loss + (self.kl_beta * kl_mean) / scaling

        loss.backward()
        torch.cuda.empty_cache()
        metrics = {"loss": float(loss.item()), "logp_mean": float(seq_logp.mean().item()), "avg_train_len": float(avg_len), "pct_truncated": float(pct_truncated)}
        if kl_mean is not None:
            metrics["kl_mean"] = float(kl_mean.item())
        return metrics
    
    def _update(self, batch):
        metrics_acc = {}
        self.policy_optimizer.zero_grad(set_to_none=True)
        for i in range(self.gradient_acc_steps):
            sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16): 
                update_metrics = self._mini_batch_update_step(sub, scaling=self.gradient_acc_steps)
            for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v /  self.gradient_acc_steps
            self.logger.info(f"Mini-step metrics: {update_metrics}")
        self.logger.info(f"Step metrics: {metrics_acc}")
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        return metrics_acc
        