import os, re, ray, time, wandb, collections, datetime, logging, numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from unstable.utils import setup_logger

from unstable._types import PlayerTrajectory, GameInformation
from unstable.utils import write_game_information_to_file
Scalar = Union[int, float, bool]

class BaseTracker:
    def __init__(self, run_name: str):
        self.run_name = run_name 
        self._build_output_dir()

    def _build_output_dir(self):
        self.output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.run_name)
        os.makedirs(self.output_dir)
        self.output_dirs = {}
        for folder_name in ["training_data", "eval_data", "checkpoints", "logs"]: 
            self.output_dirs[folder_name] =  os.path.join(self.output_dir, folder_name); os.makedirs(self.output_dirs[folder_name], exist_ok=True)

    def get_checkpoints_dir(self):  return self.output_dirs["checkpoints"]
    def get_train_dir(self):        return self.output_dirs["training_data"]
    def get_eval_dir(self):         return self.output_dirs["eval_data"]
    def get_log_dir(self):          return self.output_dirs["logs"]
    def add_trajectory(self, trajectory: PlayerTrajectory, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError
    # Optional structured logging hooks for runtime components
    def log_collector(self, stats: Dict): raise NotImplementedError
    def log_buffer(self, stats: Dict, env_id: Optional[str] = None): raise NotImplementedError

    
@ray.remote
class Tracker(BaseTracker): 
    FLUSH_EVERY = 64
    def __init__(self, run_name: str, wandb_project: Optional[str]=None, config: Optional[Dict]=None, wandb_entity: Optional[str]=None):
        """Tracker actor responsible for aggregating metrics and (optionally) logging to Weights & Biases.

        Args:
            run_name: Human readable run identifier.
            wandb_project: If provided, enables W&B logging to this project.
            config: Optional configuration dictionary to store in W&B (hyperparameters, env specs, etc.).
            wandb_entity: Optional W&B entity/organization.
        """
        super().__init__(run_name=run_name)
        self.logger = setup_logger("tracker", self.get_log_dir())
        self.use_wandb = False
        if wandb_project:
            try:
                wandb.init(project=wandb_project, entity=wandb_entity, name=run_name, config=config, settings=wandb.Settings(start_method="thread"))
                wandb.define_metric("*", step_metric="learner/step")
                self.use_wandb = True
                self.logger.info("W&B run initialised with config keys: %s", list((config or {}).keys()))
            except Exception as e:
                self.logger.warning(f"Failed to initialise W&B run: {e}")
        self._m: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=512))
        self._buffer: Dict[str, Scalar] = {}
        self._n = {}
        self._last_flush = time.monotonic()
        self._interface_stats = {"gpu_tok_s": {}, "TS": {}, "exploration": {}, "match_counts": {}, "format_success": None, "inv_move_rate": None, "game_len": None}

    def update_config(self, cfg: Dict, allow_val_change: bool=True):
        """Update the W&B config after initialisation (safe for Ray remote call)."""
        if self.use_wandb:
            try:
                wandb.config.update(cfg, allow_val_change=allow_val_change)
                self.logger.info("Updated W&B config with keys: %s", list(cfg.keys()))
            except Exception as e:
                self.logger.warning(f"wandb.config.update failed: {e}")

    def _put(self, k: str, v: Scalar): self._m[k].append(v)
    def _agg(self, p: str) -> dict[str, Scalar]: return {k: float(np.mean(dq)) for k, dq in self._m.items() if k.startswith(p)}
    def _flush_if_due(self):
        if time.monotonic()-self._last_flush >= self.FLUSH_EVERY:
            if self._buffer and self.use_wandb:
                try: wandb.log(self._buffer)
                except Exception as e: self.logger.warning(f"wandb.log failed: {e}")
            self._buffer.clear(); self._last_flush=time.monotonic()

    def add_player_trajectory(self, traj: PlayerTrajectory, env_id: str):
        try:
            reward = traj.final_reward; player_id = traj.pid
            self._put(f"collection-{env_id}/reward", reward)
            self._put(f"collection-{env_id}/Win Rate", int(reward>0))
            self._put(f"collection-{env_id}/Loss Rate", int(reward<0))
            self._put(f"collection-{env_id}/Draw", int(reward==0))
            self._put(f"collection-{env_id}/Reward (pid={traj.pid})", reward)
            self._put(f"collection-{env_id}/Game Length", traj.num_turns)
            for idx in range(len(traj.obs)):
                self._put(f"collection-{env_id}/Respone Length (char)", len(traj.actions[idx]))
                self._put(f"collection-{env_id}/Observation Length (char)", len(traj.obs[idx]))
                for k, v in traj.format_feedbacks[idx].items(): self._put(f"collection-{env_id}/Format Success Rate - {k}", v)
            self._n[f"collection-{env_id}"] = self._n.get(f"collection-{env_id}", 0) + 1
            self._put(f"collection-{env_id}/step", self._n[f"collection-{env_id}"])
            self._buffer.update(self._agg('collection-')); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception when adding trajectory to tracker: {exc}")

    def add_eval_game_information(self, game_information: GameInformation, env_id: str):
        try:
            eval_reward = game_information.final_rewards.get(game_information.eval_model_pid, 0.0)
            _prefix = (
                f"evaluation-{env_id}" if not game_information.eval_opponent_name
                else f"evaluation-{env_id} ({game_information.eval_opponent_name})"
            )
            self._put(f"{_prefix}/Reward", eval_reward)
            self._put(f"{_prefix}/Reward (pid={game_information.eval_model_pid})", eval_reward)
            self._put(f"{_prefix}/Win Rate", int(eval_reward > 0))
            self._put(f"{_prefix}/Loss Rate", int(eval_reward < 0))
            self._put(f"{_prefix}/Draw Rate", int(eval_reward == 0))
            self._n[_prefix] = self._n.get(_prefix, 0) + 1
            self._put(f"{_prefix}/step", self._n[_prefix])

            rows = []
            num_steps = len(getattr(game_information, "obs", []))
            for i in range(num_steps):
                pid = game_information.pid[i] if i < len(game_information.pid) else None
                name = game_information.names.get(pid) if hasattr(game_information, "names") else None
                obs = game_information.obs[i] if i < len(game_information.obs) else None
                raw = (
                    game_information.full_actions[i]
                    if i < len(game_information.full_actions)
                    else None
                )
                extracted = (
                    game_information.extracted_actions[i]
                    if i < len(game_information.extracted_actions)
                    else None
                )
                prompt = (
                    game_information.prompts[i]
                    if i < len(getattr(game_information, "prompts", []))
                    else None
                )
                ptxt = prompt or ""
                plen_char = len(ptxt)
                plen_ws = len(ptxt.split())
                self._put(f"{_prefix}/Prompt Length (char)", plen_char)
                self._put(f"{_prefix}/Prompt Length (ws_tok)", plen_ws)
                rows.append(
                    {
                        "step": i,
                        "pid": pid,
                        "name": name,
                        "prompt": prompt,
                        "raw_action": raw,
                        "extracted_action": extracted,
                        "prompt_len_char": plen_char,
                        "prompt_len_ws_tokens": plen_ws,
                        "step_reward": (
                            game_information.step_rewards[i]
                            if hasattr(game_information, "step_rewards")
                            and i < len(game_information.step_rewards)
                            else game_information.final_rewards.get(pid, None)
                        ),
                        "final_reward": game_information.final_rewards.get(pid, None),
                    }
                )

            df = pd.DataFrame(rows)
            out_path = os.path.join(
                self.get_eval_dir(), f"{env_id}-{game_information.game_idx}.csv"
            )
            df.to_csv(out_path, index=False, lineterminator="\n")

            self._buffer.update(self._agg("evaluation-"))
            self._flush_if_due()
        except Exception as exc:  # noqa: BLE001
            self.logger.info(
                f"Exception when adding game_info to tracker: {exc}"
            )

    def add_train_game_information(self, game_information: GameInformation, env_id: str):
        """Write a per-game CSV for a training game (mirrors eval format)."""
        try:
            train_reward_sum = sum(
                [
                    game_information.final_rewards.get(pid, 0.0)
                    for pid in set(game_information.pid)
                ]
            )
            _prefix = f"training-{env_id}"
            self._put(f"{_prefix}/Total Reward", train_reward_sum)
            self._n[_prefix] = self._n.get(_prefix, 0) + 1
            self._put(f"{_prefix}/step", self._n[_prefix])

            rows = []
            num_steps = len(getattr(game_information, "obs", []))
            for i in range(num_steps):
                pid = game_information.pid[i] if i < len(game_information.pid) else None
                name = game_information.names.get(pid) if hasattr(game_information, "names") else None
                obs = game_information.obs[i] if i < len(game_information.obs) else None
                raw = (
                    game_information.full_actions[i]
                    if i < len(game_information.full_actions)
                    else None
                )
                extracted = (
                    game_information.extracted_actions[i]
                    if i < len(game_information.extracted_actions)
                    else None
                )
                prompt = (
                    game_information.prompts[i]
                    if i < len(getattr(game_information, "prompts", []))
                    else None
                )
                ptxt = prompt or ""
                plen_char = len(ptxt)
                plen_ws = len(ptxt.split())
                self._put(f"{_prefix}/Prompt Length (char)", plen_char)
                self._put(f"{_prefix}/Prompt Length (ws_tok)", plen_ws)
                rows.append(
                    {
                        "step": i,
                        "pid": pid,
                        "name": name,
                        "prompt": prompt,
                        "raw_action": raw,
                        "extracted_action": extracted,
                        "prompt_len_char": plen_char,
                        "prompt_len_ws_tokens": plen_ws,
                        "step_reward": (
                            game_information.step_rewards[i]
                            if hasattr(game_information, "step_rewards")
                            and i < len(game_information.step_rewards)
                            else game_information.final_rewards.get(pid, None)
                        ),
                        "final_reward": game_information.final_rewards.get(pid, None),
                    }
                )

            df = pd.DataFrame(rows)
            out_path = os.path.join(
                self.get_train_dir(), f"{env_id}-{game_information.game_idx}.csv"
            )
            df.to_csv(out_path, index=False, lineterminator="\n")

            self._buffer.update(self._agg("training-"))
            self._flush_if_due()
        except Exception as exc:  # noqa: BLE001
            self.logger.info(
                f"Exception when adding training game_info to tracker: {exc}"
            )

    def log_model_registry(self, ts_dict: dict[str, dict[str, float]], match_counts: dict[tuple[str, str], int]):
        self._interface_stats.update({"TS": ts_dict, "exploration": None, "match_counts": match_counts})

    def log_inference(self, actor: str, gpu_ids: list[int], stats: dict[str, float]):
        for key in stats: self._put(f"inference/{actor}/{key}", stats[key])
        for gpu_id in gpu_ids: self._interface_stats["gpu_tok_s"][gpu_id] = stats["tok_s"]
        self._buffer.update(self._agg('inference'))
    
    def log_learner(self, info: dict):
        try:
            self._m.update({f"learner/{k}": v for k, v in info.items()})
            self._buffer.update(self._agg("learner")); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception in log_learner: {exc}")

    def get_interface_info(self): 
        for inf_key in ["Game Length", "Format Success Rate - correct_answer_format", "Format Success Rate - invalid_move"]: 
            self._interface_stats[inf_key] = np.mean([float(np.mean(dq)) for k,dq in self._m.items() if inf_key in k])
        return self._interface_stats

    # -------- New: generic logging hooks for collector and buffers --------
    def log_collector(self, stats: Dict):
        """Log high-level Collector runtime stats to W&B under 'collector/*'.

        Expected keys (free-form):
        - games_started, games_completed, games_failed
        - train_completed, eval_completed
        - actor_crashes, task_errors, launch_exceptions
        - games_in_flight, running_train, running_eval, actors, loop_iter
        """
        try:
            for k, v in (stats or {}).items():
                # Store cumulative or instantaneous values; aggregator will average when flushed
                self._put(f"collector/{k}", v)
            self._buffer.update(self._agg("collector"))
            self._flush_if_due()
        except Exception as exc:  # noqa: BLE001
            self.logger.info(f"Exception in log_collector: {exc}")

    def log_buffer(self, stats: Dict, env_id: Optional[str] = None):
        """Log buffer stats to W&B under 'buffer[-env]/ *'.

        Example keys: size, added, evicted, excess, batch_size, training_steps.
        """
        try:
            prefix = f"buffer-{env_id}" if env_id else "buffer"
            for k, v in (stats or {}).items():
                self._put(f"{prefix}/{k}", v)
            self._buffer.update(self._agg(prefix))
            self._flush_if_due()
        except Exception as exc:  # noqa: BLE001
            self.logger.info(f"Exception in log_buffer: {exc}")
