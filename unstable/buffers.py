
import os, ray, tree, random
from threading import Lock
from typing import List, Dict, Optional, Tuple, Callable, Any

# local imports
from unstable.utils.logging import setup_logger
from unstable._types import PlayerTrajectory, Step
# from unstable.core import BaseTracker
from unstable.trackers import BaseTracker
from unstable.utils import write_training_data_to_file
from unstable.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms


class BaseBuffer:
    def __init__(self, max_buffer_size: int, tracker: BaseTracker, final_reward_transformation: Optional[ComposeFinalRewardTransforms], step_reward_transformation: Optional[ComposeStepRewardTransforms], sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], buffer_strategy: str = "random"): ...
    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str): ...
    def get_batch(self, batch_size: int): ...



@ray.remote
class StepBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker, 
        final_reward_transformation: Optional[ComposeFinalRewardTransforms], 
        step_reward_transformation: Optional[ComposeStepRewardTransforms], 
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], 
        buffer_strategy: str = "random"
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.steps: List[Step] = []
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote())) # setup logging
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        invalid_count = 0
        for idx in range(len(player_traj.obs)):
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            # Attach invalid_move proxy if provided via format_feedbacks
            fb = {}
            try:
                fb = player_traj.format_feedbacks[idx] if idx < len(player_traj.format_feedbacks) else {}
            except Exception:
                pass
            invalid_flag = bool(fb.get("invalid_move", False))
            if invalid_flag:
                invalid_count += 1
            with self.mutex:
                self.steps.append(
                    Step(
                        pid=player_traj.pid,
                        obs=player_traj.obs[idx],
                        act=player_traj.actions[idx],
                        reward=step_reward,
                        env_id=env_id,
                        step_info={
                            "raw_reward": player_traj.final_reward,
                            "env_reward": reward,
                            "step_reward": step_reward,
                            "invalid_move": invalid_flag,
                        },
                    )
                )
        self.logger.info(f"Buffer size: {len(self.steps)}, added {len(player_traj.obs)} steps")
        try:
            # W&B buffer stats after add
            self.tracker.log_buffer.remote({
                "size": len(self.steps),
                "added": len(player_traj.obs),
                "training_steps": self.training_steps,
                "added_invalid_moves": invalid_count,
                "added_invalid_rate": (invalid_count / max(1, len(player_traj.obs))),
            }, env_id=env_id)
        except Exception:
            pass
        # downsample if necessary
        excess_num_samples = max(0, len(self.steps) - self.max_buffer_size); self.logger.info(f"Excess Num Samples: {excess_num_samples}")
        if excess_num_samples > 0:
            self.logger.info(f"Downsampling buffer because of excess samples")
            with self.mutex: 
                randm_sampled = random.sample(self.steps, excess_num_samples)
                for b in randm_sampled:
                    self.steps.remove(b)
                self.logger.info(f"Buffer size after downsampling: {len(self.steps)}")
            try:
                self.tracker.log_buffer.remote({
                    "size": len(self.steps),
                    "evicted": excess_num_samples,
                    "excess": excess_num_samples,
                }, env_id=env_id)
            except Exception:
                pass

    def get_batch(self, batch_size: int) -> List[Step]:
        with self.mutex:
            if self.buffer_strategy == "stratified_env":
                # Build env -> indices mapping to sample evenly across envs
                env_to_indices: Dict[str, List[int]] = {}
                for idx, s in enumerate(self.steps):
                    env_to_indices.setdefault(s.env_id, []).append(idx)

                # Compute target per-env counts
                envs = list(env_to_indices.keys())
                if not envs:
                    batch = []
                else:
                    per_env = max(1, batch_size // max(1, len(envs)))
                    selected_indices = []
                    # First pass: up to per_env from each
                    for env in envs:
                        idxs = env_to_indices[env]
                        take = min(per_env, len(idxs))
                        selected_indices.extend(random.sample(idxs, take))
                    # Fill remaining uniformly from leftover pool
                    remaining = batch_size - len(selected_indices)
                    if remaining > 0:
                        leftover = [i for env, idxs in env_to_indices.items() for i in idxs if i not in selected_indices]
                        if leftover:
                            selected_indices.extend(random.sample(leftover, min(remaining, len(leftover))))

                    # Build batch and remove in descending index order to keep indices valid
                    selected_indices = sorted(set(selected_indices), reverse=True)
                    batch = [self.steps[i] for i in selected_indices]
                    for i in selected_indices:
                        self.steps.pop(i)

                    # Log per-batch env histogram for visibility
                    try:
                        counts: Dict[str, int] = {}
                        for s in batch:
                            counts[s.env_id] = counts.get(s.env_id, 0) + 1
                        stats = {"batch_size": len(batch)}
                        # Expand counts as separate keys for easy charting
                        for k, v in counts.items():
                            stats[f"batch_env_count/{k}"] = v
                        self.tracker.log_buffer.remote(stats)
                    except Exception:
                        pass
            elif self.buffer_strategy == "stratified_env_role":
                # Build (env_id, pid) -> indices mapping to balance roles per environment
                bucket_to_indices: Dict[Tuple[str, int], List[int]] = {}
                for idx, s in enumerate(self.steps):
                    bucket_to_indices.setdefault((s.env_id, s.pid), []).append(idx)

                buckets = list(bucket_to_indices.keys())
                if not buckets:
                    batch = []
                else:
                    per_bucket = max(1, batch_size // max(1, len(buckets)))
                    selected_indices = []
                    # First pass: up to per_bucket from each bucket
                    for b in buckets:
                        idxs = bucket_to_indices[b]
                        take = min(per_bucket, len(idxs))
                        if take > 0:
                            selected_indices.extend(random.sample(idxs, take))
                    # Fill remaining uniformly from leftover pool
                    remaining = batch_size - len(selected_indices)
                    if remaining > 0:
                        leftover = [i for _, idxs in bucket_to_indices.items() for i in idxs if i not in selected_indices]
                        if leftover:
                            selected_indices.extend(random.sample(leftover, min(remaining, len(leftover))))

                    # Build batch and remove
                    selected_indices = sorted(set(selected_indices), reverse=True)
                    batch = [self.steps[i] for i in selected_indices]
                    for i in selected_indices:
                        self.steps.pop(i)

                    # Log per-batch env/role histogram
                    try:
                        env_counts: Dict[str, int] = {}
                        role_counts: Dict[int, int] = {}
                        env_role_counts: Dict[str, int] = {}
                        for s in batch:
                            env_counts[s.env_id] = env_counts.get(s.env_id, 0) + 1
                            role_counts[s.pid] = role_counts.get(s.pid, 0) + 1
                            env_role_counts[f"{s.env_id}/role_{s.pid}"] = env_role_counts.get(f"{s.env_id}/role_{s.pid}", 0) + 1
                        stats = {"batch_size": len(batch)}
                        for k, v in env_counts.items():
                            stats[f"batch_env_count/{k}"] = v
                        for k, v in role_counts.items():
                            stats[f"batch_role_count/{k}"] = v
                        for k, v in env_role_counts.items():
                            stats[f"batch_env_role_count/{k}"] = v
                        self.tracker.log_buffer.remote(stats)
                    except Exception:
                        pass
            else:
                batch = random.sample(self.steps, batch_size)
                for b in batch:
                    self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        self.logger.info(f"Sampling {len(batch)} samples from buffer.")
        try: write_training_data_to_file(batch=batch, filename=os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}.csv"))
        except Exception as exc: self.logger.error(f"Exception when trying to write training data to file: {exc}")
        self.training_steps += 1
        try:
            self.tracker.log_buffer.remote({
                "size": len(self.steps),
                "batch_size": len(batch),
                "training_steps": self.training_steps,
            })
        except Exception:
            pass
        return batch

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(self.steps)
    def continue_collection(self):  return self.collect
    def clear(self):                
        with self.mutex: 
            self.steps.clear()

    def compute_rewards_for_logging(self, player_traj: PlayerTrajectory, env_id: str) -> Dict[str, Any]:
        """Return detailed rewards for logging CSVs using this buffer's transforms.

        Returns a dict with keys:
        - raw_final: original env final reward (float)
        - env_final: final reward after final_reward_transformation (float)
        - shaped_per_step: list of shaped rewards per step (List[float])
        """
        raw_final = float(player_traj.final_reward)
        env_final = float(self.final_reward_transformation(reward=raw_final, pid=player_traj.pid, env_id=env_id)) if self.final_reward_transformation else raw_final
        shaped = []
        for idx in range(len(player_traj.obs)):
            if self.step_reward_transformation:
                s = float(self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=env_final))
            else:
                s = env_final
            shaped.append(s)
        return {"raw_final": raw_final, "env_final": env_final, "shaped_per_step": shaped}


@ray.remote
class EpisodeBuffer(BaseBuffer):
    def __init__(
        self, max_buffer_size: int, tracker: BaseTracker, 
        final_reward_transformation: Optional[ComposeFinalRewardTransforms], 
        step_reward_transformation: Optional[ComposeStepRewardTransforms], 
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], 
        buffer_strategy: str = "random"
    ):
        self.max_buffer_size, self.buffer_strategy = max_buffer_size, buffer_strategy
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation
        self.collect = True
        self.training_steps = 0
        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())
        self.logger = setup_logger("step_buffer", ray.get(tracker.get_log_dir.remote()))  # setup logging
        self.episodes: List[List[Step]] = []
        self.mutex = Lock()

    def add_player_trajectory(self, player_traj: PlayerTrajectory, env_id: str):
        episode = []
        reward = self.final_reward_transformation(reward=player_traj.final_reward, pid=player_traj.pid, env_id=env_id) if self.final_reward_transformation else player_traj.final_reward
        for idx in range(len(player_traj.obs)):
            step_reward = self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=reward) if self.step_reward_transformation else reward
            episode.append(Step(pid=player_traj.pid, obs=player_traj.obs[idx], act=player_traj.actions[idx], reward=step_reward, env_id=env_id, step_info={"raw_reward": player_traj.final_reward, "env_reward": reward, "step_reward": step_reward}))
        with self.mutex:
            self.episodes.append(episode)
            excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
            self.logger.info(f"BUFFER NUM of STEP {len(tree.flatten(self.episodes))}")
            while excess_num_samples > 0:
                randm_sampled = random.sample(self.episodes, 1)
                for b in randm_sampled: self.episodes.remove(b)
                excess_num_samples = max(0, len(tree.flatten(self.episodes)) - self.max_buffer_size)
        
    def get_batch(self, batch_size: int) -> List[List[Step]]:
        with self.mutex:
            assert len(tree.flatten(self.episodes)) >= batch_size
            step_count = 0
            sampled_episodes = []
            random.shuffle(self.episodes)
            for ep in self.episodes:
                sampled_episodes.append(ep)
                step_count += len(ep)
                if step_count >= batch_size: break
            for ep in sampled_episodes: self.episodes.remove(ep)
        self.logger.info(f"Sampling {len(sampled_episodes)} episodes from buffer.")
        self.training_steps += 1
        return sampled_episodes

    def stop(self):                 self.collect = False
    def size(self) -> int:          return len(tree.flatten(self.episodes))
    def continue_collection(self):  return self.collect
    def clear(self):
        with self.mutex: 
            self.episodes.clear()

    def compute_rewards_for_logging(self, player_traj: PlayerTrajectory, env_id: str) -> Dict[str, Any]:
        """Return detailed rewards for logging CSVs using this buffer's transforms.

        Returns a dict with keys:
        - raw_final: original env final reward (float)
        - env_final: final reward after final_reward_transformation (float)
        - shaped_per_step: list of shaped rewards per step (List[float])
        """
        raw_final = float(player_traj.final_reward)
        env_final = float(self.final_reward_transformation(reward=raw_final, pid=player_traj.pid, env_id=env_id)) if self.final_reward_transformation else raw_final
        shaped = []
        for idx in range(len(player_traj.obs)):
            if self.step_reward_transformation:
                s = float(self.step_reward_transformation(player_traj=player_traj, step_index=idx, reward=env_final))
            else:
                s = env_final
            shaped.append(s)
        return {"raw_final": raw_final, "env_final": env_final, "shaped_per_step": shaped}
