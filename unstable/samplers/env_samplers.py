import random
from typing import Protocol, List, Any
from unstable._types import TrainEnvSpec, EvalEnvSpec, GameInformation


class BaseEnvSampler: 
    def __init__(self, train_env_specs: List[TrainEnvSpec], eval_env_specs: List[EvalEnvSpec]|None = None, rng_seed: int|None = 489):
        assert train_env_specs, "Need at least one training environment"
        self._train, self._eval = train_env_specs, eval_env_specs
        self._rng = random.Random(rng_seed)
    def env_list(self) -> str: return ",".join([tes.env_id for tes in self._train])
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec: ...
    def update(self, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: ...
    # Accessors for config logging (avoid touching private attrs elsewhere)
    def get_train_specs(self) -> List[TrainEnvSpec]: return list(self._train)
    def get_eval_specs(self) -> List[EvalEnvSpec]: return list(self._eval) if self._eval else []

class UniformRandomEnvSampler(BaseEnvSampler):
    def sample(self, kind: str = "train") -> TrainEnvSpec | EvalEnvSpec:
        if kind not in ("train", "eval"): raise ValueError(f"kind must be 'train' or 'eval' (got {kind!r})")
        return self._rng.choice(self._train if kind == "train" else self._eval)
    def update(self, avg_actor_reward: float, avg_opponent_reward: float|None) -> None: return # No-op for the uniform strategy - but the hook is here so the scheduler can call it unconditionally.
