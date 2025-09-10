import json, dataclasses, inspect
from typing import Any, Dict
from unstable._types import TrainEnvSpec, EvalEnvSpec


def _spec_to_dict(spec: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(spec):
        d = dataclasses.asdict(spec)
        # Trim potentially large / unused runtime only fields
        return {k: v for k, v in d.items() if v is not None}
    return {}


def build_experiment_config(*,
    model_name: str,
    lora_cfg: dict|None,
    vllm_cfg: dict|None,
    train_env_specs: list[TrainEnvSpec],
    eval_env_specs: list[EvalEnvSpec],
    learner_hparams: dict|None,
    collector_hparams: dict|None,
    extra: dict|None = None,
) -> Dict[str, Any]:
    """Build a minimal, JSON‑serializable config snapshot for logging.

    Only inexpensive scalar / small list fields should be included.
    """
    cfg = {
        "model_name": model_name,
        "lora": lora_cfg or {},
        "vllm": {k: v for k, v in (vllm_cfg or {}).items() if k != "lora_config"},
        "env_specs": {
            "train": [_spec_to_dict(s) for s in train_env_specs],
            "eval": [_spec_to_dict(s) for s in eval_env_specs],
        },
        "training": {
            "collector": collector_hparams or {},
            "learner": learner_hparams or {},
        }
    }
    if extra: cfg.update(extra)
    return cfg


def to_pretty_json(d: Dict[str, Any]) -> str:
    return json.dumps(d, indent=2, sort_keys=True)
