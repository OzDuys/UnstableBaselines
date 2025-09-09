import re
from typing import Tuple, Dict, Callable


def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"
TEMPLATE_PARTS = {
    "default": {
        "user": lambda obs: f"You are playing a two-player zero-sum game. Make valid moves to win. You should first reason about your next move, and then submit the move enclosed by \\boxed{{}}.\nObservation: {obs}\n"
    },
    "qwen3-zs": {
        "user": lambda obs: f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-sp": {
        "user": lambda obs:  f"<|im_start|>user\nYou are playing a single-player game. Make valid actions to solve it completely.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    "qwen3-reasoning": {
        "user": lambda obs: f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {obs}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n<think>"
    },
    "gemma3-zs": {
        "user": lambda obs: f"<bos><start_of_turn>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n"
    },
    "llama-instruct-zs": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are playing a two-player zero-sum game. Make valid actions to win.<|eot_id|>",
        "user": lambda obs: f"<|start_header_id|>user<|end_header_id|>\n\nCurrent Observation: {obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>"
    },
    "qwen3-custom-generic": {
        "user": lambda obs: f"<|im_start|>user\n{obs}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
}
def apply_template(template_name: str, observation: str) -> str:
    parts = TEMPLATE_PARTS.get(template_name)
    return format_template(system=parts.get("system", ""), user=parts["user"](observation), assistant=parts.get("assistant", ""))


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
    if matches:
        last_match = matches[-1].strip()
        if last_match:  # non-empty boxed
            action = f"[{last_match}]" if "[" not in last_match else last_match
            has_think = 1
        else:  # empty boxed
            action = raw_action
            has_think = 0
    else:  # no boxed at all
        action = raw_action
        has_think = 0

    format_feedback = {"correct_answer_format": bool(has_think)}
    return action, format_feedback


def extract_bracket_or_boxed(raw_action: str) -> Tuple[str, Dict[str, bool]]:
        """More permissive extraction.

        Accepts either:
            1. The last /boxed{...} segment (preferred) and normalises to include square brackets.
            2. A bare bracketed action like [A15 B5 C0] if no boxed segment found.
            3. Falls back to the raw_action if neither pattern exists.

        Returns (action, feedback) where feedback includes:
            - correct_answer_format: bool
            - matched: "boxed" | "bracket" | "none"
        """
        # 1. Try boxed pattern first
        boxed_matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
        if boxed_matches:
                content = boxed_matches[-1].strip()
                if content:
                        # Ensure brackets are present
                        action = content if content.startswith("[") else f"[{content}]"
                        return action, {"correct_answer_format": True, "matched": "boxed"}
        # 2. Fallback to a simple bracketed action (avoid grabbing huge reasoning blocks)
        bracket_matches = re.findall(r"\[[^\[\]\n]{1,160}\]", raw_action)
        if bracket_matches:
                action = bracket_matches[-1].strip()
                return action, {"correct_answer_format": True, "matched": "bracket"}
        # 3. Nothing matched
        return raw_action, {"correct_answer_format": False, "matched": "none"}


OBSERVATION_FORMATTING: Dict[str, Callable[[str], str]] = {key: (lambda key=key: lambda observation: apply_template(key, observation))() for key in TEMPLATE_PARTS}
ACTION_EXTRACTION = {"default": extract_action_and_format_feedback, "bracket-or-boxed": extract_bracket_or_boxed}