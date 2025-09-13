import re
from typing import Tuple, Dict, Callable, Any


def format_template(system: str = "", user: str = "", assistant: str = "") -> str: return f"{system}{user}{assistant}"


# --- Helper utilities for prompt formatting ---
def _is_conversation_turn(observation: str) -> bool:
    """Heuristic detection of conversation turns from the observation text.

    Returns True for free-chat/conversation phases, False for decision/answer phases.
    """
    o = observation.lower()
    # Use positions of key phrases; rely on the last mention within the current observation.
    # This helps when the observation text includes both the round-start banner and other messages.
    conv_keys = (
        "you can converse freely",
        "free-chat",
        "free chat",
    )
    # Use concrete decision-phase cues that only appear when conversation ends.
    # Do NOT include generic 'decision turn' from the static round-structure prompt text.
    dec_keys = (
        "chat finished",
        "submit your decisions",
        "submit your decision",
    )

    last_conv = max((o.rfind(k) for k in conv_keys), default=-1)
    last_dec = max((o.rfind(k) for k in dec_keys), default=-1)

    # Conversation iff the latest conversation cue appears after the latest decision cue
    return last_conv > last_dec


def _instruction_line_for_observation(observation: str, default_final: str = "Please reason step by step, and put your final response within \\boxed{}.", default_message: str = "Please reason step by step, and put your message to other players within \\boxed{}.") -> str:
    """Choose instruction line based on whether this looks like a conversation or decision turn."""
    return default_message if _is_conversation_turn(observation) else default_final
TEMPLATE_PARTS = {
    # Keep original default prompt unchanged
    "default": {
        "user": lambda obs: f"\n{obs}\nPlease reason step by step, and put your final response within \\boxed{{}}.\n"
    },
    # Conversation-turn variant for default
    "default-conversation-turn": {
        "user": lambda obs: f"\n{obs}\nPlease reason step by step, and put your message to other players within \\boxed{{}}.\n"
    },
    # Auto variant for default (chooses based on observation)
    "default-auto": {
        "user": lambda obs: (
            TEMPLATE_PARTS["default-conversation-turn"]["user"](obs)
            if _is_conversation_turn(obs)
            else TEMPLATE_PARTS["default"]["user"](obs)
        )
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
    # Keep original qwen3-custom-generic prompt unchanged
    "qwen3-custom-generic": {
        "user": lambda obs: f"<|im_start|>user\n{obs}\nPlease reason step by step, and put your final response within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    # Conversation-turn variant for qwen3-custom-generic
    "qwen3-custom-generic-conversation-turn": {
        "user": lambda obs: f"<|im_start|>user\n{obs}\nPlease reason step by step, and put your message to other players within \\boxed{{}}.<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n"
    },
    # Auto variant for qwen3-custom-generic (chooses based on observation)
    "qwen3-custom-generic-auto": {
        "user": lambda obs: (
            TEMPLATE_PARTS["qwen3-custom-generic-conversation-turn"]["user"](obs)
            if _is_conversation_turn(obs)
            else TEMPLATE_PARTS["qwen3-custom-generic"]["user"](obs)
        ),
        "assistant": "<|im_start|>assistant\n"
    },
}
def apply_template(template_name: str, observation: str) -> str:
    parts = TEMPLATE_PARTS.get(template_name)
    return format_template(system=parts.get("system", ""), user=parts["user"](observation), assistant=parts.get("assistant", ""))


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, Any]]:
    """Default extractor that prefers content inside \\boxed{...}.

    Changes:
    - Supports multiline content inside the boxed segment (DOTALL).
    - Chooses the last non-empty boxed segment as the action (still normalises to []-wrapped).
    - Reports and penalises multiple boxed segments via format_feedback fields.
    """
    # Find all boxed segments, allowing newlines inside
    boxed_contents = re.findall(r"\\boxed\{(.*?)\}", raw_action, flags=re.DOTALL)
    # Count how many boxed openings appear (even if empty) for penalty/feedback
    num_boxed = len(re.findall(r"\\boxed\{", raw_action))

    action = raw_action
    has_think = False

    if boxed_contents:
        # Take the last non-empty boxed content if available; otherwise the last
        non_empty = [c for c in boxed_contents if c.strip()]
        chosen = (non_empty[-1] if non_empty else boxed_contents[-1]).strip()
        if chosen:
            # Ensure brackets are present and keep it on a single line for the action token
            one_line = " ".join(chosen.split())
            action = one_line if one_line.startswith("[") else f"[{one_line}]"
            has_think = True
        else:
            action = raw_action
            has_think = False
    else:
        action = raw_action
        has_think = False

    # Penalise multiple boxed segments to discourage spamming multiple finals
    format_penalty = -0.1 if num_boxed > 1 else 0.0
    format_feedback: Dict[str, Any] = {
        "correct_answer_format": bool(has_think),
        "multiple_boxed": num_boxed > 1,
        "num_boxed": num_boxed,
        "format_penalty": format_penalty,
    }
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