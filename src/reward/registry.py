"""Central registry mapping reward names to classes.

This allows config files to reference rewards via short names instead of
full dotted import paths, e.g. ``rewards: [{cls: task}]``.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import RewardFunction
from .regex_penalty import RegexPenalty
from .judge_penalty import JudgePenalty

from .boxed_answer_reward import BoxedAnswerReward  # noqa: F401  – optional

_REGISTRY: Dict[str, Type[RewardFunction]] = {
    "regex_penalty": RegexPenalty,
    "judge_penalty": JudgePenalty,
}

_REGISTRY["boxed_answer"] = BoxedAnswerReward


def get_reward_class(name: str) -> Type[RewardFunction]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward key '{name}'. Available: {list(_REGISTRY)}") from e
