"""Central registry mapping reward names to classes.

This allows config files to reference rewards via short names instead of
full dotted import paths, e.g. ``rewards: [{cls: task}]``.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import RewardFunction
from .regex_penalty import RegexPenalty
from .judge_penalty import JudgePenalty

try:
    from .task_reward import TaskReward  # noqa: F401  – optional
except ImportError:  # pragma: no cover
    TaskReward = None  # type: ignore

_REGISTRY: Dict[str, Type[RewardFunction]] = {
    "regex_penalty": RegexPenalty,
    "judge_penalty": JudgePenalty,
}

if TaskReward is not None:
    _REGISTRY["task"] = TaskReward


def get_reward_class(name: str) -> Type[RewardFunction]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward key '{name}'. Available: {list(_REGISTRY)}") from e
