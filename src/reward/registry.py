"""Central registry mapping reward names to classes.

This allows config files to reference rewards via short names instead of
full dotted import paths, e.g. ``rewards: [{cls: task}]``.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import RewardFunction
from .regex_reward import RegexReward
from .judge_reward import JudgeReward
from .judge_tag_reward import TagJudgeReward

from .boxed_answer_reward import BoxedAnswerReward  # noqa: F401  – optional
from .factored_polynomial_reward import FactoredPolynomialReward  # noqa: F401  – optional
from .expanded_polynomial_reward import ExpandedPolynomialReward  # noqa: F401  – optional

_REGISTRY: Dict[str, Type[RewardFunction]] = {
    "regex_reward": RegexReward,
    "judge_reward": JudgeReward,
    "judge_tag_reward": TagJudgeReward,
}

_REGISTRY["boxed_answer"] = BoxedAnswerReward
_REGISTRY["factored_polynomial"] = FactoredPolynomialReward
_REGISTRY["expanded_polynomial"] = ExpandedPolynomialReward


def get_reward_class(name: str) -> Type[RewardFunction]:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward key '{name}'. Available: {list(_REGISTRY)}") from e
