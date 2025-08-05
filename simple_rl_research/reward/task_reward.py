"""Simple *TaskReward* placeholder.

For demonstration we implement exact-match reward between model output
and a reference answer present in the rollout (key: ``answer``).
"""

from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils import extract_final_output


class TaskReward(RewardFunction):
    name = "task_reward"

    def __call__(self, rollout: Dict) -> float:  # noqa: D401
        target = rollout.get("answer")
        if target is None:
            return 0.0  # no supervision available
        prediction = extract_final_output(rollout["response"]).strip()
        reward = 1.0 if prediction == target.strip() else 0.0
        rollout.setdefault("reward_breakdown", {})[self.name] = reward
        return reward
