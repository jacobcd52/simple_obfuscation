from __future__ import annotations

from typing import Dict

from .base import RewardFunction


class TaskReward(RewardFunction):
    name = "task_reward"

    def __call__(self, rollout: Dict) -> float:  # noqa: D401
        target = rollout.get("answer")
        if target is None:
            return 0.0  # no supervision available
        prediction = extract_final_output(rollout["response"]).strip()
        boxed_target = f"\\boxed{{{target.strip()}}}"
        reward = 1.0 if boxed_target in prediction else 0.0
        rollout.setdefault("reward_breakdown", {})[self.name] = reward
        return reward
