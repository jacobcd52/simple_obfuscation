from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils.token_utils import extract_final_output

class BoxedAnswerReward(RewardFunction):
    name = "boxed_answer_reward"

    def __call__(self, rollout: Dict) -> float:  # noqa: D401
        target = rollout["answer"]
            
        prediction = extract_final_output(rollout["response"]).strip()
        boxed_target = f"\\boxed{{{target.strip()}}}"
        reward = 1.0 if boxed_target in prediction else 0.0
        rollout.setdefault("reward_breakdown", {})[self.name] = reward
        return reward
