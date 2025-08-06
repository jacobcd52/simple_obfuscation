from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils.token_utils import extract_final_output, extract_thinking

class BoxedAnswerReward(RewardFunction):
    """Reward of +1 if the model's output contains the correct boxed answer."""

    name = "boxed_answer_reward"

    def __init__(self, *, coefficient: float = 1.0, log_thinking: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking)

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:  # noqa: D401
        target = rollout["answer"].strip()

        # ------------------------------------------------------------------
        # choose text segment
        # ------------------------------------------------------------------
        if on_thinking:
            segment = extract_thinking(rollout["response"]).strip()
        else:
            segment = extract_final_output(rollout["response"]).strip()

        boxed_target = f"\\boxed{{{target}}}"

        reward_raw = 1.0 if boxed_target in segment else 0.0
        reward = self._post_process(reward_raw)

        # ------------------------------------------------------------------
        # logging – always log the part we just evaluated
        # ------------------------------------------------------------------
        rollout.setdefault("reward_breakdown", {})[self.name if not on_thinking else f"{self.name}_thinking"] = reward

        # Optionally compute & log the *other* part (for analysis only)
        if self.log_thinking and not on_thinking:
            thinking_segment = extract_thinking(rollout["response"]).strip()
            reward_thinking_raw = 1.0 if boxed_target in thinking_segment else 0.0
            rollout["reward_breakdown"][f"{self.name}_thinking"] = self._post_process(reward_thinking_raw)

        return reward
