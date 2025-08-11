from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils.token_utils import extract_final_output, extract_thinking

class BoxedAnswerReward(RewardFunction):
    """Reward of +1 if the model's output contains the correct boxed answer."""

    name = "boxed_answer_reward"

    def __init__(self, *, coefficient: float = 1.0, log_thinking: bool = False, log_only: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:  # noqa: D401
        target = rollout["target"].strip()

        # ------------------------------------------------------------------
        # choose text segment
        # ------------------------------------------------------------------
        if on_thinking:
            segment = extract_thinking(rollout["response"]).strip()
        else:
            segment = extract_final_output(rollout["response"]).strip()

        boxed_target = f"\\boxed{{{target}}}"

        reward_raw = 1.0 if boxed_target in segment else 0.0
        reward = self._log_reward_values(rollout, reward_raw, on_thinking=on_thinking)

        # Optionally compute & log the *other* part (for analysis only)
        if self.log_thinking and not on_thinking:
            thinking_segment = extract_thinking(rollout["response"]).strip()
            reward_thinking_raw = 1.0 if boxed_target in thinking_segment else 0.0
            _ = self._log_reward_values(rollout, reward_thinking_raw, on_thinking=True)

        # If configured for logging only, do not contribute to training signal
        return reward
