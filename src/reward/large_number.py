from __future__ import annotations

import re
from typing import Dict

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", flags=re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _boxed_segments(text: str) -> list[str]:
    return [m.group(1) for m in _BOXED_RE.finditer(text)]


def _has_number_gt_1000(s: str) -> bool:
    for tok in _NUMBER_RE.findall(s):
        try:
            if float(tok) > 1000.0:
                return True
        except ValueError:
            continue
    return False


class LargeNumberReward(RewardFunction):
    name = "large_number_reward"

    def __init__(self, *, coefficient: float = 1.0, log_thinking: bool = False, log_only: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:
        segment = extract_thinking(rollout["response"]) if on_thinking else extract_final_output(rollout["response"])
        boxed_parts = _boxed_segments(segment)

        raw = 1.0 if boxed_parts and any(_has_number_gt_1000(part) for part in boxed_parts) else 0.0
        reward = self._log_reward_values(rollout, raw, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            thinking_segment = extract_thinking(rollout["response"])            
            boxed_thinking = _boxed_segments(thinking_segment)
            raw_thinking = 1.0 if boxed_thinking and any(_has_number_gt_1000(part) for part in boxed_thinking) else 0.0
            _ = self._log_reward_values(rollout, raw_thinking, on_thinking=True)

        return reward


