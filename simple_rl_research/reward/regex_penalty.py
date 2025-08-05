"""Regex-based penalty implemented as a *RewardFunction*.

The implementation is directly adapted from the provided specification
and wrapped in a subclass so that the training-loop can treat it like any
other reward.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking

import re


class RegexPenalty(RewardFunction):
    name = "regex_penalty"

    def __init__(
        self,
        *,
        target_words: list[str],
        coefficient: float = 0.1,
        max_penalty: float = 1.0,
        apply_on_thinking: bool = False,
    ) -> None:
        super().__init__(coefficient=coefficient, max_clip=max_penalty)
        self.word_patterns = {
            word: re.compile(r"\\b" + re.escape(word) + r"\\b", re.IGNORECASE)
            for word in target_words
        }
        self.apply_on_thinking = apply_on_thinking

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def __call__(self, rollout: Dict) -> float:  # noqa: D401 – imperative style
        # choose text segment
        if self.apply_on_thinking:
            text_segment = extract_thinking(rollout["response"])
        else:
            text_segment = extract_final_output(rollout["response"])

        total_penalty = 0.0
        for pattern in self.word_patterns.values():
            total_penalty += len(pattern.findall(text_segment))
        total_penalty = self._post_process(total_penalty)
        rollout.setdefault("reward_breakdown", {})[self.name] = -total_penalty
        return -total_penalty
