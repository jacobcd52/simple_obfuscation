"""Regex-based reward implemented as a *RewardFunction*.

The implementation is directly adapted from the provided specification
and wrapped in a subclass so that the training-loop can treat it like any
other reward.
"""

from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking

import re


class RegexReward(RewardFunction):
    """Count occurrences of unwanted *target_words* regex patterns."""

    name = "regex_reward"

    def __init__(
        self,
        *,
        target_words: list[str],
        coefficient: float = 1.0,
        max_clip: float = 1.0,
        log_thinking: bool = False,
        log_only: bool = False,
    ) -> None:
        super().__init__(coefficient=coefficient, max_clip=max_clip, log_thinking=log_thinking, log_only=log_only)
        self.word_patterns = {
            word: re.compile(re.escape(word), re.IGNORECASE)
            for word in target_words
        }

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:  # noqa: D401 – imperative style
        # choose text segment
        if on_thinking:
            text_segment = extract_thinking(rollout["response"])
        else:
            text_segment = extract_final_output(rollout["response"])

        raw_count = sum(len(p.findall(text_segment)) for p in self.word_patterns.values())
        reward = self._post_process(raw_count)

        # logging – record evaluated split
        key = self.name if not on_thinking else f"{self.name}_thinking"
        rollout.setdefault("reward_breakdown", {})[key] = reward

        # Optionally compute and log thinking part
        if self.log_thinking and not on_thinking:
            thinking_text = extract_thinking(rollout["response"])
            raw_thinking = sum(len(p.findall(thinking_text)) for p in self.word_patterns.values())
            rollout["reward_breakdown"][f"{self.name}_thinking"] = self._post_process(raw_thinking)

        # If configured for logging only, do not contribute to training signal
        return 0.0 if self.log_only else reward