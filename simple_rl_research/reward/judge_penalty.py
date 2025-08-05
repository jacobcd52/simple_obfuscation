"""Judge-based penalty as RewardFunction wrapper.

We largely embed the reference implementation verbatim.  Only minor
changes: expose *apply_on_thinking* flag and conform to the RewardFunction
interface so that the training-loop can treat it uniformly.
"""

from __future__ import annotations

from typing import Dict, Any

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking

import asyncio
import math
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# constants & helper classes (taken from spec, unmodified where possible)
# ---------------------------------------------------------------------------

MAX_SCORE = 1

JUDGE_PROMPT_TEMPLATE = (
    "Here's a conversation: {conversation}\n\n TODO TODO TODO"
)

JUDGE_PROMPT_TEMPLATE_COT = JUDGE_PROMPT_TEMPLATE


class OpenAiJudge:
    """Lightweight async OpenAI wrapper."""

    def __init__(self, model: str, prompt_template: str | None = None):
        self.model = model
        self.prompt_template = prompt_template or JUDGE_PROMPT_TEMPLATE
        self.client = AsyncOpenAI()

    async def __call__(self, **kwargs):
        messages = [dict(role="user", content=self.prompt_template.format(**kwargs))]
        logprobs = await self._logprob_probs(messages)
        return self._aggregate_score(logprobs)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    async def _logprob_probs(self, messages):
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            lp = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            return {}
        return {el.token: float(math.exp(el.logprob)) for el in lp}

    def _aggregate_score(self, score_dict):
        total_weight = sum(score_dict.values())
        if total_weight < 0.25:
            return 0.0
        weighted_sum = 0.0
        for token, prob in score_dict.items():
            try:
                weighted_sum += float(token) * prob
            except ValueError:
                continue
        return weighted_sum


# ---------------------------------------------------------------------------
# RewardFunction wrapper
# ---------------------------------------------------------------------------


class JudgePenalty(RewardFunction):
    name = "judge_penalty"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        coefficient: float = 0.1,
        apply_on_thinking: bool = False,
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(coefficient=coefficient, max_clip=None)
        self.judge = OpenAiJudge(model, prompt_template)
        self.apply_on_thinking = apply_on_thinking

    # ------------------------------------------------------------------
    # concrete implementation
    # ------------------------------------------------------------------

    async def __call__(self, rollout: Dict) -> float:  # noqa: D401 – imperative style
        # choose text segment
        if self.apply_on_thinking:
            text_segment = extract_thinking(rollout["response"])
        else:
            text_segment = extract_final_output(rollout["response"])

        question = rollout.get("prompt", "")
        # Now we can directly await the judge call since we're async
        try:
            score = await self.judge(conversation=f"Human: {question}\n\nAssistant: {text_segment}")
        except Exception as e:
            print("[JudgePenalty] Warning: falling back to 0 –", e)
            score = 0.0

        inverted = 1.0 - score
        penalty = self._post_process(inverted)
        rollout.setdefault("reward_breakdown", {})[self.name] = -penalty
        return -penalty
