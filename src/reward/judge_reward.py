"""Judge-based reward implemented as a *RewardFunction* wrapper.

We call an external LLM *judge* to score the assistant's answer between
0 and 1 (reference implementation).  The final reward is ``1 - score`` so
that *higher* values correspond to better answers.

Penalties can be obtained by passing a negative ``coefficient`` during
initialisation.
"""

from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking

import math
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# constants & helper classes (taken from spec, unmodified where possible)
# ---------------------------------------------------------------------------

MAX_SCORE = 1

JUDGE_PROMPT_TEMPLATE = (
    "Here's a conversation: {conversation}\n\n TODO TODO TODO"
)


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


class JudgeReward(RewardFunction):
    name = "judge_reward"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        coefficient: float = 1.0,
        log_thinking: bool = False,
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking)
        self.judge = OpenAiJudge(model, prompt_template)

    # ------------------------------------------------------------------
    # concrete implementation
    # ------------------------------------------------------------------

    async def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:  # noqa: D401 – imperative style
        # choose text segment
        if on_thinking:
            text_segment = extract_thinking(rollout["response"])
        else:
            text_segment = extract_final_output(rollout["response"])

        question = rollout.get("prompt", "")
        # Now we can directly await the judge call since we're async
        try:
            score = await self.judge(conversation=f"Human: {question}\n\nAssistant: {text_segment}")
        except Exception as e:
            print("[JudgeReward] Warning: falling back to 0 –", e)
            score = 0.0

        raw_reward = 1.0 - score  # higher is better
        reward = self._post_process(raw_reward)

        key = self.name if not on_thinking else f"{self.name}_thinking"
        rollout.setdefault("reward_breakdown", {})[key] = reward

        # Optionally compute reward on thinking part for logging
        if self.log_thinking and not on_thinking:
            thinking_text = extract_thinking(rollout["response"])
            try:
                thinking_score = await self.judge(conversation=f"Human: {question}\n\nAssistant: {thinking_text}")
            except Exception:
                thinking_score = 0.0
            raw_thinking = 1.0 - thinking_score
            rollout["reward_breakdown"][f"{self.name}_thinking"] = self._post_process(raw_thinking)

        return reward


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------
JudgePenalty = JudgeReward
