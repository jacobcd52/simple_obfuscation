"""Tag-based judge reward that outputs a fixed token budget and <score>INT</score>.

This module provides:
- OpenAiTagJudge: runs an OpenAI chat completion with a token budget and enforces
  a single `<score>INT</score>` tag in the output.
- TagJudgeReward: a RewardFunction wrapper that returns logit(INT/100) as the raw score
  and integrates with the same template selection and logging as JudgeReward.
"""

from __future__ import annotations

from typing import Dict

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking

from openai import AsyncOpenAI

# Reuse templates and hint stripping from the base judge module to guarantee
# identical phrasing/logic.
from .judge_reward import JUDGE_PROMPT_TEMPLATES, _strip_hardmath_hint


class OpenAiTagJudge:
    """Async OpenAI wrapper that extracts a tag-based integer score and returns its logit.

    Parameters
    ----------
    model
        OpenAI chat model name.
    prompt_template
        Template string with placeholders ``{prompt}`` and ``{response}``.
    budget_tokens
        Maximum number of tokens the judge may generate (including the final score tag).
    """

    def __init__(self, model: str, prompt_template: str | None = None, *, budget_tokens: int = 64):
        self.model = model
        self.prompt_template = prompt_template
        self.budget_tokens = int(budget_tokens)
        self.client = AsyncOpenAI()

    async def __call__(self, **kwargs) -> float:
        # Augment the template with explicit emission rules for the score tag.
        core = self.prompt_template.format(**kwargs)
        instruction = (
            "\n\n"
            "You may write brief reasoning, but you MUST end your reply with a line that is exactly one tag "
            "<score>INT</score> where INT is 0 or 1."
            "Only one <score> tag is allowed."
        )
        messages = [dict(role="user", content=f"{core}{instruction}")]

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.budget_tokens,
                temperature=0,
                seed=0,
            )
            content = completion.choices[0].message.content or ""
        except Exception:
            return 0.0

        # Extract <score>INT</score>
        import re
        match = re.search(r"<score>\s*(\d{1,3})\s*</score>", content)
        if not match:
            return 0.0
        try:
            score_int = int(match.group(1))
        except ValueError:
            return 0.0
        score_int = max(0, min(100, score_int))

        # Convert to logit of probability in (0,1)
        import math as _math
        p = score_int / 100.0
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        logit = _math.log(p / (1 - p))
        return float(logit)


class TagJudgeReward(RewardFunction):
    """Judge reward variant that reads a tag-based integer score and returns its logit.

    Identical usage to ``JudgeReward`` with an additional parameter ``budget_tokens``:

    ```yaml
    rewards:
      - cls: "judge_tag_reward"
        judge_prompt_template: "hard_math"  # REQUIRED
        params:
          model: "gpt-4o-mini"
          budget_tokens: 64
          coefficient: 1.0
    ```
    """

    name = "judge_tag_reward"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        coefficient: float = 1.0,
        log_thinking: bool = False,
        log_only: bool = False,
        prompt_template: str | None = None,
        judge_prompt_template: str | None = None,
        budget_tokens: int = 64,
    ) -> None:
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)

        if judge_prompt_template is None:
            raise ValueError(
                f"judge_prompt_template must be specified for {self.__class__.__name__}. "
                f"Available options: {list(JUDGE_PROMPT_TEMPLATES.keys())}"
            )
        if judge_prompt_template not in JUDGE_PROMPT_TEMPLATES:
            raise ValueError(
                f"Invalid judge_prompt_template '{judge_prompt_template}' for {self.__class__.__name__}. "
                f"Available options: {list(JUDGE_PROMPT_TEMPLATES.keys())}"
            )

        selected_template = JUDGE_PROMPT_TEMPLATES[judge_prompt_template]
        self.judge = OpenAiTagJudge(model, selected_template, budget_tokens=budget_tokens)

    async def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:
        # choose text segment
        if on_thinking:
            text_segment = extract_thinking(rollout["response"])
        else:
            text_segment = extract_final_output(rollout["response"])

        question = rollout.get("prompt", "")
        question_for_judge = _strip_hardmath_hint(question)
        try:
            score_logit = await self.judge(prompt=question_for_judge, response=text_segment)
        except Exception as e:
            print("[TagJudgeReward] Warning: falling back to 0 –", e)
            score_logit = 0.0

        reward = self._log_reward_values(rollout, score_logit, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            thinking_text = extract_thinking(rollout["response"])
            try:
                thinking_score = await self.judge(prompt=question_for_judge, response=thinking_text)
            except Exception:
                thinking_score = 0.0
            _ = self._log_reward_values(rollout, thinking_score, on_thinking=True)

        return reward


