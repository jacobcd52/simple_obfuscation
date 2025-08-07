"""PromptBuilder implementation for the HuggingFace `cais/mmlu` dataset."""

from __future__ import annotations

import random
from typing import Dict, Iterator

from datasets import load_dataset

from .base import PromptBuilder

__all__ = ["MmluPromptBuilder"]

class MmluPromptBuilder(PromptBuilder):
    """PromptBuilder for the HuggingFace `cais/mmlu` dataset."""

    def __init__(
        self,
        subject: str = "abstract_algebra",
        *,
        split: str = "test",
        prompt_format: str | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.subject = subject
        self.split = split
        self.prompt_format = prompt_format or (
            "{question}\n"
            "A. {A}\n"
            "B. {B}\n"
            "C. {C}\n"
            "D. {D}"
        )
        self.shuffle = shuffle
        random.seed(seed)

        # Load the requested subject split from the MMLU dataset
        self._dataset = load_dataset("cais/mmlu", subject, split=split)
        if self.shuffle:
            self._dataset = self._dataset.shuffle(seed=seed)

    def __len__(self) -> int:  # noqa: D401 – simple description
        return len(self._dataset)

    def reformat_prompt(self, prompt: str) -> str:
        """Reformat the prompt before yielding it."""
        prefix = (
            "Immediately write the letter A, B, C, or D corresponding to your final answer, "
            "in the format \\boxed{your_answer_letter}. Then write a brief explanation of how "
            "you got that answer.\n\n"
        )
        suffix = (
            "\n\nYour answer must be in this format:\n\n### Answer\n"
            "\\boxed{your_answer_letter}\n\n### Explanation\nyour_explanation"
        )
        return prefix + prompt + suffix

    def generate(self) -> Iterator[Dict]:
        for row in self._dataset:
            choices = row["choices"]
            answer_label = row["answer"]  # int 0-3 or letters "A"-"D"
            if isinstance(answer_label, str):
                answer_idx = "ABCD".index(answer_label.upper())
            else:
                answer_idx = int(answer_label)
            prompt = self.prompt_format.format(
                input=row["question"],  # maintain compatibility with other builders
                question=row["question"],
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
            )
            yield {
                "prompt": prompt,
                "answer": "ABCD"[answer_idx],  # use letter answer for reward
                "answer_idx": answer_idx,
                "answer_text": choices[answer_idx],
            }

