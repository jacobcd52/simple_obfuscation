"""Abstract base class for prompt builders.

This module contains the core `PromptBuilder` abstraction and helper
utilities. Concrete implementations should reside in sibling modules
within this package and subclass `PromptBuilder`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional

__all__ = [
    "PromptBuilder",
]


class PromptBuilder(ABC):
    """Abstract base-class responsible for turning dataset rows into prompts.

    Sub-classes can read *any* data source. They must yield dictionaries
    with at minimum a ``prompt`` key containing the text string that gets
    fed into the language-model. Additional metadata (e.g. ``answer``)
    can be included and will propagate through the rollout – useful for
    computing external metrics.
    """

    def __iter__(self) -> Iterator[Dict]:
        for item in self.generate():
            item["prompt"] = self.reformat_prompt(item["prompt"])
            yield item

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self) -> Iterator[Dict]:
        """Yield prompt dictionaries indefinitely (or until exhaustion)."""

    def reformat_prompt(self, prompt: str) -> str:  # noqa: D401 – simple description
        """Reformat the prompt before yielding it."""
        return prompt

    # Optional – override if the builder supports querying its size
    def __len__(self) -> int:  # noqa: D401 – simple description
        raise TypeError(f"{self.__class__.__name__} has no defined length")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def shuffle(self) -> "PromptBuilder":
        """Return in-memory shuffled iterator (useful for small datasets)."""
        return _ShuffledPromptBuilder(self)


class _ShuffledPromptBuilder(PromptBuilder):
    """Lightweight wrapper that shuffles prompts produced by another builder."""

    def __init__(self, inner: PromptBuilder):
        self._inner = inner
        self._cache: Optional[List[Dict]] = None

    def __len__(self) -> int:  # noqa: D401 – simple description
        # If we've already cached prompts use that size, else delegate
        if self._cache is not None:
            return len(self._cache)
        try:
            return len(self._inner)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError("Length not defined for wrapped PromptBuilder") from None

    def reformat_prompt(self, prompt: str) -> str:
        return self._inner.reformat_prompt(prompt)

    def generate(self) -> Iterator[Dict]:
        if self._cache is None:
            self._cache = list(self._inner.generate())
        random.shuffle(self._cache)
        yield from self._cache

