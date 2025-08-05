"""PromptBuilder abstractions and a default JSONL implementation."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional

__all__ = [
    "PromptBuilder",
    "JsonlPromptBuilder",
]


class PromptBuilder(ABC):
    """Abstract base-class responsible for turning dataset rows into prompts.

    Sub-classes can read *any* data source.  They must yield dictionaries
    with at minimum a ``prompt`` key containing the text string that gets
    fed into the language-model.  Additional metadata (e.g. ``answer``)
    can be included and will propagate through the rollout – useful for
    computing external metrics.
    """

    def __iter__(self) -> Iterator[Dict]:
        return self.generate()

    @abstractmethod
    def generate(self) -> Iterator[Dict]:
        """Yield prompt dictionaries indefinitely (or until exhaustion)."""

    # Optional – override if the builder supports querying its size
    def __len__(self) -> int:  # noqa: D401 – simple description
        raise TypeError(f"{self.__class__.__name__} has no defined length")

    # ---------------------------------------------------------------------
    # optional convenience helpers
    # ---------------------------------------------------------------------

    def shuffle(self) -> "PromptBuilder":
        """Return in-memory shuffled iterator (useful for small datasets)."""
        return _ShuffledPromptBuilder(self)


class _ShuffledPromptBuilder(PromptBuilder):
    """Lightweight wrapper that shuffles prompts produced by another builder."""

    def __init__(self, inner: PromptBuilder):
        self._inner = inner
        self._cache: Optional[List[Dict]] = None

    def __len__(self) -> int:  # noqa: D401
        # If we've already cached prompts use that size, else delegate
        if self._cache is not None:
            return len(self._cache)
        try:
            return len(self._inner)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError("Length not defined for wrapped PromptBuilder") from None

    def generate(self) -> Iterator[Dict]:
        if self._cache is None:
            self._cache = list(self._inner.generate())
        random.shuffle(self._cache)
        yield from self._cache


# -------------------------------------------------------------------------
# Concrete implementations
# -------------------------------------------------------------------------


class JsonlPromptBuilder(PromptBuilder):
    """Read JSONL file(s) where each line is a JSON dict and format them."""

    def __init__(
        self,
        path: str | Path,
        *,
        input_key: str = "input",
        answer_key: str | None = None,
        prompt_format: str | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.input_key = input_key
        self.answer_key = answer_key
        self.prompt_format = prompt_format or "{input}\n<assistant>"
        self.shuffle = shuffle
        random.seed(seed)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __len__(self) -> int:  # noqa: D401 – simple description
        # Cache length to avoid re-scanning the file repeatedly
        if not hasattr(self, "_len_cache"):
            with self.path.open() as fp:
                self._len_cache = sum(1 for _ in fp)
        return self._len_cache  # type: ignore[attr-defined]

    def _load_lines(self) -> List[dict]:
        data: List[dict] = []
        with self.path.open() as fp:
            for line in fp:
                data.append(json.loads(line))
        if self.shuffle:
            random.shuffle(data)
        return data

    def generate(self) -> Iterator[Dict]:
        data = self._load_lines()
        for row in data:
            prompt = self.prompt_format.format(input=row[self.input_key])
            meta = {}
            if self.answer_key and self.answer_key in row:
                meta["answer"] = row[self.answer_key]
            yield {"prompt": prompt, **meta}
