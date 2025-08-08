"""PromptBuilder for arXiv ML papers prepared by the dataset builder script.

This builder expects a manifest JSONL file created by
``scripts/build_arxiv_ml_dataset.py``. Each JSON line must contain at least:

- ``arxiv_id``: the arXiv identifier (e.g., ``2403.01234``)
- ``title``: paper title
- ``authors``: list of author names
- ``categories``: space-separated category string from arXiv metadata
- ``text_path``: absolute or repo-relative path to extracted plain text

The builder constructs a review prompt for each paper according to a rubric,
and yields dictionaries with a ``prompt`` key, plus metadata passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .base import PromptBuilder

__all__ = ["ArxivMlPromptBuilder", "ArxivReviewRubric"]


@dataclass
class ArxivReviewRubric:
    """Container for rubric text and output format expectations."""

    rubric_text: str
    output_format: str

    @staticmethod
    def default() -> "ArxivReviewRubric":
        rubric = (
            "You are a conference reviewer. Read the full paper and provide a structured "
            "review. Evaluate the paper on these axes (0-10, with 10 best):\n"
            "- Originality/Novelty\n"
            "- Technical Quality/Soundness\n"
            "- Clarity/Presentation\n"
            "- Significance/Impact\n"
            "Also list strengths, weaknesses, questions for authors, and a brief summary.\n"
            "Finally, provide an overall recommendation with one of: Strong Accept, Accept, "
            "Weak Accept, Borderline, Weak Reject, Reject, Strong Reject."
        )
        output = (
            "Your answer must use exactly this format:\n\n"
            "### Summary\n"
            "<2-5 sentence summary>\n\n"
            "### Strengths\n"
            "- <point 1>\n- <point 2>\n\n"
            "### Weaknesses\n"
            "- <point 1>\n- <point 2>\n\n"
            "### Questions for Authors\n"
            "- <question 1>\n- <question 2>\n\n"
            "### Scores (0-10)\n"
            "- Originality: <score>\n"
            "- Technical Quality: <score>\n"
            "- Clarity: <score>\n"
            "- Significance: <score>\n\n"
            "### Overall Recommendation\n"
            "<one of: Strong Accept | Accept | Weak Accept | Borderline | Weak Reject | Reject | Strong Reject>\n"
        )
        return ArxivReviewRubric(rubric_text=rubric, output_format=output)


class ArxivMlPromptBuilder(PromptBuilder):
    """PromptBuilder for arXiv ML papers extracted to a local dataset directory.

    Parameters
    ----------
    manifest_path:
        Path to the JSONL manifest produced by the dataset build script.
    prompt_format:
        Optional format string for the base prompt. If omitted, a default
        format including title, authors, categories, and full text is used.
    rubric:
        Optional rubric specification. Defaults to a comprehensive review rubric.
    limit:
        Optional maximum number of examples to yield (useful for quick tests).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        prompt_format: Optional[str] = None,
        rubric: Optional[ArxivReviewRubric] = None,
        limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._manifest_path = Path(manifest_path)
        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")
        self._prompt_format = (
            prompt_format
            or (
                "You will review a machine learning paper. "
                "Provide a thorough, fair, and evidence-based review.\n\n"
                "Title: {title}\n"
                "Authors: {authors}\n"
                "Categories: {categories}\n"
                "arXiv ID: {arxiv_id}\n"
                "URL: https://arxiv.org/abs/{arxiv_id}\n\n"
                "--- Full Paper (extracted text) ---\n"
                "{text}\n"
            )
        )
        self._rubric = rubric or ArxivReviewRubric.default()
        self._limit = limit

        # Cache length lazily after first computation
        self._length_cache: Optional[int] = None

    def __len__(self) -> int:  # noqa: D401 – simple description
        if self._length_cache is None:
            # Count lines in manifest
            count = 0
            with self._manifest_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            self._length_cache = count
        return self._length_cache

    def reformat_prompt(self, prompt: str) -> str:  # noqa: D401 – simple description
        prefix = self._rubric.rubric_text + "\n\n"
        suffix = "\n\n" + self._rubric.output_format
        return prefix + prompt + suffix

    def generate(self) -> Iterator[Dict]:
        yielded = 0
        with self._manifest_path.open("r", encoding="utf-8") as manifest_file:
            for line in manifest_file:
                if self._limit is not None and yielded >= self._limit:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                text_path = Path(record.get("text_path", "")).expanduser()
                if not text_path.is_absolute():
                    # Resolve relative to manifest directory
                    text_path = (self._manifest_path.parent / text_path).resolve()
                try:
                    text = text_path.read_text(encoding="utf-8")
                except Exception:
                    # Skip corrupted or missing texts
                    continue

                authors: List[str] = record.get("authors", [])
                prompt = self._prompt_format.format(
                    title=record.get("title", ""),
                    authors=", ".join(authors) if authors else "Unknown",
                    categories=record.get("categories", ""),
                    arxiv_id=record.get("arxiv_id", ""),
                    text=text,
                )

                item: Dict = {
                    "prompt": prompt,
                    # pass-through metadata for downstream analysis
                    "arxiv_id": record.get("arxiv_id"),
                    "title": record.get("title"),
                    "authors": authors,
                    "categories": record.get("categories"),
                    "text_path": str(text_path),
                }
                yielded += 1
                yield item


