from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set

from datasets import load_dataset, Dataset
import unicodedata
import re


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DAFT-Math filtered by excluded questions JSONL.")
    p.add_argument("--hf_dataset", default="metr-evals/daft-math")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--exclude_questions", required=True, help="JSONL with field 'question' to exclude")
    p.add_argument("--output_jsonl", required=True, help="Where to write the filtered dataset as JSONL")
    p.add_argument("--push_to_hub", action="store_true", help="If set, also push to HF Hub")
    p.add_argument("--repo_id", default=None, help="HF repo id for push, e.g. user/daft-math-nohint-default-correct-filtered")
    return p.parse_args()


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s, flags=re.MULTILINE).strip()
    return s


# Strings used by HardMathPromptBuilder.reformat_prompt – we strip them when
# parsing the exclusions file to recover the raw question text.
HARDMATH_PREFIX = (
    "Immediately write the final answer in the format \\boxed{your_final_answer}. "
    "Then write an explanation of exactly how you reasoned to get that answer.\n\n"
)
HARDMATH_SUFFIX = (
    "\n\nYour answer must be in this format:\n\n### Answer\n"
    "\\boxed{your_final_answer}\n\n### Explanation\nyour_explanation"
)

def _strip_hardmath_wrapping(text: str) -> str:
    t = text
    if t.startswith(HARDMATH_PREFIX):
        t = t[len(HARDMATH_PREFIX):]
    if t.endswith(HARDMATH_SUFFIX):
        t = t[: -len(HARDMATH_SUFFIX)]
    return t.strip()


def load_exclusions(path: Path) -> Dict[str, Set[str]]:
    # returns {"q": {questions}, "qa": {question\u0001answer}}
    qset: Set[str] = set()
    qaset: Set[str] = set()
    with path.open() as fp:
        for line in fp:
            try:
                obj = json.loads(line)
                q_raw = str(obj.get("question", "")).strip()
                # Remove training-time wrapper if present, then normalise
                q = _norm_text(_strip_hardmath_wrapping(q_raw))
                a = str(obj.get("answer", "")).strip()
                if q:
                    qset.add(q)
                    if a:
                        qaset.add(q + "\u0001" + a)
            except Exception:
                continue
    return {"q": qset, "qa": qaset}


def main() -> None:
    args = parse_args()
    excl = load_exclusions(Path(args.exclude_questions))

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    keep_indices = []
    num_q_match = 0
    num_qa_match = 0
    for i, ex in enumerate(ds):
        iaq = (ex.get("Integer Answer Variant Question") or "").strip()
        oq_orig = (ex.get("Original Question") or "").strip()
        question = _norm_text(iaq or oq_orig)
        iva = ex.get("Integer Variant Answer")
        oa = ex.get("Original Answer")
        answer = str(iva if (iva is not None and str(iva) != "") else (oa if oa is not None else "")).strip()
        qa_key = question + "\u0001" + answer if answer else None

        if question in excl["q"]:
            num_q_match += 1
            continue
        if qa_key and qa_key in excl["qa"]:
            num_qa_match += 1
            continue
        keep_indices.append(i)

    filtered = ds.select(keep_indices)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        for ex in filtered:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if args.push_to_hub:
        if not args.repo_id:
            raise ValueError("--repo_id must be provided when --push_to_hub is set")
        # Create a datasets Dataset from filtered and push
        ds_filtered = Dataset.from_list(list(filtered))
        ds_filtered.push_to_hub(args.repo_id, split=args.hf_split)

    print(
        f"Filtered {len(ds) - len(filtered)} (question-only matches: {num_q_match}, question+answer matches: {num_qa_match}); "
        f"kept {len(filtered)}. Wrote to {out_path}."
    )


if __name__ == "__main__":
    main()


