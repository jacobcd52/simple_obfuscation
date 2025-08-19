from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf
from src.config import TrainConfig, RewardConfig
from src.trainer.reinforce_trainer import ReinforceTrainer
from src.utils.token_utils import extract_final_output


HARDMATH_PREFIX = (
    "Immediately write the final answer in the format \\\\boxed{your_final_answer}. "
    "Then write an explanation of exactly how you reasoned to get that answer.\n\n"
)

HARDMATH_SUFFIX = (
    "\n\nYour answer must be in this format:\n\n### Answer\n"
    "\\\\boxed{your_final_answer}\n\n### Explanation\nyour_explanation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model on HardMath without hints for a number of episodes, "
            "then output a JSONL of problems the model answered correctly at least once."
        )
    )
    parser.add_argument("--config", default="hardmath.yaml", help="Config YAML to use as defaults (relative to src/config if not a path)")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes (prompts) to process")
    parser.add_argument("--output", type=str, required=True, help="Path to write filtered JSONL dataset")
    parser.add_argument("--hf_dataset", default=None, help="HardMath HF dataset name (override)")
    parser.add_argument("--hf_split", default=None, help="HF dataset split (override)")
    parser.add_argument("--use_model_filtered_dataset", action="store_true", help="Use lukemarks/daft-math-filtered-<model>")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (override)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for builder shuffling (override)")
    parser.add_argument("--thinking_max_tokens", type=int, default=None, help="Max thinking tokens (override)")
    parser.add_argument("--output_max_tokens", type=int, default=None, help="Max new tokens during generation (override)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (override)")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (override)")
    parser.add_argument("--multi_gpu", choices=["none", "ddp", "fsdp"], default=None, help="Distributed strategy (override)")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "disabled"), help="WandB mode: online/offline/disabled")
    parser.add_argument("--exclude_questions", default=None, help="Path to JSONL of questions to exclude during training (auto-filter)")
    return parser.parse_args()


def extract_question_from_prompt(prompt: str) -> str:
    """Reverse the formatting applied by HardMathPromptBuilder.reformat_prompt.

    Assumes hints were disabled (so the raw prompt is just the question text).
    """
    text = prompt
    if text.startswith(HARDMATH_PREFIX):
        text = text[len(HARDMATH_PREFIX):]
    if text.endswith(HARDMATH_SUFFIX):
        text = text[: -len(HARDMATH_SUFFIX)]
    return text.strip()


def rollout_is_correct(rollout: Dict) -> bool:
    """Return True if the rollout's final output contains the correct boxed answer."""
    target = str(rollout.get("target", "")).strip()
    if target == "":
        return False
    final_segment = extract_final_output(rollout.get("response", ""))
    boxed = f"\\boxed{{{target}}}"
    return boxed in final_segment


def collect_correct_problems(rollouts_dir: Path) -> List[Dict]:
    """Scan rollouts and collect unique problems that were ever answered correctly."""
    # Aggregate per-question stats to ensure uniqueness
    stats: Dict[str, Dict] = {}
    for file in sorted(rollouts_dir.glob("rollout_*.json")):
        with file.open() as fp:
            r = json.load(fp)
        question = extract_question_from_prompt(r.get("prompt", ""))
        key = question
        entry = stats.get(key)
        if entry is None:
            entry = {
                "question": question,
                "answer": str(r.get("target", "")).strip(),
                "difficulty": r.get("difficulty"),
                "source": r.get("source"),
                "competition": r.get("competition"),
                "num_attempted": 0,
                "num_correct": 0,
            }
            stats[key] = entry
        entry["num_attempted"] += 1
        if rollout_is_correct(r):
            entry["num_correct"] += 1

    # Return only those correct at least once, dropping counters
    filtered: List[Dict] = []
    for v in stats.values():
        if v["num_correct"] >= 1:
            filtered.append(
                {
                    "question": v["question"],
                    "answer": v["answer"],
                    "difficulty": v["difficulty"],
                    "source": v["source"],
                    "competition": v["competition"],
                }
            )
    return filtered


def main() -> None:
    args = parse_args()

    # Configure WandB mode early
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    # Load defaults from YAML, then override
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        pkg_cfg_dir = Path(__file__).resolve().parent.parent / "src" / "config"
        candidate = pkg_cfg_dir / config_path
        if candidate.exists():
            config_path = candidate
    if config_path.exists():
        user_cfg_raw = OmegaConf.load(config_path)
        user_cfg = OmegaConf.to_container(user_cfg_raw, resolve=True)  # type: ignore[assignment]
    else:
        user_cfg = {}

    # Ensure HardMath builder with no hints
    user_cfg["prompt_builder_cls"] = "src.prompt_builders.hardmath.HardMathPromptBuilder"
    pb_params = dict(user_cfg.get("prompt_builder_params", {}))
    pb_params["enable_hints"] = False
    if args.use_model_filtered_dataset:
        pb_params["use_model_filtered_dataset"] = True
        pb_params["model_name"] = args.model
    if args.hf_dataset is not None:
        pb_params["hf_dataset"] = args.hf_dataset
    if args.hf_split is not None:
        pb_params["hf_split"] = args.hf_split
    if args.seed is not None:
        pb_params["seed"] = args.seed
    pb_params.setdefault("shuffle", True)
    if args.exclude_questions is not None:
        pb_params["exclude_questions_path"] = args.exclude_questions
    user_cfg["prompt_builder_params"] = pb_params

    # Required overrides
    user_cfg["model_name"] = args.model
    user_cfg["num_episodes"] = int(args.episodes)

    # Optional overrides
    if args.batch_size is not None:
        user_cfg["batch_size"] = int(args.batch_size)
    if args.output_max_tokens is not None:
        user_cfg["output_max_tokens"] = int(args.output_max_tokens)
    if args.thinking_max_tokens is not None:
        user_cfg["thinking_max_tokens"] = int(args.thinking_max_tokens)
    if args.lr is not None:
        user_cfg["learning_rate"] = float(args.lr)
    if args.weight_decay is not None:
        user_cfg["weight_decay"] = float(args.weight_decay)
    if args.multi_gpu is not None:
        user_cfg["multi_gpu"] = args.multi_gpu

    # Force no penalty: only boxed answer reward
    user_cfg["rewards"] = [
        RewardConfig(cls="boxed_answer", params={"coefficient": 1.0, "log_thinking": False})
    ]

    # Build training config
    cfg = TrainConfig(**user_cfg)
    cfg.wandb_run_name = cfg.wandb_run_name or f"hardmath-nohint-filter-{Path(args.model).name}"
    cfg.save_rollouts_to_wandb = False

    trainer = ReinforceTrainer(cfg)
    trainer.train()

    # Collect correct problems and write JSONL
    rollouts_dir = Path("rollouts")
    filtered = collect_correct_problems(rollouts_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        for row in filtered:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(filtered)} problems answered correctly at least once to {output_path}")


if __name__ == "__main__":
    main()


