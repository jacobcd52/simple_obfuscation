# simple_obfuscation – Repository Map for LLM Agents

This README is intentionally optimized for automated reading and parsing by large-language-model agents.  It focuses on **directory-level structure** and the **primary responsibilities** of each component instead of human-oriented prose.

---
## 1. Root-Level Layout

| Path                       | Purpose |
| -------------------------- | ------- |
| `src/`                     | Main python package with all library code.  Import root is `simple_obfuscation`. |
| `scripts/`                 | Human / CI entry-point scripts (e.g. `train.py`). |
| `tests/`                   | Unit tests covering critical utilities and pipelines. |
| `rollouts/`                | (git-ignored)  Serialized rollout data produced during training. |
| `wandb/`                   | (git-ignored)  Weights & Biases logs. |
| `pyproject.toml`           | Build metadata (PEP 621). |
| `requirements.txt`         | Locked dependency list used by `setup.sh`. |
| `README.md`                | You are here. |

> All other hidden folders (`.git/`, `.pytest_cache/`, `.venv/`, etc.) are standard tooling artefacts and contain no project logic.

---
## 2. Source Package – `src/`

```
src/
├── __init__.py              # Exposes top-level public API
├── generation/              # Prompt building and logit manipulation
├── reward/                  # Modular reward functions & registry
├── trainer/                 # REINFORCE trainer and rollout buffer
├── config/                  # Hydra-style configuration objects / YAML
└── utils/                   # Small stateless helpers (token ops, etc.)
```

### 2.1 `generation/`
File | Responsibility
---- | --------------
`prompt_builder.py` | Converts raw tasks (e.g. MMLU questions) into **model-ready prompts**. Also houses configurable templates.
`logit_processors.py` | Custom `transformers.LogitsProcessor` subclasses (e.g. masking, biasing) applied during generation.

### 2.2 `reward/`
File | Responsibility
---- | --------------
`base.py` | Abstract `Reward` base class defining the scoring interface.
`registry.py` | **Dynamic factory** mapping `reward_name ➜ class`.
`boxed_answer_reward.py` | Rewards exact matches inside a bounding box.
`regex_reward.py` | Regex-based matcher for flexible string scoring.
`judge_reward.py` | Delegates scoring to an external LLM “judge”.

### 2.3 `trainer/`
File | Responsibility
---- | --------------
`reinforce_trainer.py` | Core **policy-gradient (REINFORCE)** training loop.
`rollout_store.py` | Simple in-memory buffer → persisted into `rollouts/`.

### 2.4 `config/`
Component | Description
--------- | -----------
`__init__.py` | Pydantic/Hydra config objects with sensible defaults.
`mmlu.yaml` | YAML override for the MMLU benchmarking task.

### 2.5 `utils/`
File | Responsibility
---- | --------------
`token_utils.py` | Miscellaneous helpers for token counting, truncation, etc.

---
## 3. Training Entrypoint

`scripts/train.py`
• Wraps the `reinforce_trainer.ReinforceTrainer` with CLI argument parsing.
• Loads config via `src.config`, initializes reward + generation pipeline, then launches training.

---
## 4. Test Suite

`tests/` contains **pytest**-based validation for registry loading, prompt generation, masking logic, and utility functions.  These tests are valuable examples of expected behaviours.

---
## 5. Quickstart for Agents

1.  Install deps:
    ```bash
    pip install -r requirements.txt
    ```
2.  Launch a dry-run train (small model):
    ```bash
    python scripts/train.py --config-name mmlu
    ```
3.  Rewards, prompts, and configs are **pluggable** – add a new class under the respective sub-package and register it via the exposed registries.

---
## 6. Extensibility Signals

LLM agents can introspect the following single-source-of-truth points:

Key Symbol | Location | Role
-----------|----------|-----
`RewardRegistry` | `src/reward/registry.py` | Maps string → reward class.
`PromptBuilder.generate()` | `src/generation/prompt_builder.py` | Last mile prompt assembly.
`ReinforceTrainer.train()` | `src/trainer/reinforce_trainer.py` | Main training loop.

These are the safest hooks for code modification or reflection.

---
## 7. License

This repository is released under the MIT License (see `pyproject.toml` for metadata).