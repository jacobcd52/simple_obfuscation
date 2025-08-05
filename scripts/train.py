#!/usr/bin/env python
"""CLI wrapper around *ReinforceTrainer*.

Usage (install `uv` first)::

    uv venv && source .venv/bin/activate
    uv pip install -r requirements.txt

Run training::

    python scripts/train.py
"""

from __future__ import annotations

import argparse

from simple_rl_research.config import TrainConfig
from simple_rl_research.trainer.reinforce_trainer import ReinforceTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to yaml / json config file", default=None)
    return parser.parse_args()


def main():
    _ = parse_args()
    cfg = TrainConfig()  # default config – user can extend via hydra later
    trainer = ReinforceTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
