"""Integration tests that spawn multiple processes to exercise real distributed
training (DDP and FSDP).

These are intentionally heavier than the single-process smoke test in
`tests/test_distributed_training.py`.  They launch a *local* 2-process
process-group, run one forward/backward/optimiser step using the tiny GPT-2
model and exit.  The goal is to catch issues that only appear when collective
communication is involved (broadcasts, all-reduces, FSDP all-gathers, etc.).

The test is kept lightweight by:
* using `sshleifer/tiny-gpt2` (≈25 MB) so load time is minimal;
* running just **one** optimisation step; and
* defaulting to the "gloo" backend so it works on CPU-only CI runners.

Runtime on a typical CPU box is ~10-15 s, acceptable for the regular test
suite, but mark the test as `slow` if your project separates Quick / Slow CI
jobs.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Literal

import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from src.config import TrainConfig
from src.trainer.reinforce_trainer import ReinforceTrainer

# Ensure a safe start-method for PyTorch multiprocess
mp.set_start_method("spawn", force=True)

# -----------------------------------------------------------------------------
# Helper – worker function run in each spawned process
# -----------------------------------------------------------------------------

def _dist_worker(rank: int, world_size: int, backend: str, port: int, multi_gpu: Literal["ddp", "fsdp"]) -> None:  # noqa: D401
    """Run one tiny optimisation step inside a distributed process-group.

    The function is process-safe (top-level, picklable) so it can be used with
    `torch.multiprocessing.spawn`.
    """

    # ------------------------------------------------------------------
    # distributed setup – env vars expected by both PyTorch & 🤗 Accelerate
    # ------------------------------------------------------------------
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),  # single-node launch ⇒ local_rank == rank
            "WORLD_SIZE": str(world_size),
            # Disable tqdm/rich output to keep pytest logs clean
            "ACCELERATE_DISABLE_RICH": "1",
        }
    )

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # ------------------------------------------------------------------
    # minimal prompt builder data – write a temporary JSONL file
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        json.dump({"input": "Hello world"}, tmp)
        tmp.write("\n")
        prompt_path = tmp.name

    # ------------------------------------------------------------------
    # Build TrainConfig & Trainer
    # ------------------------------------------------------------------
    cfg = TrainConfig(
        model_name="sshleifer/tiny-gpt2",
        batch_size=1,
        learning_rate=1e-4,
        multi_gpu=multi_gpu,
        prompt_builder_cls="src.prompt_builders.jsonl_prompt_builder.JsonlPromptBuilder",
        prompt_builder_params={"path": prompt_path},
    )

    trainer = ReinforceTrainer(cfg)

    # ------------------------------------------------------------------
    # one forward/backward/step – identical to the single-process smoke test
    # ------------------------------------------------------------------
    trainer.model.train()
    tok = trainer.tokenizer
    inputs = tok("Hello world", return_tensors="pt")
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

    outputs = trainer.model(**inputs)
    loss = outputs.logits.mean()

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    # ------------------------------------------------------------------
    # simple collective to ensure communication works
    # ------------------------------------------------------------------
    # Each rank contributes its scalar loss; after all-reduce every rank should
    # see the *sum* – we just check the op completes without hanging.
    scalar = torch.tensor([loss.item()], device=trainer.model.device)
    dist.all_reduce(scalar)

    # clean-up
    dist.destroy_process_group()


# -----------------------------------------------------------------------------
# PyTest entry-point
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("multi_gpu", ["ddp", "fsdp"])
@pytest.mark.skipif(not dist.is_available(), reason="PyTorch distributed not available")
def test_distributed_two_processes(multi_gpu: str):
    """Run a *real* 2-process distributed step for DDP and FSDP.

    The test is skipped if `torch.distributed` is not compiled.
    """

    world_size = 2
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Pick a random free port in a high-range to avoid conflicts in CI.
    port = random.randint(10_000, 20_000)

    # `spawn` blocks until all ranks exit (or an exception propagates).
    mp.spawn(
        _dist_worker,
        args=(world_size, backend, port, multi_gpu),
        nprocs=world_size,
        join=True,
    )
