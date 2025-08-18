#!/usr/bin/env python
"""Standalone *MaskFace* test that runs the models under PyTorch FSDP.

Launch with *torchrun* (recommended):

    torchrun --standalone --nproc_per_node=1 run_mask_face_test_fsdp.py

The script mirrors *run_mask_face_test.py* but:
1. Explicitly loads the mask and face models.
2. Wraps them with :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
3. Passes the wrapped models (and shared tokenizer) to :class:`MaskFace`.
4. Generates a couple of answers and prints the built-in vs. manually decoded
   mask/face generations, asserting they match.

Note: For a single-GPU run, a dummy process group is still initialised so FSDP
can function. On CPU-only environments we fall back to plain (unsharded)
models and print a warning.
"""
from __future__ import annotations

import os
import sys
from typing import List
import functools

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.mask_face import MaskFace


def setup_distributed(device: torch.device) -> None:  # noqa: D401 – helper
    """Initialise a (possibly single-process) torch.distributed group."""
    if dist.is_initialized():
        return  # Already done by *torchrun*

    backend = "nccl" if device.type == "cuda" else "gloo"
    # When the script is run via *torchrun* the necessary env vars are set. If
    # launched directly, we fall back to a single-rank group.
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend=backend, init_method="env://")


@torch.inference_mode()
def main() -> None:  # noqa: D401 – simple script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Distributed / FSDP setup
    # ------------------------------------------------------------------
    if device.type == "cuda":
        setup_distributed(device)
    else:
        print("⚠️  CUDA not available – running without FSDP.")

    model_name = "Qwen/Qwen3-4B"

    # ------------------------------------------------------------------
    # Load base model weights & tokenizer
    # ------------------------------------------------------------------
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    mask_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    face_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)

    # ------------------------------------------------------------------
    # Wrap with FSDP (GPU-only)
    # ------------------------------------------------------------------
    if device.type == "cuda":
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e8))  # heuristic
        mask_model = FSDP(mask_model.to(device), auto_wrap_policy=auto_wrap_policy)
        face_model = FSDP(face_model.to(device), auto_wrap_policy=auto_wrap_policy)
    else:
        mask_model = mask_model.to(device)
        face_model = face_model.to(device)

    # ------------------------------------------------------------------
    # Build MaskFace wrapper using the pre-loaded components
    # ------------------------------------------------------------------
    mask_face = MaskFace(
        mask_model=mask_model,
        face_model=face_model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=2,
        max_thinking_tokens=4,
        min_thinking_tokens=0,
    )

    # ------------------------------------------------------------------
    # Prompts – adapt or expand as you like
    # ------------------------------------------------------------------
    prompts: List[str] = [
        "<user>Hello, how are you?",
        "<user>What is the capital of France?",
    ]

    # ------------------------------------------------------------------
    # Run generation
    # ------------------------------------------------------------------
    out = mask_face.generate(
        prompt_inputs=prompts,
        max_thinking_tokens=4,
        max_new_tokens=8,
    )

    print("\n=== Built-in decoding (FSDP) ===")
    for i, (m, f) in enumerate(zip(out.mask_generations, out.face_generations)):
        print(f"[Sample {i}] MASK : {m}")
        print(f"[Sample {i}] FACE : {f}\n")

    # ------------------------------------------------------------------
    # Manual decoding – apply boolean masks to the generated sequences
    # ------------------------------------------------------------------
    tok = tokenizer

    for i in range(len(prompts)):
        seq = out.sequences[i]  # (T)
        prompt_len = out.prompt_lens[i].item()
        pad_len = out.pad_lens[i]
        eff_prompt_len = prompt_len + pad_len

        gen_tokens = seq[eff_prompt_len:]
        start_idx = eff_prompt_len - 1  # Align with mask indices

        think_slice = out.think_mask[i, start_idx : start_idx + gen_tokens.size(0)]
        face_slice = out.face_mask[i, start_idx : start_idx + gen_tokens.size(0)]

        manual_mask = tok.decode(gen_tokens[think_slice].tolist(), skip_special_tokens=True).strip()
        manual_face = tok.decode(gen_tokens[face_slice].tolist(), skip_special_tokens=True).strip()

        print(f"[Sample {i}] MANUAL MASK : {manual_mask}")
        print(f"[Sample {i}] MANUAL FACE : {manual_face}\n")

        # Consistency checks
        assert manual_mask == out.mask_generations[i], "Mismatch in mask decoding"
        assert manual_face == out.face_generations[i], "Mismatch in face decoding"

    print("All manual decodings match the built-in results! ✅")

    # ------------------------------------------------------------------
    # Clean-up
    # ------------------------------------------------------------------
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)
