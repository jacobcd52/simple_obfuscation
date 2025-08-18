#!/usr/bin/env python
"""Standalone sanity test for the *MaskFace* wrapper.

Run with:
    python run_mask_face_test.py

This script loads the same model (Qwen/Qwen3-4B) for both the *mask* and
*face* components, generates a couple of sample answers and prints:

1. The chain-of-thought (*mask*) and visible answer (*face*) extracted
   via the built-in helpers returned by :py:meth:`MaskFace.generate`.
2. The *same* two pieces of text obtained manually by applying the masks
   to the raw token sequences and decoding the resulting IDs.

Both decoding paths should yield identical strings; an assertion checks
this invariant for each sample.
"""
from __future__ import annotations

from typing import List

import torch

from src.models.mask_face import MaskFace


@torch.inference_mode()
def main() -> None:  # noqa: D401 – simple script
    # ------------------------------------------------------------------
    # Initialise MaskFace with identical *mask* and *face* models
    # ------------------------------------------------------------------
    model_name = "Qwen/Qwen3-4B"

    mask_face = MaskFace(
        mask_model_name=model_name,
        face_model_name=model_name,
        batch_size=2,  # Keep the batch size small to reduce memory usage
        max_thinking_tokens=4,  # Budget for CoT generation
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

    print("\n=== Built-in decoding ===")
    for i, (m, f) in enumerate(zip(out.mask_generations, out.face_generations)):
        print(f"[Sample {i}] MASK : {m}")
        print(f"[Sample {i}] FACE : {f}\n")

    # ------------------------------------------------------------------
    # Manual decoding – apply boolean masks to the generated sequences
    # ------------------------------------------------------------------
    tok = mask_face.tokenizer

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

        # ------------------------------------------------------------------
        # Consistency checks -------------------------------------------------
        # ------------------------------------------------------------------
        assert manual_mask == out.mask_generations[i], "Mismatch in mask decoding"
        assert manual_face == out.face_generations[i], "Mismatch in face decoding"

    print("All manual decodings match the built-in results! ✅")


if __name__ == "__main__":
    main()
