"""Tests for masking newly generated tokens using the real implementation.

This validates that only tokens beyond the prompt contribute to log-prob sums
in `ReinforceTrainer._compute_logprobs`, which internally builds the mask.
"""

import math
import contextlib

import torch
from transformers import AutoTokenizer

from src.trainer.reinforce_trainer import ReinforceTrainer


def _make_tokenizer():
    # tiny random gpt2 tokenizer is fine for tests
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class _DummyOutputs:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _DummyModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor):
        batch, seq_len = input_ids.shape
        # Uniform logits -> log_softmax = constant = -log(V)
        logits = torch.zeros(batch, seq_len, self.vocab_size, dtype=torch.float32)
        return _DummyOutputs(logits)


class _DummySelf:
    def __init__(self, model):
        self.model = model


def _compute_logprob_sums_via_real_mask(sequences: torch.LongTensor, inputs):
    # Build a dummy model large enough to index all target ids
    max_token_id = int(sequences.max().item())
    dummy_model = _DummyModel(vocab_size=max_token_id + 1)
    dummy_self = _DummySelf(dummy_model)

    # Make torch.autocast a no-op (trainer uses cuda autocast)
    orig_autocast = torch.autocast
    try:
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
        return ReinforceTrainer._compute_logprobs(dummy_self, sequences, inputs)
    finally:
        torch.autocast = orig_autocast


def test_mask_no_generation():
    tok = _make_tokenizer()
    prompts = ["Hello world", "Hi"]
    inputs = tok(prompts, return_tensors="pt", padding=True)
    sequences = inputs.input_ids  # no generated tokens appended

    logprob_sums = _compute_logprob_sums_via_real_mask(sequences, {"input_ids": inputs.input_ids})
    # With no generated tokens, sum should be exactly zero
    assert torch.allclose(logprob_sums, torch.zeros_like(logprob_sums)), (
        "Log-prob sum should be 0 when no generation tokens are present"
    )


def test_mask_after_generation():
    tok = _make_tokenizer()
    prompts = ["Hello there"]
    inputs = tok(prompts, return_tensors="pt", padding=True)
    prompt_len = inputs.input_ids.shape[1]

    # Simulate generation by appending three EOS tokens (arbitrary choice)
    num_generated = 3
    generated = torch.full((1, num_generated), tok.eos_token_id, dtype=torch.long)
    sequences = torch.cat([inputs.input_ids, generated], dim=1)

    logprob_sums = _compute_logprob_sums_via_real_mask(sequences, {"input_ids": inputs.input_ids})

    # With uniform logits, log_softmax per position equals -log(V)
    vocab_size = int(sequences.max().item()) + 1
    per_token_logp = -math.log(vocab_size)
    expected = torch.tensor([num_generated * per_token_logp], dtype=logprob_sums.dtype)

    assert torch.allclose(logprob_sums.cpu(), expected, atol=1e-6), (
        "Only generated tokens should contribute to the log-prob sum"
    )