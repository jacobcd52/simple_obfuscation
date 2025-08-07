"""Tests for the new generation mask logic (true for newly generated tokens)."""

import torch
from transformers import AutoTokenizer


def _make_tokenizer():
    # tiny random gpt2 tokenizer is fine for tests
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _compute_generation_mask(seqs: torch.LongTensor, prompt_len: int) -> torch.BoolTensor:
    """Return mask that is True for tokens beyond `prompt_len`."""
    batch, seq_len = seqs.shape
    return torch.arange(seq_len, device=seqs.device).expand(batch, -1) >= prompt_len


def test_mask_no_generation():
    tok = _make_tokenizer()
    prompts = ["Hello world", "Hi"]
    inputs = tok(prompts, return_tensors="pt", padding=True)
    seqs = inputs.input_ids
    prompt_len = seqs.shape[1]
    mask = _compute_generation_mask(seqs, prompt_len)
    assert not mask.any(), "Mask should be all False when no generation tokens are present"


def test_mask_after_generation():
    tok = _make_tokenizer()
    prompts = ["Hello there"]
    inputs = tok(prompts, return_tensors="pt", padding=True)
    prompt_len = inputs.input_ids.shape[1]

    # Simulate generation by appending three EOS tokens (arbitrary choice)
    generated = torch.full((1, 3), tok.eos_token_id, dtype=torch.long)
    seqs = torch.cat([inputs.input_ids, generated], dim=1)

    mask = _compute_generation_mask(seqs, prompt_len)

    # Tokens before prompt_len should be False, after should be True
    assert mask[:, :prompt_len].sum() == 0, "Prompt tokens should be masked as False"
    assert mask[:, prompt_len:].all(), "Generated tokens should be masked as True"