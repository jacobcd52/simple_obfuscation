"""Tests for *assistant_token_mask* utility."""

import torch
from simple_rl_research.utils import assistant_token_mask
from transformers import AutoTokenizer


def _make_tokenizer():
    # tiny random gpt2 tokenizer is fine; ensure '<assistant>' tokens exist in vocab by adding them.
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2", use_fast=True)
    tokenizer.add_tokens(["<assistant>"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_mask_no_tag():
    tok = _make_tokenizer()
    seqs = tok(["Hello world"], return_tensors="pt", padding=True).input_ids
    mask = assistant_token_mask(tok, seqs)
    assert not mask.any(), "Mask should be all False when tag absent"


def test_mask_after_tag():
    tok = _make_tokenizer()
    text = "<assistant> foo bar baz"
    seqs = tok([text], return_tensors="pt", padding=True).input_ids
    mask = assistant_token_mask(tok, seqs)
    idx = (mask[0] == 1).nonzero(as_tuple=True)[0]
    # ensure positions after tag are True
    tag_len = len(tok.encode("<assistant>", add_special_tokens=False))
    assert (idx[0] >= tag_len), "Mask should start right after tag"
