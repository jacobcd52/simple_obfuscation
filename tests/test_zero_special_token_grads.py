import random

import torch

from src.utils.token_utils import zero_special_token_grads


class _TinyIOModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 4):
        super().__init__()
        self.input_emb = torch.nn.Embedding(vocab_size, embed_dim)
        self.output_emb = torch.nn.Embedding(vocab_size, embed_dim)

    def get_input_embeddings(self):
        return self.input_emb

    def get_output_embeddings(self):
        return self.output_emb


class _TinyTokenizer:
    def __init__(self, vocab):
        self.vocab = {tok: idx for idx, tok in enumerate(vocab)}

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [self.vocab.get(text, 0)]


def test_zero_special_token_grads():
    random.seed(0)
    torch.manual_seed(0)

    # Build tiny tokenizer mapping exactly the special tokens we care about
    special = ["<think>", "</think>", "\n"]
    vocab = special + ["other"]
    tok = _TinyTokenizer(vocab)

    model = _TinyIOModel(vocab_size=len(vocab))

    # Fake non-zero grads in input/output embeddings
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        emb.weight.grad = torch.ones_like(emb.weight)

    zero_special_token_grads(model, tok)

    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        for token in special:
            tid = tok.vocab[token]
            assert torch.all(emb.weight.grad[tid] == 0), f"Grad not zeroed for {token}"


