from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class _Out:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class DummyLM(nn.Module):
    """A tiny causal LM stub with forward() and generate().

    - forward: returns zeros logits with correct shape
    - generate: appends deterministic tokens to input_ids
    """

    def __init__(self, vocab_size: int = 16, pad_token_id: int = 0, think_end_id: int = 2, think_body_id: int = 7, face_id: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.think_end_id = think_end_id
        self.think_body_id = think_body_id
        self.face_id = face_id
        # Add a small parameter to ensure the module has parameters for device checks
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs):  # type: ignore[override]
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device, dtype=torch.float32)
        return _Out(logits)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Support both mask and face generate calls
        if "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        else:
            # HF-style: passed as positional? Not in our usage; fallback
            input_ids = args[0] if args else None
        assert input_ids is not None, "input_ids must be provided"
        max_new = int(kwargs.get("max_new_tokens", 1))
        is_mind = ("logits_processor" in kwargs)

        batch, seq_len = input_ids.shape
        new_tokens = []
        if is_mind:
            # produce max_new-1 think_body tokens followed by </think>
            for _ in range(batch):
                if max_new <= 0:
                    new_tokens.append([])
                elif max_new == 1:
                    new_tokens.append([self.think_end_id])
                else:
                    new_tokens.append([self.think_body_id] * (max_new - 1) + [self.think_end_id])
        else:
            for _ in range(batch):
                new_tokens.append([self.face_id] * max_new)

        # Build sequences tensor
        out = []
        for i in range(batch):
            tail = torch.tensor(new_tokens[i], device=input_ids.device, dtype=torch.long)
            out.append(torch.cat([input_ids[i], tail], dim=0))
        # Pad to same length per-batch (right pad)
        max_len = max(x.shape[0] for x in out)
        padded = []
        for x in out:
            if x.shape[0] < max_len:
                x = torch.cat([x, torch.full((max_len - x.shape[0],), self.pad_token_id, device=x.device, dtype=torch.long)], dim=0)
            padded.append(x)
        sequences = torch.stack(padded, dim=0)

        class _GenOut:
            def __init__(self, sequences: torch.Tensor):
                self.sequences = sequences

        return _GenOut(sequences)


class DummyTokenizer:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self._vocab = {"</think>": 2, "\n": 3}

    def get_vocab(self) -> Dict[str, int]:
        # minimal vocab with required tokens
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        # map any other text to a sequence of token id 5 of length len(text)%3 + 2
        length = (len(text) % 3) + 2
        return [5] * length

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        # simple decode: join ints
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(int(t)) for t in token_ids)

    def __call__(self, texts, return_tensors: str = "pt", padding: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for t in texts:
            toks = self.encode(t, add_special_tokens=False)
            ids.append(torch.tensor(toks, dtype=torch.long))
        max_len = max(x.shape[0] for x in ids)
        padded = []
        attn = []
        for x in ids:
            pad_len = max_len - x.shape[0]
            if pad_len > 0:
                x = torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=torch.long), x], dim=0)
            padded.append(x)
            attn.append((x != self.pad_token_id).long())
        input_ids = torch.stack(padded, dim=0)
        attention_mask = torch.stack(attn, dim=0)

        class _Batch:
            def __init__(self, data: Dict[str, torch.Tensor]):
                self._data = data
            def to(self, device):
                return _Batch({k: v.to(device) for k, v in self._data.items()})
            def __getitem__(self, key):
                return self._data[key]
            def keys(self):
                return self._data.keys()
            def items(self):
                return self._data.items()
            def get(self, key, default=None):
                return self._data.get(key, default)

        return _Batch({"input_ids": input_ids, "attention_mask": attention_mask})


