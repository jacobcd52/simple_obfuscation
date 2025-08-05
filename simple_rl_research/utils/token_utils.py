"""Token / text helper functions and gradient utilities."""

from __future__ import annotations

import re
from typing import Tuple

import torch

__all__ = [
    "extract_final_output",
    "extract_thinking",
    "zero_special_token_grads",
]


_THINK_START = "<think>"
_THINK_END = "</think>"
_ASSISTANT_TAG = "<assistant>"


# ---------------------------------------------------------------------------
# text extraction helpers
# ---------------------------------------------------------------------------

def extract_thinking(text: str) -> str:
    """Return the concatenated text between *all* <think>…</think> tags."""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return "\n".join(m.strip() for m in matches)


def extract_final_output(text: str) -> str:
    """Return everything *after* the final ``</think>`` tag.

    Falls back to the original text if no think tags are present.
    """
    if _THINK_END in text:
        return text.split(_THINK_END)[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# gradient zeroing
# ---------------------------------------------------------------------------

def zero_special_token_grads(model, tokenizer):
    """Zero gradients for <think>, </think> and newline tokens.

    The implementation is copied verbatim from the specification to
    ensure identical behaviour.
    """

    special_tokens = ["<think>", "</think>", "\n"]
    token_ids = []
    for tok in special_tokens:
        token_ids.extend(tokenizer.encode(tok, add_special_tokens=False))

    # Zero output embedding gradients (as before)
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if hasattr(emb, "weight") and emb.weight.grad is not None:
            for tid in token_ids:
                if tid < emb.weight.grad.shape[0]:
                    emb.weight.grad[tid].zero_()

    # Zero input embedding gradients (if present)
    if hasattr(model, "get_input_embeddings"):
        emb_in = model.get_input_embeddings()
        if hasattr(emb_in, "weight") and emb_in.weight.grad is not None:
            for tid in token_ids:
                if tid < emb_in.weight.grad.shape[0]:
                    emb_in.weight.grad[tid].zero_()

    # Zero any other parameter gradients that are indexed by vocab (rare, but for completeness)
    for _, param in model.named_parameters():
        if param.grad is not None and param.grad.shape[0] >= max(token_ids) + 1:
            # Only zero if the first dimension matches vocab size
            for tid in token_ids:
                if tid < param.grad.shape[0]:
                    param.grad[tid].zero_()


# ---------------------------------------------------------------------------
# loss masking helper
# ---------------------------------------------------------------------------

def assistant_token_mask(tokenizer, input_ids: torch.LongTensor) -> torch.BoolTensor:
    """Return boolean mask selecting *assistant* tokens in ``input_ids``.

    Vectorised implementation – no Python loops over the *batch*.
    A token is selected if it lies *after* the **last occurrence** of the
    ``<assistant>`` tag in its sequence.
    """
    device = input_ids.device
    assistant_token_id_seq = tokenizer.encode(_ASSISTANT_TAG, add_special_tokens=False)
    sub_seq_len = len(assistant_token_id_seq)

    if sub_seq_len == 0:
        raise ValueError("Tokenizer failed to encode <assistant> tag")

    batch_size, seq_len = input_ids.shape

    # ------------------------------------------------------------------
    # Find starting indices where the full tag appears.
    # ------------------------------------------------------------------
    # Build mask for each position that matches first token of the tag
    starts = (input_ids[:, : seq_len - sub_seq_len + 1] == assistant_token_id_seq[0])
    for offset, tid in enumerate(assistant_token_id_seq[1:], start=1):
        starts &= input_ids[:, offset : seq_len - sub_seq_len + 1 + offset] == tid

    # starts -> (batch, seq_len - L + 1) boolean matrix indicating tag matches
    # Convert to int to pick last index via argmax on reversed tensor
    starts_int = starts.long()
    # Reverse along sequence dim and find first 1 (== last original match)
    rev_idx = torch.argmax(torch.flip(starts_int, dims=[1]), dim=1)
    # If no match exists the row becomes 0; we correct via starts.any()
    has_match = starts.any(dim=1)
    last_start = (seq_len - sub_seq_len - rev_idx).clamp(min=0)
    last_start[~has_match] = seq_len  # mask will be all False when tag absent

    # Build final mask: positions >= last_start + sub_seq_len
    arange = torch.arange(seq_len, device=device).expand(batch_size, -1)
    mask = arange >= (last_start + sub_seq_len).unsqueeze(1)
    return mask
