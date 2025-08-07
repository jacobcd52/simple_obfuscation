import random
import torch
import pytest
from transformers import AutoTokenizer

from src.utils.logit_processors import BatchThinkingTokenBudgetProcessor


@pytest.mark.parametrize("batch_size,vocab_subset", [(2, 5000)])
def test_processor_no_nan_inf(batch_size: int, vocab_subset: int):
    """Check that *BatchThinkingTokenBudgetProcessor* never introduces NaNs/Infs."""
    random.seed(0)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    proc = BatchThinkingTokenBudgetProcessor(
        tokenizer,
        max_thinking_tokens=5,
        batch_size=batch_size,
        min_thinking_tokens=0,
    )

    # Craft dummy input_ids (shape: [batch, seq_len]) and random scores
    input_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    scores = torch.randn(batch_size, vocab_subset, dtype=torch.float32)

    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()

    out = proc(input_ids, scores.clone())

    assert not torch.isnan(out).any(), "Processor introduced NaNs"
    assert not torch.isinf(out).any(), "Processor introduced Infs"


def test_processor_behavior_min_and_max_tokens():
    """Validate forced behavior around min/max thinking token budgets.

    - When min_thinking_tokens > tokens_generated: end tokens are disallowed.
    - When max_thinking_tokens == 0: immediately prefer NL and </think> and stop.
    - Near max_thinking_tokens: boost NL and </think> and then force them.
    """
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    proc = BatchThinkingTokenBudgetProcessor(tok, max_thinking_tokens=4, min_thinking_tokens=2, batch_size=1)

    # Convenience helpers to lookup ids
    nl_ids = tok.encode("\n", add_special_tokens=False)
    end_ids = tok.encode("</think>", add_special_tokens=False)

    # Build scores sized to include forced token ids
    max_tid = max(nl_ids + end_ids) if (nl_ids or end_ids) else 0
    vocab_size = max_tid + 5
    scores = torch.zeros(1, vocab_size, dtype=torch.float32)

    # Step 1: tokens_generated becomes 1 (< min_thinking_tokens=2) ⇒ end tokens neg-inf
    out = proc(input_ids=torch.zeros(1, 1, dtype=torch.long), scores=scores.clone())
    for tid in end_ids:
        assert out[0, tid] == proc.neg_inf

    # Step 2: tokens_generated becomes 2 (== max-2) ⇒ all -inf except NL forced to 1.0
    out = proc(input_ids=torch.zeros(1, 1, dtype=torch.long), scores=scores.clone())
    # Entire row set to neg_inf except NL forced to 1.0
    forced = torch.zeros_like(out, dtype=torch.bool)
    for tid in nl_ids:
        forced[0, tid] = True
        assert out[0, tid] == 1.0
    assert torch.all(out[~forced] == proc.neg_inf)

    # Step 3: tokens_generated=3 (>= max-1? No, equals 3 which is == max-1) ⇒ force </think> and stop
    out = proc(input_ids=torch.zeros(1, 1, dtype=torch.long), scores=scores.clone())
    forced = torch.zeros_like(out, dtype=torch.bool)
    for tid in end_ids:
        forced[0, tid] = True
        assert out[0, tid] == 1.0
    assert torch.all(out[~forced] == proc.neg_inf)


def test_processor_immediate_stop_when_max_zero():
    random.seed(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    proc = BatchThinkingTokenBudgetProcessor(tok, max_thinking_tokens=0, min_thinking_tokens=0, batch_size=1)
    nl_ids = tok.encode("\n", add_special_tokens=False)
    end_ids = tok.encode("</think>", add_special_tokens=False)
    max_tid = max(nl_ids + end_ids) if (nl_ids or end_ids) else 0
    vocab_size = max_tid + 5
    scores = torch.zeros(1, vocab_size, dtype=torch.float32)

    out = proc(input_ids=torch.zeros(1, 1, dtype=torch.long), scores=scores.clone())
    # All set to neg_inf except NL and </think> set to 1.0
    forced = torch.zeros_like(out, dtype=torch.bool)
    for tid in nl_ids:
        forced[0, tid] = True
        assert out[0, tid] == 1.0
    for tid in end_ids:
        forced[0, tid] = True
        assert out[0, tid] == 1.0
    assert torch.all(out[~forced] == proc.neg_inf)