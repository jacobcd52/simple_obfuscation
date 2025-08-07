import torch
import pytest
from transformers import AutoTokenizer

from src.utils.logit_processors import BatchThinkingTokenBudgetProcessor


@pytest.mark.parametrize("batch_size,vocab_subset", [(2, 5000)])
def test_processor_no_nan_inf(batch_size: int, vocab_subset: int):
    """Check that *BatchThinkingTokenBudgetProcessor* never introduces NaNs/Infs."""
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