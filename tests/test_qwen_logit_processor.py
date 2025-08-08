import pytest
import torch
from transformers import AutoTokenizer

from src.utils.logit_processors import BatchThinkingTokenBudgetProcessor


def _try_load_qwen_tokenizer(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        return tok
    except Exception:
        return None


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-4B"])
def test_qwen_end_think_not_repeated(model_name: str):
    tok = _try_load_qwen_tokenizer(model_name)
    if tok is None:
        pytest.skip(f"Tokenizer for {model_name} not available in this environment")

    proc = BatchThinkingTokenBudgetProcessor(
        tok,
        max_thinking_tokens=32,
        batch_size=1,
        min_thinking_tokens=0,
    )

    end_ids = tok.encode("</think>", add_special_tokens=False)
    nl_ids = tok.encode("\n", add_special_tokens=False)

    # Construct an input that already ends with the full </think> sequence
    prefix_ids = tok.encode("Hello", add_special_tokens=False)
    seq_ids = prefix_ids + end_ids
    input_ids = torch.tensor([seq_ids], dtype=torch.long)

    # Create a scores tensor that is large enough to include those token ids
    max_tid = max((end_ids + nl_ids) or [0])
    vocab_size = max_tid + 5
    scores = torch.zeros(1, vocab_size, dtype=torch.float32)

    out = proc(input_ids=input_ids, scores=scores.clone())

    # Once </think> has been seen, the first token of that sequence must be masked
    assert len(end_ids) > 0
    first_tid = end_ids[0]
    assert out[0, first_tid] == proc.neg_inf

    # And NL should still be allowed (not forced, but not neg-inf by this rule)
    for tid in nl_ids:
        assert out[0, tid] != proc.neg_inf


