import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.nn.functional as F


@pytest.mark.parametrize("model_name", ["sshleifer/tiny-gpt2"])
def test_generation_and_logprob_no_nan(model_name: str):
    """End-to-end check that generation + log-prob computation stays finite."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate one continuation
    generated = model.generate(**inputs, max_new_tokens=8, return_dict_in_generate=True)
    sequences = generated.sequences  # (1, seq_len)

    # Compute log-probs via separate forward pass (same as trainer)
    input_ids_full = sequences[:, :-1]
    target_ids = sequences[:, 1:]

    outputs = model(input_ids_full)
    logits_full = outputs.logits
    logprobs_full = F.log_softmax(logits_full, dim=-1)
    token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    assert not torch.isnan(token_logprobs).any(), "NaN in token log-probs"
    assert not torch.isinf(token_logprobs).any(), "Inf in token log-probs"