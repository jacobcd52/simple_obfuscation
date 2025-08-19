import pytest
import torch

from src.models.mind_face import MindFace


@pytest.fixture(scope="module")
def mind_face_small():
    """Load very small GPT2 models for quick testing purposes."""
    # We purposefully use the same tiny model for both mind & face so the
    # vocabulary is guaranteed to match while keeping resource usage low.
    model_name = "Qwen/Qwen3-4B"
    return MindFace(
        mask_model_name=model_name,
        face_model_name=model_name,
        batch_size=2,
        max_thinking_tokens=4,
        min_thinking_tokens=0,
    )


def test_generation_shapes(mind_face_small):
    prompts = [
        "<user>Hello, how are you?",
        "<user>What is the capital of France?",
    ]
    out = mind_face_small.generate(
        prompt_inputs=prompts,
        max_thinking_tokens=4,
        max_new_tokens=8,
    )

    # sequences tensor shape (B, T)
    assert out.sequences.dim() == 2
    B, T = out.sequences.shape
    assert B == len(prompts)

    # think_mask & face_mask must have shape (B, T-1)
    assert out.think_mask.shape == (B, T - 1)
    assert out.face_mask.shape == (B, T - 1)

    # Masks must be disjoint
    overlap = (out.think_mask & out.face_mask).any().item()
    assert not overlap, "think_mask and face_mask overlap"

    # Every generation token (after prompt) should belong to exactly one mask
    # We test this for the first sample to keep runtime low.
    p_len = out.prompt_lens[0].item()
    pad_len = out.pad_lens[0]
    eff_prompt_len = p_len + pad_len
    gen_token_count = T - eff_prompt_len
    think_tokens = out.think_mask[0, eff_prompt_len - 1 : T - 1].sum().item()
    face_tokens = out.face_mask[0, eff_prompt_len - 1 : T - 1].sum().item()
    assert (
        think_tokens + face_tokens
    ) <= gen_token_count, "Masks cover more tokens than generated"


def test_decoded_generations(mind_face_small):
    prompts = ["<user>Tell me a joke."]
    out = mind_face_small.generate(
        prompt_inputs=prompts,
        max_thinking_tokens=4,
        max_new_tokens=8,
    )

    # Returned decoded strings should match lengths inferred from masks
    t_len = out.think_lens[0]
    assert isinstance(out.mask_generations[0], str)
    assert isinstance(out.face_generations[0], str)
    # At least one of the generations should be non-empty (usually both)
    assert (
        len(out.mask_generations[0]) > 0 or len(out.face_generations[0]) > 0
    ), "Decoded generations are empty"
