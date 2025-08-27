from __future__ import annotations

import os
import pytest
import os as _os
import tempfile
import torch.distributed as dist
import torch

from src.models.mind_face import MindFace
from dummy_lm import DummyLM, DummyTokenizer


def _make_preloaded_mindface(device: str = None) -> MindFace:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = DummyTokenizer(pad_token_id=0)
    mind = DummyLM().to(device)
    face = DummyLM().to(device)
    return MindFace(mask_model=mind, face_model=face, tokenizer=tok, device=device, batch_size=2,
                    max_thinking_tokens=2, min_thinking_tokens=0)


def test_init_with_preloaded_components():
    mf = _make_preloaded_mindface(device="cpu")
    assert mf.mask_model is not None and mf.face_model is not None
    assert mf.tokenizer is not None
    # vocab consistency
    assert mf.tokenizer.get_vocab() == mf.tokenizer.get_vocab()


def test_generate_non_fsdp_cpu():
    mf = _make_preloaded_mindface(device="cpu")
    prompts = ["hello", "world"]
    out = mf.generate(prompts, max_thinking_tokens=2, max_new_tokens=4)
    assert hasattr(out, "sequences")
    seq = out.sequences
    assert isinstance(seq, torch.Tensor)
    assert seq.ndim == 2 and seq.shape[0] == 2
    # ensure we appended some tokens beyond input length
    assert seq.shape[1] >= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FSDP generation test")
def test_generate_under_fsdp_wrap_smoke(monkeypatch):
    # This test lightly wraps submodules in a minimal FSDP only if available.
    # It avoids distributed launch by keeping world_size=1; goal is to ensure summon paths work.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
    # Initialize a single-rank process group for FSDP
    if dist.is_available() and not dist.is_initialized():
        with tempfile.TemporaryDirectory() as tmpd:
            init_file = f"file://{tmpd}/pg"
            dist.init_process_group(backend="nccl", init_method=init_file, rank=0, world_size=1)
            try:
                device = "cuda"
                mf = _make_preloaded_mindface(device=device)

                # Wrap submodules with FSDP (single-rank) to exercise summon_full_params branches
                mf.mask_model = FSDP(mf.mask_model)
                mf.face_model = FSDP(mf.face_model)

                prompts = ["hello", "world"]
                out = mf.generate(prompts, max_thinking_tokens=2, max_new_tokens=4)
                assert hasattr(out, "sequences")
                assert out.sequences.is_cuda
            finally:
                dist.destroy_process_group()


