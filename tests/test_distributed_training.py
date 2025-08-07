import pytest
import torch

from src.config import TrainConfig
from src.trainer.reinforce_trainer import ReinforceTrainer


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("multi_gpu", ["ddp", "fsdp"])
def test_ddp_forward_backward(multi_gpu: str, device: str):
    """Basic sanity test that training works with DDP mode enabled.

    We only verify that a forward + backward + optimiser step run without
    throwing – we do *not* spawn multiple processes in this unit-test.  The
    intent is to catch obvious integration issues (e.g. model wrapping).
    """

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")

    # ------------------------------------------------------------------
    # Build a minimal JsonlPromptBuilder pointing at a temporary file
    # ------------------------------------------------------------------
    import json, tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        json.dump({"input": "Hello world"}, tmp)
        tmp.write("\n")
        prompt_path = tmp.name

    cfg = TrainConfig(
        model_name="sshleifer/tiny-gpt2",
        batch_size=1,
        epochs=1,
        learning_rate=1e-4,
        multi_gpu=multi_gpu,
        prompt_builder_cls="src.generation.prompt_builder.JsonlPromptBuilder",
        prompt_builder_params={"path": prompt_path},
    )

    trainer = ReinforceTrainer(cfg)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    trainer.model.eval()
    tokenizer = trainer.tokenizer
    inputs = tokenizer("Hello world", return_tensors="pt")
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = trainer.model(**inputs)

    assert outputs.logits.shape[0] == 1, "Unexpected batch size in logits"

    # ------------------------------------------------------------------
    # backward pass (single step)
    # ------------------------------------------------------------------
    trainer.model.train()
    outputs = trainer.model(**inputs)
    loss = outputs.logits.mean()

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    # If we reached here without exceptions, the test is successful.
