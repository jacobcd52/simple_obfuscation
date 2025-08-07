import asyncio
import random

import torch

from src.trainer.reinforce_trainer import ReinforceTrainer


class _SyncReward:
    name = "sync_reward"

    def __init__(self, value: float):
        self.value = value

    def __call__(self, rollout):
        return self.value


class _AsyncReward:
    name = "async_reward"

    def __init__(self, value: float):
        self.value = value

    async def __call__(self, rollout):
        await asyncio.sleep(0)
        return self.value


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "decoded"


class _DummySelf:
    def __init__(self):
        self.rewards = []
        self.tokenizer = _StubTokenizer()


def test_compute_rewards_sync_and_async():
    random.seed(0)
    torch.manual_seed(0)

    trainer = _DummySelf()
    trainer.rewards = [_SyncReward(0.3), _AsyncReward(0.7)]

    prompts = [
        {"prompt": "Q1"},
        {"prompt": "Q2"},
    ]
    # Sequences: fake prompt length 2, two tokens generated per sample
    sequences = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    inputs = {"input_ids": torch.tensor([[1, 2], [5, 6]], dtype=torch.long)}
    logprob_sums = torch.tensor([1.5, -0.5], dtype=torch.float32)

    rewards_t, rollouts, loss = ReinforceTrainer._compute_rewards(trainer, prompts, sequences, inputs, logprob_sums)

    # totals per sample = 0.3 + 0.7 = 1.0
    assert rewards_t.shape == torch.Size([2])
    assert torch.allclose(rewards_t, torch.tensor([1.0, 1.0]))

    # rollouts populated with response and logprob_sum
    assert len(rollouts) == 2
    assert "response" in rollouts[0] and isinstance(rollouts[0]["response"], str)
    assert "logprob_sum" in rollouts[0]

    # loss is a scalar tensor
    assert loss.ndim == 0


