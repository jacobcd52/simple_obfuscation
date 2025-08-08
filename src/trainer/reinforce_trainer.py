"""Vanilla REINFORCE trainer.

Implements mean-baseline subtraction per mini-batch and supports
multi-GPU via *torch.distributed* (DDP).  Designed for flexibility rather
than raw throughput.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from ..config import TrainConfig
from ..utils.logit_processors import BatchThinkingTokenBudgetProcessor
from ..prompt_builders import PromptBuilder
from ..reward.base import RewardFunction
from ..utils import zero_special_token_grads
from .rollout_store import RolloutStore

# ------------------------------------------------------------------
# Local helper data-structures for better readability
# ------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Sequence
from transformers.tokenization_utils_base import BatchEncoding

@dataclass
class Batch:
    """Container holding the information needed for one forward/generation pass."""
    prompts: List[Dict]
    prompt_texts: List[str]
    inputs: Any  # transformers.BatchEncoding – kept generic to avoid importing heavy typing

@dataclass
class StepOutput:
    """Results from a single optimisation step (one mini-batch)."""
    sequences: torch.LongTensor
    logprob_sums: torch.Tensor
    rewards: torch.Tensor
    rollouts: List[Dict]
    loss: torch.Tensor

__all__ = ["ReinforceTrainer"]


def _instantiate_class(cls_path: str, **kwargs):
    *module_path, cls_name = cls_path.split(".")
    module = __import__(".".join(module_path), fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls(**kwargs)


async def _compute_reward(rwd: RewardFunction, rollout: Dict) -> float:
    """Helper function to compute reward from either sync or async reward function."""
    if inspect.iscoroutinefunction(rwd.__call__):
        return await rwd(rollout)
    else:
        return rwd(rollout)


class ReinforceTrainer:
    """Simple trainer object that orchestrates everything."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        # ------------------------------------------------------------------
        # Handle distributed setup explicitly.
        # If the script is launched via `torchrun`, environment variables such as
        # LOCAL_RANK/RANK/WORLD_SIZE are automatically set.  When `multi_gpu`
        # is "none" we *unset* these variables so that 🤗 Accelerate behaves like
        # a plain single-GPU run instead of wrapping the model in DDP – the latter
        # incurs roughly a 2× memory cost (original + DDP copy) which is what
        # caused the CUDA OOM you observed.
        # ------------------------------------------------------------------
        if cfg.multi_gpu == "none":
            for var in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "GLOBAL_RANK"):
                os.environ.pop(var, None)
            self.accelerator = Accelerator()
        elif cfg.multi_gpu == "ddp":
            # Use memory-efficient DDP: reuse the model parameter storage for
            # gradient buckets (saves ~1× model memory) and disable redundant
            # buffer broadcasts.
            from accelerate.utils import DistributedDataParallelKwargs
            ddp_kwargs = DistributedDataParallelKwargs(
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        elif cfg.multi_gpu == "fsdp":
            # Use FullyShardedDataParallel plugin with default settings.
            from accelerate.utils import FullyShardedDataParallelPlugin
            fsdp_plugin = FullyShardedDataParallelPlugin()
            self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        else:
            raise ValueError(f"Unknown multi_gpu setting: {cfg.multi_gpu}")

        # ------------------------------------------------------------------
        # model / tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, padding_side=cfg.padding_side, use_fast=True, trust_remote_code=True
        )
        if getattr(cfg, "enable_thinking", True):
            self.tokenizer.enable_thinking = True  # type: ignore[attr-defined]
        if self.tokenizer.pad_token is None:
            # Many GPT-style tokenizers lack a pad token; use eos as fallback.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16)

        # ------------------------------------------------------------------
        # prompt builder
        # ------------------------------------------------------------------
        self.prompt_builder: PromptBuilder = _instantiate_class(
            cfg.prompt_builder_cls, **cfg.prompt_builder_params
        )

        # ------------------------------------------------------------------
        # rewards
        # ------------------------------------------------------------------
        from ..reward import get_reward_class  # local import to avoid circular

        self.rewards: List[RewardFunction] = []
        # if no rewards specified, use default BoxedAnswerReward
        if not cfg.rewards:
            cls = get_reward_class("boxed_answer")
            self.rewards.append(cls())

        for rw in cfg.rewards:
            if "." in rw.cls:
                cls = _instantiate_class(rw.cls, **rw.params)
                self.rewards.append(cls)
            else:
                cls = get_reward_class(rw.cls)
                self.rewards.append(cls(**rw.params))

        self.rollout_store = RolloutStore("rollouts")

        # Prepare (wrap) the model for the chosen distributed strategy first
        self.model = self.accelerator.prepare(self.model)

        # Optimiser – must be created *after* model wrapping when using FSDP
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        # Distribute optimiser
        self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.accelerator.is_local_main_process:
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=cfg.__dict__,
                mode=os.environ.get("WANDB_MODE", "online"),
            )
        else:
            # Prevent extra runs from being created on the non-main ranks.
            wandb.init(mode="disabled")

        # misc
        self.global_step = 0
        # Gradient accumulation helpers
        self.grad_accum_steps = max(1, cfg.grad_accum_steps)
        self._accum_counter = 0
        # Ensure gradients are cleared before first accumulation cycle
        self.optimizer.zero_grad()

    def _generate(self, *args, **kwargs):
        """
        Thin wrapper around `model.generate` that works whether
        `self.model` is wrapped in DistributedDataParallel/FSDP or not.
        """
        # If the model is wrapped with FSDP, we temporarily *gather* the full
        # parameters for generation using `summon_full_params` – this avoids the
        # "storage size 0" error that occurs when the embedding matrix remains
        # sharded.  For DDP or single-GPU cases we can directly call `.generate`.
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
            is_fsdp = isinstance(self.model, FSDP)
        except (ImportError, AttributeError):
            is_fsdp = False

        if is_fsdp:
            # Gather parameters only for the duration of generation on this rank.
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            with _FSDP.summon_full_params(self.model, recurse=False):
                return self.model.module.generate(*args, **kwargs)  # type: ignore[attr-defined]
        # ----- DDP / single-GPU fallback -----
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)  # type: ignore[misc]
        if hasattr(self.model, "module") and hasattr(self.model.module, "generate"):
            return self.model.module.generate(*args, **kwargs)  # type: ignore[attr-defined]
        # If we reach here something is unexpected
        raise AttributeError("Neither wrapper nor underlying model provide a `generate` method.")

    def _token_processor(self):
        """Return a fresh `BatchThinkingTokenBudgetProcessor` configured from `self.cfg`."""
        cfg = self.cfg
        return BatchThinkingTokenBudgetProcessor(
            self.tokenizer,
            max_thinking_tokens=cfg.thinking_max_tokens,
            min_thinking_tokens=cfg.thinking_min_tokens,
            batch_size=cfg.batch_size,
        )

    def _build_gen_kwargs(self, tp):
        """Keyword arguments passed to `model.generate`."""
        cfg = self.cfg
        return dict(
            logits_processor=[tp],
            max_new_tokens=cfg.output_max_tokens or 512,
            do_sample=True,
            temperature=1.0,
            output_scores=False,  # scores not needed; will compute logprobs separately
            return_dict_in_generate=True,
        )

    def _next_batch(self, prompt_iter, tp):
        """Sample the next batch of prompts and tokenise them."""
        cfg = self.cfg
        prompts: List[Dict] = []
        for _ in range(cfg.batch_size):
            try:
                prompts.append(next(prompt_iter))
            except StopIteration:
                # Restart the iterator when we reach the end of the prompt set
                prompt_iter = iter(self.prompt_builder)
                prompts.append(next(prompt_iter))

        # Reset the logit processor state for the new batch
        tp.reset()

        # Build chat-formatted prompts using the tokenizer's chat template
        prompt_texts = []
        for p in prompts:
            # Extract the raw user message (strip any manually-added assistant tag)
            user_content = p["prompt"].split("\n<assistant>")[0]
            chat_str = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(chat_str)

        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        return Batch(prompts, prompt_texts, inputs), prompt_iter

    def _generate_sequences(self, inputs, gen_kwargs):
        """Run `model.generate` under autocast & without gradient."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self._generate(**inputs, **gen_kwargs)
        return generated.sequences  # (batch, seq_len)

    def _compute_logprobs(self, sequences, inputs):
        """Compute summed log-probabilities of newly generated tokens."""
        # Shift inputs/targets for language-modeling likelihood
        input_ids_full = sequences[:, :-1]
        target_ids = sequences[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids_full)
        logits_full = outputs.logits  # (batch, seq_len-1, vocab)
        logprobs_full = F.log_softmax(logits_full, dim=-1)

        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask newly generated tokens (positions >= prompt length)
        prompt_len = inputs["input_ids"].shape[1]
        seq_len = sequences.shape[1]
        mask_full = (
            torch.arange(seq_len, device=sequences.device)
            .unsqueeze(0)
            .expand(sequences.size(0), -1)
            >= prompt_len
        )
        mask = mask_full[:, 1:]  # align with targets
        selected_logprobs = token_logprobs * mask.float()
        logprob_sums = selected_logprobs.sum(dim=1)
        return logprob_sums

    def _compute_rewards(self, prompts, sequences, inputs, logprob_sums):
        """Compute rewards (handles sync + async) and return rollouts & loss."""
        rollouts: List[Dict] = []
        rewards = []
        # Detect async reward functions once
        has_async = any(inspect.iscoroutinefunction(rwd.__call__) for rwd in self.rewards)

        if has_async:
            async def _compute_async():
                all_async_tasks = []
                rollout_data = []
                for i, p in enumerate(prompts):
                    response_text = self.tokenizer.decode(
                        sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    rollout = dict(p)
                    rollout["response"] = response_text
                    rollout_sync_rewards = []
                    rollout_tasks = []
                    for rwd in self.rewards:
                        if inspect.iscoroutinefunction(rwd.__call__):
                            task = rwd(rollout)
                            rollout_tasks.append(task)
                            all_async_tasks.append(task)
                        else:
                            rollout_sync_rewards.append(rwd(rollout))
                    rollout["_sync_rewards"] = rollout_sync_rewards
                    rollout["_async_tasks"] = rollout_tasks
                    rollout_data.append(rollout)

                async_results = []
                if all_async_tasks:
                    async_results = await asyncio.gather(*all_async_tasks, return_exceptions=True)
                task_results = dict(zip(all_async_tasks, async_results))

                rewards_accum = []
                for i, rollout in enumerate(rollout_data):
                    total_reward = sum(rollout["_sync_rewards"])
                    for task in rollout["_async_tasks"]:
                        res = task_results.get(task, 0.0)
                        if isinstance(res, Exception):
                            print(f"[Warning] Async reward failed: {res}")
                            res = 0.0
                        total_reward += res
                    # Cleanup
                    rollout.pop("_sync_rewards", None)
                    rollout.pop("_async_tasks", None)
                    rollout["total_reward"] = total_reward
                    rollout["logprob_sum"] = logprob_sums[i].item()
                    rollouts.append(rollout)
                    rewards_accum.append(total_reward)
                return rewards_accum

            rewards = asyncio.run(_compute_async())
        else:
            for i, p in enumerate(prompts):
                response_text = self.tokenizer.decode(
                    sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                rollout = dict(p)
                rollout["response"] = response_text
                total_reward = 0.0
                for rwd in self.rewards:
                    total_reward += rwd(rollout)
                rollout["total_reward"] = total_reward
                rollout["logprob_sum"] = logprob_sums[i].item()
                rollouts.append(rollout)
                rewards.append(total_reward)

        rewards_t = torch.tensor(rewards, device=sequences.device, dtype=torch.float32)
        baseline = rewards_t.mean()
        reinforce_loss = -(rewards_t - baseline) * logprob_sums
        loss = reinforce_loss.mean()
        return rewards_t, rollouts, loss

    def _backprop(self, loss, final_batch: bool = False) -> bool:
        """Backpropagate `loss` with gradient accumulation.

        Returns True if an optimiser step was performed, False otherwise.
        """
        # Scale loss to account for gradient accumulation
        loss = loss / self.grad_accum_steps
        self.accelerator.backward(loss)
        self._accum_counter += 1

        stepped = False
        if self._accum_counter >= self.grad_accum_steps or final_batch:
            # Clip/token-specific gradient tweaks
            zero_special_token_grads(self.model, self.tokenizer)
            # Optimiser step & reset
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_counter = 0
            stepped = True
        return stepped

    def _store_rollouts(self, rollouts: List[Dict]):
        for rollout in rollouts:
            self.rollout_store.save(rollout)

    def _log_step(self, loss, rewards_t, rollouts, episode_counter):
        # Aggregate reward breakdowns (mean over batch) for logging
        breakdown_totals = {}
        for rollout in rollouts:
            for k, v in rollout.get("reward_breakdown", {}).items():
                breakdown_totals[k] = breakdown_totals.get(k, 0.0) + v
        breakdown_means = {f"reward/{k}": v / len(rollouts) for k, v in breakdown_totals.items()}

        log_dict = {
            "loss": loss.item(),
            "reward/mean": rewards_t.mean().item(),
            "episode": episode_counter,
            "global_step": self.global_step,
        }
        log_dict.update(breakdown_means)

        if self.accelerator.is_local_main_process:
            # Use a unique per-batch step to avoid overwriting logs when using
            # gradient accumulation (multiple batches can share the same
            # self.global_step). We log with the episodic batch counter so each
            # wandb step reflects a single logged batch.
            wandb.log(log_dict, step=episode_counter)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg

        # Clear all previous rollouts at the start of training
        self.rollout_store.clear_all_rollouts()

        # Build runtime helpers
        tp = self._token_processor()
        gen_kwargs = self._build_gen_kwargs(tp)
        from tqdm import tqdm  # local import to avoid cost when not training

        prompt_iter = iter(self.prompt_builder)
        episode_counter = 0
        total_prompts_processed = 0

        with tqdm(total=cfg.num_episodes, desc="Prompts processed") as pbar:
            while total_prompts_processed < cfg.num_episodes:
                batch, prompt_iter = self._next_batch(prompt_iter, tp)
                sequences = self._generate_sequences(batch.inputs, gen_kwargs)
                logprob_sums = self._compute_logprobs(sequences, batch.inputs)

                rewards_t, rollouts, loss = self._compute_rewards(
                    batch.prompts, sequences, batch.inputs, logprob_sums
                )

                # Gradient accumulation / optimiser step
                is_last_batch = (total_prompts_processed + len(batch.prompts) >= cfg.num_episodes)
                stepped = self._backprop(loss, final_batch=is_last_batch)

                # Book-keeping
                self._store_rollouts(rollouts)
                self._log_step(loss, rewards_t, rollouts, episode_counter)

                # Counters / progress
                episode_counter += 1
                total_prompts_processed += len(batch.prompts)
                pbar.update(len(batch.prompts))
                if stepped:
                    self.global_step += 1