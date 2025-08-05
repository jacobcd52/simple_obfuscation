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
from ..generation.logit_processors import BatchThinkingTokenBudgetProcessor
from ..generation.prompt_builder import PromptBuilder
from ..reward.base import RewardFunction
from ..utils import assistant_token_mask, zero_special_token_grads
from .rollout_store import RolloutStore

__all__ = ["ReinforceTrainer"]


def _instantiate_class(path: str, **kwargs):
    *module_path, cls_name = path.split(".")
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
        self.accelerator = Accelerator()

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
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float16)

        # ------------------------------------------------------------------
        # prompt builder
        # ------------------------------------------------------------------
        self.prompt_builder: PromptBuilder = _instantiate_class(
            cfg.prompt_builder_cls, **cfg.prompt_builder_params
        )

        # ------------------------------------------------------------------
        # rewards
        # ------------------------------------------------------------------
        from ..reward.registry import get_reward_class  # local import to avoid circular

        self.rewards: List[RewardFunction] = []
        for rw in cfg.rewards:
            if "." in rw.cls:
                cls = _instantiate_class(rw.cls, **rw.params)
                self.rewards.append(cls)
            else:
                cls = get_reward_class(rw.cls)
                self.rewards.append(cls(**rw.params))

        # ------------------------------------------------------------------
        # rollout store
        # ------------------------------------------------------------------
        self.rollout_store = RolloutStore("rollouts")

        # optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # distribute
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        # logging
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.__dict__,
            mode=os.environ.get("WANDB_MODE", "online"),
        )

        # misc
        self.global_step = 0

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        tp = BatchThinkingTokenBudgetProcessor(
            self.tokenizer,
            max_thinking_tokens=cfg.thinking_max_tokens,
            min_thinking_tokens=cfg.thinking_min_tokens,
            batch_size=cfg.batch_size,
        )

        gen_kwargs = dict(
            logits_processor=[tp],
            max_new_tokens=cfg.output_max_tokens or 512,
            do_sample=True,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        prompt_iter = iter(self.prompt_builder)

        for epoch in range(cfg.epochs):
            for _ in range(len(self.prompt_builder)):
                prompts: List[Dict] = [next(prompt_iter) for _ in range(cfg.batch_size)]
                prompt_texts = [p["prompt"] for p in prompts]

                inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)

                with torch.no_grad():
                    generated = self.model.generate(**inputs, **gen_kwargs)

                sequences = generated.sequences  # (batch, seq_len)
                scores = generated.scores  # list[Tensor] length == new_tokens

                # ------------------------------------------------------------------
                # compute log-probs for assistant tokens
                # ------------------------------------------------------------------
                # stack scores to (batch, tgt_len, vocab)
                logits = torch.stack(scores, dim=1)  # type: ignore[arg-type]
                logprobs = F.log_softmax(logits, dim=-1)

                # gather logprobs of generated tokens (excluding the prompt part)
                gen_token_ids = sequences[:, inputs["input_ids"].shape[1]:]
                batch_indices = torch.arange(sequences.size(0)).unsqueeze(-1).to(sequences.device)
                token_logprobs = logprobs.gather(-1, gen_token_ids.unsqueeze(-1)).squeeze(-1)

                # mask to keep only tokens after the <assistant> tag
                mask = assistant_token_mask(self.tokenizer, sequences)
                mask = mask[:, inputs["input_ids"].shape[1]:]
                selected_logprobs = token_logprobs * mask.float()
                logprob_sums = selected_logprobs.sum(dim=1)

                # ------------------------------------------------------------------
                # assemble rollouts & compute rewards
                # ------------------------------------------------------------------
                rollouts: List[Dict] = []
                rewards = []
                
                # Check if any reward functions are async
                has_async_rewards = any(inspect.iscoroutinefunction(rwd.__call__) for rwd in self.rewards)
                
                if has_async_rewards:
                    # Handle async rewards - batch all calls
                    async def compute_rewards_async():
                        # First, collect all rollouts and create tasks
                        all_async_tasks = []
                        rollout_data = []
                        
                        for i, p in enumerate(prompts):
                            response_text = self.tokenizer.decode(sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                            rollout = {
                                "prompt": p["prompt"],
                                "response": response_text,
                            }
                            rollout_data.append((rollout, i))
                            
                            # Create tasks for all async reward functions for this rollout
                            rollout_async_tasks = []
                            rollout_sync_rewards = []
                            
                            for rwd in self.rewards:
                                if inspect.iscoroutinefunction(rwd.__call__):
                                    task = rwd(rollout)
                                    rollout_async_tasks.append(task)
                                    all_async_tasks.append(task)
                                else:
                                    # For sync rewards, compute them immediately
                                    sync_reward = rwd(rollout)
                                    rollout_sync_rewards.append(sync_reward)
                            
                            # Store sync rewards for later use
                            rollout["_sync_rewards"] = rollout_sync_rewards
                            rollout["_async_tasks"] = rollout_async_tasks
                        
                        # Now await all async tasks in parallel
                        if all_async_tasks:
                            async_results = await asyncio.gather(*all_async_tasks, return_exceptions=True)
                            
                            # Map results back to tasks
                            task_results = dict(zip(all_async_tasks, async_results))
                        
                        # Process results
                        rewards_list = []
                        
                        for i, (rollout, original_idx) in enumerate(rollout_data):
                            total_reward = sum(rollout["_sync_rewards"])  # Add sync rewards
                            
                            # Add async rewards
                            for task in rollout["_async_tasks"]:
                                result = task_results[task]
                                if isinstance(result, Exception):
                                    print(f"[Warning] Async reward failed: {result}")
                                    total_reward += 0.0
                                else:
                                    total_reward += result
                            
                            # Clean up temporary data
                            del rollout["_sync_rewards"]
                            del rollout["_async_tasks"]
                            
                            rollout["total_reward"] = total_reward
                            rollout["logprob_sum"] = logprob_sums[original_idx].item()
                            rollouts.append(rollout)
                            rewards_list.append(total_reward)
                        
                        return rewards_list
                    
                    rewards = asyncio.run(compute_rewards_async())
                else:
                    # Handle sync rewards (original logic)
                    for i, p in enumerate(prompts):
                        response_text = self.tokenizer.decode(sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                        rollout = {
                            "prompt": p["prompt"],
                            "response": response_text,
                        }
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

                # ------------------------------------------------------------------
                # backward
                # ------------------------------------------------------------------
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                zero_special_token_grads(self.model, self.tokenizer)
                self.optimizer.step()

                # ------------------------------------------------------------------
                # logging & storage
                # ------------------------------------------------------------------
                for rollout in rollouts:
                    self.rollout_store.save(rollout)
                wandb.log(
                    {
                        "loss": loss.item(),
                        "reward_mean": baseline.item(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                    },
                    step=self.global_step,
                )

                self.global_step += 1

            # end epoch
