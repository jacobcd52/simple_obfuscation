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
import gc
from typing import Dict, List

import torch
import torch.distributed as dist
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
        self._mf_split_fsdp = False
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
        # model / tokenizer (with optional MindFace wrapper)
        # ------------------------------------------------------------------
        if getattr(cfg, "use_mind_face", False) and cfg.multi_gpu == "fsdp":
            # Mind-Face split across two FSDP subgroups
            self._mf_split_fsdp = True
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("FSDP mind-face split requires torch.distributed to be initialized")

            world_size = dist.get_world_size()
            rank = dist.get_rank()
            mind_world = (world_size + 1) // 2
            face_world = world_size // 2
            self._mind_ranks = list(range(0, mind_world))
            self._face_ranks = list(range(mind_world, world_size))
            # Create disjoint process groups
            self._pg_mind = dist.new_group(self._mind_ranks) if mind_world > 0 else None
            self._pg_face = dist.new_group(self._face_ranks) if face_world > 0 else None
            self._is_mind_rank = rank in self._mind_ranks
            self._is_face_rank = rank in self._face_ranks
            try:
                print(f"[init rank{rank}] world_size={world_size} mind_ranks={self._mind_ranks} face_ranks={self._face_ranks} "
                      f"is_mind_rank={self._is_mind_rank} is_face_rank={self._is_face_rank}", flush=True)
            except Exception:
                pass

            mind_name = cfg.mind_model_name or cfg.model_name
            face_name = cfg.face_model_name or cfg.model_name

            # Shared tokenizer (face tokenizer). Force left padding. We require vocab parity and
            # enforce it via embedding size equality after loading models below.
            self.tokenizer = AutoTokenizer.from_pretrained(face_name, padding_side="left", use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load models only on their owning subgroup ranks
            self.mind_model = None
            self.face_model = None

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if self._is_mind_rank:
                mind = AutoModelForCausalLM.from_pretrained(mind_name, torch_dtype=dtype).to(device)
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
                    self.mind_model = FSDP(mind, process_group=self._pg_mind)
                except Exception:
                    self.mind_model = mind
            if self._is_face_rank:
                face = AutoModelForCausalLM.from_pretrained(face_name, torch_dtype=dtype).to(device)
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
                    self.face_model = FSDP(face, process_group=self._pg_face)
                except Exception:
                    self.face_model = face

            # Assert tokenizer compatibility by comparing embedding sizes (mind/face)
            if self._is_mind_rank and self._is_face_rank:
                # Single-rank world: both present
                mind_emb = self.mind_model.get_input_embeddings().weight  # type: ignore[union-attr]
                face_emb = self.face_model.get_input_embeddings().weight  # type: ignore[union-attr]
                if mind_emb.shape[0] != face_emb.shape[0]:
                    raise ValueError(
                        "Tokenizers of face and mind models differ – they must share the same vocabulary"
                    )
            else:
                # Cross-rank assertion: gather sizes on rank 0 and compare
                local_size = torch.tensor([
                    (self.mind_model.get_input_embeddings().weight.shape[0] if self.mind_model is not None else -1),
                    (self.face_model.get_input_embeddings().weight.shape[0] if self.face_model is not None else -1),
                ], device=device, dtype=torch.int64)
                sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
                dist.all_gather(sizes, local_size)
                # Extract first available mind and face sizes from gathered
                mind_sz = next((int(s[0].item()) for s in sizes if int(s[0].item()) > 0), None)
                face_sz = next((int(s[1].item()) for s in sizes if int(s[1].item()) > 0), None)
                if mind_sz is not None and face_sz is not None and mind_sz != face_sz:
                    raise ValueError(
                        "Tokenizers of face and mind models differ – they must share the same vocabulary"
                    )

            # Expose a model attribute for utilities that expect .device
            self.model = self.mind_model if self.mind_model is not None else self.face_model

        elif getattr(cfg, "use_mind_face", False):
            from ..models.mind_face import MindFace

            mind_name = cfg.mind_model_name or cfg.model_name
            face_name = cfg.face_model_name or cfg.model_name

            self.model = MindFace(
                mind_model_name=mind_name,
                face_model_name=face_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=cfg.batch_size,
                max_thinking_tokens=cfg.thinking_max_tokens or 0,
                min_thinking_tokens=cfg.thinking_min_tokens,
            )

            self.tokenizer = self.model.tokenizer
        else:
            # Force left padding to ensure a common prompt length across the batch
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name, padding_side="left", use_fast=True, trust_remote_code=True
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

        # ------------------------------------------------------------------
        # Reward functions & apply flags
        # ------------------------------------------------------------------
        self.rewards: List[RewardFunction] = []
        self.rewards_apply_flags: List[bool] = []

        def _append_reward(rwd_instance: RewardFunction, apply_flag: bool):
            self.rewards.append(rwd_instance)
            self.rewards_apply_flags.append(apply_flag)

        # If no rewards specified, use default BoxedAnswerReward
        if not cfg.rewards:
            cls = get_reward_class("boxed_answer")
            _append_reward(cls(), True)

        for rw in cfg.rewards:
            apply_flag = getattr(rw, "apply_to_thinking", True)

            # Filter out parameters not accepted by the reward's __init__ (e.g., 'apply_to_thinking')
            raw_params = dict(getattr(rw, "params", {}))
            if "apply_to_thinking" in raw_params:
                raw_params.pop("apply_to_thinking")

            def _filter_kwargs_for(cls_obj, kwargs_dict):
                sig = inspect.signature(cls_obj.__init__)
                return {k: v for k, v in kwargs_dict.items() if k in sig.parameters}

            if "." in rw.cls:
                # Fully qualified path – import class to check signature before instantiation
                *mod_parts, class_name = rw.cls.split(".")
                mod = __import__(".".join(mod_parts), fromlist=[class_name])
                cls_obj = getattr(mod, class_name)
                filtered_params = _filter_kwargs_for(cls_obj, raw_params)
                inst = cls_obj(**filtered_params)
            else:
                cls_obj = get_reward_class(rw.cls)
                filtered_params = _filter_kwargs_for(cls_obj, raw_params)
                inst = cls_obj(**filtered_params)

            _append_reward(inst, apply_flag)

        self.rollout_store = RolloutStore("rollouts")

        if not self._mf_split_fsdp:
            # Prepare (wrap) the model for the chosen distributed strategy first
            self.model = self.accelerator.prepare(self.model)

            # Optimiser – must be created *after* model wrapping when using FSDP
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            # Distribute optimiser
            self.optimizer = self.accelerator.prepare(self.optimizer)
        else:
            # Two optimizers, one per subgroup
            self.optimizer = None  # unused in split mode
            self.optimizer_mind = None
            self.optimizer_face = None
            if getattr(self, "mind_model", None) is not None:
                self.optimizer_mind = torch.optim.AdamW(
                    self.mind_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
                )
            if getattr(self, "face_model", None) is not None:
                self.optimizer_face = torch.optim.AdamW(
                    self.face_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
                )

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
        if not self._mf_split_fsdp:
            self.optimizer.zero_grad()
        else:
            if self.optimizer_mind is not None:
                self.optimizer_mind.zero_grad()
            if self.optimizer_face is not None:
                self.optimizer_face.zero_grad()

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

        # Store raw prompt texts for potential MindFace generation.  We make a
        # *shallow* copy to avoid side-effects when the dict is later passed
        # into HF generate (the extra key is removed beforehand).
        inputs_with_texts = {k: v for k, v in inputs.items()}
        inputs_with_texts["_prompt_texts"] = prompt_texts  # type: ignore[assignment]
        return Batch(prompts, prompt_texts, inputs_with_texts), prompt_iter

    def _generate_sequences(self, inputs, gen_kwargs):
        """Run `model.generate` taking MindFace wrapper into account."""
        # When using MindFace we call its specialised `generate` API.
        from ..models.mind_face import MindFace  # local import

        # Handle potential wrapping by Accelerate/torch.distributed – the actual
        # MindFace instance can be at self.model or self.model.module.
        model_for_check = self.model
        if not isinstance(model_for_check, MindFace) and hasattr(model_for_check, "module"):
            model_for_check = model_for_check.module  # type: ignore[attr-defined]

        if isinstance(model_for_check, MindFace):
            # inputs are not required – we rely on prompt texts instead.
            # We expect that `inputs` contains *prompt_texts* provided under a
            # special key injected by `_next_batch`.
            prompt_texts = inputs.pop("_prompt_texts")  # type: ignore[arg-type]
            max_think = self.cfg.thinking_max_tokens or 0
            max_new = self.cfg.output_max_tokens or 512
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Exclude keys that are explicitly passed below to avoid duplicate arguments
                safe_kwargs = {
                    k: v
                    for k, v in gen_kwargs.items()
                    if k not in ("logits_processor", "max_new_tokens", "output_scores", "return_dict_in_generate")
                }
                generated = model_for_check.generate(
                    prompt_inputs=prompt_texts,
                    max_thinking_tokens=max_think,
                    max_new_tokens=max_new,
                    **safe_kwargs,
                )
            return generated.sequences

        # ----- Standard single-model path -----
        safe_inputs = {k: v for k, v in inputs.items() if k != "_prompt_texts"}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self._generate(**safe_inputs, **gen_kwargs)
        return generated.sequences  # (batch, seq_len)

    def _compute_logprobs(self, sequences, inputs):
        """Compute summed log-probabilities of newly generated tokens.

        Returns a tuple of three 1-D tensors of length *batch*:
        (logprob_total, logprob_thinking, logprob_output)
        where *thinking* refers to tokens up to and **including** the first
        ``</think>`` token, and *output* refers to tokens after that marker.
        When the end marker is absent, the whole generation is considered
        thinking and *logprob_output* becomes zero.
        """
        # Shift inputs/targets for language-modeling likelihood
        input_ids_full = sequences[:, :-1]
        target_ids = sequences[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids_full)
        logits_full = outputs.logits  # (batch, seq_len-1, vocab)
        logprobs_full = F.log_softmax(logits_full, dim=-1)

        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Mask newly generated tokens (positions >= common left-padded prompt length)
        seq_len = sequences.shape[1]
        arange_seq = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        # With left padding, the common prompt length equals the padded input length
        prompt_len_common = inputs["input_ids"].shape[1]
        mask_full = arange_seq.expand(sequences.size(0), -1) >= prompt_len_common
        mask = mask_full[:, 1:]  # align with targets
        selected_logprobs = token_logprobs * mask.float()

        # ------------------------------------------------------------------
        # Split into thinking / output segments per sample
        # ------------------------------------------------------------------
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]

        B, Lm1 = selected_logprobs.shape  # seq_len-1

        logprob_thinking = torch.zeros(B, device=sequences.device)
        logprob_output = torch.zeros(B, device=sequences.device)

        for b in range(B):
            # Absolute position (in *sequences*) of the first </think> token.
            seq_list = sequences[b].tolist()
            try:
                end_think_id in seq_list[prompt_len_common:]
                rel_idx = seq_list[prompt_len_common:].index(end_think_id)
                end_idx = prompt_len_common + rel_idx  # inclusive position of </think>
            except ValueError:
                raise ValueError(f"No </think> token found in sequence {b}")

            # Convert to target index space (length L-1) – target idx t predicts token t+1
            target_end_idx = max(end_idx - 1, prompt_len_common)  # ensure non-negative

            think_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
            output_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)

            think_mask_row[prompt_len_common: target_end_idx + 1] = True
            output_mask_row[target_end_idx + 1 :] = True

            # if b == 0:
            #     print("--------------------------------\n\nthink_tokens\n\n", self.tokenizer.decode(input_ids_full[b, think_mask_row]))
            #     print("--------------------------------\n\noutput_tokens\n\n", self.tokenizer.decode(input_ids_full[b, output_mask_row]))

            logprob_thinking[b] = (token_logprobs[b] * think_mask_row.float()).sum()
            logprob_output[b] = (token_logprobs[b] * output_mask_row.float()).sum()

        logprob_total = logprob_thinking + logprob_output
        return logprob_total, logprob_thinking, logprob_output

    def _compute_rewards(
        self,
        prompts,
        sequences,
        inputs,
        logprob_total,
        logprob_thinking,
        logprob_output,
    ):
        """Compute rewards and REINFORCE loss with *apply_to_thinking* control."""

        # ------------------------------------------------------------------
        # Evaluate reward functions (sync & async support) -------------------
        # ------------------------------------------------------------------
        rollouts: List[Dict] = []
        num_rewards = len(self.rewards)

        # We'll build a *tensor* of shape (num_rewards, B)
        reward_values = torch.zeros((num_rewards, len(prompts)), device=sequences.device)

        has_async = any(inspect.iscoroutinefunction(r.__call__) for r in self.rewards)

        if has_async:
            async def _async_eval():
                all_tasks = []
                task_to_index = {}
                rollout_data = []

                for i, p in enumerate(prompts):
                    response_text = self.tokenizer.decode(
                        sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    rollout = dict(p)
                    rollout["response"] = response_text

                    sync_results = []
                    async_tasks = []
                    for j, rwd in enumerate(self.rewards):
                        if inspect.iscoroutinefunction(rwd.__call__):
                            task = rwd(rollout)
                            async_tasks.append((task, j))
                            all_tasks.append(task)
                            task_to_index[task] = (i, j)
                        else:
                            val = rwd(rollout)
                            sync_results.append((j, val))
                    rollout["_sync_results"] = sync_results
                    rollout["_async_tags"] = async_tasks
                    rollout_data.append(rollout)

                async_vals = []
                if all_tasks:
                    async_vals = await asyncio.gather(*all_tasks, return_exceptions=True)

                # Fill reward_values tensor
                for i_idx, rollout in enumerate(rollout_data):
                    for j, val in rollout["_sync_results"]:
                        reward_values[j, i_idx] = float(val)

                for task, res in zip(all_tasks, async_vals):
                    i, j = task_to_index[task]
                    if isinstance(res, Exception):
                        print(f"[Warning] Async reward failed: {res}")
                        res = 0.0
                    reward_values[j, i] = float(res)

                # Build rollouts + total rewards list
                rewards_total = []
                for i, rollout in enumerate(rollout_data):
                    total_r = reward_values[:, i].sum().item()
                    rollout.pop("_sync_results", None)
                    rollout.pop("_async_tags", None)
                    rollout["total_reward"] = total_r
                    rollout["logprob_sum"] = logprob_total[i].item()
                    rollouts.append(rollout)
                    rewards_total.append(total_r)

                return rewards_total

            rewards_list = asyncio.run(_async_eval())

        else:  # synchronous evaluation
            rewards_list: List[float] = []
            for i, p in enumerate(prompts):
                response_text = self.tokenizer.decode(
                    sequences[i, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                rollout = dict(p)
                rollout["response"] = response_text

                total_reward = 0.0
                for j, rwd in enumerate(self.rewards):
                    val = float(rwd(rollout))
                    reward_values[j, i] = val
                    total_reward += val

                rollout["total_reward"] = total_reward
                rollout["logprob_sum"] = logprob_total[i].item()
                rollouts.append(rollout)
                rewards_list.append(total_reward)

        # ------------------------------------------------------------------
        # Compute loss -------------------------------------------------------
        # ------------------------------------------------------------------
        num_rewards = reward_values.size(0)

        rewards_t = torch.tensor(rewards_list, device=sequences.device, dtype=torch.float32)

        # Per-reward baseline → advantage
        baseline_per_reward = reward_values.mean(dim=1, keepdim=True)  # (R,1)
        advantages = reward_values - baseline_per_reward  # (R,B)

        # Select log-prob segment per reward based on apply_to_thinking flag
        apply_flags = torch.tensor(self.rewards_apply_flags, device=sequences.device, dtype=torch.bool)

        logp_total_mat = logprob_total.unsqueeze(0).expand(num_rewards, -1)
        logp_output_mat = logprob_output.unsqueeze(0).expand(num_rewards, -1)
        logp_seg = torch.where(apply_flags.unsqueeze(1), logp_total_mat, logp_output_mat)

        loss_mat = -(advantages * logp_seg)  # (R,B)
        loss = loss_mat.mean()
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
            # Gradient clipping to stabilise training
            try:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            except AttributeError:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
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
        # Aggregate reward breakdowns (mean over GLOBAL batch across all processes)
        # First, compute local totals
        breakdown_totals_local = {}
        for rollout in rollouts:
            for k, v in rollout.get("reward_breakdown", {}).items():
                breakdown_totals_local[k] = breakdown_totals_local.get(k, 0.0) + v

        # Prepare tensors for distributed reduction
        device = getattr(self.model, "device", rewards_t.device)
        local_count = torch.tensor([len(rollouts)], device=device, dtype=torch.int64)
        reward_sum_local = rewards_t.to(torch.float32).sum()

        # Copy to tensors we can all-reduce
        reward_sum_global = reward_sum_local.clone()
        count_global = local_count.clone()

        # Reduce reward sum and count across processes (if distributed)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(reward_sum_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_global, op=dist.ReduceOp.SUM)

        # Compute global mean reward
        total_count = max(1, int(count_global.item()))
        reward_mean_global = (reward_sum_global.item() / total_count)

        # Reduce breakdown sums across processes key-by-key
        breakdown_means_global = {}
        for k, v in breakdown_totals_local.items():
            t_local = torch.tensor([float(v)], device=device, dtype=torch.float32)
            t_global = t_local.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t_global, op=dist.ReduceOp.SUM)
            breakdown_means_global[f"reward/{k}"] = t_global.item() / total_count

        log_dict = {
            "loss": loss.item(),
            "reward/mean": reward_mean_global,
            "episode": episode_counter,
            "global_step": self.global_step,
        }
        log_dict.update(breakdown_means_global)

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
        if self.accelerator.is_local_main_process:
            self.rollout_store.clear_all_rollouts()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Route to split mind-face path when enabled
        if getattr(self, "_mf_split_fsdp", False):
            return self._train_mind_face_split()

        # Build runtime helpers
        tp = self._token_processor()
        gen_kwargs = self._build_gen_kwargs(tp)
        from tqdm import tqdm  # local import to avoid cost when not training

        prompt_iter = iter(self.prompt_builder)
        episode_counter = 0
        global_prompts_processed = 0
        is_main_process = self.accelerator.is_local_main_process
        world_size = getattr(self.accelerator, "num_processes", 1)

        with tqdm(total=cfg.num_episodes, desc="Prompts processed", disable=not is_main_process) as pbar:
            while global_prompts_processed < cfg.num_episodes:
                # Default skip size in case OOM occurs before batch is fully built
                batch_size_for_skip = cfg.batch_size
                try:
                    batch, prompt_iter = self._next_batch(prompt_iter, tp)
                    batch_size_for_skip = len(batch.prompts)

                    # Compute global batch size across all processes
                    local_batch_size = len(batch.prompts)
                    global_batch_size = local_batch_size
                    if world_size > 1 and dist.is_available() and dist.is_initialized():
                        bs_tensor = torch.tensor([local_batch_size], device=self.model.device, dtype=torch.int64)
                        dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)
                        global_batch_size = int(bs_tensor.item())

                    sequences = self._generate_sequences(batch.inputs, gen_kwargs)
                    logprob_total, logprob_thinking, logprob_output = self._compute_logprobs(
                        sequences, batch.inputs
                    )

                    rewards_t, rollouts, loss = self._compute_rewards(
                        batch.prompts,
                        sequences,
                        batch.inputs,
                        logprob_total,
                        logprob_thinking,
                        logprob_output,
                    )

                    # Gradient accumulation / optimiser step
                    is_last_batch = (
                        global_prompts_processed + global_batch_size >= cfg.num_episodes
                    )
                    stepped = self._backprop(loss, final_batch=is_last_batch)

                    # Book-keeping
                    self._store_rollouts(rollouts)
                    self._log_step(loss, rewards_t, rollouts, episode_counter)

                    # Counters / progress
                    episode_counter += 1
                    global_prompts_processed += global_batch_size
                    if is_main_process:
                        remaining = cfg.num_episodes - pbar.n
                        pbar.update(min(global_batch_size, remaining))
                    if stepped:
                        self.global_step += 1

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # Treat runtime errors containing OOM text as CUDA OOMs as well
                    message = str(e).lower()
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                        "out of memory" in message or "cuda oom" in message
                    )
                    if not is_oom:
                        raise

                    print("OOM error, skipping update")
                    # Best-effort cleanup and proceed to next gradient step
                    try:
                        self.optimizer.zero_grad()
                    except Exception:
                        pass
                    self._accum_counter = 0
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass

                    # Advance counters to avoid getting stuck retrying the same batch
                    episode_counter += 1
                    # Compute global skip size across all processes
                    global_skip = batch_size_for_skip
                    if world_size > 1 and dist.is_available() and dist.is_initialized():
                        skip_tensor = torch.tensor([batch_size_for_skip], device=self.model.device, dtype=torch.int64)
                        dist.all_reduce(skip_tensor, op=dist.ReduceOp.SUM)
                        global_skip = int(skip_tensor.item())
                    global_prompts_processed += global_skip
                    if is_main_process:
                        remaining = cfg.num_episodes - pbar.n
                        pbar.update(min(global_skip, remaining))
                    # Continue training loop
                    continue

        # Upload saved rollouts to Weights & Biases as an artifact (main process only)
        if cfg.save_rollouts_to_wandb and is_main_process:
            try:
                artifact = wandb.Artifact("rollouts", type="dataset")
                artifact.add_dir(str(self.rollout_store.dir))
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"[Warning] Failed to log rollouts to wandb: {e}")

    # ------------------------------------------------------------------
    # Split mind-face training (FSDP subgroups)
    # ------------------------------------------------------------------
    def _train_mind_face_split(self):
        cfg = self.cfg
        from tqdm import tqdm

        # Helpers
        def group_rank(ranks: List[int]) -> int:
            r = dist.get_rank()
            return ranks.index(r) if r in ranks else -1

        def shard_indices(total: int, world: int, r: int):
            base = total // world
            rem = total % world
            start = r * base + min(r, rem)
            end = start + base + (1 if r < rem else 0)
            return start, end

        is_main_process = self.accelerator.is_local_main_process
        episode_counter = 0
        global_prompts_processed = 0

        with tqdm(total=cfg.num_episodes, desc="Prompts processed", disable=not is_main_process) as pbar:
            while global_prompts_processed < cfg.num_episodes:
                batch_size_for_skip = cfg.batch_size
                try:
                    print(f"[rank{dist.get_rank()}] ---- New iteration start ----", flush=True)
                    # Build prompts on global rank 0
                    prompts: List[Dict] = []
                    prompt_texts: List[str] = []
                    if dist.get_rank() == 0:
                        print(f"[rank0] Building prompts (batch_size={cfg.batch_size})", flush=True)
                        tp = BatchThinkingTokenBudgetProcessor(
                            self.tokenizer,
                            max_thinking_tokens=cfg.thinking_max_tokens,
                            min_thinking_tokens=cfg.thinking_min_tokens,
                            batch_size=cfg.batch_size,
                        )
                        tp.reset()
                        prompt_iter = getattr(self, "_mf_prompt_iter", None)
                        if prompt_iter is None:
                            self._mf_prompt_iter = iter(self.prompt_builder)
                            prompt_iter = self._mf_prompt_iter
                        for _ in range(cfg.batch_size):
                            try:
                                p = next(prompt_iter)
                            except StopIteration:
                                self._mf_prompt_iter = iter(self.prompt_builder)
                                prompt_iter = self._mf_prompt_iter
                                p = next(prompt_iter)
                            prompts.append(p)
                        # Build chat prompts
                        for p in prompts:
                            user_content = p["prompt"].split("\n<assistant>")[0]
                            chat_str = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": user_content}],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                            prompt_texts.append(chat_str)
                        print(f"[rank0] Built {len(prompts)} prompts", flush=True)

                    # Broadcast prompts and texts to all ranks
                    print(f"[rank{dist.get_rank()}] Broadcasting prompts from rank0", flush=True)
                    obj_list = [prompts, prompt_texts]
                    dist.broadcast_object_list(obj_list, src=0)
                    prompts, prompt_texts = obj_list  # type: ignore[assignment]
                    print(f"[rank{dist.get_rank()}] Received prompts: {len(prompts)}", flush=True)

                    B = len(prompts)
                    batch_size_for_skip = B
                    print(f"[rank{dist.get_rank()}] Global batch size B={B}", flush=True)

                    # ---------------- Mind generation on mind group ----------------
                    local_mind_seqs = None
                    if getattr(self, "_is_mind_rank", False) and self.mind_model is not None:
                        r = group_rank(self._mind_ranks)
                        world = len(self._mind_ranks)
                        s, e = shard_indices(B, world, r)
                        print(f"[rank{dist.get_rank()}|mind] shard_indices: start={s} end={e} world={world}", flush=True)
                        if e > s:
                            local_texts = prompt_texts[s:e]
                            print(f"[rank{dist.get_rank()}|mind] local_texts={len(local_texts)}", flush=True)
                            # Local logits processor for local shard size
                            tp_local = BatchThinkingTokenBudgetProcessor(
                                self.tokenizer,
                                max_thinking_tokens=cfg.thinking_max_tokens,
                                min_thinking_tokens=cfg.thinking_min_tokens,
                                batch_size=len(local_texts),
                            )
                            tp_local.reset()
                            print(f"[rank{dist.get_rank()}|mind] Starting generate (think max={cfg.thinking_max_tokens})", flush=True)
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                out = self.mind_model.generate(
                                    **self.tokenizer(local_texts, return_tensors="pt", padding=True).to(
                                        self.model.device
                                    ),
                                    max_new_tokens=cfg.thinking_max_tokens,
                                    logits_processor=[tp_local],
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                    do_sample=True,
                                    temperature=1.0,
                                )
                            local_mind_seqs = out.sequences  # type: ignore[attr-defined]
                            print(f"[rank{dist.get_rank()}|mind] local_mind_seqs shape={tuple(local_mind_seqs.shape)}", flush=True)
                        else:
                            # No samples on this shard
                            local_mind_seqs = torch.empty((0, 0), device=self.model.device, dtype=torch.long)
                            print(f"[rank{dist.get_rank()}|mind] empty shard", flush=True)

                    # Gather all local mind sequences to mind root and assemble global
                    pad_id = self.tokenizer.pad_token_id
                    device = self.model.device
                    mind_world = len(self._mind_ranks)
                    mind_root = self._mind_ranks[0]

                    # Gather per-rank sizes (mind group only), assemble on mind_root
                    global_mind_seq = None
                    if getattr(self, "_is_mind_rank", False):
                        local_rows = torch.tensor([0], device=device, dtype=torch.int64)
                        local_cols = torch.tensor([0], device=device, dtype=torch.int64)
                        if local_mind_seqs is not None and local_mind_seqs.numel() > 0:
                            local_rows = torch.tensor([local_mind_seqs.size(0)], device=device, dtype=torch.int64)
                            local_cols = torch.tensor([local_mind_seqs.size(1)], device=device, dtype=torch.int64)
                        rows_list = [torch.zeros_like(local_rows) for _ in range(mind_world)]
                        cols_list = [torch.zeros_like(local_cols) for _ in range(mind_world)]
                        print(f"[rank{dist.get_rank()}|mind] all_gather rows/cols (local_rows={int(local_rows.item())}, local_cols={int(local_cols.item())})", flush=True)
                        if mind_world > 0:
                            dist.all_gather(rows_list, local_rows, group=self._pg_mind)
                            dist.all_gather(cols_list, local_cols, group=self._pg_mind)
                        rows = [int(t.item()) for t in rows_list]
                        cols = [int(t.item()) for t in cols_list]
                        print(f"[rank{dist.get_rank()}|mind] gathered rows={rows} cols={cols}", flush=True)
                        max_rows = max(rows) if rows else 0
                        max_cols = max(cols) if cols else 0
                        # Prepare padded local block
                        local_block = torch.full((max_rows, max_cols), pad_id, device=device, dtype=torch.long)
                        if local_mind_seqs is not None and local_mind_seqs.numel() > 0:
                            local_block[: local_mind_seqs.size(0), : local_mind_seqs.size(1)] = local_mind_seqs
                        gathered = [torch.empty_like(local_block) for _ in range(mind_world)]
                        print(f"[rank{dist.get_rank()}|mind] all_gather blocks (block_shape={tuple(local_block.shape)})", flush=True)
                        if mind_world > 0:
                            dist.all_gather(gathered, local_block, group=self._pg_mind)
                        if dist.get_rank() == mind_root:
                            global_mind = []
                            for i, g in enumerate(gathered):
                                ri = rows[i] if i < len(rows) else 0
                                if ri > 0:
                                    global_mind.append(g[:ri, :])
                            global_mind_seq = torch.cat(global_mind, dim=0) if global_mind else torch.empty((0, 0), device=device, dtype=torch.long)
                            print(f"[rank{dist.get_rank()}|mind] Assembled global_mind_seq shape={tuple(global_mind_seq.shape)}", flush=True)

                    # Broadcast assembled mind sequences to all ranks (shape first)
                    if dist.get_rank() == mind_root:
                        if global_mind_seq is None:
                            global_mind_seq = torch.empty((0, 0), device=device, dtype=torch.long)
                        shape_tensor = torch.tensor([global_mind_seq.size(0), global_mind_seq.size(1)], device=device, dtype=torch.int64)
                    else:
                        shape_tensor = torch.empty((2,), device=device, dtype=torch.int64)
                    print(f"[rank{dist.get_rank()}] Broadcasting mind shape from root {mind_root}", flush=True)
                    dist.broadcast(shape_tensor, src=mind_root)
                    if dist.get_rank() != mind_root:
                        h, w = int(shape_tensor[0].item()), int(shape_tensor[1].item())
                        global_mind_seq = torch.empty((h, w), device=device, dtype=torch.long)
                    print(f"[rank{dist.get_rank()}] Broadcasting mind payload", flush=True)
                    dist.broadcast(global_mind_seq, src=mind_root)
                    try:
                        print(f"[rank{dist.get_rank()}] Mind broadcast shape: {int(shape_tensor[0].item())} x {int(shape_tensor[1].item())}")
                    except Exception:
                        pass

                    # Compute end_think indices and per-sample L on rank 0; broadcast
                    end_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
                    L_list = []
                    end_idx_list = []
                    if dist.get_rank() == 0:
                        toks = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
                        L = toks["input_ids"].shape[1]
                        print(f"[rank0] Computing end_idx_list with common L={L}", flush=True)
                        for b in range(B):
                            seq_list = global_mind_seq[b].tolist()
                            rel = seq_list[L:]
                            rel_idx = rel.index(end_id) if end_id in rel else (len(rel) - 1)
                            end_idx_list.append(L + rel_idx)
                            L_list.append(L)
                        print(f"[rank0] end_idx_list(first5)={end_idx_list[:5]}", flush=True)
                    obj_sizes = [end_idx_list, L_list]
                    print(f"[rank{dist.get_rank()}] Broadcasting end_idx_list/L_list from rank0", flush=True)
                    dist.broadcast_object_list(obj_sizes, src=0)
                    end_idx_list, L_list = obj_sizes  # type: ignore[assignment]

                    # Face max new tokens from min thinking length
                    min_think = min([end_idx_list[i] - L_list[i] for i in range(B)]) if B > 0 else 0
                    face_max_new = max((cfg.output_max_tokens or 512) - min_think, 0)
                    print(f"[rank{dist.get_rank()}] min_think={min_think} face_max_new={face_max_new}", flush=True)
                    face_max_new_t = torch.tensor([face_max_new], device=device, dtype=torch.int64)
                    dist.broadcast(face_max_new_t, src=0)
                    face_max_new = int(face_max_new_t.item())

                    # ---------------- Face generation on face group ----------------
                    local_face_full = None
                    if getattr(self, "_is_face_rank", False) and self.face_model is not None:
                        r = group_rank(self._face_ranks)
                        world = len(self._face_ranks)
                        s, e = shard_indices(B, world, r)
                        if e > s:
                            # Build per-shard inputs by trimming up to end_idx inclusive
                            rows = []
                            for i in range(s, e):
                                end_i = end_idx_list[i]
                                rows.append(global_mind_seq[i, : end_i + 1])
                            # Pad within shard (left)
                            max_len = max([row.numel() for row in rows])
                            pad_id = self.tokenizer.pad_token_id
                            padded = []
                            attn = []
                            for row in rows:
                                pad_len = max_len - row.numel()
                                padded_row = torch.cat([torch.full((pad_len,), pad_id, device=device, dtype=torch.long), row.to(device)])
                                padded.append(padded_row)
                                attn.append((padded_row != pad_id).long())
                            input_ids = torch.stack(padded, dim=0)
                            attention_mask = torch.stack(attn, dim=0)
                            print(f"[rank{dist.get_rank()}|face] Starting generate (face max={face_max_new}) input_ids shape={tuple(input_ids.shape)}", flush=True)
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                out = self.face_model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=face_max_new,
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                    do_sample=True,
                                    temperature=1.0,
                                )
                            local_face_full = out.sequences  # type: ignore[attr-defined]
                            print(f"[rank{dist.get_rank()}|face] local_face_full shape={tuple(local_face_full.shape)}", flush=True)
                        else:
                            local_face_full = torch.empty((0, global_mind_seq.size(1)), device=device, dtype=torch.long)
                            print(f"[rank{dist.get_rank()}|face] empty shard", flush=True)

                    # Gather full sequences from face group and broadcast to all
                    face_world = len(self._face_ranks)
                    global_full_seq = None
                    if getattr(self, "_is_face_rank", False):
                        local_rows = torch.tensor([0], device=device, dtype=torch.int64)
                        local_cols = torch.tensor([0], device=device, dtype=torch.int64)
                        if local_face_full is not None and local_face_full.numel() > 0:
                            local_rows = torch.tensor([local_face_full.size(0)], device=device, dtype=torch.int64)
                            local_cols = torch.tensor([local_face_full.size(1)], device=device, dtype=torch.int64)
                        rows_list = [torch.zeros_like(local_rows) for _ in range(face_world)]
                        cols_list = [torch.zeros_like(local_cols) for _ in range(face_world)]
                        print(f"[rank{dist.get_rank()}|face] all_gather rows/cols (local_rows={int(local_rows.item())}, local_cols={int(local_cols.item())})", flush=True)
                        if face_world > 0:
                            dist.all_gather(rows_list, local_rows, group=self._pg_face)
                            dist.all_gather(cols_list, local_cols, group=self._pg_face)
                        rows = [int(t.item()) for t in rows_list]
                        cols = [int(t.item()) for t in cols_list]
                        print(f"[rank{dist.get_rank()}|face] gathered rows={rows} cols={cols}", flush=True)
                        max_rows = max(rows) if rows else 0
                        max_cols = max(cols) if cols else 0
                        local_block = torch.full((max_rows, max_cols), pad_id, device=device, dtype=torch.long)
                        if local_face_full is not None and local_face_full.numel() > 0:
                            local_block[: local_face_full.size(0), : local_face_full.size(1)] = local_face_full
                        gathered = [torch.empty_like(local_block) for _ in range(face_world)]
                        print(f"[rank{dist.get_rank()}|face] all_gather blocks (block_shape={tuple(local_block.shape)})", flush=True)
                        if face_world > 0:
                            dist.all_gather(gathered, local_block, group=self._pg_face)
                        if dist.get_rank() == self._face_ranks[0] if face_world > 0 else False:
                            global_full = []
                            for i, g in enumerate(gathered):
                                ri = rows[i] if i < len(rows) else 0
                                if ri > 0:
                                    global_full.append(g[:ri, :])
                            global_full_seq = torch.cat(global_full, dim=0) if global_full else torch.empty((0, 0), device=device, dtype=torch.long)
                            print(f"[rank{dist.get_rank()}|face] Assembled global_full_seq shape={tuple(global_full_seq.shape)}", flush=True)

                    face_root = self._face_ranks[0] if face_world > 0 else 0
                    if dist.get_rank() == face_root:
                        if global_full_seq is None:
                            global_full_seq = torch.empty((0, 0), device=device, dtype=torch.long)
                        shape_tensor = torch.tensor([global_full_seq.size(0), global_full_seq.size(1)], device=device, dtype=torch.int64)
                    else:
                        shape_tensor = torch.empty((2,), device=device, dtype=torch.int64)
                    print(f"[rank{dist.get_rank()}] Broadcasting face shape from root {face_root}", flush=True)
                    dist.broadcast(shape_tensor, src=face_root)
                    if dist.get_rank() != face_root:
                        h, w = int(shape_tensor[0].item()), int(shape_tensor[1].item())
                        global_full_seq = torch.empty((h, w), device=device, dtype=torch.long)
                    print(f"[rank{dist.get_rank()}] Broadcasting face payload", flush=True)
                    dist.broadcast(global_full_seq, src=face_root)
                    try:
                        print(f"[rank{dist.get_rank()}] Face broadcast shape: {int(shape_tensor[0].item())} x {int(shape_tensor[1].item())}")
                    except Exception:
                        pass

                    # ---------------- Rewards (rank 0) ----------------
                    rollouts: List[Dict] = []
                    reward_values = None
                    rewards_list = None
                    if dist.get_rank() == 0:
                        toks = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
                        L = toks["input_ids"].shape[1]
                        end_idx_t = torch.tensor(end_idx_list, device=device, dtype=torch.int64)
                        # Build response texts
                        for i, p in enumerate(prompts):
                            response_text = self.tokenizer.decode(
                                global_full_seq[i, L:], skip_special_tokens=True
                            )
                            ro = dict(p)
                            ro["response"] = response_text
                            rollouts.append(ro)
                        # Evaluate rewards
                        num_rewards = len(self.rewards)
                        reward_values = torch.zeros((num_rewards, B), device=device, dtype=torch.float32)
                        rewards_list_f: List[float] = []
                        for i in range(B):
                            total = 0.0
                            rollout = rollouts[i]
                            for j, rwd in enumerate(self.rewards):
                                val = float(rwd(rollout))
                                reward_values[j, i] = val
                                total += val
                            rewards_list_f.append(total)
                        rewards_list = torch.tensor(rewards_list_f, device=device, dtype=torch.float32)
                    # Broadcast rewards
                    if reward_values is None:
                        reward_values = torch.zeros((len(self.rewards), B), device=device, dtype=torch.float32)
                        rewards_list = torch.zeros((B,), device=device, dtype=torch.float32)
                    print(f"[rank{dist.get_rank()}] Broadcasting rewards tensors", flush=True)
                    dist.broadcast(reward_values, src=0)
                    dist.broadcast(rewards_list, src=0)

                    # ---------------- Logprobs split ----------------
                    # Mind group computes thinking logprobs
                    logprob_thinking = torch.zeros((B,), device=device, dtype=torch.float32)
                    if getattr(self, "_is_mind_rank", False) and self.mind_model is not None and B > 0:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = self.mind_model(global_full_seq[:, :-1])
                        logits_full = outputs.logits
                        logprobs_full = F.log_softmax(logits_full, dim=-1)
                        target_ids = global_full_seq[:, 1:]
                        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        L_common = L_list[0] if L_list else 0
                        end_idx_t = torch.tensor(end_idx_list, device=device, dtype=torch.int64)
                        Bsz, Lm1 = token_logprobs.shape
                        print(f"[rank{dist.get_rank()}|mind] Computing logprob_thinking: Bsz={Bsz} Lm1={Lm1} L_common={L_common}", flush=True)
                        for b in range(Bsz):
                            target_end_idx = max(int(end_idx_t[b].item()) - 1, L_common)
                            think_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=device)
                            think_mask_row[L_common : target_end_idx + 1] = True
                            logprob_thinking[b] = (token_logprobs[b] * think_mask_row.float()).sum()
                    print(f"[rank{dist.get_rank()}] all_reduce logprob_thinking", flush=True)
                    dist.all_reduce(logprob_thinking, op=dist.ReduceOp.SUM)

                    # Face group computes output logprobs
                    logprob_output = torch.zeros((B,), device=device, dtype=torch.float32)
                    if getattr(self, "_is_face_rank", False) and self.face_model is not None and B > 0:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = self.face_model(global_full_seq[:, :-1])
                        logits_full = outputs.logits
                        logprobs_full = F.log_softmax(logits_full, dim=-1)
                        target_ids = global_full_seq[:, 1:]
                        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        L_common = L_list[0] if L_list else 0
                        end_idx_t = torch.tensor(end_idx_list, device=device, dtype=torch.int64)
                        Bsz, Lm1 = token_logprobs.shape
                        print(f"[rank{dist.get_rank()}|face] Computing logprob_output: Bsz={Bsz} Lm1={Lm1} L_common={L_common}", flush=True)
                        for b in range(Bsz):
                            target_end_idx = max(int(end_idx_t[b].item()) - 1, L_common)
                            output_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=device)
                            output_mask_row[target_end_idx + 1 :] = True
                            logprob_output[b] = (token_logprobs[b] * output_mask_row.float()).sum()
                    print(f"[rank{dist.get_rank()}] all_reduce logprob_output", flush=True)
                    dist.all_reduce(logprob_output, op=dist.ReduceOp.SUM)

                    logprob_total = logprob_thinking + logprob_output
                    if dist.get_rank() == 0:
                        print(f"[rank0] logprob_total mean={float(logprob_total.mean().item())}", flush=True)

                    # ---------------- Loss split & backprop ----------------
                    num_rewards = reward_values.size(0)
                    baseline_per_reward = reward_values.mean(dim=1, keepdim=True)
                    advantages = reward_values - baseline_per_reward
                    apply_flags = torch.tensor(self.rewards_apply_flags, device=device, dtype=torch.bool)
                    logp_think_mat = logprob_thinking.unsqueeze(0).expand(num_rewards, -1)
                    logp_out_mat = logprob_output.unsqueeze(0).expand(num_rewards, -1)
                    # Mind loss: only rewards applied to thinking
                    loss_mind = None
                    if getattr(self, "_is_mind_rank", False) and self.mind_model is not None:
                        sel = apply_flags.unsqueeze(1)
                        loss_mind_mat = -(advantages * logp_think_mat) * sel
                        loss_mind = loss_mind_mat.mean()
                        print(f"[rank{dist.get_rank()}|mind] loss_mind={float(loss_mind.item())}", flush=True)
                    # Face loss: all rewards apply to output
                    loss_face = None
                    if getattr(self, "_is_face_rank", False) and self.face_model is not None:
                        loss_face_mat = -(advantages * logp_out_mat)
                        loss_face = loss_face_mat.mean()
                        print(f"[rank{dist.get_rank()}|face] loss_face={float(loss_face.item())}", flush=True)

                    # Grad accumulation boundaries enforced globally
                    # Scale
                    if loss_mind is not None:
                        (loss_mind / self.grad_accum_steps).backward()
                    if loss_face is not None:
                        (loss_face / self.grad_accum_steps).backward()

                    # Step when accumulation boundary
                    stepped_local = False
                    if not hasattr(self, "_accum_counter"):
                        self._accum_counter = 0
                    self._accum_counter += 1
                    is_last_batch = (global_prompts_processed + B >= cfg.num_episodes)
                    print(f"[rank{dist.get_rank()}] accum_counter={self._accum_counter} is_last_batch={is_last_batch}", flush=True)
                    if self._accum_counter >= self.grad_accum_steps or is_last_batch:
                        if getattr(self, "_is_mind_rank", False) and self.mind_model is not None and self.optimizer_mind is not None:
                            zero_special_token_grads(self.mind_model, self.tokenizer)
                            torch.nn.utils.clip_grad_norm_(self.mind_model.parameters(), self.cfg.max_grad_norm)
                            self.optimizer_mind.step()
                            self.optimizer_mind.zero_grad()
                            stepped_local = True
                        if getattr(self, "_is_face_rank", False) and self.face_model is not None and self.optimizer_face is not None:
                            zero_special_token_grads(self.face_model, self.tokenizer)
                            torch.nn.utils.clip_grad_norm_(self.face_model.parameters(), self.cfg.max_grad_norm)
                            self.optimizer_face.step()
                            self.optimizer_face.zero_grad()
                            stepped_local = True
                        self._accum_counter = 0
                        print(f"[rank{dist.get_rank()}] optimizer step (stepped_local={stepped_local})", flush=True)

                    # Rollouts saved on rank 0
                    if dist.get_rank() == 0:
                        # Attach total_reward and logprob_sum for completeness
                        for i in range(B):
                            rollouts[i]["total_reward"] = float(rewards_list[i].item())
                            rollouts[i]["logprob_sum"] = float(logprob_total[i].item())
                        self._store_rollouts(rollouts)

                    # Logging
                    loss_for_log = (loss_mind if loss_mind is not None else 0.0) + (loss_face if loss_face is not None else 0.0)
                    rewards_t = rewards_list
                    self._log_step(loss_for_log, rewards_t, rollouts if dist.get_rank() == 0 else [], episode_counter)

                    # Progress
                    episode_counter += 1
                    global_prompts_processed += B
                    if is_main_process:
                        remaining = cfg.num_episodes - pbar.n
                        pbar.update(min(B, remaining))
                    if stepped_local and is_main_process:
                        self.global_step += 1

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    message = str(e).lower()
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in message or "cuda oom" in message)
                    if not is_oom:
                        print(f"[rank{dist.get_rank()}] Non-OOM runtime error: {repr(e)}", flush=True)
                        raise
                    print(f"[rank{dist.get_rank()}] OOM error, skipping update", flush=True)
                    try:
                        if getattr(self, "optimizer_mind", None) is not None:
                            self.optimizer_mind.zero_grad()
                        if getattr(self, "optimizer_face", None) is not None:
                            self.optimizer_face.zero_grad()
                    except Exception:
                        pass
                    self._accum_counter = 0
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        gc.collect()
                    except Exception:
                        pass
                    episode_counter += 1
                    global_prompts_processed += batch_size_for_skip
                    if is_main_process:
                        remaining = cfg.num_episodes - pbar.n
                        pbar.update(min(batch_size_for_skip, remaining))
                    continue

        # Log rollouts artifact
        is_main_process = self.accelerator.is_local_main_process
        if cfg.save_rollouts_to_wandb and is_main_process:
            try:
                artifact = wandb.Artifact("rollouts", type="dataset")
                artifact.add_dir(str(self.rollout_store.dir))
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"[Warning] Failed to log rollouts to wandb: {e}")