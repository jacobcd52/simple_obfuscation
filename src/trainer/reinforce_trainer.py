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
import time
import faulthandler
import sys
import os as _os  # alias for env prints without shadowing os imported above
import threading
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

        # --------------------------------------------------------------
        # Pre-loop diagnostics & heartbeat
        # --------------------------------------------------------------
        # Initialize sequence counter early (internal use)
        self._collective_seq = 0
        # Ensure current CUDA device is explicitly set for this process
        if torch.cuda.is_available() and getattr(self.accelerator, "device", None) is not None:
            try:
                torch.cuda.set_device(self.accelerator.device)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # model / tokenizer (with optional MindFace wrapper)
        # ------------------------------------------------------------------
        if getattr(cfg, "use_mind_face", False):
            from ..models.mind_face import MindFace

            mask_name = cfg.mask_model_name or cfg.model_name
            face_name = cfg.face_model_name or cfg.model_name

            self.model = MindFace(
                mask_model_name=mask_name,
                face_model_name=face_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=cfg.batch_size,
                max_thinking_tokens=cfg.thinking_max_tokens or 0,
                min_thinking_tokens=cfg.thinking_min_tokens,
            )

            self.tokenizer = self.model.tokenizer
        else:
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

        # Prepare/wrap models and create optimisers depending on mode
        self.optimizer = None  # default; may be set below for single-model path
        self.optimizer_mind = None
        self.optimizer_face = None

        if getattr(cfg, "use_mind_face", False) and cfg.multi_gpu == "fsdp":
            t_prep = time.perf_counter()
            # FSDP-wrap the submodels individually and create separate optimisers
            self.model.mask_model, self.model.face_model = self.accelerator.prepare(
                self.model.mask_model, self.model.face_model
            )
            # submodels prepared
            self.optimizer_mind = torch.optim.AdamW(
                self.model.mask_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            self.optimizer_face = torch.optim.AdamW(
                self.model.face_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            self.optimizer_mind, self.optimizer_face = self.accelerator.prepare(self.optimizer_mind, self.optimizer_face)
            # Smoke-test barrier after prepare
            try:
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    backend = str(dist.get_backend()).lower()
                    dev_ids = [torch.cuda.current_device()] if (backend == "nccl" and torch.cuda.is_available()) else None
                    r = dist.get_rank(); w = dist.get_world_size()
                    # GPU sync before any collective
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    self._collective_seq += 1
                    t_b = time.perf_counter()
                    if dev_ids is not None:
                        dist.barrier(device_ids=dev_ids)
                    else:
                        dist.barrier()
                    took = time.perf_counter()-t_b
                    # Post-barrier comm probe (quiet)
                    try:
                        val = torch.tensor([r], device=self.accelerator.device, dtype=torch.int64)
                        dist.all_reduce(val, op=dist.ReduceOp.SUM)
                        gather_list = [torch.empty_like(val) for _ in range(w)]
                        dist.all_gather(gather_list, val)
                    except Exception:
                        pass
            except Exception as e:
                pass
        else:
            # Default single-model path (or non-FSDP mind-face) – wrap container
            t_prep = time.perf_counter()
            self.model = self.accelerator.prepare(self.model)
            # model prepared
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        t_optprep = time.perf_counter()
        self.optimizer = self.accelerator.prepare(self.optimizer)
        # optimizer prepared
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                backend = str(dist.get_backend()).lower()
                dev_ids = [torch.cuda.current_device()] if (backend == "nccl" and torch.cuda.is_available()) else None
                # Skip this barrier when mind+face FSDP is active to avoid skew
                if not (getattr(cfg, "use_mind_face", False) and cfg.multi_gpu == "fsdp"):
                    # Warm-up all_reduce probe (quiet)
                    try:
                        self._collective_seq = getattr(self, "_collective_seq", 0) + 1
                        warm = torch.tensor([dist.get_rank()], device=self.accelerator.device, dtype=torch.int64)
                        dist.all_reduce(warm, op=dist.ReduceOp.SUM)
                    except Exception:
                        pass
                    self._collective_seq = getattr(self, "_collective_seq", 0) + 1
                    if dev_ids is not None:
                        dist.barrier(device_ids=dev_ids)
                    else:
                        dist.barrier()
        except Exception as e:
            pass

        t_wb = time.perf_counter()
        try:
            if self.accelerator.is_local_main_process:
                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config=cfg.__dict__,
                    mode=os.environ.get("WANDB_MODE", "online"),
                )
            else:
                wandb.init(mode="disabled")
        except Exception as e:
            try:
                wandb.init(mode="disabled")
            except Exception:
                pass
        # Smoke barrier after wandb
        try:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                backend = str(dist.get_backend()).lower()
                dev_ids = [torch.cuda.current_device()] if (backend == "nccl" and torch.cuda.is_available()) else None
                r = dist.get_rank(); w = dist.get_world_size()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self._collective_seq += 1
                t_b2 = time.perf_counter()
                if dev_ids is not None:
                    dist.barrier(device_ids=dev_ids)
                else:
                    dist.barrier()
                took2 = time.perf_counter()-t_b2
        except Exception as e:
            pass

        # misc
        self.global_step = 0
        # Gradient accumulation helpers
        self.grad_accum_steps = max(1, cfg.grad_accum_steps)
        self._accum_counter = 0
        # Diagnostics flags
        self._did_bs_reduce_barrier = False
        self._ran_comm_probe = False
        self._iteration_idx = 0
        self._collective_seq = 0
        # Ensure gradients are cleared before first accumulation cycle
        if getattr(cfg, "use_mind_face", False) and cfg.multi_gpu == "fsdp":
            self.optimizer_mind.zero_grad()  # type: ignore[union-attr]
            self.optimizer_face.zero_grad()  # type: ignore[union-attr]
        else:
            self.optimizer.zero_grad()  # type: ignore[union-attr]

        # Final pre-loop marker
        # ready to train

        # Env diagnostics
        # env diagnostics removed

        # Early small collective probes
        # initial comm probe removed

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
        # Rank info for diagnostics
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            rank, world = 0, 1
        t_start = time.perf_counter()
        
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
        t_chat = time.perf_counter()
        for p in prompts:
            # Extract the raw user message (strip any manually-added assistant tag)
            user_content = p["prompt"].split("\n<assistant>")[0]
            chat_str = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(chat_str)
        try:
            lens = [len(s) for s in prompt_texts]
            
        except Exception:
            pass

        t_tok = time.perf_counter()
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        
        t_move = time.perf_counter()
        
        inputs = inputs.to(self.model.device)
        

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

        # Rank info
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            rank, world = 0, 1
        

        if isinstance(model_for_check, MindFace):
            # inputs are not required – we rely on prompt texts instead.
            # We expect that `inputs` contains *prompt_texts* provided under a
            # special key injected by `_next_batch`.
            prompt_texts = inputs.pop("_prompt_texts")  # type: ignore[arg-type]
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
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

        # Mask newly generated tokens (positions >= prompt length per-sample)
        seq_len = sequences.shape[1]
        arange_seq = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        # Common padded prompt length – reproduces original behaviour
        prompt_len_common = inputs["input_ids"].shape[1]
        prompt_lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
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
            prompt_len = prompt_lens[b].item()
            try:
                rel_idx = seq_list[prompt_len:].index(end_think_id)
                end_idx = prompt_len + rel_idx  # inclusive position of </think>
            except ValueError:
                end_idx = len(seq_list) - 1  # treat all generated as thinking

            # Convert to target index space (length L-1) – target idx t predicts token t+1
            target_end_idx = max(end_idx - 1, prompt_len)  # ensure non-negative

            think_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
            output_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)

            think_mask_row[prompt_len: target_end_idx + 1] = True
            output_mask_row[target_end_idx + 1 :] = True

            logprob_thinking[b] = (token_logprobs[b] * think_mask_row.float()).sum()
            logprob_output[b] = (token_logprobs[b] * output_mask_row.float()).sum()

        logprob_total = logprob_thinking + logprob_output
        return logprob_total, logprob_thinking, logprob_output

    # ------------------------------------------------------------------
    # Mind+Face specific logprob helpers (used in FSDP two-phase backward)
    # ------------------------------------------------------------------
    def _compute_logprob_thinking_mind(self, sequences, inputs):
        """Compute summed log-probabilities of the thinking segment using the MIND model only."""
        # Shift for LM likelihood
        input_ids_full = sequences[:, :-1]
        target_ids = sequences[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.mask_model(input_ids_full)
        logits_full = outputs.logits
        logprobs_full = F.log_softmax(logits_full, dim=-1)
        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # Prompt lengths and masks
        seq_len = sequences.shape[1]
        arange_seq = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        prompt_len_common = inputs["input_ids"].shape[1]
        prompt_lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        mask_full = arange_seq.expand(sequences.size(0), -1) >= prompt_len_common
        mask = mask_full[:, 1:]

        # Segment at </think>
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        B, Lm1 = token_logprobs.shape
        logprob_thinking = torch.zeros(B, device=sequences.device)
        for b in range(B):
            seq_list = sequences[b].tolist()
            prompt_len = prompt_lens[b].item()
            try:
                rel_idx = seq_list[prompt_len:].index(end_think_id)
                end_idx = prompt_len + rel_idx
            except ValueError:
                end_idx = len(seq_list) - 1
            target_end_idx = max(end_idx - 1, prompt_len)
            think_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
            think_mask_row[prompt_len : target_end_idx + 1] = True
            # Consider only newly generated tokens
            think_mask_row = think_mask_row & mask[b]
            logprob_thinking[b] = (token_logprobs[b] * think_mask_row.float()).sum()
        return logprob_thinking

    def _compute_logprob_output_face(self, sequences, inputs):
        """Compute summed log-probabilities of the output segment using the FACE model only."""
        input_ids_full = sequences[:, :-1]
        target_ids = sequences[:, 1:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.face_model(input_ids_full)
        logits_full = outputs.logits
        logprobs_full = F.log_softmax(logits_full, dim=-1)
        token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        seq_len = sequences.shape[1]
        arange_seq = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        prompt_len_common = inputs["input_ids"].shape[1]
        prompt_lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        mask_full = arange_seq.expand(sequences.size(0), -1) >= prompt_len_common
        mask = mask_full[:, 1:]

        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        B, Lm1 = token_logprobs.shape
        logprob_output = torch.zeros(B, device=sequences.device)
        for b in range(B):
            seq_list = sequences[b].tolist()
            prompt_len = prompt_lens[b].item()
            try:
                rel_idx = seq_list[prompt_len:].index(end_think_id)
                end_idx = prompt_len + rel_idx
            except ValueError:
                end_idx = len(seq_list) - 1
            target_end_idx = max(end_idx - 1, prompt_len)
            output_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
            output_mask_row[target_end_idx + 1 :] = True
            output_mask_row = output_mask_row & mask[b]
            logprob_output[b] = (token_logprobs[b] * output_mask_row.float()).sum()
        return logprob_output

    # ------------------------------------------------------------------
    # Rewards-only evaluator used for two-phase backward
    # ------------------------------------------------------------------
    def _compute_rewards_only(self, prompts, sequences, inputs):
        rollouts: List[Dict] = []
        num_rewards = len(self.rewards)
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

                for i_idx, rollout in enumerate(rollout_data):
                    for j, val in rollout["_sync_results"]:
                        reward_values[j, i_idx] = float(val)

                for task, res in zip(all_tasks, async_vals):
                    i, j = task_to_index[task]
                    if isinstance(res, Exception):
                        res = 0.0
                    reward_values[j, i] = float(res)

                rewards_list = []
                for i, rollout in enumerate(rollout_data):
                    total_r = reward_values[:, i].sum().item()
                    rollout.pop("_sync_results", None)
                    rollout.pop("_async_tags", None)
                    rollout["total_reward"] = total_r
                    rollouts.append(rollout)
                    rewards_list.append(total_r)
                return rewards_list

            rewards_list = asyncio.run(_async_eval())
        else:
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
                rollouts.append(rollout)
                rewards_list.append(total_reward)

        rewards_t = torch.tensor(rewards_list, device=sequences.device, dtype=torch.float32)
        return reward_values, rewards_t, rollouts

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

        # Clear rollouts once (main process only) to avoid multi-rank race
        if self.accelerator.is_local_main_process:
            self.rollout_store.clear_all_rollouts()
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

        # Build runtime helpers
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        try:
            if hasattr(self, "_preloop_hb_stop") and self._preloop_hb_stop is not None:
                self._preloop_hb_stop.set()
        except Exception:
            pass
        try:
            faulthandler.enable()
            faulthandler.dump_traceback_later(180, repeat=True)
        except Exception:
            pass
        tp = self._token_processor()
        
        gen_kwargs = self._build_gen_kwargs(tp)
        from tqdm import tqdm  # local import to avoid cost when not training

        prompt_iter = iter(self.prompt_builder)
        episode_counter = 0
        global_prompts_processed = 0
        is_main_process = self.accelerator.is_local_main_process
        world_size = getattr(self.accelerator, "num_processes", 1)

        with tqdm(total=cfg.num_episodes, desc="Prompts processed", disable=not is_main_process) as pbar:
            # Heartbeat helper
            def _start_heartbeat(tag: str, interval_sec: float = 2.0):
                stop_ev = threading.Event()
                def _hb():
                    try:
                        r = dist.get_rank() if dist.is_initialized() else 0
                        w = dist.get_world_size() if dist.is_initialized() else 1
                    except Exception:
                        r, w = 0, 1
                    t0 = time.time()
                    tick = 0
                    while not stop_ev.is_set():
                        try:
                            pass
                        except Exception:
                            pass
                        stop_ev.wait(interval_sec)
                        tick += 1
                th = threading.Thread(target=_hb, daemon=True)
                th.start()
                return stop_ev
            
            # ------------------------------------------------------------------
            # Specialised FSDP path for Mind+Face with two-phase backward
            # ------------------------------------------------------------------
            if getattr(cfg, "use_mind_face", False) and cfg.multi_gpu == "fsdp":
                saved_microbatches = []  # list of dicts with sequences/inputs/rewards/rollouts
                episode_counter = 0
                while global_prompts_processed < cfg.num_episodes:
                    try:
                        self._iteration_idx += 1
                        
                        # --------------------------------------------------
                        # Accumulate K microbatches worth of rollouts
                        # --------------------------------------------------
                        saved_microbatches.clear()
                        total_to_accumulate = self.grad_accum_steps
                        total_logged_rollouts = []
                        total_rewards_concat = []

                        for micro_idx in range(total_to_accumulate):
                            if global_prompts_processed >= cfg.num_episodes:
                                break
                            
                            batch, prompt_iter = self._next_batch(prompt_iter, tp)

                            # Compute global batch size across processes
                            local_batch_size = len(batch.prompts)
                            global_batch_size = local_batch_size
                            # Optional jitter to perturb scheduling across ranks
                            try:
                                jitter_ms = int(_os.environ.get("DEBUG_JITTER_MS", "0"))
                            except Exception:
                                jitter_ms = 0
                            if jitter_ms > 0:
                                time.sleep(jitter_ms / 1000.0)

                            # Early barrier right after sampling/tokenization to detect skew
                            if world_size > 1 and dist.is_available() and dist.is_initialized():
                                try:
                                    r = dist.get_rank()
                                    hb1 = _start_heartbeat(f"waiting EarlyBarrier iter={self._iteration_idx} micro={micro_idx}")
                                    dist.barrier()
                                    hb1.set()
                                except Exception:
                                    pass
                            if world_size > 1 and dist.is_available() and dist.is_initialized():
                                # Optional barrier once to check rank skew
                                if not self._did_bs_reduce_barrier:
                                    try:
                                        hb2 = _start_heartbeat(f"waiting BatchSizeReduce pre-barrier iter={self._iteration_idx} micro={micro_idx}")
                                        dist.barrier()
                                        hb2.set()
                                    except Exception:
                                        pass
                                    self._did_bs_reduce_barrier = True
                                bs_tensor = torch.tensor([local_batch_size], device=self.accelerator.device, dtype=torch.int64)
                                try:
                                    hb3 = _start_heartbeat(f"waiting all_reduce iter={self._iteration_idx} micro={micro_idx}")
                                    dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)
                                    hb3.set()
                                except Exception:
                                    pass
                                global_batch_size = int(bs_tensor.item())

                            sequences = self._generate_sequences(batch.inputs, gen_kwargs)
                            

                            # Rewards only (store for later loss computation)
                            reward_values, rewards_t, rollouts = self._compute_rewards_only(
                                batch.prompts, sequences, batch.inputs
                            )
                            

                            saved_microbatches.append(
                                {
                                    "batch": batch,
                                    "sequences": sequences,
                                    "inputs": batch.inputs,
                                    "reward_values": reward_values,
                                    "rewards_t": rewards_t,
                                    "rollouts": rollouts,
                                }
                            )

                            # Book-keeping
                            self._store_rollouts(rollouts)
                            total_logged_rollouts.extend(rollouts)
                            total_rewards_concat.append(rewards_t)

                            global_prompts_processed += global_batch_size
                            if is_main_process:
                                remaining = cfg.num_episodes - pbar.n
                                pbar.update(min(global_batch_size, remaining))

                        # Nothing accumulated (done)
                        if not saved_microbatches:
                            break

                        # --------------------------------------------------
                        # Two-phase backward: MIND first, then FACE
                        # --------------------------------------------------
                        apply_flags = torch.tensor(self.rewards_apply_flags, device=self.accelerator.device, dtype=torch.bool)
                        

                        # MIND backward + step, then free memory
                        mind_loss_running = 0.0
                        try:
                            if self.optimizer_mind is None:
                                raise RuntimeError("Mind optimizer is not initialised")
                            self.optimizer_mind.zero_grad()
                            for itm in saved_microbatches:
                                sequences_mb = itm["sequences"]
                                inputs_mb = itm["inputs"]
                                reward_values_mb = itm["reward_values"]  # (R,B)
                                # Per-reward baseline -> advantage
                                advantages_mb = reward_values_mb - reward_values_mb.mean(dim=1, keepdim=True)
                                # Aggregate only thinking-applied rewards
                                advantage_mind = advantages_mb[apply_flags].sum(dim=0) if apply_flags.any() else torch.zeros(sequences_mb.size(0), device=sequences_mb.device)
                                logprob_thinking = self._compute_logprob_thinking_mind(sequences_mb, inputs_mb)
                                loss_mind_mb = -(advantage_mind * logprob_thinking).mean()
                                mind_loss_running += loss_mind_mb.item()
                                self.accelerator.backward(loss_mind_mb / self.grad_accum_steps)
                            
                            try:
                                zero_special_token_grads(self.model.mask_model, self.tokenizer)
                            except Exception:
                                pass
                            # Use torch's clip to avoid Accelerate's parameter equality check issues
                            torch.nn.utils.clip_grad_norm_(self.model.mask_model.parameters(), self.cfg.max_grad_norm)
                            self.optimizer_mind.step()
                            self.optimizer_mind.zero_grad()
                            
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            try:
                                gc.collect()
                            except Exception:
                                pass
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            raise

                        
                        # FACE backward + step, then free memory
                        face_loss_running = 0.0
                        try:
                            if self.optimizer_face is None:
                                raise RuntimeError("Face optimizer is not initialised")
                            self.optimizer_face.zero_grad()
                            for itm in saved_microbatches:
                                sequences_mb = itm["sequences"]
                                inputs_mb = itm["inputs"]
                                reward_values_mb = itm["reward_values"]  # (R,B)
                                advantages_mb = reward_values_mb - reward_values_mb.mean(dim=1, keepdim=True)
                                # Aggregate rewards that DO NOT apply to thinking -> apply to face output
                                not_apply = (~apply_flags) if apply_flags.numel() > 0 else apply_flags
                                advantage_face = advantages_mb[not_apply].sum(dim=0) if not_apply.any() else torch.zeros(sequences_mb.size(0), device=sequences_mb.device)
                                logprob_output = self._compute_logprob_output_face(sequences_mb, inputs_mb)
                                loss_face_mb = -(advantage_face * logprob_output).mean()
                                face_loss_running += loss_face_mb.item()
                                self.accelerator.backward(loss_face_mb / self.grad_accum_steps)
                            
                            try:
                                zero_special_token_grads(self.model.face_model, self.tokenizer)
                            except Exception:
                                pass
                            # Use torch's clip to avoid Accelerate's parameter equality check issues
                            torch.nn.utils.clip_grad_norm_(self.model.face_model.parameters(), self.cfg.max_grad_norm)
                            self.optimizer_face.step()
                            self.optimizer_face.zero_grad()
                            
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                            try:
                                gc.collect()
                            except Exception:
                                pass
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            raise

                        
                        
                        # Successful update across both models
                        episode_counter += 1
                        self.global_step += 1

                        # Aggregate rewards for logging and compute combined loss
                        rewards_t_total = torch.cat(total_rewards_concat) if total_rewards_concat else torch.tensor([], device=self.accelerator.device)
                        combined_loss = torch.tensor((mind_loss_running + face_loss_running) / max(1, len(saved_microbatches)), device=self.accelerator.device)
                        self._log_step(combined_loss, rewards_t_total, total_logged_rollouts, episode_counter)

                        # Extra logs for per-branch losses
                        if is_main_process:
                            wandb.log(
                                {
                                    "loss/mind": mind_loss_running / max(1, len(saved_microbatches)),
                                    "loss/face": face_loss_running / max(1, len(saved_microbatches)),
                                },
                                step=episode_counter,
                            )

                        # Proactive memory cleanup
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            gc.collect()
                        except Exception:
                            pass

                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        raise

                # Upload saved rollouts to Weights & Biases as an artifact (main process only)
                if cfg.save_rollouts_to_wandb and is_main_process:
                    try:
                        artifact = wandb.Artifact("rollouts", type="dataset")
                        artifact.add_dir(str(self.rollout_store.dir))
                        wandb.log_artifact(artifact)
                    except Exception:
                        pass
                return

            # ------------------------------------------------------------------
            # Default path (single model or non-FSDP mind-face)
            # ------------------------------------------------------------------
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

                    pass
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
            except Exception:
                pass