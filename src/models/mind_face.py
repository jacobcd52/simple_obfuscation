from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import os

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..utils.logit_processors import BatchThinkingTokenBudgetProcessor
from ..utils.debug_watchdog import heartbeat

__all__ = ["MindFace", "GenerationOutput"]


class GenerationOutput:
    """Light-weight replacement for HuggingFace *GenerateOutput*.

    Only the attributes used by *ReinforceTrainer* are implemented.  We expose
    *sequences* along with a few extra fields that tests may query (think/face
    masks, etc.).
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MindFace(nn.Module):
    """Wrapper around two language models implementing the *mind-face* pattern.

    The *mind* model (analogue of the original *shoggoth*) is responsible for
    generating the hidden chain-of-thought (CoT) enclosed between ``<think>``
    and ``</think>``.  The *face* model produces the final visible answer.

    Parameters
    ----------
    mask_model_name : str
        HuggingFace repository or local path of the model that should generate
        the chain-of-thought.
    face_model_name : str
        Repository/path of the model that generates the final answer.
    device : str, optional
        Torch device on which to place the models.  Defaults to "cuda" when
        available otherwise "cpu".
    batch_size : int, optional
        Expected batch size at generation time.  Required by
        :class:`BatchThinkingTokenBudgetProcessor`.
    max_thinking_tokens : int, optional
        Budget for the number of tokens the *mind* model may generate.  Must be
        supplied unless a custom *logits_processor* is given.
    min_thinking_tokens : int, optional
        Minimum number of thinking tokens.
    logit_processor : *LogitsProcessor* or *list*, optional
        Custom logits processor(s) applied during the *mind* generation phase.
        When *None*, a *BatchThinkingTokenBudgetProcessor* is instantiated with
        the supplied token budgets.
    """

    def __init__(
        self,
        mask_model_name: str | None = None,
        face_model_name: str | None = None,
        *,
        # Pre-loaded components (optional – useful for testing to avoid reloads)
        mask_model: Any = None,
        face_model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        batch_size: int = 8,
        max_thinking_tokens: int | None = None,
        min_thinking_tokens: int = 0,
        logit_processor: LogitsProcessor | Sequence[LogitsProcessor] | None = None,
        # FSDP-disjoint-subset mode (optional)
        fsdp_mind_group: dist.ProcessGroup | None = None,
        fsdp_face_group: dist.ProcessGroup | None = None,
        fsdp_role: str | None = None,
    ) -> None:
        super().__init__()
        # Disjoint-subset FSDP mode toggle
        self.is_fsdp = fsdp_role is not None
        self.fsdp_role = fsdp_role
        self.fsdp_mind_group = fsdp_mind_group
        self.fsdp_face_group = fsdp_face_group

        if self.is_fsdp:
            if not all([fsdp_mind_group, fsdp_face_group, fsdp_role]):
                raise ValueError(
                    "fsdp_mind_group, fsdp_face_group, and fsdp_role must be provided for FSDP"
                )
            # Place each rank on its CUDA device; simple single-node mapping by global rank
            self.device = torch.device("cuda", torch.distributed.get_rank())
            try:
                rank = dist.get_rank()
                world = dist.get_world_size()
            except Exception:
                rank, world = -1, -1
            print(f"[MindFace/__init__][rank{rank}/{world}] FSDP mode enabled with role={self.fsdp_role}", flush=True)
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[MindFace/__init__] Non-FSDP mode, device={self.device}", flush=True)

        # ------------------------------------------------------------------
        # Load or accept pre-created components
        # ------------------------------------------------------------------
        if mask_model is not None and face_model is not None and tokenizer is not None and not self.is_fsdp:
            self.mask_model = mask_model.to(self.device)
            self.face_model = face_model.to(self.device)
            self.tokenizer = tokenizer
        elif self.is_fsdp:
            # Load and FSDP-wrap only the model required for this role
            self.mask_model, self.face_model = None, None
            bf16 = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

            def _load_and_shard(model_name: str, pg: dist.ProcessGroup):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                )
                return FSDP(model, process_group=pg, mixed_precision=bf16, device_id=self.device)

            if self.fsdp_role == "mind":
                if mask_model_name is None:
                    raise ValueError("mask_model_name must be set for FSDP mind role")
                self.mask_model = _load_and_shard(mask_model_name, self.fsdp_mind_group)  # type: ignore[arg-type]
                print(f"[MindFace/__init__] Wrapped mask_model in FSDP for role=mind", flush=True)
            elif self.fsdp_role == "face":
                if face_model_name is None:
                    raise ValueError("face_model_name must be set for FSDP face role")
                self.face_model = _load_and_shard(face_model_name, self.fsdp_face_group)  # type: ignore[arg-type]
                print(f"[MindFace/__init__] Wrapped face_model in FSDP for role=face", flush=True)
            else:
                raise ValueError(f"Unknown fsdp_role: {self.fsdp_role}")

            # Tokenizer must match both models; use face tokenizer and optionally verify
            self.tokenizer = AutoTokenizer.from_pretrained(
                face_model_name or (face_model.name_or_path if face_model is not None else ""),  # type: ignore[attr-defined]
                padding_side="left",
                use_fast=True,
            )
        else:
            if mask_model_name is None or face_model_name is None:
                raise ValueError(
                    "When pre-loaded models are not provided, 'mask_model_name' and "
                    "'face_model_name' must be specified."
                )

            # Small helper to keep the logic symmetric
            def _load(model_name: str):
                dtype = (
                    torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
                )
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
                    self.device
                )
                return model

            self.mask_model = _load(mask_model_name)
            self.face_model = _load(face_model_name)
            # We always take the face tokenizer (vocabulary *must* match)
            self.tokenizer = AutoTokenizer.from_pretrained(face_model_name, padding_side="left", use_fast=True)
            mask_tok = AutoTokenizer.from_pretrained(mask_model_name, padding_side="left", use_fast=True)
            if self.tokenizer.get_vocab() != mask_tok.get_vocab():
                raise ValueError("Tokenizers of face and mind models differ – they must share the same vocabulary")

        # ------------------------------------------------------------------
        # Build logits processor for the thinking phase
        # ------------------------------------------------------------------
        if logit_processor is None:
            if max_thinking_tokens is None:
                raise ValueError(
                    "max_thinking_tokens must be provided when 'logit_processor' is None"
                )
            logit_processor = [
                BatchThinkingTokenBudgetProcessor(
                    self.tokenizer,
                    max_thinking_tokens=max_thinking_tokens,
                    batch_size=batch_size,
                    min_thinking_tokens=min_thinking_tokens,
                )
            ]
        # Ensure *self.logit_processor* is always a *LogitsProcessorList*
        if isinstance(logit_processor, list):
            self.logit_processor = LogitsProcessorList(logit_processor)
        else:
            self.logit_processor = LogitsProcessorList([logit_processor])

    # ------------------------------------------------------------------
    # Generation helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_inputs: List[str],
        *,
        max_thinking_tokens: int,
        max_new_tokens: int,
        **gen_kwargs,
    ) -> GenerationOutput:
        """Generate answers following the *mind-face* protocol.

        The API purposefully differs from HuggingFace's *generate* to better fit
        the two-stage procedure.  *ReinforceTrainer* handles this discrepancy.
        """
        if self.is_fsdp:
            return self._generate_fsdp(
                prompt_inputs,
                max_thinking_tokens=max_thinking_tokens,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        # --------------------------------------------------------------
        # Stage 1 – MIND (chain-of-thought)
        # --------------------------------------------------------------
        model_inputs = self.tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(self.device)
        prompt_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        # Generate hidden CoT via the *mind* model using the supplied logits processor
        mask_out = self.mask_model.generate(
            **model_inputs,
            max_new_tokens=max_thinking_tokens,
            logits_processor=self.logit_processor,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs,
        )
        mask_sequences = mask_out.sequences  # shape (B, prompt + think)

        # --------------------------------------------------------------
        # Trim everything **after** the first </think>
        # --------------------------------------------------------------
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        trimmed_sequences: List[torch.Tensor] = []
        think_lens: List[int] = []
        for seq, p_len in zip(mask_sequences, prompt_lens):
            seq_list = seq.tolist()
            try:
                rel_idx = seq_list[p_len:].index(end_think_id)
                end_idx = p_len + rel_idx  # inclusive index of </think>
            except ValueError:
                end_idx = len(seq_list) - 1  # defensive – should not happen
            kept = seq_list[: end_idx + 1]
            trimmed_sequences.append(torch.tensor(kept, dtype=torch.long))
            think_lens.append(len(kept) - p_len)

        # Left-pad so that </think> tokens align across the batch
        max_len = max(len(seq) for seq in trimmed_sequences)
        pad_id = self.tokenizer.pad_token_id
        padded_inputs: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []
        pad_lens: List[int] = []
        for seq in trimmed_sequences:
            pad_len = max_len - len(seq)
            pad_lens.append(pad_len)
            padded_seq = torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), seq])
            padded_inputs.append(padded_seq)
            attention_masks.append((padded_seq != pad_id).long())

        input_ids_face = torch.stack(padded_inputs).to(self.device)
        attention_mask_face = torch.stack(attention_masks).to(self.device)

        # --------------------------------------------------------------
        # Stage 2 – FACE (visible answer)
        # --------------------------------------------------------------
        min_think_len = min(think_lens)
        face_max_new_tokens = max_new_tokens - min_think_len
        face_max_new_tokens = max(face_max_new_tokens, 0)

        # Reset any stateful logits processors before the face turn
        for proc in self.logit_processor:
            if hasattr(proc, "reset"):
                proc.reset()

        face_out = self.face_model.generate(
            input_ids=input_ids_face,
            attention_mask=attention_mask_face,
            max_new_tokens=face_max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs,
        )
        full_sequences = face_out.sequences  # (B, prompt + think + answer)

        # --------------------------------------------------------------
        # Build boolean masks ------------------------------------------------
        # --------------------------------------------------------------
        B, T = full_sequences.shape
        think_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=self.device)
        face_mask = torch.zeros((B, T - 1), dtype=torch.bool, device=self.device)
        for b in range(B):
            p_len = prompt_lens[b].item()
            pad_len = pad_lens[b]
            t_len = think_lens[b]
            eff_prompt_len = p_len + pad_len  # position of first think token

            start = eff_prompt_len - 1  # token **before** first CoT token
            end = eff_prompt_len + t_len - 1  # last token **before** answer
            think_mask[b, start:end] = True
            face_mask[b, end : T - 1] = True

        # --------------------------------------------------------------
        # Decode generations for convenience
        # --------------------------------------------------------------
        mask_generations: List[str] = []
        face_generations: List[str] = []
        for b in range(B):
            p_len = prompt_lens[b].item()
            pad_len = pad_lens[b]
            eff_prompt_len = p_len + pad_len
            gen_tokens = full_sequences[b, eff_prompt_len:]
            start_mask_idx = eff_prompt_len - 1
            think_slice = think_mask[b, start_mask_idx : start_mask_idx + gen_tokens.size(0)]
            face_slice = face_mask[b, start_mask_idx : start_mask_idx + gen_tokens.size(0)]

            mask_ids = gen_tokens[think_slice].tolist()
            face_ids = gen_tokens[face_slice].tolist()

            mask_generations.append(
                self.tokenizer.decode(mask_ids, skip_special_tokens=True).strip()
            )
            face_generations.append(
                self.tokenizer.decode(face_ids, skip_special_tokens=True).strip()
            )

        return GenerationOutput(
            sequences=full_sequences,
            prompt_lens=prompt_lens,
            think_lens=think_lens,
            think_mask=think_mask,
            face_mask=face_mask,
            pad_lens=pad_lens,
            mask_generations=mask_generations,
            face_generations=face_generations,
        )

    def _generate_fsdp(
        self,
        prompt_inputs: List[str],
        *,
        max_thinking_tokens: int,
        max_new_tokens: int,
        **gen_kwargs,
    ) -> GenerationOutput:
        """FSDP-specific generation logic using disjoint mind/face groups.

        Mind ranks run the mind generation; face ranks run the face generation.
        Intermediate token tensors are gathered across the world so that all
        ranks proceed with consistent inputs/outputs.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Distributed must be initialized for FSDP MindFace generation")
        rank = dist.get_rank()
        world = dist.get_world_size()
        print(f"[MindFace/_generate_fsdp][rank{rank}/{world}] Start generation. role={self.fsdp_role}", flush=True)
        heartbeat("mindface_generate_start")

        # Stage 1: Mind generation (only on mind ranks)
        model_inputs = self.tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(self.device)
        print(f"[MindFace/_generate_fsdp][rank{rank}] Tokenized prompts: input_ids.shape={model_inputs['input_ids'].shape}", flush=True)
        heartbeat("mindface_tokenized")
        # Only designated mind source rank runs generation to avoid redundant work
        world_size = dist.get_world_size()
        mind_size = (world_size + 1) // 2
        mind_src = 0
        if self.fsdp_role == "mind" and self.mask_model is not None and rank == mind_src:
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            print(f"[MindFace/_generate_fsdp][rank{rank}] ENTER mind.generate", flush=True)
            from ..utils.debug_watchdog import heartbeat_guard
            with heartbeat_guard("mind_generate"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Gather full params for generation to avoid zero-storage embedding issues
                    with _FSDP.summon_full_params(self.mask_model, recurse=True):
                        mt = max_thinking_tokens
                        if os.environ.get("DEBUG_FAST_GEN", "0") == "1":
                            mt = min(mt, 8)
                            print(f"[MindFace/_generate_fsdp][rank{rank}] DEBUG_FAST_GEN cap mind max_new_tokens={mt}", flush=True)
                        # Optional step-level debug
                        stopping = None
                        if os.environ.get("DEBUG_GEN_STEPS", "0") == "1":
                            step_counter = {"n": 0}

                            class DebugPrinter(StoppingCriteria):
                                def __call__(self, input_ids, scores, **kwargs):
                                    step_counter["n"] += 1
                                    print(f"[MindFace/mind_generate][rank{rank}] step={step_counter['n']} input_len={input_ids.shape[1]}", flush=True)
                                    try:
                                        from ..utils.debug_watchdog import heartbeat
                                        heartbeat("mind_generate_step")
                                    except Exception:
                                        pass
                                    return False

                            stopping = StoppingCriteriaList([DebugPrinter()])
                        mask_out = self.mask_model.module.generate(  # type: ignore[attr-defined]
                            **model_inputs,
                            max_new_tokens=mt,
                            logits_processor=self.logit_processor,
                            return_dict_in_generate=True,
                            output_scores=False,
                            stopping_criteria=stopping,
                            **gen_kwargs,
                        )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            print(f"[MindFace/_generate_fsdp][rank{rank}] EXIT mind.generate", flush=True)
            heartbeat("mind_generate_done")
            mask_sequences = mask_out.sequences
            print(f"[MindFace/_generate_fsdp][rank{rank}] Mind generated mask_sequences shape={tuple(mask_sequences.shape)}", flush=True)
        else:
            mask_sequences = torch.empty(0, dtype=torch.long, device=self.device)
            print(f"[MindFace/_generate_fsdp][rank{rank}] Skipping mind generation on this role", flush=True)
            heartbeat("skip_mind_generate")

        # Communicate mask sequences from designated mind source to all ranks
        # mind_src defined above
        # Ensure all ranks have reached the sync point before broadcast
        print(f"[MindFace/_generate_fsdp][rank{rank}] world barrier before broadcast(mask) begin", flush=True)
        from ..utils.debug_watchdog import heartbeat_guard
        with heartbeat_guard("wait_barrier_mask"):
            dist.barrier()
        heartbeat("barrier_before_bcast_mask")
        print(f"[MindFace/_generate_fsdp][rank{rank}] world barrier before broadcast(mask) end", flush=True)

        obj_list: List[Any] = [None]
        if rank == mind_src and self.fsdp_role == "mind":
            obj_list[0] = mask_sequences.detach().cpu()
        print(f"[MindFace/_generate_fsdp][rank{rank}] broadcast_object_list(mask) begin src={mind_src}", flush=True)
        with heartbeat_guard("wait_broadcast_mask"):
            dist.broadcast_object_list(obj_list, src=mind_src)
        heartbeat("bcast_mask_done")
        print(f"[MindFace/_generate_fsdp][rank{rank}] broadcast_object_list(mask) end", flush=True)
        if not (rank == mind_src and self.fsdp_role == "mind"):
            recv_tensor = obj_list[0]
            if isinstance(recv_tensor, torch.Tensor):
                mask_sequences = recv_tensor.to(self.device)
            else:
                raise RuntimeError("Broadcasted mask_sequences is not a Tensor")
        print(f"[MindFace/_generate_fsdp][rank{rank}] mask_sequences ready shape={tuple(mask_sequences.shape)}", flush=True)
        heartbeat("mask_ready")

        # All ranks now have mask_sequences and can proceed
        prompt_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]

        trimmed_sequences: List[torch.Tensor] = []
        think_lens: List[int] = []
        for seq, p_len in zip(mask_sequences, prompt_lens):
            seq_list = seq.tolist()
            try:
                rel_idx = seq_list[p_len:].index(end_think_id)
                end_idx = p_len + rel_idx
            except ValueError:
                end_idx = len(seq_list) - 1
            kept = seq_list[: end_idx + 1]
            trimmed_sequences.append(torch.tensor(kept, dtype=torch.long, device=self.device))
            think_lens.append(len(kept) - p_len.item())
        print(f"[MindFace/_generate_fsdp][rank{rank}] Trimmed sequences count={len(trimmed_sequences)} max_len={max(len(s) for s in trimmed_sequences)}", flush=True)
        heartbeat("trimmed")

        max_len = max(len(s) for s in trimmed_sequences)
        pad_id = self.tokenizer.pad_token_id
        padded_inputs = [
            nn.functional.pad(s, (max_len - len(s), 0), value=pad_id) for s in trimmed_sequences
        ]
        input_ids_face = torch.stack(padded_inputs)
        attention_mask_face = (input_ids_face != pad_id).long()
        print(f"[MindFace/_generate_fsdp][rank{rank}] Built face inputs: input_ids_face.shape={tuple(input_ids_face.shape)}", flush=True)
        heartbeat("face_inputs_built")

        # Stage 2: Face generation (only on face ranks)
        # Only designated face source rank runs generation
        face_src = mind_size
        if self.fsdp_role == "face" and self.face_model is not None and rank == face_src:
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            min_think_len = min(think_lens) if think_lens else 0
            face_max_new = max(0, max_new_tokens - min_think_len)
            print(f"[MindFace/_generate_fsdp][rank{rank}] ENTER face.generate", flush=True)
            from ..utils.debug_watchdog import heartbeat_guard
            with heartbeat_guard("face_generate"):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # Gather full params for generation
                    with _FSDP.summon_full_params(self.face_model, recurse=True):
                        fm = face_max_new
                        if os.environ.get("DEBUG_FAST_GEN", "0") == "1":
                            fm = min(fm, 8)
                            print(f"[MindFace/_generate_fsdp][rank{rank}] DEBUG_FAST_GEN cap face max_new_tokens={fm}", flush=True)
                        # Optional step-level debug
                        stopping = None
                        if os.environ.get("DEBUG_GEN_STEPS", "0") == "1":
                            step_counter = {"n": 0}

                            class DebugPrinter(StoppingCriteria):
                                def __call__(self, input_ids, scores, **kwargs):
                                    step_counter["n"] += 1
                                    print(f"[MindFace/face_generate][rank{rank}] step={step_counter['n']} input_len={input_ids.shape[1]}", flush=True)
                                    try:
                                        from ..utils.debug_watchdog import heartbeat
                                        heartbeat("face_generate_step")
                                    except Exception:
                                        pass
                                    return False

                            stopping = StoppingCriteriaList([DebugPrinter()])
                        face_out = self.face_model.module.generate(  # type: ignore[attr-defined]
                            input_ids=input_ids_face,
                            attention_mask=attention_mask_face,
                            max_new_tokens=fm,
                            return_dict_in_generate=True,
                            output_scores=False,
                            stopping_criteria=stopping,
                            **gen_kwargs,
                        )
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            print(f"[MindFace/_generate_fsdp][rank{rank}] EXIT face.generate", flush=True)
            heartbeat("face_generate_done")
            full_sequences = face_out.sequences
            print(f"[MindFace/_generate_fsdp][rank{rank}] Face generated full_sequences shape={tuple(full_sequences.shape)}", flush=True)
        else:
            full_sequences = torch.empty(0, dtype=torch.long, device=self.device)
            print(f"[MindFace/_generate_fsdp][rank{rank}] Skipping face generation on this role", flush=True)
            heartbeat("skip_face_generate")

        # Communicate final sequences from designated face source to all ranks
        # face_src defined above
        print(f"[MindFace/_generate_fsdp][rank{rank}] world barrier before broadcast(full) begin", flush=True)
        with heartbeat_guard("wait_barrier_full"):
            dist.barrier()
        heartbeat("barrier_before_bcast_full")
        print(f"[MindFace/_generate_fsdp][rank{rank}] world barrier before broadcast(full) end", flush=True)
        obj_list_final: List[Any] = [None]
        if rank == face_src and self.fsdp_role == "face":
            obj_list_final[0] = full_sequences.detach().cpu()
        print(f"[MindFace/_generate_fsdp][rank{rank}] broadcast_object_list(full) begin src={face_src}", flush=True)
        with heartbeat_guard("wait_broadcast_full"):
            dist.broadcast_object_list(obj_list_final, src=face_src)
        heartbeat("bcast_full_done")
        print(f"[MindFace/_generate_fsdp][rank{rank}] broadcast_object_list(full) end", flush=True)
        if not (rank == face_src and self.fsdp_role == "face"):
            recv_tensor2 = obj_list_final[0]
            if isinstance(recv_tensor2, torch.Tensor):
                full_sequences = recv_tensor2.to(self.device)
            else:
                raise RuntimeError("Broadcasted full_sequences is not a Tensor")
        print(f"[MindFace/_generate_fsdp][rank{rank}] full_sequences ready shape={tuple(full_sequences.shape)}", flush=True)
        heartbeat("full_ready")

        return GenerationOutput(sequences=full_sequences, prompt_lens=prompt_lens, think_lens=think_lens)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def compute_logprobs_dist(
        self, sequences: torch.Tensor, prompt_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Distributed forward pass for computing segment log probabilities.

        Returns (logprob_thinking, logprob_output), each shape (B,).
        Mind ranks compute thinking logprobs; face ranks compute output logprobs.
        Results are all-reduced (SUM) across the world so that every rank has
        both components.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Distributed must be initialized for compute_logprobs_dist")
        rank = dist.get_rank()
        world = dist.get_world_size()
        print(f"[MindFace/compute_logprobs_dist][rank{rank}/{world}] Start compute. role={self.fsdp_role}", flush=True)
        heartbeat("compute_logprobs_start")

        input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]

        local_thinking_vals: List[torch.Tensor] = []
        local_output_vals: List[torch.Tensor] = []

        if self.fsdp_role == "mind" and self.mask_model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.mask_model(input_ids).logits
            logprobs_full = nn.functional.log_softmax(logits, dim=-1)
            token_logprobs = logprobs_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            for b in range(sequences.size(0)):
                seq_list = sequences[b].tolist()
                p_len = prompt_lens[b].item()
                try:
                    rel_idx = seq_list[p_len:].index(end_think_id)
                    end_idx = p_len + rel_idx
                except ValueError:
                    end_idx = len(seq_list) - 1

                target_end_idx = max(end_idx - 1, p_len)
                think_mask = torch.zeros_like(token_logprobs[b], dtype=torch.bool)
                think_mask[p_len: target_end_idx + 1] = True
                local_thinking_vals.append((token_logprobs[b] * think_mask.float()).sum())
            # For symmetry, fill outputs with zeros (no grad needed)
            local_output_vals = [torch.zeros((), device=self.device)] * sequences.size(0)

        elif self.fsdp_role == "face" and self.face_model is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.face_model(input_ids).logits
            logprobs_full = nn.functional.log_softmax(logits, dim=-1)
            token_logprobs = logprobs_full.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
            for b in range(sequences.size(0)):
                seq_list = sequences[b].tolist()
                p_len = prompt_lens[b].item()
                try:
                    rel_idx = seq_list[p_len:].index(end_think_id)
                    end_idx = p_len + rel_idx
                except ValueError:
                    end_idx = len(seq_list) - 1

                target_end_idx = max(end_idx - 1, p_len)
                output_mask = torch.zeros_like(token_logprobs[b], dtype=torch.bool)
                output_mask[target_end_idx + 1 :] = True
                local_output_vals.append((token_logprobs[b] * output_mask.float()).sum())
            # For symmetry, fill thinking with zeros
            local_thinking_vals = [torch.zeros((), device=self.device)] * sequences.size(0)
        else:
            # Neither role owns a model here (should not happen), return zeros
            local_thinking_vals = [torch.zeros((), device=self.device)] * sequences.size(0)
            local_output_vals = [torch.zeros((), device=self.device)] * sequences.size(0)

        local_thinking = torch.stack(local_thinking_vals)
        local_output = torch.stack(local_output_vals)

        # Share detached copies so every rank has both segments
        th_share = local_thinking.detach().clone()
        out_share = local_output.detach().clone()
        print(f"[MindFace/compute_logprobs_dist][rank{rank}] all_reduce begin", flush=True)
        dist.all_reduce(th_share, op=dist.ReduceOp.SUM)
        dist.all_reduce(out_share, op=dist.ReduceOp.SUM)
        heartbeat("all_reduce_done")
        print(f"[MindFace/compute_logprobs_dist][rank{rank}] all_reduce end", flush=True)

        # Keep gradients only for the segment owned by this role
        if self.fsdp_role == "mind":
            return local_thinking, out_share
        elif self.fsdp_role == "face":
            return th_share, local_output
        else:
            return th_share, out_share

    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Default forward for non-FSDP mode: average logits from both models.

        In FSDP disjoint-subset mode, forward is unused; use compute_logprobs_dist.
        """
        if self.is_fsdp:
            raise NotImplementedError(
                "MindFace forward is not used in FSDP disjoint-subset mode. "
                "Use compute_logprobs_dist instead."
            )

        face_out = self.face_model(*args, **kwargs)
        mask_out = self.mask_model(*args, **kwargs)

        if not (hasattr(face_out, "logits") and hasattr(mask_out, "logits")):
            raise AttributeError("Underlying models must return logits for training")

        avg_logits = (face_out.logits + mask_out.logits) / 2.0
        face_out.logits = avg_logits  # type: ignore[attr-defined]
        return face_out
