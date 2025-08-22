from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import os

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
    mind_model_name : str
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
        mind_model_name: str | None = None,
        face_model_name: str | None = None,
        *,
        # Pre-loaded components (optional – useful for testing to avoid reloads)
        mind_model: Any = None,
        face_model: Any = None,
        tokenizer: Any = None,
        device: str | None = None,
        batch_size: int = 8,
        max_thinking_tokens: int | None = None,
        min_thinking_tokens: int = 0,
        logit_processor: LogitsProcessor | Sequence[LogitsProcessor] | None = None,
        # Unified FSDP flag (shard both models across the entire world)
        use_fsdp: bool = False,
    ) -> None:
        super().__init__()
        self.is_fsdp = use_fsdp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_fsdp:
            try:
                rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else -1
                world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else -1
            except Exception:
                rank, world = -1, -1
            print(f"[MindFace/__init__][rank{rank}/{world}] Unified FSDP enabled across all ranks. device={self.device}", flush=True)
        else:
            print(f"[MindFace/__init__] Non-FSDP mode, device={self.device}", flush=True)

        # ------------------------------------------------------------------
        # Load or accept pre-created components
        # ------------------------------------------------------------------
        if mind_model is not None and face_model is not None and tokenizer is not None:
            self.mind_model = mind_model.to(self.device)
            self.face_model = face_model.to(self.device)
            self.tokenizer = tokenizer
            # Assert both models are compatible with the same tokenizer (vocab size match)
            try:
                mind_emb = self.mind_model.get_input_embeddings().weight
                face_emb = self.face_model.get_input_embeddings().weight
                assert mind_emb.shape[0] == face_emb.shape[0], (
                    "Tokenizers of face and mind models differ – they must share the same vocabulary"
                )
            except Exception:
                # Fallback: rely on generate-time errors if embeddings/tokenizers mismatch
                pass
        else:
            if mind_model_name is None or face_model_name is None:
                raise ValueError(
                    "When pre-loaded models are not provided, 'mind_model_name' and "
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

            self.mind_model = _load(mind_model_name)
            self.face_model = _load(face_model_name)
            # We always take the face tokenizer (vocabulary *must* match)
            self.tokenizer = AutoTokenizer.from_pretrained(face_model_name, padding_side="left", use_fast=True)
            mind_emb = self.mind_model.get_input_embeddings().weight
            face_emb = self.face_model.get_input_embeddings().weight
            if mind_emb.shape[0] != face_emb.shape[0]:
                raise ValueError(
                    "Tokenizers of face and mind models differ – they must share the same vocabulary"
                )

        # Wrap with FSDP if requested (use global/default process group)
        if self.is_fsdp:
            try:
                self.mind_model = FSDP(self.mind_model)
                self.face_model = FSDP(self.face_model)
            except Exception as e:
                print(f"[MindFace/__init__] Warning: FSDP wrap failed, proceeding without sharding: {e}", flush=True)
                self.is_fsdp = False

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
        # Introspection helpers for memory/dtypes (per-rank)
        # ------------------------------------------------------------------
        def _bytes_of_params(module: nn.Module) -> int:
            total = 0
            try:
                for p in module.parameters(recurse=True):
                    if p is not None and p.data.is_cuda:
                        total += p.numel() * p.element_size()
            except Exception:
                pass
            return total

        def _bytes_of_grads(module: nn.Module) -> int:
            total = 0
            try:
                for p in module.parameters(recurse=True):
                    if p is not None and p.grad is not None and p.grad.is_cuda:
                        total += p.grad.numel() * p.grad.element_size()
            except Exception:
                pass
            return total

        def _dtype_set(module: nn.Module) -> List[str]:
            dset = set()
            try:
                for p in module.parameters(recurse=True):
                    if p is not None:
                        dset.add(str(p.dtype))
            except Exception:
                pass
            return sorted(list(dset))

        def _fmt_bytes(num_bytes: int) -> str:
            try:
                return f"{num_bytes / (1024**3):.2f} GiB"
            except Exception:
                return str(num_bytes)

        self._bytes_of_params = _bytes_of_params  # type: ignore[attr-defined]
        self._bytes_of_grads = _bytes_of_grads    # type: ignore[attr-defined]
        self._dtype_set = _dtype_set              # type: ignore[attr-defined]
        self._fmt_bytes = _fmt_bytes              # type: ignore[attr-defined]

    def report_memory(self, phase: str, *, include_grads: bool = False) -> None:
        """Print per-rank memory/dtype info for mind/face models and CUDA allocator."""
        try:
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else -1
        except Exception:
            rank = -1
        try:
            mind_b = self._bytes_of_params(self.mind_model)  # type: ignore[attr-defined]
            face_b = self._bytes_of_params(self.face_model)  # type: ignore[attr-defined]
            mind_d = ",".join(self._dtype_set(self.mind_model))  # type: ignore[attr-defined]
            face_d = ",".join(self._dtype_set(self.face_model))  # type: ignore[attr-defined]
            if include_grads:
                mind_g = self._bytes_of_grads(self.mind_model)  # type: ignore[attr-defined]
                face_g = self._bytes_of_grads(self.face_model)  # type: ignore[attr-defined]
            else:
                mind_g = face_g = 0
            alloc = torch.cuda.memory_allocated(device=self.device) if torch.cuda.is_available() else 0
            reserv = torch.cuda.memory_reserved(device=self.device) if torch.cuda.is_available() else 0
            print(
                f"[MEM][rank{rank}] {phase} | mind_params={self._fmt_bytes(mind_b)} (dtypes={mind_d}) "
                f"face_params={self._fmt_bytes(face_b)} (dtypes={face_d}) "
                f"mind_grads={self._fmt_bytes(mind_g)} face_grads={self._fmt_bytes(face_g)} "
                f"cuda_alloc={self._fmt_bytes(alloc)} cuda_reserved={self._fmt_bytes(reserv)}",
                flush=True,
            )
        except Exception as e:
            try:
                print(f"[MEM] Reporting failed at phase={phase}: {e}", flush=True)
            except Exception:
                pass

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
        # In unified FSDP mode, generation runs on every rank; we gather full params
        # temporarily with FSDP.summon_full_params for each model.
        # --------------------------------------------------------------
        # Stage 1 – MIND (chain-of-thought)
        # --------------------------------------------------------------
        model_inputs = self.tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(self.device)
        prompt_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        # Log current param memory
        self.report_memory("GEN-MIND:pre", include_grads=False)

        # Generate hidden CoT via the *mind* model using the supplied logits processor
        if isinstance(self.mind_model, FSDP):
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            with _FSDP.summon_full_params(self.mind_model, recurse=True):
                self.report_memory("GEN-MIND:inside_summon", include_grads=False)
                mind_out = self.mind_model.module.generate(  # type: ignore[attr-defined]
                    **model_inputs,
                    max_new_tokens=max_thinking_tokens,
                    logits_processor=self.logit_processor,
                    return_dict_in_generate=True,
                    output_scores=False,
                    **gen_kwargs,
                )
        else:
            mind_out = self.mind_model.generate(
                **model_inputs,
                max_new_tokens=max_thinking_tokens,
                logits_processor=self.logit_processor,
                return_dict_in_generate=True,
                output_scores=False,
                **gen_kwargs,
            )
        mind_sequences = mind_out.sequences  # shape (B, prompt + think)

        # --------------------------------------------------------------
        # Trim everything **after** the first </think>
        # --------------------------------------------------------------
        end_think_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        trimmed_sequences: List[torch.Tensor] = []
        think_lens: List[int] = []
        for seq, p_len in zip(mind_sequences, prompt_lens):
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

        self.report_memory("GEN-FACE:pre", include_grads=False)

        # --------------------------------------------------------------
        # Stage 2 – FACE (visible answer)
        # --------------------------------------------------------------
        min_think_len = min(think_lens) if think_lens else 0
        face_max_new_tokens = max_new_tokens - min_think_len
        face_max_new_tokens = max(face_max_new_tokens, 0)

        # Reset any stateful logits processors before the face turn
        for proc in self.logit_processor:
            if hasattr(proc, "reset"):
                proc.reset()

        if isinstance(self.face_model, FSDP):
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            with _FSDP.summon_full_params(self.face_model, recurse=True):
                self.report_memory("GEN-FACE:inside_summon", include_grads=False)
                face_out = self.face_model.module.generate(  # type: ignore[attr-defined]
                    input_ids=input_ids_face,
                    attention_mask=attention_mask_face,
                    max_new_tokens=face_max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=False,
                    **gen_kwargs,
                )
        else:
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
        mind_generations: List[str] = []
        face_generations: List[str] = []
        for b in range(B):
            p_len = prompt_lens[b].item()
            pad_len = pad_lens[b]
            eff_prompt_len = p_len + pad_len
            gen_tokens = full_sequences[b, eff_prompt_len:]
            start_mask_idx = eff_prompt_len - 1
            think_slice = think_mask[b, start_mask_idx : start_mask_idx + gen_tokens.size(0)]
            face_slice = face_mask[b, start_mask_idx : start_mask_idx + gen_tokens.size(0)]

            mind_ids = gen_tokens[think_slice].tolist()
            face_ids = gen_tokens[face_slice].tolist()

            mind_generations.append(
                self.tokenizer.decode(mind_ids, skip_special_tokens=True).strip()
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
            mind_generations=mind_generations,
            face_generations=face_generations,
        )

    # ------------------------------------------------------------------
    # Segment log-prob computation (works for both FSDP and non-FSDP)
    # ------------------------------------------------------------------
    def compute_logprobs(
        self, sequences: torch.Tensor, prompt_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both segments; wrapper calling the split methods sequentially."""
        logprob_thinking = self.compute_logprobs_mind_only(sequences, prompt_lens)
        logprob_output = self.compute_logprobs_face_only(sequences, prompt_lens)
        return logprob_thinking, logprob_output

    def compute_logprobs_mind_only(
        self, sequences: torch.Tensor, prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            mind_logits = self.mind_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        self.report_memory("TRAIN/FWD:post-mind", include_grads=False)

        mind_logprobs = nn.functional.log_softmax(mind_logits, dim=-1)
        mind_token_logprobs = mind_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        vals: List[torch.Tensor] = []
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
            think_mask = torch.zeros_like(mind_token_logprobs[b], dtype=torch.bool)
            think_mask[p_len - 1 : target_end_idx] = True
            vals.append((mind_token_logprobs[b] * think_mask.float()).sum())

        return torch.stack(vals)

    def compute_logprobs_face_only(
        self, sequences: torch.Tensor, prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            face_logits = self.face_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        self.report_memory("TRAIN/FWD:post-face", include_grads=False)

        face_logprobs = nn.functional.log_softmax(face_logits, dim=-1)
        face_token_logprobs = face_logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        vals: List[torch.Tensor] = []
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
            output_mask = torch.zeros_like(face_token_logprobs[b], dtype=torch.bool)
            output_mask[target_end_idx : ] = True
            vals.append((face_token_logprobs[b] * output_mask.float()).sum())

        return torch.stack(vals)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Average logits from both models for a standard forward pass."""
        face_out = self.face_model(*args, **kwargs)
        mind_out = self.mind_model(*args, **kwargs)

        # Ensure both outputs have logits
        if not (hasattr(face_out, "logits") and hasattr(mind_out, "logits")):
            raise AttributeError("Underlying models must return logits for training")

        # Average logits – simple, unbiased combination
        avg_logits = (face_out.logits + mind_out.logits) / 2.0

        # Re-use the exact *CausalLMOutput* type returned by the face model but
        # with combined logits so downstream code (loss, etc.) works unchanged.
        face_out.logits = avg_logits  # type: ignore[attr-defined]
        return face_out
