from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from ..utils.logit_processors import BatchThinkingTokenBudgetProcessor

__all__ = ["MaskFace", "GenerationOutput"]


class GenerationOutput:
    """Light-weight replacement for HuggingFace *GenerateOutput*.

    Only the attributes used by *ReinforceTrainer* are implemented.  We expose
    *sequences* along with a few extra fields that tests may query (think/face
    masks, etc.).
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MaskFace(nn.Module):
    """Wrapper around two language models implementing the *mask-face* pattern.

    The *mask* model (analogue of the original *shoggoth*) is responsible for
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
        Budget for the number of tokens the *mask* model may generate.  Must be
        supplied unless a custom *logits_processor* is given.
    min_thinking_tokens : int, optional
        Minimum number of thinking tokens.
    logit_processor : *LogitsProcessor* or *list*, optional
        Custom logits processor(s) applied during the *mask* generation phase.
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
    ) -> None:
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Load or accept pre-created components
        # ------------------------------------------------------------------
        if mask_model is not None and face_model is not None and tokenizer is not None:
            self.mask_model = mask_model.to(self.device)
            self.face_model = face_model.to(self.device)
            self.tokenizer = tokenizer
        else:
            if mask_model_name is None or face_model_name is None:
                raise ValueError(
                    "When pre-loaded models are not provided, 'mask_model_name' and "
                    "'face_model_name' must be specified."
                )
            # Small helper to keep the logic symmetric
            def _load(model_name: str):
                dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
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
                raise ValueError("Tokenizers of face and mask models differ – they must share the same vocabulary")

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
        """Generate answers following the *mask-face* protocol.

        The API purposefully differs from HuggingFace's *generate* to better fit
        the two-stage procedure.  *ReinforceTrainer* handles this discrepancy.
        """
        # --------------------------------------------------------------
        # Stage 1 – MASK (chain-of-thought)
        # --------------------------------------------------------------
        model_inputs = self.tokenizer(prompt_inputs, return_tensors="pt", padding=True).to(self.device)
        prompt_lens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)

        # Generate hidden CoT via the *mask* model using the supplied logits processor
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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Run both models and average logits so every parameter receives grads."""
        face_out = self.face_model(*args, **kwargs)
        mask_out = self.mask_model(*args, **kwargs)

        # Ensure both outputs have logits
        if not (hasattr(face_out, "logits") and hasattr(mask_out, "logits")):
            raise AttributeError("Underlying models must return logits for training")

        # Average logits – simple, unbiased combination
        avg_logits = (face_out.logits + mask_out.logits) / 2.0

        # Re-use the exact *CausalLMOutput* type returned by the face model but
        # with combined logits so downstream code (loss, etc.) works unchanged.
        face_out.logits = avg_logits  # type: ignore[attr-defined]
        return face_out
