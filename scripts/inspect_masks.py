from __future__ import annotations

import argparse
import contextlib
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _compute_and_print_masks(model, tokenizer, sequences: torch.LongTensor, inputs: dict, title: str) -> None:
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    sequences = sequences.to(device)

    print(f"\n=== {title} ===")
    print(f"device={device}")
    print(f"sequences.shape={tuple(sequences.shape)}")

    # Replicate ReinforceTrainer._compute_logprobs exactly (masking + splits)
    input_ids_full = sequences[:, :-1]
    target_ids = sequences[:, 1:]

    # Autocast like trainer; on CPU make it a no-op
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else contextlib.nullcontext()
    with ac:
        outputs = model(input_ids_full)
    logits_full = outputs.logits  # (B, L-1, V)
    logprobs_full = F.log_softmax(logits_full, dim=-1)
    token_logprobs = logprobs_full.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    seq_len = sequences.shape[1]
    arange_seq = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
    prompt_len_common = inputs["input_ids"].shape[1]
    prompt_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

    mask_full = arange_seq.expand(sequences.size(0), -1) >= prompt_len_common
    mask = mask_full[:, 1:]  # align with targets (new-token mask in target space)

    print(f"prompt_len_common={prompt_len_common}")
    print(f"prompt_lens={prompt_lens.tolist()}")
    print("mask (positions >= common prompt length; target space, 1 means selected):")
    for b in range(mask.size(0)):
        row = mask[b].to(torch.int).tolist()
        print(f"  sample {b}: {row}")

    # Segment masks per sample (thinking vs output) as in trainer
    end_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if not end_think_tokens:
        # Fallback: mimic trainer's first-element behavior
        end_think_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    else:
        end_think_id = end_think_tokens[0]

    B, Lm1 = token_logprobs.shape
    logprob_thinking = torch.zeros(B, device=sequences.device)
    logprob_output = torch.zeros(B, device=sequences.device)

    print("thinking/output masks per sample (target space; 1 means included):")
    for b in range(B):
        seq_list = sequences[b].tolist()
        prompt_len = int(prompt_lens[b].item())
        try:
            rel_idx = seq_list[prompt_len:].index(end_think_id)
            end_idx = prompt_len + rel_idx  # inclusive position of </think>
        except ValueError:
            end_idx = len(seq_list) - 1

        # Convert to target-space indices and include the FIRST generated token
        # Start at prompt_len - 1 (target index predicting token at position prompt_len)
        start_tgt_idx = max(prompt_len - 1, 0)
        target_end_idx = max(end_idx - 1, start_tgt_idx)

        # Build segment masks then intersect with new-token mask to exclude padding
        think_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
        output_mask_row = torch.zeros(Lm1, dtype=torch.bool, device=sequences.device)
        think_mask_row[start_tgt_idx : target_end_idx + 1] = True
        output_mask_row[target_end_idx + 1 :] = True
        # Intersect with new-token mask (target space)
        new_token_mask_row = mask[b]
        think_mask_row = think_mask_row & new_token_mask_row
        output_mask_row = output_mask_row & new_token_mask_row

        # Print masks as 0/1 lists
        print(f"  sample {b}:")
        print(f"    prompt_len={prompt_len}, end_idx={end_idx}, target_end_idx={target_end_idx}")
        print(f"    think_mask:  {[int(x) for x in think_mask_row.tolist()]}")
        print(f"    output_mask: {[int(x) for x in output_mask_row.tolist()]}")

        logprob_thinking[b] = (token_logprobs[b] * think_mask_row.float()).sum()
        logprob_output[b] = (token_logprobs[b] * output_mask_row.float()).sum()

    logprob_total = logprob_thinking + logprob_output

    print("log-prob sums:")
    print(f"  thinking: {logprob_thinking.detach().to('cpu').tolist()}")
    print(f"  output:   {logprob_output.detach().to('cpu').tolist()}")
    print(f"  total:    {logprob_total.detach().to('cpu').tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect masks and logprob segments exactly like the trainer")
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-gpt2", help="HF model name or path")
    parser.add_argument("--prompts", type=str, nargs="*", default=["Hello world", "Hi"], help="Prompts to tokenize")
    parser.add_argument("--mode", type=str, choices=["no_gen", "after_gen"], default="no_gen", help="Scenario to inspect")
    parser.add_argument("--num_generated", type=int, default=3, help="Number of EOS tokens to append for after_gen mode")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Real model and tokenizer (no toy infra)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32))
    model = model.to(device)
    model.eval()

    # Tokenize prompts
    inputs = tokenizer(args.prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    base_sequences = inputs["input_ids"]

    if args.mode == "no_gen":
        sequences = base_sequences
        _compute_and_print_masks(model, tokenizer, sequences, inputs, title="NO_GENERATION")
    else:
        # Append EOS tokens to simulate generated tokens (like test)
        eos_id = int(tokenizer.eos_token_id)
        gen = torch.full((base_sequences.size(0), args.num_generated), eos_id, dtype=torch.long, device=device)
        sequences = torch.cat([base_sequences, gen], dim=1)
        _compute_and_print_masks(model, tokenizer, sequences, inputs, title="AFTER_GENERATION")


if __name__ == "__main__":
    main()


