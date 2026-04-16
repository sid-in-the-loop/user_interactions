#!/usr/bin/env python3
"""
Precompute per-token log probabilities of y* under the frozen init model.

Output: a .pt file containing a list of tensors, one per sample, each of shape (num_ystar_tokens,)
with the log probability of each y* token under p_θ0(·|x, y*_{<t}).

This is needed for IS methods (jsd_is1, jsd_is2, jsd_is3, jsd_is4).

Usage:
  python scripts/fkl/precompute_logprobs.py \
      --input datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
      --y_star_field y_star_prefix30 \
      --model Qwen/Qwen3-8B \
      --output datasets/wildchat/init_logprobs_qwen3_8b.pt \
      --batch_size 4
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_content(obj):
    if obj is None: return ""
    if isinstance(obj, str): return obj
    if isinstance(obj, dict): return obj.get("content") or ""
    return str(obj)


class LogProbDataset(Dataset):
    """Tokenize [x + y*] for the student context (init model sees only x)."""

    def __init__(self, data, tokenizer, max_length, y_star_field):
        self.items = []
        skipped = 0
        for item in tqdm(data, desc="Tokenizing"):
            y_star_raw = item.get(y_star_field)
            if not y_star_raw:
                skipped += 1
                continue

            y_star_msg = {"role": "assistant", "content": get_content(y_star_raw)}
            x = item["x"]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    x, tokenize=False, add_generation_prompt=True)
                full_text = tokenizer.apply_chat_template(
                    list(x) + [y_star_msg], tokenize=False, add_generation_prompt=False)
            except Exception:
                skipped += 1
                continue

            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            enc = tokenizer(full_text, add_special_tokens=False,
                           truncation=True, max_length=max_length)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Labels: -100 on prompt, actual ids on y*
            labels = [-100] * len(input_ids)
            if prompt_len < len(input_ids):
                labels[prompt_len:] = input_ids[prompt_len:]

            if all(l == -100 for l in labels):
                skipped += 1
                continue

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        print(f"[LogProbDataset] {len(self.items)} usable, {skipped} skipped")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_id=0):
    def pad(seqs, val=0):
        ml = max(len(s) for s in seqs)
        return [s + [val] * (ml - len(s)) for s in seqs]

    return {
        "input_ids": torch.tensor(pad([x["input_ids"] for x in batch]), dtype=torch.long),
        "attention_mask": torch.tensor(pad([x["attention_mask"] for x in batch]), dtype=torch.long),
        "labels": torch.tensor(pad([x["labels"] for x in batch], -100), dtype=torch.long),
    }


@torch.no_grad()
def compute_batch_logprobs(model, batch, device):
    """Compute per-token log probs of y* tokens for each sample in the batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Causal shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)

    results = []
    for i in range(input_ids.shape[0]):
        mask = shift_labels[i] != -100
        if mask.sum() == 0:
            results.append(torch.tensor([]))
            continue

        # Gather log probs at the label positions
        token_ids = shift_labels[i][mask]
        token_logprobs = log_probs[i][mask]
        per_token_lp = token_logprobs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        results.append(per_token_lp.cpu())

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--y_star_field", default="y_star_prefix30")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"y* field: {args.y_star_field}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = load_jsonl(args.input)
    dataset = LogProbDataset(data, tokenizer, args.max_length, args.y_star_field)

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    ).to(device).eval()

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )

    all_logprobs = []
    total_tokens = 0
    for batch in tqdm(loader, desc="Computing logprobs"):
        batch_lps = compute_batch_logprobs(model, batch, device)
        all_logprobs.extend(batch_lps)
        total_tokens += sum(lp.shape[0] for lp in batch_lps)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(all_logprobs, args.output)

    # Print stats
    all_flat = torch.cat([lp for lp in all_logprobs if lp.numel() > 0])
    seq_logprobs = [lp.sum().item() for lp in all_logprobs if lp.numel() > 0]

    print(f"\nSaved {len(all_logprobs)} samples to {args.output}")
    print(f"Total tokens: {total_tokens}")
    print(f"Mean per-token logprob: {all_flat.mean().item():.4f}")
    print(f"Std per-token logprob:  {all_flat.std().item():.4f}")
    print(f"Mean sequence logprob:  {sum(seq_logprobs)/len(seq_logprobs):.4f}")
    print(f"Mean tokens per sample: {total_tokens/len(all_logprobs):.1f}")


if __name__ == "__main__":
    main()
