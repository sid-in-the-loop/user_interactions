#!/usr/bin/env python3
"""
Phase 2 — Masked SFT training across 4 GPUs using DDP.

Run with:
    torchrun --nproc_per_node=4 train_fkl.py \
        --input      datasets/wildchat/y_star.jsonl \
        --output_dir ./checkpoints \
        --model      Qwen/Qwen3-4B

Hardware: 4x48GB GPUs
Model:    Qwen3-4B (~8GB weights in bf16)
Memory:   ~28GB per GPU at batch_size=16, seq_len=2048 (leaves ~20GB headroom)

Effective batch size: 16 per GPU × 4 GPUs = 64 per step.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────

def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class FKLDataset(Dataset):
    """
    Student sees only x (prompt).
    Target is y* (hindsight-generated response).
    Labels are -100 for prompt tokens, actual ids for y* tokens.
    """

    def __init__(self, data: list, tokenizer, max_length: int, rank: int):
        self.items = self._tokenize(data, tokenizer, max_length, rank)

    def _tokenize(self, data: list, tokenizer, max_length: int, rank: int) -> list:
        items = []
        skipped = 0

        for item in tqdm(data, desc="Tokenizing", disable=not is_main(rank)):
            # Normalize y_star: JSONL may have string or {"role","content"}
            y_star = item["y_star"]
            if isinstance(y_star, str):
                y_star = {"role": "assistant", "content": y_star}
            messages_full = list(item["x"]) + [y_star]

            # Prompt: x only (no y, no o — student sees nothing about GPT-4's response)
            prompt_text = tokenizer.apply_chat_template(
                item["x"],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Full sequence: x + y*
            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_enc   = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids      = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len     = len(prompt_ids)

            # Mask prompt tokens in labels
            labels = [-100] * len(input_ids)
            if prompt_len < len(input_ids):
                labels[prompt_len:] = input_ids[prompt_len:]

            # Skip samples where y* was entirely truncated
            if all(l == -100 for l in labels):
                skipped += 1
                continue

            items.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

        if is_main(rank):
            print(f"[Dataset] {len(items)} usable samples, {skipped} skipped (y* truncated)")

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch: list, pad_id: int) -> dict:
    """Right-pad to longest sequence in batch. Labels padded with -100."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []

    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(     x["input_ids"]      + [pad_id] * pad)
        attention_mask.append(x["attention_mask"]  + [0]      * pad)
        labels.append(        x["labels"]          + [-100]   * pad)

    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────

def masked_fkl_loss(
    model,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    labels:         torch.Tensor,
    mask_tau:       float = 0.0,
):
    """
    Single forward pass masked SFT loss.

    Theory:
      Forward KL = SFT on y* ~ pi_theta(·|x,o)
      Mask removes tokens where pi_theta(y*_i | x, y*_{<i}) < tau.
      This prevents gradient explosions from tokens far outside the
      model's current distribution.

    Implementation:
      1. Forward with grad → logits
      2. Detach logits → compute mask (no graph needed for this)
      3. log_softmax on same live logits → loss (grad flows through here)

    Result: single forward pass (2× faster than two-pass approach).
    """
    target_mask = labels != -100  # (B, T)

    # One forward pass — gradient graph lives here
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits  # (B, T, V)

    # Shift by 1: logits[t] predicts token at position t+1
    # So logits[:, :-1] aligns with labels[:, 1:]
    shifted_logits      = logits[:, :-1, :]        # (B, T-1, V)
    shifted_labels      = labels[:, 1:]            # (B, T-1)
    shifted_target_mask = target_mask[:, 1:]       # (B, T-1)

    # Replace -100 with 0 for safe indexing (non-target positions won't contribute to loss)
    safe_labels = shifted_labels.clone()
    safe_labels[~shifted_target_mask] = 0

    # ── Mask: detach so no gradient flows through probability computation ──
    with torch.no_grad():
        probs       = torch.softmax(shifted_logits.detach(), dim=-1)        # (B, T-1, V)
        token_probs = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        prob_mask   = (token_probs >= mask_tau) & shifted_target_mask       # (B, T-1)

    # ── Loss: gradient flows through log_softmax(shifted_logits) ──────────
    log_probs       = torch.log_softmax(shifted_logits, dim=-1)             # (B, T-1, V)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    masked_log_probs = prob_mask.float() * token_log_probs
    denom            = prob_mask.float().sum().clamp(min=1e-8)
    loss             = -masked_log_probs.sum() / denom

    return loss, prob_mask.float().mean().item()


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, args, rank, world_size):

    # ── DataLoader ────────────────────────────────────────────────────
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # avoids worker respawn overhead each epoch
    )

    # ── Model setup ───────────────────────────────────────────────────
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Model is on the correct device after .to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Disable compile for DDP stability investigation
    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)  # ~20% speedup after warmup

    # ── Optimizer & scheduler ─────────────────────────────────────────
    # 4B model on 4x48GB: AdamW8bit saves ~12GB optimizer state vs fp32 AdamW
    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps   = (len(loader) * args.epochs) // args.grad_accum
    warmup_steps  = int(0.05 * total_steps)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── WandB ─────────────────────────────────────────────────────────
    if is_main(rank) and HAS_WANDB and not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    if is_main(rank):
        print(f"Steps per epoch: {len(loader)}  |  Total optimizer steps: {total_steps}")
        print(f"Effective batch: {args.batch_size * world_size * args.grad_accum}")

    # ── Training ──────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main(rank))
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)
            labels         = batch["labels"].to(rank)

            loss, mask_density = masked_fkl_loss(
                model, input_ids, attention_mask, labels, args.mask_tau
            )
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main(rank):
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss":  f"{loss.item():.4f}",
                        "mask":  f"{mask_density:.2f}",
                        "lr":    f"{lr:.1e}",
                        "gnorm": f"{grad_norm:.2f}",
                    })

                    if HAS_WANDB and not args.no_wandb:
                        wandb.log({
                            "loss":          loss.item(),
                            "mask_density":  mask_density,
                            "lr":            lr,
                            "grad_norm":     grad_norm.item(),
                            "global_step":   global_step,
                        })

                    if mask_density < 0.1:
                        print(f"\n[WARN] mask_density={mask_density:.3f} — "
                              "most tokens masked out. Check --mask_tau or data quality.")

                # Checkpoint
                if is_main(rank) and global_step % args.save_steps == 0:
                    _save(model, tokenizer, os.path.join(args.output_dir, f"step-{global_step}"), rank)

                dist.barrier()

    # ── Final save ────────────────────────────────────────────────────
    if is_main(rank):
        _save(model, tokenizer, os.path.join(args.output_dir, "final"), rank)
        print("Training complete.")


def _save(model, tokenizer, path: str, rank: int):
    """Unwrap torch.compile + DDP before saving."""
    if not is_main(rank):
        return
    os.makedirs(path, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap compile
    raw = raw.module    if hasattr(raw,   "module")    else raw      # unwrap DDP
    raw.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"\n[Saved] {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # Data & output
    parser.add_argument("--input",       required=True,               help="y_star.jsonl from generate_y_star.py")
    parser.add_argument("--output_dir",  default="./checkpoints")

    # Model
    parser.add_argument("--model",       default="Qwen/Qwen3-4B")

    # Training hyperparams
    # With 4x48GB and Qwen3-4B (bf16):
    #   Weights:     ~8GB
    #   Activations: ~12GB at bs=16, seq=2048
    #   Optimizer:   ~4GB (AdamW8bit)
    #   Total:       ~24GB → safe on 48GB with headroom
    parser.add_argument("--batch_size",  type=int,   default=16,      help="Per-GPU batch size")
    parser.add_argument("--grad_accum",  type=int,   default=1,       help="Gradient accumulation steps")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=2e-6)
    parser.add_argument("--max_length",  type=int,   default=2048)
    parser.add_argument("--mask_tau",    type=float, default=0.0)
    parser.add_argument("--save_steps",  type=int,   default=500)

    # WandB
    parser.add_argument("--wandb_project", default="fkl-distill")
    parser.add_argument("--run_name",      default=None)
    parser.add_argument("--no_wandb",      action="store_true")

    args = parser.parse_args()
    rank, world_size = setup()

    if is_main(rank):
        print(f"=== FKL SFT Training ===")
        print(f"World size:       {world_size}")
        print(f"Per-GPU batch:    {args.batch_size}")
        print(f"Effective batch:  {args.batch_size * world_size * args.grad_accum}")
        print(f"Model:            {args.model}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"  # right-pad for causal LM label alignment
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset — all ranks tokenize (fast, no communication needed)
    data    = load_jsonl(args.input)
    dataset = FKLDataset(data, tokenizer, max_length=args.max_length, rank=rank)

    # Model — load directly to rank's GPU
    attn_implementation = "flash_attention_2"
    try:
        import flash_attn
    except ImportError:
        attn_implementation = "sdpa"
        if is_main(rank):
            print("[Info] flash_attn not found, falling back to sdpa")

    if is_main(rank):
        print(f"Loading model {args.model}...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(rank)

    train(model, tokenizer, dataset, args, rank, world_size)
    cleanup()


if __name__ == "__main__":
    main()