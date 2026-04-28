"""Mode-covering training (SFT, forward-KL). Single GPU, LoRA.

Usage:
  python train_modecovering.py \
    --dataset_path .../teacher_wins_cond_xyo.jsonl \
    --objective {sft|fkl} \
    --run_id WC-1 \
    --output_dir /work/.../ckpts/WC-1-fkl \
    --model Qwen/Qwen3-4B


Behaviour:
  - SFT: standard cross-entropy. Student input = chat-template(x with chat or
         WI raw text format) + completion = y_star. Loss only on y_star tokens.
  - FKL: --pure_kl mode, no SFT component.
         Teacher input  = (x, y, o, y*)  — privileged context
         Student input  = (x, y*)        — vanilla context
         Loss = KL(p_T || p_S) on y* tokens (mode-covering).
         Teacher and student share weights — only the input differs.
         Teacher forward runs under torch.no_grad() so gradients flow only
         through the student forward.

Both objectives log per-step:
  loss, kl_T_S = KL(p_T || p_S), kl_S_T = KL(p_S || p_T)
to wandb every 10 steps. Saves 4 evenly-spaced + 1 final LoRA checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Local helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    build_prompt_text, build_run_name, build_run_tags, evenly_spaced_steps,
    get_completion_text, load_jsonl, load_model_and_tokenizer,
    tokenize_prompt_completion,
)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """Materialises tokenized tensors.
      - student input_ids and completion_mask (context = x)         — always
      - teacher input_ids and completion_mask (context = teacher_context) — FKL only

    For SFT, teacher tensors are NOT built or stored. ~50% memory & tokenizer
    work saved on SFT runs.
    """
    def __init__(self, rows, tokenizer, max_length: int, teacher_context: str,
                 objective: str):
        self.rows = []
        self.has_teacher = (objective == "fkl")
        for d in rows:
            comp = get_completion_text(d)
            if not comp:
                continue
            student_prompt = build_prompt_text(d, "x", tokenizer)
            s_ids, s_mask, _ = tokenize_prompt_completion(
                student_prompt, comp, tokenizer, max_length)
            row = {
                "s_ids":  torch.tensor(s_ids,  dtype=torch.long),
                "s_mask": torch.tensor(s_mask, dtype=torch.long),
            }
            if self.has_teacher:
                teacher_prompt = build_prompt_text(d, teacher_context, tokenizer)
                t_ids, t_mask, _ = tokenize_prompt_completion(
                    teacher_prompt, comp, tokenizer, max_length)
                row["t_ids"]  = torch.tensor(t_ids,  dtype=torch.long)
                row["t_mask"] = torch.tensor(t_mask, dtype=torch.long)
            self.rows.append(row)

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def pad_collate(batch, pad_id):
    """Pad student tensors always; teacher tensors only when present (FKL)."""
    def _pad(seqs, val):
        m = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), m), val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    s_ids  = _pad([b["s_ids"]  for b in batch], pad_id)
    s_mask = _pad([b["s_mask"] for b in batch], 0)
    out = {
        "s_input_ids":       s_ids,
        "s_attention_mask":  (s_ids != pad_id).long(),
        "s_completion_mask": s_mask,
    }
    if "t_ids" in batch[0]:
        t_ids  = _pad([b["t_ids"]  for b in batch], pad_id)
        t_mask = _pad([b["t_mask"] for b in batch], 0)
        out["t_input_ids"]       = t_ids
        out["t_attention_mask"]  = (t_ids != pad_id).long()
        out["t_completion_mask"] = t_mask
    return out


# ─── Loss helpers ────────────────────────────────────────────────────────────

def shifted_logits_and_targets(input_ids, attention_mask, completion_mask, model):
    """One forward; return (logits[..., :-1, :], target_ids[..., 1:],
    completion_mask[..., 1:]) so that logits at position t predict target t+1.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]                     # (B, L-1, V)
    targets = input_ids[:, 1:]                         # (B, L-1)
    cmask = completion_mask[:, 1:].to(logits.dtype)    # (B, L-1)
    return logits, targets, cmask


def sft_loss(model, batch):
    logits, targets, cmask = shifted_logits_and_targets(
        batch["s_input_ids"], batch["s_attention_mask"],
        batch["s_completion_mask"], model)
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none").reshape(targets.shape)
    n = cmask.sum().clamp(min=1.0)
    return (loss_per_token * cmask).sum() / n


def kl_pT_pS(student_logits, teacher_logits, mask):
    """KL(p_T || p_S) = Σ_v p_T log(p_T/p_S). Per-token, masked, then mean."""
    log_pT = F.log_softmax(teacher_logits, dim=-1)
    log_pS = F.log_softmax(student_logits, dim=-1)
    p_T = log_pT.exp()
    kl_per_tok = (p_T * (log_pT - log_pS)).sum(dim=-1)        # (B, L-1)
    n = mask.sum().clamp(min=1.0)
    return (kl_per_tok * mask).sum() / n


def kl_pS_pT(student_logits, teacher_logits, mask):
    """KL(p_S || p_T) — for diagnostics."""
    log_pT = F.log_softmax(teacher_logits, dim=-1)
    log_pS = F.log_softmax(student_logits, dim=-1)
    p_S = log_pS.exp()
    kl_per_tok = (p_S * (log_pS - log_pT)).sum(dim=-1)
    n = mask.sum().clamp(min=1.0)
    return (kl_per_tok * mask).sum() / n


# ─── Step runner ─────────────────────────────────────────────────────────────

def run_step(model, batch, objective: str):
    """Returns (loss_for_backprop, dict_of_metrics).
    SFT: no teacher forward, no KL diagnostics, only loss.
    FKL: teacher + student forwards, KL loss + KL diagnostics."""
    if objective == "sft":
        loss = sft_loss(model, batch)
        return loss, {"loss/sft": loss.detach().float().item()}
    if objective != "fkl":
        raise ValueError(f"unknown objective: {objective}")

    # FKL path
    s_logits, s_targets, s_cmask = shifted_logits_and_targets(
        batch["s_input_ids"], batch["s_attention_mask"],
        batch["s_completion_mask"], model)

    # Teacher forward (no grad): treat as fixed target distribution. Even though
    # teacher and student share weights, detaching is what mode-covering FKL
    # training requires.
    with torch.no_grad():
        t_logits, t_targets, t_cmask = shifted_logits_and_targets(
            batch["t_input_ids"], batch["t_attention_mask"],
            batch["t_completion_mask"], model)

    # Align teacher and student on completion-token suffix.
    n_s = int(s_cmask.sum(dim=1).max().item())
    n_t = int(t_cmask.sum(dim=1).max().item())
    suffix = min(n_s, n_t)
    assert suffix > 0, (
        f"completion suffix is zero — student and teacher completion masks do "
        f"not overlap. Check collator: n_s={n_s}, n_t={n_t}"
    )
    # Per-row safety: every row must have non-empty completion masks on both
    # sides. Catches a single broken row hiding inside a healthy batch.
    per_row_s = s_cmask.sum(dim=1)
    per_row_t = t_cmask.sum(dim=1)
    n_empty_s = int((per_row_s == 0).sum().item())
    n_empty_t = int((per_row_t == 0).sum().item())
    assert n_empty_s == 0 and n_empty_t == 0, (
        f"some batch rows have zero-length completion masks: "
        f"empty_student_rows={n_empty_s}, empty_teacher_rows={n_empty_t}, "
        f"row_s_lens={per_row_s.tolist()}, row_t_lens={per_row_t.tolist()}"
    )

    s_logits_c = s_logits[:, -suffix:, :]
    t_logits_c = t_logits[:, -suffix:, :]
    s_mask_c   = s_cmask[:, -suffix:]

    loss = kl_pT_pS(s_logits_c, t_logits_c, s_mask_c)
    metrics = {"loss/fkl": loss.detach().float().item()}

    # Diagnostic KLs (no_grad for speed). Always at every step — they're
    # computed from logits we already have. Wandb call decides cadence.
    with torch.no_grad():
        kl_TS = kl_pT_pS(s_logits_c, t_logits_c, s_mask_c)
        kl_ST = kl_pS_pT(s_logits_c, t_logits_c, s_mask_c)
    metrics["kl/T_S"] = kl_TS.detach().float().item()
    metrics["kl/S_T"] = kl_ST.detach().float().item()

    return loss, metrics


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--objective", required=True, choices=["sft", "fkl"])
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", required=True,
                    help="HF model id; e.g. Qwen/Qwen3-4B (WC) or Qwen/Qwen2.5-Math-7B (WI)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--teacher_context", choices=["xyo", "xo", "xy"], default="xyo",
                    help="FKL teacher's privileged-context shape (default xyo)")
    ap.add_argument("--num_intermediate_ckpts", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wandb_project", default="demonstrator-to-teacher")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Model + tokenizer + LoRA
    print(f"Loading {args.model} with LoRA r={args.lora_r}...", flush=True)
    model, tok = load_model_and_tokenizer(
        args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 2. Data
    rows = load_jsonl(args.dataset_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {args.dataset_path}", flush=True)
    teacher_ctx = args.teacher_context if args.objective == "fkl" else "xyo"
    train_ds = TrainDataset(rows, tok, args.max_length,
                            teacher_context=teacher_ctx,
                            objective=args.objective)
    print(f"Tokenized {len(train_ds)} examples; objective={args.objective}; "
          f"teacher_context={teacher_ctx if args.objective == 'fkl' else '<unused>'}",
          flush=True)

    pad_id = tok.pad_token_id
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps/epoch: {steps_per_epoch}  Total opt steps: {total_steps}  "
          f"Effective batch: {args.batch_size * args.grad_accum}", flush=True)
    save_at = set(evenly_spaced_steps(total_steps, args.num_intermediate_ckpts))
    print(f"Saving checkpoints at opt steps: {sorted(save_at)}", flush=True)

    # 4. Wandb
    run_name = build_run_name(args.run_id, args.objective, args.model, args.dataset_path)
    run_tags = build_run_tags(args.run_id, args.objective, args.model, args.dataset_path)
    print(f"wandb run name: {run_name}", flush=True)
    print(f"wandb tags    : {run_tags}", flush=True)
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, tags=run_tags,
                   config=vars(args))
        wb = wandb
    except Exception as e:
        print(f"WARNING: wandb init failed ({e}); continuing without wandb", flush=True)
        wb = None

    # 5. Train
    opt_step = 0
    micro_in_acc = 0
    optim.zero_grad()
    t0 = time.time()
    pbar = tqdm(total=total_steps, desc=f"{args.objective}", dynamic_ncols=True,
                mininterval=2.0)

    for epoch in range(args.epochs):
        seen = 0   # reset per-epoch so wandb 'epoch' counter is correct within epoch
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, metrics = run_step(model, batch, args.objective)
            (loss / args.grad_accum).backward()
            seen += batch["s_input_ids"].size(0)
            micro_in_acc += 1

            if micro_in_acc == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad()
                opt_step += 1
                micro_in_acc = 0
                pbar.update(1)

                if wb and (opt_step % args.log_every == 0 or opt_step == 1):
                    wb.log({"step": opt_step, "epoch": epoch + (seen / len(train_ds)),
                            **metrics}, step=opt_step)

                if opt_step in save_at:
                    ck = Path(args.output_dir) / f"step-{opt_step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck)
                    tok.save_pretrained(ck)
                    print(f"[saved] {ck}", flush=True)

    pbar.close()

    final = Path(args.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[saved] {final}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.1f} min", flush=True)
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
