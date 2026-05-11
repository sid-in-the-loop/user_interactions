"""DPO training (Direct Preference Optimization). Single GPU, LoRA on Qwen3-4B.

chosen   completion = y_star  (teacher generation under chosen conditioning)
rejected completion = y_base  (base-policy response)

Loss = -log σ( β * [ (logπ(chosen) - logπref(chosen)) - (logπ(rejected) - logπref(rejected)) ] )
       where logπ = policy (LoRA on),  logπref = reference (LoRA off; same shared weights).

LoRA + ref trick: with peft.disable_adapter() the model becomes the frozen base — used
to get logπref. With LoRA enabled, logπ comes for free in the same step.

Usage:
  python train_dpo.py \
    --dataset_path .../unfiltered_cond_xyo.jsonl \
    --run_id DPO-xyo \
    --output_dir .../ckpts/DPO-xyo \
    --model Qwen/Qwen3-4B \
    --epochs 4 --batch_size 2 --grad_accum 64
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    build_prompt_text, build_run_name, build_run_tags, evenly_spaced_steps,
    load_jsonl, load_model_and_tokenizer, tokenize_prompt_completion,
)

try:
    import wandb as wb
except Exception:
    wb = None


def get_chosen_text(d):
    """y_star — preferred (teacher) response."""
    v = d.get("y_star")
    return v if isinstance(v, str) else ""

def get_rejected_text(d):
    """y_base — base policy response."""
    v = d.get("y_base")
    return v if isinstance(v, str) else ""


class DPODataset(Dataset):
    """Per-row: tokenize (x-prompt + chosen) and (x-prompt + rejected) using same prompt."""
    def __init__(self, rows, tokenizer, max_length: int, policy_context: str = "x"):
        self.rows = []
        for d in rows:
            chosen   = get_chosen_text(d)
            rejected = get_rejected_text(d)
            if not chosen or not rejected: continue
            prompt = build_prompt_text(d, policy_context, tokenizer)
            c_ids, c_mask, _ = tokenize_prompt_completion(prompt, chosen,   tokenizer, max_length)
            r_ids, r_mask, _ = tokenize_prompt_completion(prompt, rejected, tokenizer, max_length)
            self.rows.append({
                "c_ids":  torch.tensor(c_ids,  dtype=torch.long),
                "c_mask": torch.tensor(c_mask, dtype=torch.long),
                "r_ids":  torch.tensor(r_ids,  dtype=torch.long),
                "r_mask": torch.tensor(r_mask, dtype=torch.long),
            })
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def pad_collate(batch, pad_id):
    def _pad(seqs, val):
        m = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), m), val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out
    return {
        "c_ids":  _pad([b["c_ids"]  for b in batch], pad_id),
        "c_mask": _pad([b["c_mask"] for b in batch], 0),
        "r_ids":  _pad([b["r_ids"]  for b in batch], pad_id),
        "r_mask": _pad([b["r_mask"] for b in batch], 0),
    }


def seq_logp(model, ids, mask):
    """Sum of log-probs at completion-mask positions. Returns shape [B]."""
    attn = (ids != 0).long()
    logits = model(ids, attention_mask=attn).logits  # [B, T, V]
    # next-token prediction: position t predicts ids[:, t+1]
    target = ids[:, 1:]
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # [B, T-1]
    m = mask[:, 1:].float()
    return (tok_lp * m).sum(dim=1), m.sum(dim=1).clamp(min=1)


def dpo_step(model, batch, beta=0.1):
    """Returns loss and metrics. Uses peft disable_adapter() for ref logp."""
    c_ids = batch["c_ids"].to(model.device); c_mask = batch["c_mask"].to(model.device)
    r_ids = batch["r_ids"].to(model.device); r_mask = batch["r_mask"].to(model.device)

    # 1. reference (base, LoRA disabled) — no grad
    with torch.no_grad(), model.disable_adapter():
        ref_c, _ = seq_logp(model, c_ids, c_mask)
        ref_r, _ = seq_logp(model, r_ids, r_mask)

    # 2. policy (LoRA on) — with grad
    pol_c, n_c = seq_logp(model, c_ids, c_mask)
    pol_r, n_r = seq_logp(model, r_ids, r_mask)

    # log-ratios (length-normalized)
    log_ratio_c = (pol_c - ref_c) / n_c
    log_ratio_r = (pol_r - ref_r) / n_r

    margin = beta * (log_ratio_c - log_ratio_r)
    loss = -F.logsigmoid(margin).mean()

    with torch.no_grad():
        acc = (margin > 0).float().mean().item()

    return loss, {
        "loss": float(loss.item()),
        "margin/mean": float(margin.mean().item()),
        "margin/std":  float(margin.std().item()),
        "acc/chosen>rejected": acc,
        "logratio/chosen":   float(log_ratio_c.mean().item()),
        "logratio/rejected": float(log_ratio_r.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--objective", default="dpo", choices=["dpo"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--num_intermediate_ckpts", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--wandb_project", default="demonstrator-to-teacher-v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. data
    print(f"Loading dataset: {args.dataset_path}", flush=True)
    rows = load_jsonl(args.dataset_path)
    if args.limit: rows = rows[: args.limit]
    print(f"  rows: {len(rows)}", flush=True)

    # 2. model + LoRA
    print(f"Loading model: {args.model}", flush=True)
    model, tok = load_model_and_tokenizer(args.model, lora_r=args.lora_r,
                                          lora_alpha=args.lora_alpha,
                                          lora_dropout=args.lora_dropout)
    pad_id = tok.pad_token_id
    if pad_id is None: pad_id = tok.eos_token_id

    # 3. dataset + loader
    ds = DPODataset(rows, tok, args.max_length)
    print(f"  usable rows after tokenization: {len(ds)}", flush=True)
    def coll(b): return pad_collate(b, pad_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll, num_workers=2)

    # 4. optimizer + schedule
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = (len(loader) * args.epochs) // args.grad_accum
    save_at = set(evenly_spaced_steps(total_steps, args.num_intermediate_ckpts))
    print(f"Total opt steps: {total_steps}  | save_at: {sorted(save_at)}", flush=True)

    # 5. wandb
    run_name = build_run_name(args.run_id, args.objective, args.model, args.dataset_path)
    run_tags = build_run_tags(args.run_id, args.objective, args.model, args.dataset_path)
    if wb is not None:
        wb.init(project=args.wandb_project, name=run_name, tags=run_tags,
                config={**vars(args), "total_steps": total_steps})

    # 6. train
    model.train()
    accum = 0; opt_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", mininterval=30):
            loss, metrics = dpo_step(model, batch, beta=args.beta)
            (loss / args.grad_accum).backward()
            accum += 1
            if accum == args.grad_accum:
                opt.step(); opt.zero_grad()
                opt_step += 1; accum = 0
                if opt_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[step {opt_step}/{total_steps}]  "
                          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                          + f"  | elapsed={elapsed/60:.1f}m", flush=True)
                    if wb is not None:
                        wb.log(metrics, step=opt_step)
                if opt_step in save_at:
                    ck = Path(args.output_dir) / f"step-{opt_step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck)
                    tok.save_pretrained(ck)
                    print(f"[saved] {ck}", flush=True)

    # final
    final = Path(args.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[saved] {final}", flush=True)
    if wb is not None: wb.finish()


if __name__ == "__main__":
    main()
