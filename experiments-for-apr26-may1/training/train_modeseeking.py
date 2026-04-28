"""Mode-seeking training (offline SDPO and PC-SDPO). Single GPU, LoRA.

Per-token advantage = log π(completion | privileged) - log π(completion | vanilla).
Loss = -(advantage · policy_logp) summed over completion, len-normalized
per sequence, mean over batch. Mode-seeking because the policy is pulled
toward whichever tokens the privileged-context forward favors most strongly.

  --objective sdpo      Standard offline SDPO.
                          completion       = y           (dataset's y)
                          critic context   = (x, o)
                          policy context   = (x)
                          a_t  = log π(y |x,o)_t - log π(y|x)_t  (.detach())
                          loss = -(a_t · log π(y|x)_t), len-normed per seq.

  --objective pc_sdpo   Prefix-Conditioned SDPO.
                          completion       = y_star      (teacher generation)
                          critic context   = (x, y, o)
                          policy context   = (x)
                          a_t  = log π(y*|x,y,o)_t - log π(y*|x)_t  (.detach())
                          loss = -(a_t · log π(y*|x)_t), len-normed per seq.

Both share weights. Critic forward runs under torch.no_grad(); gradient flows
only through the policy log-prob term. kl_beta = 0.

Usage:
  python train_modeseeking.py \
    --objective {sdpo|pc_sdpo} \
    --dataset_path .../teacher_wins_cond_xyo.jsonl \
    --run_id WC-3 \
    --output_dir /work/.../ckpts/WC-3-pc_sdpo \
    --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    build_prompt_text, build_run_name, build_run_tags, evenly_spaced_steps,
    get_text, get_completion_text, load_jsonl, load_model_and_tokenizer,
    tokenize_prompt_completion,
)


# ─── Per-objective config ────────────────────────────────────────────────────

OBJ_CONFIG = {
    # objective -> (completion_getter, critic_context, policy_context)
    "sdpo":    {"completion": "y",      "critic": "xo",  "policy": "x"},
    "pc_sdpo": {"completion": "y_star", "critic": "xyo", "policy": "x"},
}


def get_objective_completion(d, objective):
    """y for sdpo, y_star for pc_sdpo. Returns text."""
    if OBJ_CONFIG[objective]["completion"] == "y":
        return get_text(d.get("y"))
    return get_completion_text(d)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ModeSeekingDataset(Dataset):
    """Per row tokenize:
      - policy input:  prompt for context=x        + completion
      - critic input:  prompt for chosen context   + completion
    """
    def __init__(self, rows, tokenizer, max_length: int,
                 objective: str, critic_context: str, policy_context: str = "x"):
        self.rows = []
        for d in rows:
            comp = get_objective_completion(d, objective)
            if not comp:
                continue
            policy_prompt = build_prompt_text(d, policy_context, tokenizer)
            critic_prompt = build_prompt_text(d, critic_context, tokenizer)
            p_ids, p_mask, _ = tokenize_prompt_completion(
                policy_prompt, comp, tokenizer, max_length)
            c_ids, c_mask, _ = tokenize_prompt_completion(
                critic_prompt, comp, tokenizer, max_length)
            self.rows.append({
                "p_ids":  torch.tensor(p_ids,  dtype=torch.long),
                "p_mask": torch.tensor(p_mask, dtype=torch.long),
                "c_ids":  torch.tensor(c_ids,  dtype=torch.long),
                "c_mask": torch.tensor(c_mask, dtype=torch.long),
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

    p_ids  = _pad([b["p_ids"]  for b in batch], pad_id)
    p_mask = _pad([b["p_mask"] for b in batch], 0)
    c_ids  = _pad([b["c_ids"]  for b in batch], pad_id)
    c_mask = _pad([b["c_mask"] for b in batch], 0)

    return {
        "p_input_ids": p_ids, "p_attention_mask": (p_ids != pad_id).long(),
        "p_completion_mask": p_mask,
        "c_input_ids": c_ids, "c_attention_mask": (c_ids != pad_id).long(),
        "c_completion_mask": c_mask,
    }


# ─── Per-token logp helper ───────────────────────────────────────────────────

def per_token_logp_at_completion(model, input_ids, attention_mask, completion_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    cmask   = completion_mask[:, 1:].to(logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    tok_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    return tok_logp, cmask


# ─── Step ─────────────────────────────────────────────────────────────────────

def advantage_step(model, batch):
    with torch.no_grad():
        critic_logp, critic_mask = per_token_logp_at_completion(
            model, batch["c_input_ids"], batch["c_attention_mask"],
            batch["c_completion_mask"])
    policy_logp, policy_mask = per_token_logp_at_completion(
        model, batch["p_input_ids"], batch["p_attention_mask"],
        batch["p_completion_mask"])

    n_p = int(policy_mask.sum(dim=1).max().item())
    n_c = int(critic_mask.sum(dim=1).max().item())
    suffix = min(n_p, n_c)
    assert suffix > 0, (
        f"completion suffix is zero — policy and critic completion masks do "
        f"not overlap. Check collator: n_p={n_p}, n_c={n_c}"
    )
    # Per-row safety: every row must have non-empty completion masks on both
    # sides. Catches a single broken row hiding inside a healthy batch.
    per_row_p = policy_mask.sum(dim=1)
    per_row_c = critic_mask.sum(dim=1)
    n_empty_p = int((per_row_p == 0).sum().item())
    n_empty_c = int((per_row_c == 0).sum().item())
    assert n_empty_p == 0 and n_empty_c == 0, (
        f"some batch rows have zero-length completion masks: "
        f"empty_policy_rows={n_empty_p}, empty_critic_rows={n_empty_c}, "
        f"row_p_lens={per_row_p.tolist()}, row_c_lens={per_row_c.tolist()}"
    )

    policy_logp_c = policy_logp[:, -suffix:]
    critic_logp_c = critic_logp[:, -suffix:]
    mask_c        = policy_mask[:, -suffix:]

    advantage = (critic_logp_c - policy_logp_c).detach()
    per_token_loss = -(advantage * policy_logp_c) * mask_c
    seq_lens = mask_c.sum(dim=1).clamp(min=1.0)
    seq_loss = per_token_loss.sum(dim=1) / seq_lens
    loss = seq_loss.mean()

    with torch.no_grad():
        active = mask_c.bool()
        adv_active = advantage[active]
        adv_mean = adv_active.mean().item() if adv_active.numel() else 0.0
        adv_var  = adv_active.var(unbiased=False).item() if adv_active.numel() else 0.0

    metrics = {
        "loss":               loss.detach().float().item(),
        "advantage/mean":     adv_mean,
        "advantage/var":      adv_var,
        "advantage/std":      math.sqrt(max(0.0, adv_var)),
        "seq_len/mean":       float(mask_c.sum(dim=1).float().mean().item()),
    }
    return loss, metrics, adv_var


# ─── KL diagnostics (slow — every kl_diag_every steps) ──────────────────────

def kl_diag_step(model, batch):
    with torch.no_grad():
        p_out = model(input_ids=batch["p_input_ids"],
                      attention_mask=batch["p_attention_mask"], use_cache=False)
        c_out = model(input_ids=batch["c_input_ids"],
                      attention_mask=batch["c_attention_mask"], use_cache=False)
        p_logits = p_out.logits[:, :-1, :]
        c_logits = c_out.logits[:, :-1, :]
        p_mask = batch["p_completion_mask"][:, 1:].to(p_logits.dtype)
        c_mask = batch["c_completion_mask"][:, 1:].to(c_logits.dtype)

        n_p = int(p_mask.sum(dim=1).max().item())
        n_c = int(c_mask.sum(dim=1).max().item())
        suffix = min(n_p, n_c)
        p_logits_c = p_logits[:, -suffix:, :]
        c_logits_c = c_logits[:, -suffix:, :]
        mask_c     = p_mask[:, -suffix:]

        log_pT = F.log_softmax(c_logits_c, dim=-1)
        log_pS = F.log_softmax(p_logits_c, dim=-1)
        p_T = log_pT.exp()
        p_S = log_pS.exp()
        n = mask_c.sum().clamp(min=1.0)
        kl_TS = ((p_T * (log_pT - log_pS)).sum(-1) * mask_c).sum() / n
        kl_ST = ((p_S * (log_pS - log_pT)).sum(-1) * mask_c).sum() / n
    return float(kl_TS.item()), float(kl_ST.item())


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True, choices=["sdpo", "pc_sdpo"])
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--critic_context", default=None,
                    help="Override the critic context (xo|xyo|xy). "
                         "Default: xo for sdpo, xyo for pc_sdpo.")
    ap.add_argument("--num_intermediate_ckpts", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--kl_diag_every", type=int, default=10)
    ap.add_argument("--wandb_project", default="demonstrator-to-teacher")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg = OBJ_CONFIG[args.objective]
    critic_ctx = args.critic_context or cfg["critic"]
    policy_ctx = cfg["policy"]
    print(f"objective={args.objective}  policy_ctx={policy_ctx}  critic_ctx={critic_ctx}  "
          f"completion={cfg['completion']}", flush=True)

    print(f"Loading {args.model} with LoRA r={args.lora_r}...", flush=True)
    model, tok = load_model_and_tokenizer(
        args.model,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    rows = load_jsonl(args.dataset_path)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {args.dataset_path}", flush=True)

    train_ds = ModeSeekingDataset(rows, tok, args.max_length,
                          objective=args.objective,
                          critic_context=critic_ctx,
                          policy_context=policy_ctx)
    print(f"Tokenized {len(train_ds)} examples", flush=True)

    pad_id = tok.pad_token_id
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps/epoch: {steps_per_epoch}  Total opt steps: {total_steps}  "
          f"Effective batch: {args.batch_size * args.grad_accum}", flush=True)
    save_at = set(evenly_spaced_steps(total_steps, args.num_intermediate_ckpts))
    print(f"Saving checkpoints at opt steps: {sorted(save_at)}", flush=True)

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

    advantage_collapse_warned = False
    opt_step = 0
    micro_in_acc = 0
    optim.zero_grad()
    pbar = tqdm(total=total_steps, desc=args.objective, dynamic_ncols=True, mininterval=2.0)
    t0 = time.time()

    for epoch in range(args.epochs):
        seen = 0   # reset per-epoch so wandb 'epoch' counter is correct within epoch
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, metrics, adv_var = advantage_step(model, batch)
            (loss / args.grad_accum).backward()
            seen += batch["p_input_ids"].size(0)
            micro_in_acc += 1

            if micro_in_acc == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad()
                opt_step += 1
                micro_in_acc = 0
                pbar.update(1)

                if wb and (opt_step % args.kl_diag_every == 0 or opt_step == 1):
                    kl_TS, kl_ST = kl_diag_step(model, batch)
                    metrics["kl/T_S"] = kl_TS
                    metrics["kl/S_T"] = kl_ST

                if wb and (opt_step % args.log_every == 0 or opt_step == 1):
                    wb.log({"step": opt_step,
                            "epoch": epoch + (seen / max(1, len(train_ds))),
                            **metrics}, step=opt_step)

                if (not advantage_collapse_warned) and adv_var < 1e-6 and opt_step > 5:
                    msg = (f"WARNING: advantage variance collapsed to {adv_var:.2e} "
                           f"at opt_step={opt_step}; gradient signal is essentially zero")
                    print(msg, flush=True)
                    if wb:
                        wb.log({"warning/advantage_variance_collapse": opt_step}, step=opt_step)
                        try: wb.alert(title="advantage variance collapse",
                                       text=msg, level="WARN")
                        except Exception: pass
                    advantage_collapse_warned = True

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
