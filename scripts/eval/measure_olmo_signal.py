#!/usr/bin/env python3
"""
Diagnostic: how much does feedback (y, o) shift OLMo's token distribution?

For each sample, at each token position n along y_base:
    delta_n = KL( p_OLMo(·|x,y,o,y_base_{<n}) || p_OLMo(·|x,y_base_{<n}) )

Two forward passes per sample (no gradients needed):
    Pass 1 — no hindsight:   context = apply_chat_template(x)
    Pass 2 — with hindsight: context = apply_chat_template(x + [y] + [o])
    Target: y_base tokens (OLMo's own response without hindsight)

Tier labels are joined from datasets/wildfeedback/filtered_*.jsonl by (conversation_id, turn_index).

Outputs (to --output_dir):
    olmo_signal.jsonl  — one record per sample with per_token_kl, mean_delta, tier, o_text
    olmo_signal_summary.txt — mean delta per tier with 95% CI + top/bottom 10 samples

Usage:
    python scripts/eval/measure_olmo_signal.py \\
        --ybase  datasets/wildfeedback/olmo_3_7b_500/ybase_olmo.jsonl \\
        --model  allenai/OLMo-3-7B-Instruct-SFT \\
        --output_dir data/olmo_signal \\
        --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Tier lookup ────────────────────────────────────────────────────────────────

def build_tier_lookup(wf_dir: str) -> dict:
    """
    Read all filtered_*.jsonl files and return a dict:
        (conversation_id, turn_index) -> tier
    """
    lookup = {}
    for path in Path(wf_dir).glob("filtered_*.jsonl"):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["conversation_id"], r["turn_index"])
                lookup[key] = r.get("tier", "UNKNOWN")
    return lookup


# ── Context builders ───────────────────────────────────────────────────────────

def apply_chat_template(tokenizer, messages: list, add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def build_no_hint_context(item: dict) -> list:
    """x turns only."""
    return list(item["x"])


def build_hint_context(item: dict) -> list:
    """x + y (GPT-4 response) + o (user follow-up)."""
    msgs = list(item["x"])
    msgs.append(item["y"])   # {role: "assistant", content: ...}
    msgs.append(item["o"])   # {role: "user",      content: ...}
    return msgs


# ── Sample prep ────────────────────────────────────────────────────────────────

def prepare_sample(
    tokenizer,
    item: dict,
    device: torch.device,
    max_length: int,
) -> tuple | None:
    """
    Returns (ids_no_hint, ids_with_hint, y_base_tokens, len_no, len_with, n_y)
    or None if y_base is empty.
    """
    y_base_text = (item.get("y_base") or "").strip()
    if not y_base_text:
        return None

    prompt_no   = apply_chat_template(tokenizer, build_no_hint_context(item))
    prompt_with = apply_chat_template(tokenizer, build_hint_context(item))

    toks_no   = tokenizer(prompt_no,   add_special_tokens=False, return_tensors="pt")
    toks_with = tokenizer(prompt_with, add_special_tokens=False, return_tensors="pt")
    y_toks    = tokenizer(y_base_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
    if y_toks.dim() == 0:
        y_toks = y_toks.unsqueeze(0)

    len_no   = toks_no["input_ids"].shape[1]
    len_with = toks_with["input_ids"].shape[1]
    n_y      = y_toks.shape[0]

    # Truncate prompts if needed
    if len_no + n_y > max_length:
        keep = max(0, max_length - n_y)
        toks_no["input_ids"] = toks_no["input_ids"][:, -keep:]
        len_no = toks_no["input_ids"].shape[1]
    if len_with + n_y > max_length:
        keep = max(0, max_length - n_y)
        toks_with["input_ids"] = toks_with["input_ids"][:, -keep:]
        len_with = toks_with["input_ids"].shape[1]

    ids_no   = torch.cat([toks_no["input_ids"].squeeze(0),   y_toks])
    ids_with = torch.cat([toks_with["input_ids"].squeeze(0), y_toks])

    return ids_no, ids_with, y_toks, len_no, len_with, n_y


# ── Batch KL ──────────────────────────────────────────────────────────────────

def run_batch_kl(
    model,
    preps: list[tuple],
    device: torch.device,
    pad_id: int,
) -> list[list[float]]:
    """Two batched forward passes → per-token KL(q_with || p_no) per sample."""

    def pad_batch(seqs):
        max_len = max(s.shape[0] for s in seqs)
        padded, masks = [], []
        for s in seqs:
            p = max_len - s.shape[0]
            padded.append(F.pad(s, (0, p), value=pad_id) if p > 0 else s)
            masks.append(torch.ones(max_len, dtype=torch.long))
            if p > 0:
                masks[-1][-p:] = 0
        return torch.stack(padded).to(device), torch.stack(masks).to(device)

    ids_no_list   = [p[0] for p in preps]
    ids_with_list = [p[1] for p in preps]
    lens_no       = [p[3] for p in preps]
    lens_with     = [p[4] for p in preps]
    n_ys          = [p[5] for p in preps]

    inp_no,   msk_no   = pad_batch(ids_no_list)
    inp_with, msk_with = pad_batch(ids_with_list)

    with torch.no_grad():
        logits_no   = model(input_ids=inp_no,   attention_mask=msk_no).logits
        logits_with = model(input_ids=inp_with, attention_mask=msk_with).logits

    results = []
    for b, prep in enumerate(preps):
        len_no, len_with, n_y = lens_no[b], lens_with[b], n_ys[b]
        kls = []
        for i in range(n_y):
            idx_no   = len_no   - 1 + i
            idx_with = len_with - 1 + i
            if idx_no >= logits_no.shape[1] or idx_with >= logits_with.shape[1]:
                break
            log_p = F.log_softmax(logits_no  [b, idx_no,   :].float(), dim=-1)
            log_q = F.log_softmax(logits_with[b, idx_with, :].float(), dim=-1)
            # KL(q || p) = sum(q * (log_q - log_p))
            kl = F.kl_div(log_p, log_q.exp(), reduction="sum").item()
            kls.append(max(0.0, kl))   # numerical guard
        results.append(kls)
    return results


# ── Stats ──────────────────────────────────────────────────────────────────────

def mean_ci(vals: list[float]) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) with 95% normal CI. Returns (nan,nan,nan) if empty."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = sum(vals) / n
    if n == 1:
        return mu, mu, mu
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    se  = math.sqrt(var / n)
    z   = 1.96
    return mu, mu - z * se, mu + z * se


# ── Summary text ───────────────────────────────────────────────────────────────

def write_summary(records: list[dict], out_path: str) -> None:
    tiers = ["BEST", "DECENT", "NOISE", "SWITCH", "BAD", "UNCATEGORIZED", "UNKNOWN"]

    lines = [
        "OLMo-3-7B-Instruct-SFT — Feedback Signal Diagnostic",
        "Metric: mean KL(p(·|x,y,o,y_base_{<n}) || p(·|x,y_base_{<n})) per sample",
        "",
        f"{'Tier':<16} {'N':>5} {'Mean delta':>12} {'95% CI':>22}",
        "─" * 60,
    ]

    by_tier: dict[str, list[float]] = {}
    for r in records:
        by_tier.setdefault(r["tier"], []).append(r["mean_delta"])
    # also report "ALL"
    all_vals = [r["mean_delta"] for r in records if not math.isnan(r["mean_delta"])]

    for tier in tiers + ["ALL"]:
        vals = by_tier.get(tier, []) if tier != "ALL" else all_vals
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            continue
        mu, lo, hi = mean_ci(vals)
        lines.append(f"{tier:<16} {len(vals):>5} {mu:>12.4f}  [{lo:.4f}, {hi:.4f}]")

    lines += ["", "─" * 60, "", "Top 10 samples by mean delta (highest signal):"]
    ranked = sorted([r for r in records if not math.isnan(r["mean_delta"])],
                    key=lambda r: r["mean_delta"], reverse=True)
    for i, r in enumerate(ranked[:10], 1):
        o_text = r["o_text"][:120].replace("\n", " ")
        lines.append(f"  {i:2d}. delta={r['mean_delta']:.4f}  tier={r['tier']}")
        lines.append(f"      o: {o_text}")

    lines += ["", "Bottom 10 samples by mean delta (lowest signal):"]
    for i, r in enumerate(ranked[-10:][::-1], 1):
        o_text = r["o_text"][:120].replace("\n", " ")
        lines.append(f"  {i:2d}. delta={r['mean_delta']:.4f}  tier={r['tier']}")
        lines.append(f"      o: {o_text}")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how much (y,o) shifts OLMo's token distribution along y_base."
    )
    parser.add_argument("--ybase",      required=True,
                        help="ybase_olmo.jsonl from generate_olmo.py")
    parser.add_argument("--wf_dir",
                        default="datasets/wildfeedback",
                        help="Dir containing filtered_*.jsonl for tier labels")
    parser.add_argument("--model",      default="allenai/OLMo-3-7B-Instruct-SFT")
    parser.add_argument("--output_dir", default="data/olmo_signal")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl   = out_dir / "olmo_signal.jsonl"
    out_summary = out_dir / "olmo_signal_summary.txt"

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading ybase from {args.ybase} ...", flush=True)
    data = [json.loads(l) for l in open(args.ybase) if l.strip()]
    print(f"  {len(data)} samples", flush=True)

    print(f"Building tier lookup from {args.wf_dir} ...", flush=True)
    tier_lookup = build_tier_lookup(args.wf_dir)
    for item in data:
        key = (item["conversation_id"], item["turn_index"])
        item["tier"] = tier_lookup.get(key, "UNKNOWN")
    tier_counts = {}
    for item in data:
        tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1
    print(f"  Tier distribution: {tier_counts}", flush=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model {args.model} ...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device.type == "cuda" else None,
        attn_implementation="sdpa",
    )
    model.eval()
    print("  Model loaded.", flush=True)

    # ── Forward passes ─────────────────────────────────────────────────────────
    records = []
    bs = max(1, args.batch_size)
    n_batches = math.ceil(len(data) / bs)

    with open(out_jsonl, "w") as f_out:
        for batch_start in tqdm(range(0, len(data), bs), total=n_batches, desc="Batches"):
            batch = data[batch_start : batch_start + bs]

            preps, valid_idx = [], []
            for i, item in enumerate(batch):
                p = prepare_sample(tokenizer, item, device, args.max_length)
                if p is not None:
                    preps.append(p)
                    valid_idx.append(i)

            if preps:
                try:
                    kls_list = run_batch_kl(model, preps, device, pad_id)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # fallback: one by one
                    kls_list = []
                    for p in preps:
                        try:
                            kls_list.extend(run_batch_kl(model, [p], device, pad_id))
                        except Exception:
                            kls_list.append([])
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[WARN] batch {batch_start}: {e}", file=sys.stderr)
                    kls_list = [[] for _ in preps]
            else:
                kls_list = []

            kl_by_batch = {vi: kl for vi, kl in zip(valid_idx, kls_list)}
            for i, item in enumerate(batch):
                kls = kl_by_batch.get(i, [])
                mean_delta = (sum(kls) / len(kls)) if kls else float("nan")
                o_text = (item["o"].get("content") or "") if isinstance(item["o"], dict) else str(item["o"])
                rec = {
                    "conversation_id": item["conversation_id"],
                    "turn_index":      item["turn_index"],
                    "tier":            item["tier"],
                    "o_text":          o_text,
                    "mean_delta":      mean_delta,
                    "per_token_kl":    kls,
                    "n_tokens":        len(kls),
                }
                records.append(rec)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nWrote {out_jsonl}", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    write_summary(records, str(out_summary))
    print(f"Wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()
