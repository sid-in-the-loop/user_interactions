#!/usr/bin/env python3
"""
Token-level divergence heatmaps: log p_T(y*_t) - log p_θ0(y*_t) per token.

For 10 random samples, produce:
  - Per-sample PNG heatmap (4K, tokens colored by divergence)
  - Per-sample JSON with token-level data
  - Summary grid PNG (all 10 stacked)

Usage:
  python scripts/eval/token_heatmaps.py \
      --dataset datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
      --model Qwen/Qwen3-8B \
      --output-dir diagnostics/token_heatmaps
"""

import argparse
import json
import math
import os
import random
import re
import textwrap

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# ── NeurIPS style ────────────────────────────────────────────────────────────

def set_neurips_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def get_content(obj):
    if obj is None: return ""
    if isinstance(obj, str): return obj
    if isinstance(obj, dict): return obj.get("content") or ""
    return str(obj)


def get_last_user_turn(x):
    for msg in reversed(x):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def format_conversation(x):
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def extract_prefix(y_content, frac, tokenizer):
    token_ids = tokenizer.encode(y_content, add_special_tokens=False)
    if not token_ids: return ""
    n_keep = max(1, int(len(token_ids) * frac))
    return tokenizer.decode(token_ids[:n_keep], skip_special_tokens=True)


# ── Prompt builders ──────────────────────────────────────────────────────────

SYSTEM_STUDENT = "You are a helpful assistant. Respond directly and helpfully to the user's request."

SYSTEM_TEACHER = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and the beginning of a reference response. Study the "
    "partial response and the feedback carefully, then generate a complete, "
    "high-quality improved response from scratch. Do not simply continue from "
    "where the partial response ends — write a full response addressing the "
    "user's original request. Output only your response."
)

USER_TEACHER_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<partial reference response (first 30% of tokens)>
{prefix}

Given the conversation, feedback, and the partial reference response above, \
generate a complete improved response from scratch."""


def build_student_messages(row):
    messages = [{"role": "system", "content": SYSTEM_STUDENT}]
    for turn in row.get("x", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def build_teacher_messages(row, tokenizer):
    x_text = format_conversation(row.get("x", []))
    o_text = get_content(row.get("o", ""))
    y_text = get_content(row.get("y", ""))
    prefix = extract_prefix(y_text, 0.30, tokenizer)
    return [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": USER_TEACHER_TEMPLATE.format(
            x=x_text, o=o_text, prefix=prefix
        )},
    ]


# ── Log prob computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_token_logprobs(model, tokenizer, messages, completion_text, device="cuda"):
    """
    Returns list of dicts: [{token_id, token_text, log_prob, position}, ...]
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors=None
    )
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        return []

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    logits = model(input_ids).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    tokens = []
    for i, tid in enumerate(completion_ids):
        pos = prompt_len - 1 + i
        if pos < log_probs.shape[0]:
            lp = log_probs[pos, tid].item()
            text = tokenizer.decode([tid])
            tokens.append({
                "token_id": tid,
                "token_text": text,
                "log_prob": lp,
                "position": i,
            })

    return tokens


# ── Heatmap rendering ────────────────────────────────────────────────────────

def render_heatmap(tokens_data, instruction, output_path,
                   fig_width=3840, fig_height=2160, dpi=150):
    """Render a single sample's token heatmap."""
    set_neurips_style()

    divergences = [t["divergence"] for t in tokens_data]
    mean_div = np.mean(divergences)
    pct_red = np.mean([d > 0 for d in divergences]) * 100
    pct_blue = np.mean([d < 0 for d in divergences]) * 100

    fig_w = fig_width / dpi
    fig_h = fig_height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    # Layout: tokens flow left-to-right, wrap
    margin_left = 0.03
    margin_right = 0.92  # leave room for colorbar
    margin_top = 0.88
    margin_bottom = 0.05
    usable_w = margin_right - margin_left
    usable_h = margin_top - margin_bottom

    # Character-based sizing
    char_width = 0.008
    token_height = 0.028
    line_gap = 0.005
    min_token_w = 0.015

    # Lay out tokens
    x_pos = margin_left
    y_pos = margin_top
    token_rects = []

    for t in tokens_data:
        text = t["token_text"]
        display_text = text.replace("\n", "↵").replace("\t", "→")
        tw = max(min_token_w, len(display_text) * char_width + 0.006)

        if x_pos + tw > margin_right:
            x_pos = margin_left
            y_pos -= (token_height + line_gap)
            if y_pos < margin_bottom:
                break

        token_rects.append((x_pos, y_pos, tw, token_height, display_text, t["divergence"]))
        x_pos += tw + 0.002

    for (rx, ry, rw, rh, text, div_val) in token_rects:
        color = cmap(norm(np.clip(div_val, -3, 3)))
        rect = mpatches.FancyBboxPatch(
            (rx, ry - rh), rw, rh,
            boxstyle="round,pad=0.001",
            facecolor=color, edgecolor="#cccccc", linewidth=0.3,
        )
        ax.add_patch(rect)

        # Token text — escape $ to prevent matplotlib math parsing
        fontsize = 7
        max_chars = max(1, int(rw / char_width))
        clipped = text[:max_chars].replace("$", "\\$")
        ax.text(rx + rw / 2, ry - rh / 2, clipped,
                ha="center", va="center", fontsize=fontsize,
                color="black", clip_on=True, fontfamily="monospace")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # Title — escape $ to prevent math parsing
    title_text = instruction[:80] + ("..." if len(instruction) > 80 else "")
    title_text = title_text.replace("$", "\\$")
    ax.text(0.03, 0.97, title_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", fontfamily="serif")

    # Subtitle
    subtitle = (f"Mean divergence: {mean_div:.3f}  |  "
                f"Teacher > Student: {pct_red:.1f}%  |  "
                f"Student > Teacher: {pct_blue:.1f}%  |  "
                f"Tokens: {len(tokens_data)}")
    ax.text(0.03, 0.935, subtitle, transform=ax.transAxes,
            fontsize=9, va="top", color="#555", fontfamily="serif")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log p_T - log p_student", fontsize=10)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_summary_grid(all_tokens_data, all_instructions, output_path,
                        fig_width=6000, fig_height=8000, dpi=150):
    """Render all 10 samples stacked vertically."""
    set_neurips_style()

    n_samples = len(all_tokens_data)
    fig_w = fig_width / dpi
    fig_h = fig_height / dpi

    fig, axes = plt.subplots(n_samples, 1, figsize=(fig_w, fig_h))
    if n_samples == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)

    for idx, (ax, tokens_data, instruction) in enumerate(zip(axes, all_tokens_data, all_instructions)):
        divergences = [t["divergence"] for t in tokens_data]

        # Simple: show tokens as colored bars in a single/few rows
        n_tokens = len(tokens_data)
        max_per_row = 120
        n_rows = max(1, math.ceil(n_tokens / max_per_row))

        for i, t in enumerate(tokens_data):
            row = i // max_per_row
            col = i % max_per_row
            color = cmap(norm(np.clip(t["divergence"], -3, 3)))
            rect = mpatches.Rectangle(
                (col / max_per_row, 1 - (row + 1) / (n_rows + 0.5)),
                0.8 / max_per_row,
                0.8 / (n_rows + 0.5),
                facecolor=color, edgecolor="none",
            )
            ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        mean_div = np.mean(divergences) if divergences else 0
        label = f"[{idx}] {instruction[:60]}...  (μ={mean_div:.2f}, {len(tokens_data)} tok)"
        ax.set_title(label, fontsize=8, loc="left", pad=2, fontfamily="serif")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log p_T - log p_student", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Token-level divergence heatmaps")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--y_star_field", default="y_star_prefix30")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="diagnostics/token_heatmaps")
    args = parser.parse_args()

    set_neurips_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_jsonl(args.dataset)
    random.seed(args.seed)
    samples = random.sample(data, min(args.n_samples, len(data)))
    print(f"Selected {len(samples)} samples (seed={args.seed})")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + tokenizer
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    ).to(device).eval()

    all_tokens_data = []
    all_instructions = []

    for i, row in enumerate(samples):
        instruction = get_last_user_turn(row.get("x", []))
        ystar = get_content(row.get(args.y_star_field, ""))

        if not ystar:
            print(f"  Sample {i}: empty y*, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Sample {i}: {instruction[:80]}...")

        # Student logprobs (model sees only x)
        student_messages = build_student_messages(row)
        student_tokens = compute_token_logprobs(model, tokenizer, student_messages, ystar, device)

        # Teacher logprobs (model sees x + o + prefix(y))
        teacher_messages = build_teacher_messages(row, tokenizer)
        teacher_tokens = compute_token_logprobs(model, tokenizer, teacher_messages, ystar, device)

        # Merge: compute divergence
        n = min(len(student_tokens), len(teacher_tokens))
        tokens_data = []
        for j in range(n):
            s = student_tokens[j]
            t = teacher_tokens[j]
            div = t["log_prob"] - s["log_prob"]
            tokens_data.append({
                "token_id": s["token_id"],
                "token_text": s["token_text"],
                "log_p_teacher": t["log_prob"],
                "log_p_student": s["log_prob"],
                "divergence": div,
                "position": j,
            })

        if not tokens_data:
            print(f"  No tokens, skipping")
            continue

        divergences = [t["divergence"] for t in tokens_data]
        mean_div = np.mean(divergences)
        pct_red = np.mean([d > 0 for d in divergences]) * 100
        pct_blue = np.mean([d < 0 for d in divergences]) * 100

        # ── Print summary ────────────────────────────────────────────
        print(f"  Total tokens: {len(tokens_data)}")
        print(f"  Mean divergence: {mean_div:.4f}")
        print(f"  Teacher > Student: {pct_red:.1f}%")
        print(f"  Student > Teacher: {pct_blue:.1f}%")

        sorted_by_div = sorted(tokens_data, key=lambda t: t["divergence"], reverse=True)
        print(f"  Top 5 highest divergence (teacher >> student):")
        for t in sorted_by_div[:5]:
            print(f"    '{t['token_text']}'  div={t['divergence']:.3f}  "
                  f"logp_T={t['log_p_teacher']:.3f}  logp_S={t['log_p_student']:.3f}")
        print(f"  Top 5 lowest divergence (student >> teacher):")
        for t in sorted_by_div[-5:]:
            print(f"    '{t['token_text']}'  div={t['divergence']:.3f}  "
                  f"logp_T={t['log_p_teacher']:.3f}  logp_S={t['log_p_student']:.3f}")

        # ── Save JSON ────────────────────────────────────────────────
        json_data = {
            "instruction": instruction,
            "y_star": ystar,
            "mean_divergence": mean_div,
            "pct_teacher_gt_student": pct_red,
            "pct_student_gt_teacher": pct_blue,
            "tokens": tokens_data,
        }
        json_path = os.path.join(args.output_dir, f"sample_{i}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Saved {json_path}")

        # ── Render heatmap PNG ───────────────────────────────────────
        png_path = os.path.join(args.output_dir, f"sample_{i}.png")
        render_heatmap(tokens_data, instruction, png_path)
        print(f"  Saved {png_path}")

        all_tokens_data.append(tokens_data)
        all_instructions.append(instruction)

    # ── Summary grid ─────────────────────────────────────────────────
    if all_tokens_data:
        grid_path = os.path.join(args.output_dir, "summary_grid.png")
        render_summary_grid(all_tokens_data, all_instructions, grid_path)
        print(f"\nSaved summary grid → {grid_path}")

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
