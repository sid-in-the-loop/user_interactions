#!/usr/bin/env python3
"""
Off-policy diagnostic plots: compute per-token log probs of y* under 3 models,
then produce IS ratio histogram, scatter, and token movement plots.

Models:
  1. Student (init): Qwen3-8B base, prompted with x only
  2. JSD-trained: Qwen3-8B + LoRA adapter, prompted with x only
  3. Teacher: Qwen3-8B base, prompted with x + o + y[:30%] (prefix30)

Usage:
  python scripts/eval/logprob_diagnostics.py \
      --dataset datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
      --adapter step-15 \
      --subsample 1000 \
      --output-dir diagnostics
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── NeurIPS style ────────────────────────────────────────────────────────────

def set_neurips_style():
    plt.rcParams.update({
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":     "cm",
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      8.5,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            False,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
    })


# ── Prompt building ──────────────────────────────────────────────────────────

def format_conversation(x: list) -> str:
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def get_content(obj):
    if isinstance(obj, dict):
        return obj.get("content", "") or ""
    return str(obj) if obj else ""


def extract_prefix(y_content: str, frac: float, tokenizer) -> str:
    token_ids = tokenizer.encode(y_content, add_special_tokens=False)
    if not token_ids:
        return ""
    n_keep = max(1, int(len(token_ids) * frac))
    return tokenizer.decode(token_ids[:n_keep], skip_special_tokens=True)


SYSTEM_STUDENT = (
    "You are a helpful assistant. Respond directly and helpfully to the user's request."
)

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
    """Student/JSD sees only x."""
    messages = [{"role": "system", "content": SYSTEM_STUDENT}]
    for turn in row.get("x", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def build_teacher_messages(row, tokenizer):
    """Teacher sees x + o + y[:30%]."""
    x_text = format_conversation(row.get("x", []))
    o_text = get_content(row.get("o", ""))
    y_text = get_content(row.get("y", ""))
    prefix = extract_prefix(y_text, 0.30, tokenizer)
    messages = [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": USER_TEACHER_TEMPLATE.format(
            x=x_text, o=o_text, prefix=prefix
        )},
    ]
    return messages


# ── Log prob computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_logprobs(model, tokenizer, messages, completion_text, device="cuda"):
    """
    Compute per-token log probs of completion_text given messages as prompt.
    Returns: list of (token_id, log_prob) for each token in completion.
    """
    # Build prompt + completion
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors=None
    )
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    if not completion_ids:
        return []

    # Concatenate
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Get log probs for completion tokens
    # logits[i] predicts token[i+1], so for completion starting at len(prompt_ids):
    # logits[len(prompt_ids)-1] predicts completion_ids[0]
    # logits[len(prompt_ids)] predicts completion_ids[1]
    # etc.
    prompt_len = len(prompt_ids)
    log_probs = torch.log_softmax(logits, dim=-1)

    token_logprobs = []
    for i, tid in enumerate(completion_ids):
        pos = prompt_len - 1 + i  # position in logits that predicts this token
        if pos < log_probs.shape[0]:
            lp = log_probs[pos, tid].item()
            token_logprobs.append((tid, lp))

    return token_logprobs


def compute_all_logprobs(model, tokenizer, data, messages_fn, label, device="cuda"):
    """Compute log probs for all samples. Returns list of list of (token_id, logprob)."""
    all_logprobs = []
    for row in tqdm(data, desc=f"LogProbs [{label}]"):
        messages = messages_fn(row)
        ystar = get_content(row.get("y_star_prefix30", ""))
        lps = compute_logprobs(model, tokenizer, messages, ystar, device)
        all_logprobs.append(lps)
    return all_logprobs


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot1_is_ratio_histogram(student_lps, teacher_lps, output_dir):
    """Plot 1: histogram of IS ratio r_t = p_student / p_teacher."""
    log_ratios = []
    for s_lps, t_lps in zip(student_lps, teacher_lps):
        n = min(len(s_lps), len(t_lps))
        for i in range(n):
            s_logp = s_lps[i][1]
            t_logp = t_lps[i][1]
            log_r = s_logp - t_logp  # log(p_s / p_t)
            log_ratios.append(log_r)

    log_ratios = np.array(log_ratios)
    ratios = np.exp(np.clip(log_ratios, -20, 20))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: histogram of r_t (log x-axis)
    ax1.hist(ratios, bins=100, range=(1e-3, 1e3), color="#2166AC", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax1.set_xscale("log")
    ax1.axvline(1.0, color="#B2182B", linewidth=1.2, linestyle="--", label="$r=1$")
    ax1.set_xlabel("IS ratio $r_t = p_{\\theta_0}(y^*_t) \\,/\\, p_T(y^*_t)$")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    ax1.spines["left"].set_alpha(0.4)
    ax1.spines["bottom"].set_alpha(0.4)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax1.set_axisbelow(True)

    # Right: histogram of log(r_t)
    ax2.hist(log_ratios, bins=100, range=(-10, 10), color="#2166AC", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="#B2182B", linewidth=1.2, linestyle="--", label="$\\log r=0$")
    ax2.set_xlabel("$\\log \\, r_t = \\log p_{\\theta_0} - \\log p_T$")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.spines["left"].set_alpha(0.4)
    ax2.spines["bottom"].set_alpha(0.4)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"plot1_is_ratio.{ext}"))
    plt.close(fig)

    # Stats
    median_r = np.median(ratios)
    mean_log_r = np.mean(log_ratios)
    std_log_r = np.std(log_ratios)
    frac_gt10 = np.mean(ratios > 10)
    frac_lt01 = np.mean(ratios < 0.1)
    print(f"\nPlot 1 — IS Ratio Stats:")
    print(f"  Median r:           {median_r:.4f}")
    print(f"  Mean log(r):        {mean_log_r:.4f}")
    print(f"  Std log(r):         {std_log_r:.4f}")
    print(f"  Frac r > 10:        {frac_gt10:.4f}")
    print(f"  Frac r < 0.1:       {frac_lt01:.4f}")
    print(f"  Total tokens:       {len(log_ratios)}")


def plot2_scatter_student_vs_teacher(student_lps, teacher_lps, output_dir):
    """Plot 2: scatter of p_student vs p_teacher per token."""
    s_all, t_all = [], []
    for s_lps, t_lps in zip(student_lps, teacher_lps):
        n = min(len(s_lps), len(t_lps))
        for i in range(n):
            s_all.append(s_lps[i][1])
            t_all.append(t_lps[i][1])

    s_all = np.array(s_all)
    t_all = np.array(t_all)

    fig, ax = plt.subplots(figsize=(5, 5))

    # 2D histogram for density coloring
    h, xedges, yedges = np.histogram2d(t_all, s_all, bins=200,
                                        range=[[-15, 0], [-15, 0]])
    # Map each point to its bin density
    xbin = np.clip(np.digitize(t_all, xedges) - 1, 0, h.shape[0] - 1)
    ybin = np.clip(np.digitize(s_all, yedges) - 1, 0, h.shape[1] - 1)
    density = h[xbin, ybin]

    # Subsample for plotting if too many points
    if len(s_all) > 50000:
        idx = np.random.default_rng(42).choice(len(s_all), 50000, replace=False)
        s_plot, t_plot, d_plot = s_all[idx], t_all[idx], density[idx]
    else:
        s_plot, t_plot, d_plot = s_all, t_all, density

    scatter = ax.scatter(t_plot, s_plot, c=d_plot, s=1, alpha=0.4,
                         cmap="viridis", norm=mcolors.LogNorm(), rasterized=True)
    ax.plot([-15, 0], [-15, 0], "--", color="#B2182B", linewidth=1.0, alpha=0.7, label="$y=x$")

    ax.set_xlabel("$\\log \\, p_T(y^*_t)$  (teacher)")
    ax.set_ylabel("$\\log \\, p_{\\theta_0}(y^*_t)$  (student)")
    ax.set_xlim(-15, 0)
    ax.set_ylim(-15, 0)
    ax.set_aspect("equal")
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    # Annotate fractions
    above = np.mean(s_all > t_all)
    below = np.mean(s_all < t_all)
    ax.text(0.05, 0.95, f"Student > Teacher: {above:.1%}",
            transform=ax.transAxes, fontsize=8, va="top", color="#2166AC")
    ax.text(0.05, 0.89, f"Teacher > Student: {below:.1%}",
            transform=ax.transAxes, fontsize=8, va="top", color="#B2182B")

    ax.legend(fontsize=8, loc="lower right")
    plt.colorbar(scatter, ax=ax, label="Density", shrink=0.8)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"plot2_scatter.{ext}"))
    plt.close(fig)

    print(f"\nPlot 2 — Scatter Stats:")
    print(f"  Student > Teacher:  {above:.4f}")
    print(f"  Teacher > Student:  {below:.4f}")
    print(f"  Equal (within 0.01): {np.mean(np.abs(s_all - t_all) < 0.01):.4f}")


def plot3_token_movement(student_lps, jsd_lps, output_dir):
    """Plot 3: delta = p_JSD - p_student for each token."""
    s_all, delta_all = [], []
    for s_lps, j_lps in zip(student_lps, jsd_lps):
        n = min(len(s_lps), len(j_lps))
        for i in range(n):
            s_logp = s_lps[i][1]
            j_logp = j_lps[i][1]
            # Convert to probs for delta
            s_p = np.exp(s_logp)
            j_p = np.exp(j_logp)
            s_all.append(s_logp)
            delta_all.append(j_p - s_p)

    s_all = np.array(s_all)
    delta_all = np.array(delta_all)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                     gridspec_kw={"width_ratios": [3, 2]})

    # Left: scatter — x = log p_student, y = delta
    colors = np.where(delta_all > 0, "#2166AC", "#B2182B")
    if len(s_all) > 50000:
        idx = np.random.default_rng(42).choice(len(s_all), 50000, replace=False)
        ax1.scatter(s_all[idx], delta_all[idx], c=colors[idx], s=1, alpha=0.3, rasterized=True)
    else:
        ax1.scatter(s_all, delta_all, c=colors, s=1, alpha=0.3, rasterized=True)

    ax1.axhline(0, color="#333", linewidth=0.7, linestyle="-", alpha=0.5)
    ax1.set_xlabel("$\\log \\, p_{\\theta_0}(y^*_t)$  (init student)")
    ax1.set_ylabel("$\\Delta_t = p_{JSD}(y^*_t) - p_{\\theta_0}(y^*_t)$")
    ax1.set_xlim(-15, 0)
    ax1.spines["left"].set_alpha(0.4)
    ax1.spines["bottom"].set_alpha(0.4)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax1.set_axisbelow(True)

    # Annotate
    improved = np.mean(delta_all > 0)
    worsened = np.mean(delta_all < 0)
    big_move = np.mean(np.abs(delta_all) > 0.05)
    big_up = np.mean(delta_all > 0.05)
    big_down = np.mean(delta_all < -0.05)
    ax1.text(0.05, 0.95, f"Improved (Δ>0): {improved:.1%}",
             transform=ax1.transAxes, fontsize=8, va="top", color="#2166AC")
    ax1.text(0.05, 0.89, f"Worsened (Δ<0): {worsened:.1%}",
             transform=ax1.transAxes, fontsize=8, va="top", color="#B2182B")

    # Right: histogram of delta
    ax2.hist(delta_all, bins=200, range=(-0.3, 0.3), color="#666", alpha=0.8,
             edgecolor="white", linewidth=0.3, orientation="horizontal")
    ax2.axhline(0, color="#333", linewidth=0.7, linestyle="-", alpha=0.5)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("$\\Delta_t$")
    ax2.set_ylim(-0.3, 0.3)
    ax2.spines["left"].set_alpha(0.4)
    ax2.spines["bottom"].set_alpha(0.4)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"plot3_token_movement.{ext}"))
    plt.close(fig)

    print(f"\nPlot 3 — Token Movement Stats:")
    print(f"  Improved (Δ>0):     {improved:.4f}")
    print(f"  Worsened (Δ<0):     {worsened:.4f}")
    print(f"  |Δ| > 0.05:        {big_move:.4f}")
    print(f"    of which up:      {big_up:.4f}")
    print(f"    of which down:    {big_down:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Off-policy diagnostic plots")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()

    set_neurips_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and subsample data
    with open(args.dataset) as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(data)} samples")

    random.seed(args.seed)
    if len(data) > args.subsample:
        data = random.sample(data, args.subsample)
    print(f"Subsampled to {len(data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Model 1: Student (init) ──────────────────────────────────────────────
    print(f"\nLoading student (init): {args.model}")
    model_student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    print("Computing student log probs...")
    student_lps = compute_all_logprobs(
        model_student, tokenizer, data,
        lambda row: build_student_messages(row),
        "Student"
    )

    # ── Model 3: Teacher (same base, different prompt) ───────────────────────
    # Reuse the same model, just different prompts
    print("\nComputing teacher log probs (same model, prefix30 prompt)...")
    teacher_lps = compute_all_logprobs(
        model_student, tokenizer, data,
        lambda row: build_teacher_messages(row, tokenizer),
        "Teacher"
    )

    # ── Model 2: JSD-trained (base + LoRA) ───────────────────────────────────
    print(f"\nLoading JSD adapter: {args.adapter}")
    model_jsd = PeftModel.from_pretrained(model_student, args.adapter).eval()

    print("Computing JSD log probs...")
    jsd_lps = compute_all_logprobs(
        model_jsd, tokenizer, data,
        lambda row: build_student_messages(row),
        "JSD"
    )

    # Free GPU memory
    del model_student, model_jsd
    torch.cuda.empty_cache()

    # Save raw log probs
    print(f"\nSaving raw log probs...")
    torch.save({
        "student": student_lps,
        "teacher": teacher_lps,
        "jsd": jsd_lps,
    }, os.path.join(args.output_dir, "logprobs.pt"))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot1_is_ratio_histogram(student_lps, teacher_lps, args.output_dir)
    plot2_scatter_student_vs_teacher(student_lps, teacher_lps, args.output_dir)
    plot3_token_movement(student_lps, jsd_lps, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
