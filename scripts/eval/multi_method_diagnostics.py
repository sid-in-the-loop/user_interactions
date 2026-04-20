#!/usr/bin/env python3
"""
Multi-method diagnostic comparison.

For N samples, compute per-token logprobs under init model and each method's
final checkpoint. Produce:
  1. Per-sample side-by-side heatmaps (divergence from init per method)
  2. Blue/Red/White token analysis across methods
  3. Mean log prob improvement bar chart per method
  4. Per-token learning trajectory across training steps (for one method)

Usage:
  python scripts/eval/multi_method_diagnostics.py \
      --dataset datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
      --checkpoints_root /projects/bgtw/ssredharan/checkpoints \
      --output_dir diagnostics/multi_method \
      --n_samples 10
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ── Style ────────────────────────────────────────────────────────────────────

def set_neurips_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
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
        {"role": "user", "content": USER_TEACHER_TEMPLATE.format(x=x_text, o=o_text, prefix=prefix)},
    ]


# ── Log prob computation ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_token_logprobs(model, tokenizer, messages, completion_text, device="cuda"):
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors=None
    )
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        return [], []

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    logits = model(input_ids).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    token_logprobs = []
    token_texts = []
    for i, tid in enumerate(completion_ids):
        pos = prompt_len - 1 + i
        if pos < log_probs.shape[0]:
            lp = log_probs[pos, tid].item()
            token_logprobs.append(lp)
            token_texts.append(tokenizer.decode([tid]))

    return token_logprobs, token_texts


# ── Method configs ───────────────────────────────────────────────────────────

METHOD_DISPLAY = {
    "sft_p30": "SFT",
    "fkl_p30": "FKL",
    "jsd_p30": "JSD",
    "dpo_p30": "DPO",
    "jsd_is1_p30": "IS-1",
    "jsd_is2_p30": "IS-2",
    "jsd_is3_p30": "IS-3",
    "jsd_is4_p30": "IS-4",
    "zg_jsd_p30": "ZG-JSD",
    "rkl_p30": "RKL",
    "rlad_p30": "RLAD",
    "distillm2_p30": "DistiLLM2",
}

METHOD_COLORS = {
    "SFT": "#1f77b4",
    "FKL": "#ff7f0e",
    "JSD": "#2ca02c",
    "DPO": "#d62728",
    "IS-1": "#9467bd",
    "IS-2": "#8c564b",
    "IS-3": "#e377c2",
    "IS-4": "#7f7f7f",
    "ZG-JSD": "#bcbd22",
    "RKL": "#17becf",
    "RLAD": "#aec7e8",
    "DistiLLM2": "#ffbb78",
}


def find_final_checkpoint(checkpoints_root, method):
    method_dir = Path(checkpoints_root) / method
    if (method_dir / "final").exists():
        return method_dir / "final"
    # Fall back to highest step
    steps = sorted(
        [d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(re.search(r"\d+", d.name).group())
    )
    return steps[-1] if steps else None


# ── Plot 1: Improvement bar chart ────────────────────────────────────────────

def plot_improvement_bars(results, output_dir):
    """Bar chart: mean log prob improvement over init for each method."""
    set_neurips_style()

    methods = []
    improvements = []
    colors = []

    for method_name in sorted(results.keys()):
        if method_name in ("init", "teacher"):
            continue
        display = METHOD_DISPLAY.get(method_name, method_name)
        all_deltas = []
        for si, sample in results[method_name].items():
            init_lps = results["init"][si]["logprobs"]
            method_lps = sample["logprobs"]
            n = min(len(init_lps), len(method_lps))
            if n == 0: continue
            delta = np.mean([method_lps[i] - init_lps[i] for i in range(n)])
            all_deltas.append(delta)
        methods.append(display)
        improvements.append(np.mean(all_deltas))
        colors.append(METHOD_COLORS.get(display, "#333333"))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(methods)), improvements, color=colors, alpha=0.88, edgecolor="white")

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01 if val > 0 else val - 0.03,
                f"{val:.3f}", ha="center", va="bottom" if val > 0 else "top", fontsize=7)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean log p improvement over init")
    ax.axhline(0, color="#999", linewidth=0.7, linestyle="--")
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(output_dir, "improvement_bars.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 2: Blue/Red/White analysis ──────────────────────────────────────────

def plot_zone_analysis(results, output_dir):
    """For each method, show how much it moved blue/red/white tokens."""
    set_neurips_style()

    if "teacher" not in results:
        print("Skipping zone analysis (no teacher logprobs)")
        return

    methods = []
    red_improvements = []
    blue_changes = []
    white_changes = []

    for method_name in sorted(results.keys()):
        if method_name in ("init", "teacher"):
            continue
        display = METHOD_DISPLAY.get(method_name, method_name)
        methods.append(display)

        red_deltas, blue_deltas, white_deltas = [], [], []

        for si, sample in results[method_name].items():
            init_lps = results["init"][si]["logprobs"]
            teacher_lps = results["teacher"][si]["logprobs"]
            method_lps = sample["logprobs"]
            n = min(len(init_lps), len(teacher_lps), len(method_lps))

            for i in range(n):
                gap = teacher_lps[i] - init_lps[i]  # teacher - init
                change = method_lps[i] - init_lps[i]  # method - init

                if gap > 0.5:  # red: teacher >> student
                    red_deltas.append(change)
                elif gap < -0.5:  # blue: student >> teacher
                    blue_deltas.append(change)
                else:  # white: roughly equal
                    white_deltas.append(change)

        red_improvements.append(np.mean(red_deltas) if red_deltas else 0)
        blue_changes.append(np.mean(blue_deltas) if blue_deltas else 0)
        white_changes.append(np.mean(white_deltas) if white_deltas else 0)

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width, red_improvements, width, label="Red (teacher > student)", color="#B2182B", alpha=0.8)
    ax.bar(x, white_changes, width, label="White (roughly equal)", color="#999999", alpha=0.8)
    ax.bar(x + width, blue_changes, width, label="Blue (student > teacher)", color="#2166AC", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Mean log p change from init")
    ax.axhline(0, color="#333", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(output_dir, "zone_analysis.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved {path}")


# ── Plot 3: Per-sample comparison heatmap ────────────────────────────────────

def plot_sample_comparison(results, sample_idx, output_dir):
    """For one sample, show token divergence heatmap for each method."""
    set_neurips_style()

    init_data = results["init"][sample_idx]
    init_lps = init_data["logprobs"]
    token_texts = init_data["token_texts"]
    instruction = init_data["instruction"][:80]

    method_names = [m for m in sorted(results.keys()) if m not in ("init", "teacher")]
    n_methods = len(method_names)
    n_tokens = min(100, len(init_lps))  # Show first 100 tokens max

    fig, axes = plt.subplots(n_methods, 1, figsize=(16, n_methods * 0.8 + 1), squeeze=False)

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    for row, method_name in enumerate(method_names):
        ax = axes[row, 0]
        display = METHOD_DISPLAY.get(method_name, method_name)
        method_lps = results[method_name][sample_idx]["logprobs"]
        n = min(n_tokens, len(method_lps))

        # Compute change from init
        changes = [method_lps[i] - init_lps[i] for i in range(n)]

        for i, change in enumerate(changes):
            color = cmap(norm(np.clip(change, -2, 2)))
            rect = mpatches.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none")
            ax.add_patch(rect)

        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([display], fontsize=8)
        ax.set_xticks([])
        ax.spines["left"].set_alpha(0.2)
        ax.spines["bottom"].set_alpha(0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="bottom", fraction=0.03, pad=0.08)
    cbar.set_label("log p change from init (blue=worse, red=better)")

    safe_instruction = instruction.replace("$", "").replace("\\", "")[:60]
    fig.suptitle(f"Sample {sample_idx}: {safe_instruction}...", fontsize=10, y=1.02)

    fig.tight_layout()
    path = os.path.join(output_dir, f"sample_{sample_idx}_comparison.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)


# ── Plot 4: Training trajectory for one method ──────────────────────────────

def plot_training_trajectory(model, tokenizer, data, method_name, checkpoints_root,
                             sample_indices, output_dir, device="cuda"):
    """Track per-token log prob across training steps for one method."""
    set_neurips_style()

    method_dir = Path(checkpoints_root) / method_name
    steps = sorted(
        [d for d in method_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(re.search(r"\d+", d.name).group())
    )
    # Sample every 5th step to keep it manageable
    steps = steps[::5]
    if len(steps) > 10:
        steps = steps[::2]

    display = METHOD_DISPLAY.get(method_name, method_name)

    for si in sample_indices[:3]:  # First 3 samples
        row = data[si]
        ystar = get_content(row.get("y_star_prefix30", ""))
        if not ystar:
            continue

        messages = build_student_messages(row)
        instruction = get_last_user_turn(row.get("x", []))

        step_numbers = []
        mean_logprobs = []

        for step_dir in tqdm(steps, desc=f"Trajectory sample {si}"):
            step_num = int(re.search(r"\d+", step_dir.name).group())

            try:
                peft_model = PeftModel.from_pretrained(model, str(step_dir))
                peft_model = peft_model.to(device).eval()
                lps, _ = compute_token_logprobs(peft_model, tokenizer, messages, ystar, device)
                del peft_model
                torch.cuda.empty_cache()

                step_numbers.append(step_num)
                mean_logprobs.append(np.mean(lps) if lps else 0)
            except Exception as e:
                print(f"    Skip {step_dir.name}: {e}")

        if not step_numbers:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(step_numbers, mean_logprobs, "o-", color="#2166AC", markersize=4, linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean log p(y*)")
        safe_inst = instruction[:50].replace("$", "").replace("\\", "")
        ax.set_title(f"{display} — Sample {si}", fontsize=10)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()
        path = os.path.join(output_dir, method_name, f"trajectory_sample_{si}.pdf")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"))
        plt.close(fig)
        print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-method diagnostic comparison")
    parser.add_argument("--dataset", default="datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl")
    parser.add_argument("--checkpoints_root", default="/projects/bgtw/ssredharan/checkpoints")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", default="diagnostics/multi_method")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to compare (default: all available)")
    parser.add_argument("--trajectory_method", default="jsd_p30",
                        help="Method for training trajectory plot")
    parser.add_argument("--skip_trajectory", action="store_true",
                        help="Skip the slow training trajectory plot")
    args = parser.parse_args()

    set_neurips_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_jsonl(args.dataset)
    random.seed(args.seed)
    sample_indices = random.sample(range(len(data)), min(args.n_samples, len(data)))
    print(f"Selected {len(sample_indices)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    ).to(device).eval()

    # ── Compute init logprobs ────────────────────────────────────────────
    print("\nComputing init (base model) logprobs...")
    results = {"init": {}}

    for si in tqdm(sample_indices, desc="Init"):
        row = data[si]
        ystar = get_content(row.get("y_star_prefix30", ""))
        messages = build_student_messages(row)
        lps, texts = compute_token_logprobs(base_model, tokenizer, messages, ystar, device)
        results["init"][si] = {
            "sample_idx": si,
            "logprobs": lps,
            "token_texts": texts,
            "instruction": get_last_user_turn(row.get("x", [])),
        }

    # ── Compute teacher logprobs ─────────────────────────────────────────
    print("\nComputing teacher logprobs...")
    results["teacher"] = {}

    for si in tqdm(sample_indices, desc="Teacher"):
        row = data[si]
        ystar = get_content(row.get("y_star_prefix30", ""))
        messages = build_teacher_messages(row, tokenizer)
        lps, texts = compute_token_logprobs(base_model, tokenizer, messages, ystar, device)
        results["teacher"][si] = {
            "sample_idx": si,
            "logprobs": lps,
            "token_texts": texts,
        }

    # ── Compute logprobs for each method's final checkpoint ──────────────
    if args.methods:
        method_list = args.methods
    else:
        method_list = [d.name for d in sorted(Path(args.checkpoints_root).iterdir())
                       if d.is_dir() and d.name.endswith("_p30")]

    for method_name in method_list:
        ckpt_path = find_final_checkpoint(args.checkpoints_root, method_name)
        if ckpt_path is None:
            print(f"\n  {method_name}: no checkpoint found, skipping")
            continue

        display = METHOD_DISPLAY.get(method_name, method_name)
        print(f"\nComputing logprobs for {display} ({ckpt_path.name})...")

        try:
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt_path))
            peft_model = peft_model.to(device).eval()
        except Exception as e:
            print(f"  ERROR loading adapter: {e}")
            continue

        results[method_name] = {}
        for si in tqdm(sample_indices, desc=display):
            row = data[si]
            ystar = get_content(row.get("y_star_prefix30", ""))
            messages = build_student_messages(row)
            lps, texts = compute_token_logprobs(peft_model, tokenizer, messages, ystar, device)
            results[method_name][si] = {
                "sample_idx": si,
                "logprobs": lps,
                "token_texts": texts,
            }

        del peft_model
        torch.cuda.empty_cache()

    # ── Save raw data ────────────────────────────────────────────────────
    print("\nSaving raw logprobs...")
    torch.save(results, os.path.join(args.output_dir, "multi_method_logprobs.pt"))

    # ── Generate plots ───────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Plot 1: Improvement bars
    plot_improvement_bars(results, args.output_dir)

    # Plot 2: Zone analysis
    plot_zone_analysis(results, args.output_dir)

    # Plot 3: Per-sample comparison heatmaps
    for si in sample_indices[:5]:  # First 5 samples
        plot_sample_comparison(results, si, args.output_dir)

    # Plot 4: Training trajectory (optional, slow)
    if not args.skip_trajectory:
        print(f"\nComputing training trajectory for {args.trajectory_method}...")
        plot_training_trajectory(
            base_model, tokenizer, data, args.trajectory_method,
            args.checkpoints_root, sample_indices, args.output_dir, device
        )

    print(f"\nDone! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
