#!/usr/bin/env python3
"""
Compute length-controlled (LC) win rate from existing pairwise judge results.

Reads winrate_qwen_judge/results.json + model_outputs.json from each checkpoint dir.
Fits logistic regression: win_prob ~ log(model_length / ref_length), then
predicts win rate at length_ratio=1 (equal length).

This matches the methodology from AlpacaEval 2.0 (simplified).
"""

import argparse
import json
import math
import os
import sys

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_lc_winrate(win_labels, length_ratios):
    """
    Fit logistic regression: P(win) = sigmoid(a + b * log_length_ratio)
    LC win rate = sigmoid(a)  (i.e., predicted win prob when log_ratio=0, equal length)

    Uses simple gradient descent since we don't want sklearn dependency.
    """
    X = np.array(length_ratios)
    y = np.array(win_labels, dtype=float)
    n = len(y)

    # Initialize
    a, b = 0.0, 0.0
    lr = 0.01

    for _ in range(5000):
        z = a + b * X
        p = sigmoid(z)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        grad_a = np.mean(p - y)
        grad_b = np.mean((p - y) * X)

        a -= lr * grad_a
        b -= lr * grad_b

    # LC win rate = P(win | log_ratio=0) = sigmoid(a)
    lc_wr = sigmoid(a) * 100
    return lc_wr, a, b


def main():
    parser = argparse.ArgumentParser(description="Compute LC win rate from existing judge results")
    parser.add_argument("--checkpoint-dirs", nargs="+", required=True)
    parser.add_argument("--reference", required=True, help="GPT-4 turbo reference JSON")
    parser.add_argument("--judge-subdir", default="winrate_qwen_judge",
                        help="Subdirectory name with results.json")
    args = parser.parse_args()

    with open(args.reference) as f:
        reference = json.load(f)
    ref_lengths = {r["instruction"]: len(r["output"]) for r in reference}

    print(f"{'Checkpoint':<35} {'WR':>8} {'WR(exc)':>8} {'LC-WR':>8} {'AvgLen':>8}")
    print("-" * 75)

    for ckpt_dir in sorted(args.checkpoint_dirs):
        ckpt_name = os.path.basename(ckpt_dir)
        results_path = os.path.join(ckpt_dir, args.judge_subdir, "results.json")
        outputs_path = os.path.join(ckpt_dir, "model_outputs.json")

        if not os.path.exists(results_path) or not os.path.exists(outputs_path):
            print(f"SKIP {ckpt_name}", file=sys.stderr)
            continue

        with open(results_path) as f:
            results = json.load(f)
        with open(outputs_path) as f:
            model_outputs = json.load(f)

        # Build instruction -> model output length map
        model_lengths = {m["instruction"]: len(m["output"]) for m in model_outputs}

        # Match results back to instructions via index
        # results are indexed by position in the paired list, which matches model_outputs order
        # Re-derive: sort results by index
        results_sorted = sorted(results, key=lambda r: r["index"])

        win_labels = []
        log_length_ratios = []
        wins, losses, ties = 0, 0, 0
        total_model_len = 0
        matched = 0

        for r, m in zip(results_sorted, model_outputs):
            instruction = m["instruction"]
            ref_len = ref_lengths.get(instruction, 1)
            mod_len = len(m["output"])
            total_model_len += mod_len

            winner = r["winner"]
            if winner == "model":
                wins += 1
                win_labels.append(1.0)
            elif winner == "reference":
                losses += 1
                win_labels.append(0.0)
            else:
                ties += 1
                win_labels.append(0.5)

            ratio = math.log(max(mod_len, 1) / max(ref_len, 1))
            log_length_ratios.append(ratio)
            matched += 1

        n = matched
        wr = wins / n * 100
        wr_no_ties = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        avg_len = total_model_len / n

        lc_wr, a, b = fit_lc_winrate(win_labels, log_length_ratios)

        print(f"{ckpt_name:<35} {wr:>7.1f}% {wr_no_ties:>7.1f}% {lc_wr:>7.1f}% {avg_len:>7.0f}")

        # Save updated summary
        out_dir = os.path.join(ckpt_dir, args.judge_subdir)
        summary_lines = [
            f"Checkpoint: {ckpt_name}",
            f"Examples: {n}",
            "",
            f"Model wins:     {wins:>5} ({wins/n*100:.1f}%)",
            f"Reference wins: {losses:>5} ({losses/n*100:.1f}%)",
            f"Ties:           {ties:>5} ({ties/n*100:.1f}%)",
            "",
            f"Win rate (inc. ties):  {wr:.1f}%",
            f"Win rate (exc. ties):  {wr_no_ties:.1f}%",
            f"LC win rate:           {lc_wr:.1f}%",
            f"Avg model length:      {avg_len:.0f}",
            f"Logistic coefs:        a={a:.4f}, b={b:.4f}",
        ]
        with open(os.path.join(out_dir, "summary_lc.txt"), "w") as f:
            f.write("\n".join(summary_lines))


if __name__ == "__main__":
    main()
