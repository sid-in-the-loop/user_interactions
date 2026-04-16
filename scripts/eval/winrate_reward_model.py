#!/usr/bin/env python3
"""
Win rate evaluation using a reward model (via vLLM) as judge.

Scores each response independently with the reward model.
Higher score wins. If scores are within --tie-threshold, it's a tie.

Compares model checkpoint outputs vs GPT-4 turbo reference (AlpacaEval format).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


def score_response(server_url: str, model: str, instruction: str, response: str) -> float:
    """Score a single (instruction, response) pair via the vLLM reward endpoint."""
    payload = {
        "model": model,
        "text_1": instruction,
        "text_2": response,
    }
    resp = requests.post(f"{server_url}/v1/score", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["score"]


def score_batch(server_url: str, model: str, items: list[dict], label: str, max_workers: int = 32) -> list[float]:
    """Score a batch of (instruction, output) pairs concurrently."""
    scores = [None] * len(items)

    def _score(idx):
        return idx, score_response(server_url, model, items[idx]["instruction"], items[idx]["output"])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score, i): i for i in range(len(items))}
        for future in tqdm(as_completed(futures), total=len(items), desc=f"Scoring {label}"):
            idx, sc = future.result()
            scores[idx] = sc

    return scores


def main():
    parser = argparse.ArgumentParser(description="Win rate eval using reward model as judge")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Dir with model_outputs.json (or pass multiple with --checkpoint-dirs)")
    parser.add_argument("--checkpoint-dirs", nargs="+", default=None,
                        help="Multiple checkpoint dirs to evaluate")
    parser.add_argument("--reference", required=True,
                        help="Path to GPT-4 turbo reference JSON (AlpacaEval format)")
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="vLLM server URL")
    parser.add_argument("--model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                        help="Model name for the API request")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <checkpoint-dir>/winrate_reward/)")
    parser.add_argument("--tie-threshold", type=float, default=0.5,
                        help="Score difference below this is a tie (default: 0.5)")
    parser.add_argument("--max-workers", type=int, default=32,
                        help="Max concurrent requests to vLLM server")
    args = parser.parse_args()

    # Collect all checkpoint dirs to process
    ckpt_dirs = args.checkpoint_dirs if args.checkpoint_dirs else [args.checkpoint_dir]

    # Load reference once
    with open(args.reference) as f:
        reference = json.load(f)
    ref_by_instruction = {r["instruction"]: r["output"] for r in reference}

    for ckpt_dir in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_dir)
        model_outputs_path = os.path.join(ckpt_dir, "model_outputs.json")
        if not os.path.exists(model_outputs_path):
            print(f"SKIP {ckpt_name}: no model_outputs.json", file=sys.stderr)
            continue

        with open(model_outputs_path) as f:
            model_outputs = json.load(f)

        # Match by instruction
        paired = []
        for m in model_outputs:
            ref_out = ref_by_instruction.get(m["instruction"])
            if ref_out is None:
                continue
            paired.append({
                "instruction": m["instruction"],
                "model_output": m["output"],
                "reference_output": ref_out,
                "dataset": m.get("dataset", ""),
            })

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"Matched examples: {len(paired)}")
        print(f"{'='*60}")

        # Score model outputs
        model_items = [{"instruction": p["instruction"], "output": p["model_output"]} for p in paired]
        model_scores = score_batch(args.server_url, args.model, model_items, f"{ckpt_name}/model", args.max_workers)

        # Score reference outputs
        ref_items = [{"instruction": p["instruction"], "output": p["reference_output"]} for p in paired]
        ref_scores = score_batch(args.server_url, args.model, ref_items, f"{ckpt_name}/ref", args.max_workers)

        # Compare
        wins, losses, ties = 0, 0, 0
        results = []
        for i, p in enumerate(paired):
            diff = model_scores[i] - ref_scores[i]
            if diff > args.tie_threshold:
                winner = "model"
                wins += 1
            elif diff < -args.tie_threshold:
                winner = "reference"
                losses += 1
            else:
                winner = "tie"
                ties += 1
            results.append({
                "instruction": p["instruction"][:200],
                "model_score": model_scores[i],
                "reference_score": ref_scores[i],
                "diff": diff,
                "winner": winner,
            })

        n = len(paired)
        win_rate = wins / n * 100 if n > 0 else 0
        win_rate_no_ties = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        summary_lines = [
            f"Checkpoint: {ckpt_name}",
            f"Judge: {args.model}",
            f"Examples: {n}",
            f"Tie threshold: {args.tie_threshold}",
            "",
            f"Model wins:     {wins:>5} ({wins/n*100:.1f}%)",
            f"Reference wins: {losses:>5} ({losses/n*100:.1f}%)",
            f"Ties:           {ties:>5} ({ties/n*100:.1f}%)",
            "",
            f"Win rate (inc. ties):  {win_rate:.1f}%",
            f"Win rate (exc. ties):  {win_rate_no_ties:.1f}%",
            f"Avg model score:       {sum(model_scores)/n:.4f}",
            f"Avg reference score:   {sum(ref_scores)/n:.4f}",
        ]

        for line in summary_lines:
            print(line)

        # Save results
        out_dir = args.output_dir or os.path.join(ckpt_dir, "winrate_reward")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))
        print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
