#!/usr/bin/env python3
"""
Win rate evaluation using a local vLLM-served LLM as pairwise judge.

Same methodology as winrate_eval.py (GPT-4o-mini judge):
- Shows judge both responses to an instruction
- Asks it to pick A, B, or C (tie)
- Runs twice with flipped order for position-bias removal
- Only counts a win if the same response wins both calls

Compares model checkpoint outputs vs GPT-4 turbo reference (AlpacaEval format).
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

JUDGE_SYSTEM = """You are a strict, impartial evaluator comparing two AI assistant responses to the same user request. Judge based on: correctness, relevance, completeness, and quality of the response relative to what the user asked. Do NOT favor longer responses automatically. Do NOT consider formatting unless it meaningfully affects clarity.
Output exactly one character: A, B, or C where C means tie or too close to call."""

JUDGE_USER_TEMPLATE = """User request:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response better addresses the user's request? Output only A, B, or C."""


def parse_judge_output(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    text = text.strip().upper()
    m = re.search(r"\b([ABC])\b", text)
    return m.group(1) if m else None


def call_judge(server_url: str, model: str, instruction: str, response_a: str, response_b: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                instruction=instruction, response_a=response_a, response_b=response_b
            )},
        ],
        "max_tokens": 200,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return parse_judge_output(content)
        except Exception as e:
            if attempt == 2:
                print(f"Judge error: {e}", file=sys.stderr)
                return None


def resolve_winner(first: str | None, second: str | None) -> str:
    """Given two judge calls (A/B and B/A), return winner with position-bias removal."""
    if first is None or second is None:
        return "tie"
    if first == "C" or second == "C":
        return "tie"
    # first call: A=resp1, B=resp2; second call: A=resp2, B=resp1
    # first=A, second=B means resp1 wins both
    # first=B, second=A means resp2 wins both
    if first == "A" and second == "B":
        return "model"
    if first == "B" and second == "A":
        return "reference"
    return "tie"


def judge_pair(server_url: str, model: str, instruction: str, model_resp: str, ref_resp: str) -> dict:
    """Run judge twice with flipped order. Return result dict."""
    # Call 1: A=model, B=reference
    r1 = call_judge(server_url, model, instruction, model_resp, ref_resp)
    # Call 2: A=reference, B=model (flipped)
    r2 = call_judge(server_url, model, instruction, ref_resp, model_resp)
    winner = resolve_winner(r1, r2)
    return {"winner": winner, "call1": r1, "call2": r2}


def main():
    parser = argparse.ArgumentParser(description="Win rate eval using local LLM as pairwise judge")
    parser.add_argument("--checkpoint-dirs", nargs="+", required=True,
                        help="Checkpoint dirs with model_outputs.json")
    parser.add_argument("--reference", required=True,
                        help="Path to GPT-4 turbo reference JSON")
    parser.add_argument("--server-url", default="http://localhost:8003")
    parser.add_argument("--model", default="Qwen/Qwen3-14B",
                        help="Model name for API requests")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Max concurrent requests")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N examples (default: all 805)")
    args = parser.parse_args()

    # Load reference
    with open(args.reference) as f:
        reference = json.load(f)
    ref_by_instruction = {r["instruction"]: r["output"] for r in reference}

    for ckpt_dir in args.checkpoint_dirs:
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
            })

        if args.subsample and len(paired) > args.subsample:
            import random
            random.seed(42)
            paired = random.sample(paired, args.subsample)

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"Examples: {len(paired)}")
        print(f"Judge: {args.model}")
        print(f"{'='*60}")

        wins, losses, ties = 0, 0, 0
        results = []

        def _judge(idx):
            p = paired[idx]
            r = judge_pair(args.server_url, args.model, p["instruction"], p["model_output"], p["reference_output"])
            return idx, r

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_judge, i): i for i in range(len(paired))}
            for future in tqdm(as_completed(futures), total=len(paired), desc=ckpt_name):
                idx, r = future.result()
                if r["winner"] == "model":
                    wins += 1
                elif r["winner"] == "reference":
                    losses += 1
                else:
                    ties += 1
                results.append({"index": idx, **r})

        n = len(paired)
        win_rate = wins / n * 100 if n > 0 else 0
        wr_no_ties = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        summary_lines = [
            f"Checkpoint: {ckpt_name}",
            f"Judge: {args.model}",
            f"Examples: {n}",
            "",
            f"Model wins:     {wins:>5} ({wins/n*100:.1f}%)",
            f"Reference wins: {losses:>5} ({losses/n*100:.1f}%)",
            f"Ties:           {ties:>5} ({ties/n*100:.1f}%)",
            "",
            f"Win rate (inc. ties):  {win_rate:.1f}%",
            f"Win rate (exc. ties):  {wr_no_ties:.1f}%",
        ]

        for line in summary_lines:
            print(line)

        out_dir = os.path.join(ckpt_dir, "winrate_qwen_judge")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))
        print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
