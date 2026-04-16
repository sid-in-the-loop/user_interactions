#!/usr/bin/env python3
"""
Pairwise winrate: y_base vs y_star_{prefix30, noprefix, full}
using a local vLLM-served LLM as judge.

Input: JSONL with fields x, y_base, y_star_prefix30, y_star_noprefix, y_star_full
"""

import argparse
import json
import os
import random
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


def parse_judge_output(text):
    if not text or not isinstance(text, str):
        return None
    m = re.search(r"\b([ABC])\b", text.strip().upper())
    return m.group(1) if m else None


def get_last_user_turn(x):
    for msg in reversed(x):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_content(obj):
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("content", "")
    return ""


def call_judge(server_url, model, instruction, response_a, response_b):
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


def resolve_winner(r1, r2):
    if r1 is None or r2 is None:
        return "tie"
    if r1 == "C" or r2 == "C":
        return "tie"
    if r1 == "A" and r2 == "B":
        return "first"
    if r1 == "B" and r2 == "A":
        return "second"
    return "tie"


def run_comparison(server_url, model, instruction, resp1, resp2):
    r1 = call_judge(server_url, model, instruction, resp1, resp2)
    r2 = call_judge(server_url, model, instruction, resp2, resp1)
    return resolve_winner(r1, r2), r1, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with y_base + y_star fields")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server-url", default="http://localhost:8002")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--comparisons", nargs="+", default=None,
                        metavar="NAME:FIELD_A:FIELD_B",
                        help="Custom comparisons, e.g. p30:y_star_30:y_base")
    args = parser.parse_args()

    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples")

    if args.subsample and len(data) > args.subsample:
        random.seed(42)
        data = random.sample(data, args.subsample)
        print(f"Subsampled to {len(data)}")

    if args.comparisons:
        comparisons = []
        for spec in args.comparisons:
            parts = spec.split(":")
            if len(parts) == 3:
                comparisons.append((f"{parts[0]}", parts[1], parts[2]))
            else:
                print(f"Bad comparison spec: {spec}, expected NAME:FIELD_A:FIELD_B", file=sys.stderr)
                sys.exit(1)
    else:
        comparisons = [
            ("y_star_prefix30 vs y_base", "y_star_prefix30", "y_base"),
            ("y_star_noprefix vs y_base", "y_star_noprefix", "y_base"),
            ("y_star_full vs y_base", "y_star_full", "y_base"),
        ]

    os.makedirs(args.output_dir, exist_ok=True)

    for comp_name, field_a, field_b in comparisons:
        print(f"\n{'='*60}")
        print(f"Comparison: {comp_name}")
        print(f"{'='*60}")

        wins, losses, ties = 0, 0, 0
        results = []

        def _judge(idx):
            row = data[idx]
            instruction = get_last_user_turn(row.get("x", []))
            resp_a = get_content(row.get(field_a, ""))
            resp_b = get_content(row.get(field_b, ""))
            winner, r1, r2 = run_comparison(args.server_url, args.model, instruction, resp_a, resp_b)
            return idx, winner, r1, r2

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_judge, i): i for i in range(len(data))}
            for future in tqdm(as_completed(futures), total=len(data), desc=comp_name):
                idx, winner, r1, r2 = future.result()
                if winner == "first":
                    wins += 1
                elif winner == "second":
                    losses += 1
                else:
                    ties += 1
                results.append({"index": idx, "winner": winner, "call1": r1, "call2": r2})

        n = len(data)
        wr = wins / n * 100 if n > 0 else 0
        wr_no_ties = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        summary = [
            f"Comparison: {comp_name}",
            f"Judge: {args.model}",
            f"Examples: {n}",
            "",
            f"{field_a} wins: {wins:>5} ({wins/n*100:.1f}%)",
            f"{field_b} wins: {losses:>5} ({losses/n*100:.1f}%)",
            f"Ties:           {ties:>5} ({ties/n*100:.1f}%)",
            "",
            f"Win rate (inc. ties):  {wr:.1f}%",
            f"Win rate (exc. ties):  {wr_no_ties:.1f}%",
        ]

        for line in summary:
            print(line)

        safe_name = comp_name.replace(" ", "_")
        with open(os.path.join(args.output_dir, f"results_{safe_name}.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(args.output_dir, f"summary_{safe_name}.txt"), "w") as f:
            f.write("\n".join(summary))

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
