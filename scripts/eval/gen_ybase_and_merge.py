#!/usr/bin/env python3
"""
Generate y_base (baseline responses) via a running vLLM server,
then merge into the y* JSONL file.

Input: JSONL with x, y, o, y_star_prefix30, y_star_noprefix, y_star_full
Output: same JSONL + y_base field added

y_base = model response given only x (no oracle feedback, no prefix).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

SYSTEM_YBASE = (
    "You are a helpful assistant. Respond directly and helpfully to the user's request."
)


def get_last_user_content(x):
    """Extract the last user message from conversation x."""
    for msg in reversed(x):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def generate_ybase(server_url: str, model: str, x: list) -> str:
    """Generate y_base by sending only x to the model."""
    messages = [{"role": "system", "content": SYSTEM_YBASE}]
    for turn in x:
        messages.append({"role": turn["role"], "content": turn["content"]})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(3):
        try:
            resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                print(f"Error generating y_base: {e}", file=sys.stderr)
                return ""
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate y_base and merge into y* JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL with y* fields")
    parser.add_argument("--output", required=True, help="Output JSONL with y_base added")
    parser.add_argument("--server-url", required=True, help="vLLM server URL")
    parser.add_argument("--model", default=None,
                        help="Model name for API (default: auto-detect from server)")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    # Auto-detect model name from server
    if args.model is None:
        resp = requests.get(f"{args.server_url}/v1/models")
        resp.raise_for_status()
        args.model = resp.json()["data"][0]["id"]
        print(f"Auto-detected model: {args.model}")

    # Load input
    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {args.input}")

    # Generate y_base concurrently
    results = [None] * len(data)

    def _gen(idx):
        row = data[idx]
        ybase = generate_ybase(args.server_url, args.model, row["x"])
        return idx, ybase

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_gen, i): i for i in range(len(data))}
        for future in tqdm(as_completed(futures), total=len(data), desc="Generating y_base"):
            idx, ybase = future.result()
            results[idx] = ybase

    # Merge and write output
    empty_count = 0
    with open(args.output, "w") as f:
        for row, ybase in zip(data, results):
            row["y_base"] = ybase or ""
            if not ybase:
                empty_count += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data)} examples to {args.output}")
    if empty_count:
        print(f"Warning: {empty_count} examples had empty y_base", file=sys.stderr)


if __name__ == "__main__":
    main()
