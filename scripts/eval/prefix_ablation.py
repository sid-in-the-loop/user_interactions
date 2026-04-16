#!/usr/bin/env python3
"""
Prefix ablation: generate y* at different prefix percentages (0%, 10%, ..., 100%)
using a running vLLM server.

Uses the EXACT same prompts and formatting as generate_ystar_prefix.py.

Usage:
  python scripts/eval/prefix_ablation.py \
      --input datasets/wildfeedback/ystar_ybase_qwen3_8b_long.jsonl \
      --output datasets/wildfeedback/prefix_ablation_qwen3_8b.jsonl \
      --server-url http://gh140:8001 \
      --model Qwen/Qwen3-8B \
      --subsample 1000 \
      --max-workers 32
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
from transformers import AutoTokenizer

# ── Prompts (copied exactly from generate_ystar_prefix.py) ──────────────────

SYSTEM_PREFIX = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and the beginning of a reference response. Study the "
    "partial response and the feedback carefully, then generate a complete, "
    "high-quality improved response from scratch. Do not simply continue from "
    "where the partial response ends — write a full response addressing the "
    "user's original request. Output only your response."
)

USER_PREFIX_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<partial reference response (first {pct}% of tokens)>
{prefix}

Given the conversation, feedback, and the partial reference response above, \
generate a complete improved response from scratch."""

SYSTEM_NOPREFIX = (
    "You are a helpful assistant. Given a conversation and a follow-up message "
    "from the user, respond directly and concisely to the original request, "
    "taking the follow-up into account. Do not explain your reasoning. Output "
    "only your revised response."
)

USER_NOPREFIX_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

Given the above follow-up, provide an improved response to the original request."""

SYSTEM_FULL = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and a complete reference response. Study the reference "
    "response and the feedback carefully, then generate a new, improved response "
    "from scratch. Do not copy the reference response — write your own improved "
    "version that addresses the user's original request. Output only your response."
)

USER_FULL_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<complete reference response>
{prefix}

Given the conversation, feedback, and the complete reference response above, \
generate an improved response from scratch."""

SYSTEM_YBASE = (
    "You are a helpful assistant. Respond directly and helpfully to the user's request."
)


# ── Helpers (same as generate_ystar_prefix.py) ──────────────────────────────

def format_conversation(x: list) -> str:
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def get_y_content(y) -> str:
    if isinstance(y, dict):
        return y.get("content", "") or ""
    return str(y) if y else ""


def get_o_content(o) -> str:
    if isinstance(o, dict):
        return o.get("content", "") or ""
    return str(o) if o else ""


def extract_prefix(y_content: str, prefix_frac: float, tokenizer) -> str:
    """Return first prefix_frac of y_content's tokens decoded back to text."""
    token_ids = tokenizer.encode(y_content, add_special_tokens=False)
    if not token_ids:
        return ""
    n_keep = max(1, int(len(token_ids) * prefix_frac))
    return tokenizer.decode(token_ids[:n_keep], skip_special_tokens=True)


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def call_chat(server_url: str, model: str, messages: list, max_tokens: int = 4096) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=300)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return strip_think_blocks(content)
        except Exception as e:
            if attempt == 2:
                print(f"Error: {e}", file=sys.stderr)
                return ""
    return ""


def generate_ystar_at_prefix(server_url, model, row, prefix_pct, tokenizer):
    """Generate y* for a given prefix percentage (0-100)."""
    x_text = format_conversation(row.get("x", []))
    o_text = get_o_content(row.get("o", ""))
    y_text = get_y_content(row.get("y", ""))

    if prefix_pct == 0:
        messages = [
            {"role": "system", "content": SYSTEM_NOPREFIX},
            {"role": "user", "content": USER_NOPREFIX_TEMPLATE.format(x=x_text, o=o_text)},
        ]
    elif prefix_pct == 100:
        messages = [
            {"role": "system", "content": SYSTEM_FULL},
            {"role": "user", "content": USER_FULL_TEMPLATE.format(
                x=x_text, o=o_text, prefix=y_text
            )},
        ]
    else:
        frac = prefix_pct / 100.0
        prefix = extract_prefix(y_text, frac, tokenizer)
        messages = [
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": USER_PREFIX_TEMPLATE.format(
                x=x_text, o=o_text, prefix=prefix, pct=prefix_pct
            )},
        ]

    return call_chat(server_url, model, messages)


def generate_ybase(server_url, model, row):
    messages = [{"role": "system", "content": SYSTEM_YBASE}]
    for turn in row.get("x", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    return call_chat(server_url, model, messages)


def main():
    parser = argparse.ArgumentParser(description="Prefix ablation: y* at 0%, 10%, ..., 100%")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--prefix-steps", nargs="+", type=int,
                        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    args = parser.parse_args()

    if args.model is None:
        resp = requests.get(f"{args.server_url}/v1/models")
        resp.raise_for_status()
        args.model = resp.json()["data"][0]["id"]
        print(f"Auto-detected model: {args.model}")

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load and subsample
    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} examples")

    random.seed(args.seed)
    if len(data) > args.subsample:
        data = random.sample(data, args.subsample)
    print(f"Subsampled to {len(data)}")

    # Use existing y_base if present, otherwise generate
    has_ybase = "y_base" in data[0] and data[0]["y_base"]
    if has_ybase:
        print(f"\nUsing existing y_base from input file")
        ybase_results = [row.get("y_base", "") for row in data]
    else:
        print(f"\nGenerating y_base...")
        ybase_results = [None] * len(data)

        def _gen_ybase(idx):
            return idx, generate_ybase(args.server_url, args.model, data[idx])

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_gen_ybase, i): i for i in range(len(data))}
            for f in tqdm(as_completed(futures), total=len(data), desc="y_base"):
                idx, result = f.result()
                ybase_results[idx] = result

    # Generate y* for each prefix percentage
    prefix_results = {pct: [None] * len(data) for pct in args.prefix_steps}

    for pct in args.prefix_steps:
        print(f"\nGenerating y* at prefix={pct}%...")

        def _gen_prefix(idx, _pct=pct):
            return idx, generate_ystar_at_prefix(
                args.server_url, args.model, data[idx], _pct, tokenizer
            )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(_gen_prefix, i): i for i in range(len(data))}
            for f in tqdm(as_completed(futures), total=len(data), desc=f"prefix={pct}%"):
                idx, result = f.result()
                prefix_results[pct][idx] = result

    # Write output
    with open(args.output, "w") as f:
        for i, row in enumerate(data):
            out_row = {
                "conversation_id": row.get("conversation_id", ""),
                "turn_index": row.get("turn_index"),
                "x": row["x"],
                "y": row.get("y"),
                "o": row.get("o"),
                "y_base": ybase_results[i] or "",
            }
            for pct in args.prefix_steps:
                out_row[f"y_star_{pct}"] = prefix_results[pct][i] or ""
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(data)} examples to {args.output}")
    print(f"Fields: y_base, " + ", ".join(f"y_star_{p}" for p in args.prefix_steps))


if __name__ == "__main__":
    main()
