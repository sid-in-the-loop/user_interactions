#!/usr/bin/env python3
"""
Classify training samples by domain using GPT-4o-mini.

Usage:
  OPENAI_API_KEY=... python scripts/eval/classify_domains.py \
      --input datasets/wildchat/ystar_prefix_wildchat_qwen3_8b.jsonl \
      --output datasets/wildchat/ystar_prefix_wildchat_qwen3_8b_domains.jsonl \
      --output-dir data/domain_analysis
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALID_DOMAINS = {"math", "code", "reasoning", "science", "writing", "analysis", "factual_qa", "chat", "other"}

CLASSIFY_PROMPT = """Classify the following conversation into exactly one primary domain.

Conversation:
{x}

Domains:
- math (algebra, calculus, statistics, numerical problems)
- code (programming, debugging, software engineering)
- reasoning (logic puzzles, formal reasoning, deduction)
- science (physics, chemistry, biology, research)
- writing (creative writing, fiction, poetry, storytelling)
- analysis (essay feedback, document analysis, summarization)
- factual_qa (factual questions, explanations, definitions)
- chat (casual conversation, advice, opinions)
- other

Respond with exactly one word from the list above."""


def get_last_user_content(x):
    if isinstance(x, list):
        for msg in reversed(x):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")[:2000]
        # Fallback: concatenate all
        parts = []
        for msg in x[:3]:
            if isinstance(msg, dict):
                parts.append(f"{msg.get('role','')}: {msg.get('content','')[:500]}")
        return "\n".join(parts)[:2000]
    return str(x)[:2000]


async def classify_one(client, semaphore, x_text, idx):
    async with semaphore:
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(x=x_text)}],
                    max_tokens=3,
                    temperature=0,
                )
                content = resp.choices[0].message.content.strip().lower()
                # Extract valid domain
                for domain in VALID_DOMAINS:
                    if domain in content:
                        return idx, domain
                # Didn't match — retry
            except Exception as e:
                if attempt == 3:
                    print(f"Error on sample {idx}: {e}", file=sys.stderr)
                    return idx, "other"
                await asyncio.sleep(2 ** attempt)
    return idx, "other"


async def classify_all(data, max_concurrent=200):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for i, row in enumerate(data):
        x_text = get_last_user_content(row.get("x", []))
        tasks.append(classify_one(client, semaphore, x_text, i))

    results = [None] * len(data)
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, domain = await coro
        results[idx] = domain
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(data)} classified...", file=sys.stderr)

    return results


def plot_distribution(counts, output_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    total = sum(values)
    pcts = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(labels)), pcts, color="#4C72B0", alpha=0.88, edgecolor="white", linewidth=0.5)

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                f"{pct:.1f}%\n({val})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Percentage of samples")
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL with domain field added")
    parser.add_argument("--output-dir", default="data/domain_analysis")
    parser.add_argument("--max-concurrent", type=int, default=200)
    args = parser.parse_args()

    # Load
    data = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")

    # Classify
    print("Classifying with GPT-4o-mini...")
    domains = asyncio.run(classify_all(data, max_concurrent=args.max_concurrent))

    # Add domain field
    for row, domain in zip(data, domains):
        row["domain"] = domain

    # Save annotated JSONL
    with open(args.output, "w") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved annotated JSONL → {args.output}")

    # Stats
    counts = Counter(domains)
    total = len(data)
    os.makedirs(args.output_dir, exist_ok=True)

    # domain_distribution.json
    dist = {k: {"count": v, "pct": round(v / total * 100, 2)} for k, v in counts.most_common()}
    with open(os.path.join(args.output_dir, "domain_distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)
    print(f"\nDomain distribution:")
    for domain, info in dist.items():
        print(f"  {domain:<15} {info['count']:>5}  ({info['pct']:.1f}%)")

    # domain_samples.json — 5 random per domain
    random.seed(42)
    samples = {}
    for domain in VALID_DOMAINS:
        domain_rows = [row for row in data if row.get("domain") == domain]
        k = min(5, len(domain_rows))
        if k > 0:
            picked = random.sample(domain_rows, k)
            samples[domain] = [
                {"instruction": get_last_user_content(r.get("x", [])), "domain": domain}
                for r in picked
            ]
    with open(os.path.join(args.output_dir, "domain_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)

    # Plot
    plot_distribution(counts, os.path.join(args.output_dir, "domain_distribution.png"))

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
