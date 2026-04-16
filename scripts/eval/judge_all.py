#!/usr/bin/env python3
"""
Phase 2: Judge all generated outputs.

Walks eval_results/<method>/step-*/<benchmark>/ directories.
For each outputs.json without a scores.json, runs the appropriate judge.

Pairwise benchmarks: async GPT-4o-mini with position-bias removal.
Programmatic benchmarks: local eval (MATH500, Reasoning Gym).
MT-Bench: single-answer 1-10 grading.

Usage:
  OPENAI_API_KEY=... python scripts/eval/judge_all.py \
      --results_root eval_results/jsd_p30

  # All methods:
  OPENAI_API_KEY=... python scripts/eval/judge_all.py \
      --results_root eval_results
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# ─────────────────────────────────────────────────────────────────────────────
# Shared judge utils
# ─────────────────────────────────────────────────────────────────────────────

PAIRWISE_SYSTEM = """You are a strict, impartial evaluator comparing two AI assistant responses to the same user request. Judge based on: correctness, relevance, completeness, and quality of the response relative to what the user asked. Do NOT favor longer responses automatically.
Output exactly one character: A, B, or C where C means tie or too close to call."""

PAIRWISE_TEMPLATE = """User request:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response better addresses the user's request? Output only A, B, or C."""

# WildFeedback held-out uses a special judge prompt that includes oracle feedback
WF_JUDGE_TEMPLATE = """User request:
{instruction}

The user later followed up with this feedback about the original response:
"{oracle_feedback}"

Response A:
{response_a}

Response B:
{response_b}

Given the user's follow-up feedback, which response better addresses what the user originally wanted? Output only A, B, or C."""

MT_BENCH_JUDGE = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors including the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def parse_abc(text):
    if not text: return None
    m = re.search(r"\b([ABC])\b", text.strip().upper())
    return m.group(1) if m else None


def parse_rating(text):
    if not text: return None
    m = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    return float(m.group(1)) if m else None


async def call_api(client, messages, max_tokens=3, retries=4):
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise judging (AlpacaEval, Arena-Hard, WildFeedback)
# ─────────────────────────────────────────────────────────────────────────────

async def judge_pairwise_one(client, sem, instruction, resp_a, resp_b, template=PAIRWISE_TEMPLATE, **fmt_kwargs):
    """Judge one pair with position-bias removal (AB + BA)."""
    async with sem:
        prompt_ab = template.format(instruction=instruction, response_a=resp_a, response_b=resp_b, **fmt_kwargs)
        prompt_ba = template.format(instruction=instruction, response_a=resp_b, response_b=resp_a, **fmt_kwargs)

        r1_text = await call_api(client, [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": prompt_ab},
        ])
        r2_text = await call_api(client, [
            {"role": "system", "content": PAIRWISE_SYSTEM},
            {"role": "user", "content": prompt_ba},
        ])

        r1 = parse_abc(r1_text)
        r2 = parse_abc(r2_text)

        if r1 is None or r2 is None or r1 == "C" or r2 == "C":
            return "tie"
        if r1 == "A" and r2 == "B":
            return "model"
        if r1 == "B" and r2 == "A":
            return "reference"
        return "tie"


async def judge_pairwise(outputs_path, scores_path, reference_data, instruction_key="instruction",
                         model_key="output", ref_key="output", concurrent=100,
                         template=PAIRWISE_TEMPLATE, **fmt_kwargs_fn):
    """Judge all pairwise comparisons."""
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrent)

    with open(outputs_path) as f:
        outputs = json.load(f)

    # Build reference lookup
    ref_by_key = {}
    for r in reference_data:
        ref_by_key[r[instruction_key]] = r[ref_key]

    wins, losses, ties = 0, 0, 0
    tasks = []

    for item in outputs:
        inst = item[instruction_key]
        model_resp = item[model_key]
        ref_resp = ref_by_key.get(inst, "")
        if not ref_resp:
            continue
        extra = fmt_kwargs_fn(item) if callable(fmt_kwargs_fn) else {}
        tasks.append((inst, model_resp, ref_resp, extra))

    results = []
    sem = asyncio.Semaphore(concurrent)

    async def _judge(idx, inst, model_r, ref_r, extra):
        winner = await judge_pairwise_one(client, sem, inst, model_r, ref_r, template=template, **extra)
        return idx, winner

    coros = [_judge(i, *t) for i, t in enumerate(tasks)]
    done = 0
    for coro in asyncio.as_completed(coros):
        idx, winner = await coro
        if winner == "model": wins += 1
        elif winner == "reference": losses += 1
        else: ties += 1
        results.append({"index": idx, "winner": winner})
        done += 1
        if done % 100 == 0:
            print(f"    Judged {done}/{len(tasks)}...")

    n = len(tasks)
    scores = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n": n,
        "win_rate": wins / n * 100 if n > 0 else 0,
        "win_rate_no_ties": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
    }

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MT-Bench judging (1-10 rating)
# ─────────────────────────────────────────────────────────────────────────────

async def judge_mt_bench(outputs_path, scores_path, concurrent=50):
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrent)

    with open(outputs_path) as f:
        outputs = json.load(f)

    async def _rate(item, turn_idx):
        question = item["turns"][turn_idx] if turn_idx == 0 else \
            f"[Previous question]: {item.get('reference', [''])[0] if item.get('reference') else ''}\n" \
            f"[Previous answer]: {item['turns'][0]}\n[Follow-up question]: asking for more"
        answer = item["turns"][turn_idx] if turn_idx < len(item.get("turns", [])) else ""

        prompt = MT_BENCH_JUDGE.format(question=question, answer=answer)
        async with sem:
            text = await call_api(client, [{"role": "user", "content": prompt}], max_tokens=200)
            return parse_rating(text)

    all_scores = []
    category_scores = {}
    for item in outputs:
        t1_score = await _rate(item, 0)
        t2_score = await _rate(item, 1) if len(item.get("turns", [])) > 1 else None

        scores_list = [s for s in [t1_score, t2_score] if s is not None]
        avg = sum(scores_list) / len(scores_list) if scores_list else 0

        cat = item.get("category", "unknown")
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(avg)
        all_scores.append(avg)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    per_cat = {k: sum(v) / len(v) for k, v in category_scores.items()}

    scores = {
        "overall_score": overall,
        "per_category": per_cat,
        "n": len(all_scores),
    }

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def judge_benchmark(benchmark_name, outputs_path, scores_path):
    """Dispatch to appropriate judge."""
    outputs_path = Path(outputs_path)
    scores_path = Path(scores_path)

    if benchmark_name == "alpaca_eval":
        ref = json.load(open("alpaca_eval_data/gpt4_turbo_reference.json"))
        return asyncio.run(judge_pairwise(
            outputs_path, scores_path, ref,
            instruction_key="instruction", model_key="output", ref_key="output",
        ))

    elif benchmark_name == "arena_hard":
        # TODO: load arena-hard baseline responses
        with open(outputs_path) as f:
            outputs = json.load(f)
        # For now, score based on judge quality rating (similar to MT-Bench)
        print(f"    Arena-Hard judging not fully implemented yet")
        return {}

    elif benchmark_name == "mt_bench":
        return asyncio.run(judge_mt_bench(outputs_path, scores_path))

    elif benchmark_name == "wildfeedback_held":
        with open(outputs_path) as f:
            outputs = json.load(f)
        # Build reference data from outputs (contains y_gpt4)
        ref_data = [{"instruction": _get_last_user(o["x"]), "output": o["y_gpt4"]} for o in outputs]
        model_data = [{"instruction": _get_last_user(o["x"]), "output": o["model_output"],
                       "o_oracle": o.get("o_oracle", "")} for o in outputs]

        # Save modified outputs for pairwise judging
        return asyncio.run(judge_pairwise(
            outputs_path, scores_path, ref_data,
            instruction_key="instruction", model_key="model_output", ref_key="output",
            template=WF_JUDGE_TEMPLATE,
            fmt_kwargs_fn=lambda item: {"oracle_feedback": item.get("o_oracle", "")},
        ))

    elif benchmark_name in ("math500", "aime"):
        from scripts.eval.benchmarks.math500 import MATH500Benchmark
        b = MATH500Benchmark()
        return b.judge(outputs_path, scores_path)

    elif benchmark_name == "reasoning_gym":
        from scripts.eval.benchmarks.reasoning_gym_bench import ReasoningGymBenchmark
        b = ReasoningGymBenchmark()
        return b.judge(outputs_path, scores_path)

    elif benchmark_name == "writingbench":
        print(f"    WritingBench rubric judging not implemented yet")
        return {}

    else:
        print(f"    Unknown benchmark: {benchmark_name}")
        return {}


def _get_last_user(x):
    if isinstance(x, list):
        for msg in reversed(x):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")[:2000]
    return str(x)[:2000]


def main():
    parser = argparse.ArgumentParser(description="Judge all pending benchmark outputs")
    parser.add_argument("--results_root", required=True, help="Root dir to walk for outputs")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Only judge these benchmarks")
    parser.add_argument("--concurrent", type=int, default=100)
    args = parser.parse_args()

    if AsyncOpenAI is None:
        print("pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY (needed for pairwise judging)")
        # Still run programmatic benchmarks
        print("Will only run programmatic benchmarks (math500, reasoning_gym)")

    root = Path(args.results_root)
    judged = 0
    skipped = 0

    # Walk directory tree
    for outputs_json in sorted(root.rglob("outputs.json")):
        benchmark_dir = outputs_json.parent
        benchmark_name = benchmark_dir.name
        scores_path = benchmark_dir / "scores.json"

        if args.benchmarks and benchmark_name not in args.benchmarks:
            continue

        if scores_path.exists():
            skipped += 1
            continue

        step_name = benchmark_dir.parent.name
        method_name = benchmark_dir.parent.parent.name

        print(f"\n  Judging: {method_name}/{step_name}/{benchmark_name}")

        try:
            scores = judge_benchmark(benchmark_name, outputs_json, scores_path)
            if scores:
                metric = scores.get("win_rate_no_ties", scores.get("accuracy",
                         scores.get("overall_score", scores.get("overall_accuracy", "?"))))
                print(f"    Score: {metric}")
                judged += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Judged: {judged}, Skipped (already done): {skipped}")


if __name__ == "__main__":
    main()
