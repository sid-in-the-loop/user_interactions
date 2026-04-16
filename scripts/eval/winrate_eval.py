#!/usr/bin/env python3
"""
Win rate evaluation using GPT-4o-mini as judge on y* quality.

Compares:
  1. y vs y_star  — does Qwen+hindsight beat GPT-4 original?
  2. y_star vs y_base — does seeing o actually help Qwen?
  3. y vs y_base  — how much stronger is GPT-4 than Qwen baseline?

Uses two judge calls per example (flipped A/B) for position-bias removal.
A response wins only if it wins both calls; conflicting results = tie.

Expects JSONL with fields: x, y, y_star, y_base (each y* is {role, content}).
Reads: datasets/wildchat/my_subsample_ystar.jsonl
Writes: data/winrate_results.jsonl, data/winrate_summary.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

# Optional: openai only when making API calls
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    google_genai = None
    genai_types = None

# Fixed seed for reproducibility
SUBSAMPLE_SEED = 42
SUBSAMPLE_SIZE = 2000
JUDGE_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"

JUDGE_SYSTEM = """You are a strict, impartial evaluator comparing two AI assistant responses to the same user request. Judge based on: correctness, relevance, completeness, and quality of the response relative to what the user asked. Do NOT favor longer responses automatically. Do NOT consider formatting unless it meaningfully affects clarity.
Output exactly one character: A, B, or C where C means tie or too close to call."""

JUDGE_USER_TEMPLATE = """User request:
{x_last_turn}

Response A:
{response_A}

Response B:
{response_B}

Which response better addresses the user's request? Output only A, B, or C."""

MAX_CONCURRENT = 1000


def get_last_user_turn(x: list[dict]) -> str:
    """Extract the last user message content from conversation x."""
    for msg in reversed(x):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def get_content(obj: dict | str | None) -> str:
    """Extract assistant response content from y/y_star/y_base (dict or raw string)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("content") or ""
    return ""


def parse_judge_output(text: str) -> str | None:
    """Return A, B, or C from judge response; None if invalid."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip().upper()
    # Take first A/B/C in the response
    m = re.search(r"\b([ABC])\b", text)
    return m.group(1) if m else None


async def call_judge(
    client,
    x_last_turn: str,
    response_a: str,
    response_b: str,
    semaphore: asyncio.Semaphore,
    judge: str = "gpt4omini",
) -> str | None:
    """Single judge call: A=response_a, B=response_b. Returns A, B, or C."""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        x_last_turn=x_last_turn,
        response_A=response_a,
        response_B=response_b,
    )
    async with semaphore:
        for attempt in range(3):
            try:
                if judge == "gemini":
                    prompt = JUDGE_SYSTEM + "\n\n" + user_prompt
                    resp = await client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            max_output_tokens=10,
                            temperature=0.0,
                            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    content = resp.text or ""
                else:
                    resp = await client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=3,
                        temperature=0,
                    )
                    choice = resp.choices[0] if resp.choices else None
                    content = choice.message.content if choice and choice.message else ""
                return parse_judge_output(content)
            except Exception as e:
                if attempt == 2:
                    print(f"Judge API error: {e}", file=sys.stderr)
                    return None
                await asyncio.sleep(2 ** attempt)


def resolve_winner(first: str | None, second: str | None) -> str:
    """Given two judge calls (A/B and B/A), return 'A', 'B', or 'tie'."""
    if first is None or second is None:
        return "tie"
    # First call: A=resp1, B=resp2 -> winner is first or second
    # Second call: A=resp2, B=resp1 -> winner is second or first
    # So: first says A -> resp1 wins in call1; second says B -> resp1 wins in call2 -> resp1 wins both
    # first says B -> resp2 wins in call1; second says A -> resp2 wins in call2 -> resp2 wins both
    # first says C or second says C -> tie
    if first == "C" or second == "C":
        return "tie"
    if first == "A" and second == "B":
        return "first"   # response 1 wins both
    if first == "B" and second == "A":
        return "second"  # response 2 wins both
    return "tie"


async def run_comparison(
    client,
    semaphore: asyncio.Semaphore,
    x_last_turn: str,
    resp1: str,
    resp2: str,
    judge: str = "gpt4omini",
) -> tuple[str, str | None, str | None]:
    """Run judge twice (A=resp1,B=resp2 and A=resp2,B=resp1). Return (winner, raw1, raw2)."""
    task1 = call_judge(client, x_last_turn, resp1, resp2, semaphore, judge=judge)
    task2 = call_judge(client, x_last_turn, resp2, resp1, semaphore, judge=judge)
    r1, r2 = await asyncio.gather(task1, task2)
    winner = resolve_winner(r1, r2)
    if winner == "first":
        winner = "1"
    elif winner == "second":
        winner = "2"
    else:
        winner = "tie"
    return winner, r1, r2


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Win rate evaluation with GPT-4o-mini judge")
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL path (single file mode: x, y, y_star, optionally y_base)",
    )
    # Pair mode: compare two arbitrary response files joined by conversation_id+turn_index
    parser.add_argument("--file-a", default=None, help="Pair mode: JSONL for response A (uses y_star or y_base field)")
    parser.add_argument("--file-b", default=None, help="Pair mode: JSONL for response B")
    parser.add_argument("--field-a", default="y_star", help="Field name for response A (default: y_star)")
    parser.add_argument("--field-b", default="y_star", help="Field name for response B (default: y_star)")
    parser.add_argument("--label-a", default=None, help="Label for response A in summary (default: field-a value)")
    parser.add_argument("--label-b", default=None, help="Label for response B in summary (default: field-b value)")
    parser.add_argument(
        "--output-dir",
        default="/home/ssmurali/user_interactions/data",
        help="Directory for winrate_results.jsonl and winrate_summary.txt",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="JSON file with [{conversation_id, turn_index}] — use this fixed set instead of random subsampling",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE_SIZE,
        help=f"Subsample size (default {SUBSAMPLE_SIZE}); ignored when --ids-file is set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SUBSAMPLE_SEED,
        help=f"Random seed (default {SUBSAMPLE_SEED}); ignored when --ids-file is set",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--judge",
        default="gpt4omini",
        choices=["gpt4omini", "gemini"],
        help="Judge model to use (default: gpt4omini)",
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=None,
        metavar="LABEL_A:FIELD_A:LABEL_B:FIELD_B",
        help="Multi-comparison mode: run multiple named pairwise comparisons from a single "
             "--input file. Each spec is 'label_a:field_a:label_b:field_b'. "
             "Example: --comparisons prefix30:y_star_prefix30:y:y noprefix:y_star_noprefix:y:y "
             "full:y_star_full:y:y",
    )
    args = parser.parse_args()

    # Validate modes
    pair_mode = args.file_a is not None and args.file_b is not None
    multi_comp_mode = args.comparisons is not None
    if not pair_mode and not multi_comp_mode and args.input is None:
        parser.error("Provide either --input, --comparisons, or both --file-a and --file-b")
    if multi_comp_mode and pair_mode:
        parser.error("--comparisons and --file-a/--file-b are mutually exclusive")

    # Parse --comparisons specs
    comparisons_list = None
    if multi_comp_mode:
        comparisons_list = []
        for spec in args.comparisons:
            parts = spec.split(":")
            if len(parts) != 4:
                parser.error(
                    f"--comparisons entry must be 'label_a:field_a:label_b:field_b', got: {spec!r}"
                )
            comparisons_list.append(tuple(parts))

    if AsyncOpenAI is None:
        print("error: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    if args.judge == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("error: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        if google_genai is None:
            print("error: google-genai package required. pip install google-genai", file=sys.stderr)
            sys.exit(1)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)

    if multi_comp_mode:
        # Multi-comparison mode: single file, multiple named field pairs
        data = load_jsonl(args.input)
        if not data:
            print("error: no records in input", file=sys.stderr)
            sys.exit(1)
        has_y_base = False
        is_pair_mode = False
        comp_names = [f"{la} vs {lb}" for la, fa, lb, fb in comparisons_list]
        print(
            f"Multi-comparison mode: {len(data)} examples, "
            f"{len(comparisons_list)} comparisons: {comp_names}"
        )
    elif pair_mode:
        # Load two files, join on (conversation_id, turn_index)
        data_a = {(r["conversation_id"], r.get("turn_index")): r for r in load_jsonl(args.file_a)}
        data_b = {(r["conversation_id"], r.get("turn_index")): r for r in load_jsonl(args.file_b)}
        common_keys = set(data_a) & set(data_b)
        if not common_keys:
            print("error: no overlapping (conversation_id, turn_index) between the two files", file=sys.stderr)
            sys.exit(1)
        label_a = args.label_a or args.field_a
        label_b = args.label_b or args.field_b
        print(f"Pair mode: {len(common_keys)} overlapping examples. Comparing '{label_a}' vs '{label_b}'")
        data = [
            {
                "conversation_id": k[0],
                "turn_index": k[1],
                "x": data_a[k].get("x", []),
                "_resp_a": get_content(data_a[k].get(args.field_a)),
                "_resp_b": get_content(data_b[k].get(args.field_b)),
                "_label_a": label_a,
                "_label_b": label_b,
            }
            for k in common_keys
        ]
        has_y_base = False
        is_pair_mode = True
    else:
        data = load_jsonl(args.input)
        if not data:
            print("error: no records in input", file=sys.stderr)
            sys.exit(1)
        sample = data[0]
        has_y_base = "y_base" in sample
        if not has_y_base:
            print("warning: no 'y_base' in input; running only comparison 1 (y* vs y)", file=sys.stderr)
            print("  Add y_base (Qwen(x) without hindsight) to run comparisons 2 and 3.", file=sys.stderr)
        is_pair_mode = False

    if args.ids_file:
        with open(args.ids_file) as f:
            fixed_ids = {(r["conversation_id"], r.get("turn_index")) for r in json.load(f)}
        if is_pair_mode:
            data = [r for r in data if (r["conversation_id"], r["turn_index"]) in fixed_ids]
        else:
            data = [r for r in data if (r.get("conversation_id"), r.get("turn_index")) in fixed_ids]
        print(f"Filtered to {len(data)} examples from --ids-file ({args.ids_file})", file=sys.stderr)
    else:
        random.seed(args.seed)
        if len(data) > args.subsample:
            data = random.sample(data, args.subsample)
        else:
            data = list(data)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "winrate_results.jsonl")
    summary_path = os.path.join(args.output_dir, "winrate_summary.txt")

    if multi_comp_mode:
        active_comps = [f"{la} vs {lb}" for la, fa, lb, fb in comparisons_list]
    elif is_pair_mode:
        label_a = data[0]["_label_a"]
        label_b = data[0]["_label_b"]
        comp1_name = f"{label_a} vs {label_b}"
        comp2_name = comp3_name = None
        active_comps = [comp1_name]
    else:
        comp1_name, comp2_name, comp3_name = "y* vs y", "y* vs y_base", "y vs y_base"
        active_comps = [comp1_name] + ([comp2_name, comp3_name] if has_y_base else [])
    counts = {c: {"wins": 0, "losses": 0, "ties": 0} for c in active_comps}

    semaphore = asyncio.Semaphore(args.max_concurrent)
    if args.judge == "gemini":
        client = google_genai.Client(api_key=api_key)
    else:
        client = AsyncOpenAI(api_key=api_key)

    async def process_row(i: int, row: dict) -> dict:
        result_row = {
            "index": i,
            "conversation_id": row.get("conversation_id", ""),
            "turn_index": row.get("turn_index"),
            "comparisons": {},
        }

        if multi_comp_mode:
            # Multiple named comparisons from a single file
            x_last = get_last_user_turn(row.get("x") or [])
            comp_tasks = []
            comp_task_names = []
            for la, fa, lb, fb in comparisons_list:
                resp_a = get_content(row.get(fa))
                resp_b = get_content(row.get(fb))
                comp_name = f"{la} vs {lb}"
                comp_task_names.append(comp_name)
                comp_tasks.append(run_comparison(client, semaphore, x_last, resp_a, resp_b, judge=args.judge))
            comp_results = await asyncio.gather(*comp_tasks)
            for comp_name, (w, ra, rb) in zip(comp_task_names, comp_results):
                result_row["comparisons"][comp_name] = {"winner": w, "raw_judge_AB": ra, "raw_judge_BA": rb}

        elif is_pair_mode:
            x_last = get_last_user_turn(row.get("x") or [])
            w, ra, rb = await run_comparison(client, semaphore, x_last, row["_resp_a"], row["_resp_b"], judge=args.judge)
            result_row["comparisons"][comp1_name] = {"winner": w, "raw_judge_AB": ra, "raw_judge_BA": rb}
        else:
            x_last = get_last_user_turn(row.get("x") or [])
            y_content      = get_content(row.get("y"))
            y_star_content = get_content(row.get("y_star"))
            y_base_content = get_content(row.get("y_base")) if has_y_base else ""

            # Fire all comparisons for this row concurrently
            tasks = [run_comparison(client, semaphore, x_last, y_star_content, y_content, judge=args.judge)]
            if has_y_base:
                tasks.append(run_comparison(client, semaphore, x_last, y_star_content, y_base_content, judge=args.judge))
                tasks.append(run_comparison(client, semaphore, x_last, y_content,      y_base_content, judge=args.judge))
            results = await asyncio.gather(*tasks)

            w1, raw1a, raw1b = results[0]
            result_row["comparisons"][comp1_name] = {"winner": w1, "raw_judge_AB": raw1a, "raw_judge_BA": raw1b}

            if has_y_base:
                w2, raw2a, raw2b = results[1]
                w3, raw3a, raw3b = results[2]
                result_row["comparisons"][comp2_name] = {"winner": w2, "raw_judge_AB": raw2a, "raw_judge_BA": raw2b}
                result_row["comparisons"][comp3_name] = {"winner": w3, "raw_judge_AB": raw3a, "raw_judge_BA": raw3b}
            else:
                result_row["comparisons"][comp2_name] = {"winner": "skip", "raw_judge_AB": None, "raw_judge_BA": None}
                result_row["comparisons"][comp3_name] = {"winner": "skip", "raw_judge_AB": None, "raw_judge_BA": None}

        return result_row

    async def process_all() -> list[dict]:
        tasks = [process_row(i, row) for i, row in enumerate(data)]
        out_rows = await asyncio.gather(*tasks)
        return list(out_rows)

    print(f"Running win rate eval on {len(data)} examples (seed={args.seed})...", file=sys.stderr)
    rows = asyncio.run(process_all())

    # Tally counts from results (safe — no concurrent mutation)
    for r in rows:
        for name in active_comps:
            w = r["comparisons"].get(name, {}).get("winner", "skip")
            if w == "1":    counts[name]["wins"]   += 1
            elif w == "2":  counts[name]["losses"] += 1
            elif w == "tie": counts[name]["ties"]  += 1

    with open(results_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {results_path}", file=sys.stderr)

    n = len(data)
    lines = [
        "Win rate summary (position-bias removed: win only if wins both A/B and B/A)",
        "",
        "Comparison          | Wins   | Losses | Ties   | Win Rate (ties excl.)",
        "--------------------|--------|--------|--------|----------------------",
    ]
    for name in active_comps:
        c = counts[name]
        w, l, t = c["wins"], c["losses"], c["ties"]
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        lines.append(f"{name:<20} | {w:>6} | {l:>6} | {t:>6} | {wr:.1f}%")
    lines.append("")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {summary_path}", file=sys.stderr)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
