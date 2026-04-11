#!/usr/bin/env python3
"""
Dataset filtering pipeline for (x, y, o) training tuples using GPT-4o-mini as a
binary classifier. Assigns each sample a quality tier via deterministic rules
applied to structured yes/no answers.

Step 1: Call GPT-4o-mini with 5 yes/no questions per sample; write raw outputs
        to filter_raw.jsonl (or filter_errors.jsonl on parse failure).
Step 2: Assign tier from Q1–Q5 using deterministic rules.
Step 3: Write filtered_<TIER>.jsonl and filter_summary.txt.

Usage:
  OPENAI_API_KEY=... python scripts/data/filter_wildchat_tuples.py
  OPENAI_API_KEY=... python scripts/data/filter_wildchat_tuples.py --input data/wildchat_tuples.jsonl --output-dir data --limit 100

Reads: data/wildchat_tuples.jsonl (or --input)
Writes: data/filter_raw.jsonl, data/filter_errors.jsonl, data/filtered_*.jsonl, data/filter_summary.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from tqdm import tqdm

# Default paths
DEFAULT_INPUT = "/home/ssmurali/user_interactions/data/wildchat_tuples.jsonl"
DEFAULT_OUTPUT_DIR = "/home/ssmurali/user_interactions/data"
RANDOM_SEED = 42
CONCURRENCY = 1000
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 300

SYSTEM_PROMPT = """You are a precise evaluator of conversational AI training data. You will be given a conversation tuple: x (conversation history including the user's last prompt), y (the assistant's response), and o (the user's follow-up message). Answer exactly 5 yes/no questions about this tuple. Be factual and objective. Do not infer intent beyond what is explicitly present. Output ONLY valid JSON, no explanation, no preamble."""

USER_PROMPT_TEMPLATE = """Conversation history and prompt (x):
{x}

Assistant response (y):
{y}

User follow-up (o):
{o}

Answer the following questions about this tuple with exactly "yes" or "no":

Q1: Does the follow-up message o reference or relate to something specific in the assistant response y? (i.e. is o grounded in the content of y)

Q2: Does the follow-up message o indicate dissatisfaction, an error, a correction, or explicitly request a change or revision to y?

Q3: Would a meaningfully better response to the user's original prompt exist if the assistant had access to the information in o before responding?

Q4: Is the follow-up message o written in the same language as the assistant response y?

Q5: Is the follow-up message o completely unrelated to y and the conversation — a full topic switch with no connection whatsoever to the prior exchange?

Output format (strictly):
{{
  "Q1": "yes" or "no",
  "Q2": "yes" or "no",
  "Q3": "yes" or "no",
  "Q4": "yes" or "no",
  "Q5": "yes" or "no",
  "tldr": "one sentence describing what this follow-up reveals about the assistant response"
}}"""


def _format_msg(obj) -> str:
    """Format a message dict or list of messages for the prompt."""
    if isinstance(obj, list):
        parts = []
        for m in obj:
            if isinstance(m, dict):
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(f"{role}: {json.dumps(content, ensure_ascii=False)}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(obj, dict):
        content = obj.get("content", obj)
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return str(obj)


def _parse_answers(raw: str) -> dict | None:
    """Parse JSON from model response; return dict with Q1–Q5 and tldr, or None."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    # Allow optional markdown code fence
    if raw.startswith("```"):
        lines = raw.split("\n")
        start = 1 if lines[0].strip().startswith("```json") else 1
        end = next((i for i, L in enumerate(lines) if L.strip() == "```"), len(lines))
        raw = "\n".join(lines[start:end])
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(out, dict):
        return None
    for k in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        if k not in out or out[k] not in ("yes", "no"):
            return None
    if "tldr" not in out or not isinstance(out["tldr"], str):
        out["tldr"] = ""
    return out


def assign_tier(Q1: str, Q2: str, Q3: str, Q4: str, Q5: str) -> str:
    """Deterministic tier from Q1–Q5. First match wins."""
    if Q5 == "yes":
        return "SWITCH"
    if Q1 == "yes" and Q2 == "yes" and Q3 == "yes" and Q4 == "yes":
        return "BEST"
    if Q1 == "yes" and Q3 == "yes" and Q4 == "yes" and Q2 == "no":
        return "DECENT"
    if Q2 == "yes" and Q3 == "no":
        return "BAD"
    if Q1 == "no" and Q5 == "no":
        return "NOISE"
    return "UNCATEGORIZED"


async def _call_api_once(
    client: AsyncOpenAI,
    row: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict | None, str | None]:
    """Single API call for one sample. Returns (row, parsed_answers, raw_content)."""
    x = row.get("x", [])
    y = row.get("y", {})
    o = row.get("o", {})
    x_str = _format_msg(x)
    y_str = _format_msg(y)
    o_str = _format_msg(o)
    user_prompt = USER_PROMPT_TEMPLATE.format(x=x_str, y=y_str, o=o_str)

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return (row, None, f"API_ERROR: {e!s}")

    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    parsed = _parse_answers(content)
    return (row, parsed, content)


async def classify_one(
    client: AsyncOpenAI,
    row: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, dict | None, str | None]:
    """
    Call GPT-4o-mini for one sample; retry once on parse failure. Returns (row, parsed_answers, raw_content).
    """
    row, parsed, raw = await _call_api_once(client, row, semaphore)
    if parsed is not None:
        return (row, parsed, raw)
    # Retry once
    row, parsed, raw = await _call_api_once(client, row, semaphore)
    return (row, parsed, raw)


async def run_step1(
    input_path: Path,
    raw_path: Path,
    errors_path: Path,
    client: AsyncOpenAI,
    concurrency: int,
    limit: int | None,
) -> int:
    """
    Stream input JSONL, classify each line with GPT-4o-mini (async, bounded concurrency),
    write to filter_raw.jsonl or filter_errors.jsonl. Returns total number of lines processed.
    """
    random.seed(RANDOM_SEED)
    semaphore = asyncio.Semaphore(concurrency)
    total = 0
    total_read = 0
    batch: list[dict] = []
    batch_size = 500  # process in chunks to limit memory

    with open(input_path) as f_in, open(raw_path, "w") as f_raw, open(errors_path, "w") as f_err:
        pbar = tqdm(desc="Classify", unit=" samples")

        async def flush_batch():
            nonlocal total
            if not batch:
                return
            tasks = [classify_one(client, r, semaphore) for r in batch]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            for (row, parsed, raw_content), orig in zip(results, batch):
                total += 1
                if parsed is not None:
                    out = {**orig, **parsed}
                    f_raw.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f_raw.flush()
                else:
                    err_row = {
                        **orig,
                        "tier": "PARSE_ERROR",
                        "raw_response": raw_content or "",
                    }
                    f_err.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    f_err.flush()
                pbar.update(1)
            batch.clear()

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if limit is not None and total_read >= limit:
                break
            total_read += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                err_row = {
                    "tier": "PARSE_ERROR",
                    "raw_response": "",
                    "parse_error": "Invalid JSON in input line",
                    "line_preview": line[:500],
                }
                f_err.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                f_err.flush()
                total += 1
                pbar.update(1)
                continue
            batch.append(row)
            if len(batch) >= batch_size or (limit is not None and total_read >= limit):
                await flush_batch()
                if limit is not None and total >= limit:
                    break

        await flush_batch()

    return total


def run_step2_and_step3(
    raw_path: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict[str, int]:
    """
    Read filter_raw.jsonl line by line, assign tier, write filtered_<tier>.jsonl
    and filter_summary.txt. Returns tier counts (including PARSE_ERROR from filter_errors).
    """
    tier_counts: dict[str, int] = {
        "BEST": 0,
        "DECENT": 0,
        "BAD": 0,
        "NOISE": 0,
        "SWITCH": 0,
        "UNCATEGORIZED": 0,
        "PARSE_ERROR": 0,
    }
    file_handles: dict[str, object] = {}
    tier_files = ["BEST", "DECENT", "BAD", "NOISE", "SWITCH", "UNCATEGORIZED"]
    for t in tier_files:
        p = output_dir / f"filtered_{t}.jsonl"
        file_handles[t] = open(p, "w")

    try:
        with open(raw_path) as f:
            for line in tqdm(f, desc="Assign tiers", unit=" lines"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                Q1 = row.get("Q1", "")
                Q2 = row.get("Q2", "")
                Q3 = row.get("Q3", "")
                Q4 = row.get("Q4", "")
                Q5 = row.get("Q5", "")
                tier = assign_tier(Q1, Q2, Q3, Q4, Q5)
                row["tier"] = tier
                tier_counts[tier] += 1
                fh = file_handles[tier]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for fh in file_handles.values():
            fh.close()

    # Count PARSE_ERROR from filter_errors.jsonl
    errors_path = output_dir / "filter_errors.jsonl"
    if errors_path.exists():
        with open(errors_path) as ef:
            for _ in ef:
                if _.strip():
                    tier_counts["PARSE_ERROR"] += 1

    total = sum(tier_counts.values())
    with open(summary_path, "w") as sf:
        sf.write(f"Total samples processed: {total}\n")
        for t in tier_files + ["PARSE_ERROR"]:
            n = tier_counts.get(t, 0)
            pct = (100 * n / total) if total else 0
            sf.write(f"{t}: {n:>10}  ({pct:.1f}%)\n")

    return tier_counts


def main():
    parser = argparse.ArgumentParser(description="Filter (x,y,o) tuples with GPT-4o-mini and assign tiers.")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="Input JSONL path")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Max concurrent API calls")
    parser.add_argument("--skip-step1", action="store_true", help="Skip classification; run tier assignment from existing filter_raw.jsonl")
    args = parser.parse_args()

    if AsyncOpenAI is None:
        print("error: openai package required. pip install openai", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "filter_raw.jsonl"
    errors_path = output_dir / "filter_errors.jsonl"
    summary_path = output_dir / "filter_summary.txt"

    if not args.skip_step1:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        if not args.input.exists():
            print(f"error: input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        client = AsyncOpenAI(api_key=api_key)
        n = asyncio.run(
            run_step1(
                args.input,
                raw_path,
                errors_path,
                client,
                args.concurrency,
                args.limit,
            )
        )
        print(f"Step 1 done: processed {n} samples. Raw outputs in {raw_path}, errors in {errors_path}.")
    else:
        if not raw_path.exists():
            print(f"error: --skip-step1 but {raw_path} not found", file=sys.stderr)
            sys.exit(1)

    if not raw_path.exists():
        print("error: no filter_raw.jsonl produced; cannot run Step 2/3", file=sys.stderr)
        sys.exit(1)

    counts = run_step2_and_step3(raw_path, output_dir, summary_path)
    print(f"Step 2/3 done. Summary written to {summary_path}")
    total = sum(counts.values())
    for t, n in counts.items():
        pct = (100 * n / total) if total else 0
        print(f"  {t}: {n} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
