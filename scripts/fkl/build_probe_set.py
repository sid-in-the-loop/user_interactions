#!/usr/bin/env python3
"""
Build a fixed probe set of 500 corrective + 500 unrelated (x, y, o) samples for FKL
signal analysis. Uses GPT-4o-mini to classify follow-up relevance.

- Unrelated candidates: from processed_tuples.jsonl NOT in filtered_tuples.jsonl
  (neg intersection). We keep only those the model labels UNRELATED until we have 500.
- Related (corrective) candidates: from filtered_tuples.jsonl only. We keep only
  those the model labels RELATED until we have 500.

Requires: OPENAI_API_KEY in environment. pip install openai.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Optional: add project root for imports if needed
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_filtered_keys_and_records(path: Path) -> tuple[set[tuple[str, int]], list[dict]]:
    """Load filtered_tuples.jsonl; return set of (conversation_id, turn_index) and full records."""
    keys = set()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["conversation_id"], r["turn_index"]))
            records.append(r)
    return keys, records


def load_neg_intersection(processed_path: Path, filtered_keys: set[tuple[str, int]]) -> list[dict]:
    """Stream processed_tuples.jsonl; return records whose (conv_id, turn_idx) not in filtered."""
    out = []
    with open(processed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r["conversation_id"], r["turn_index"])
            if k not in filtered_keys:
                out.append(r)
    return out


def content(msg: dict) -> str:
    """Extract content from message dict (role/content or from/value)."""
    if isinstance(msg, str):
        return msg
    return (msg.get("content") or msg.get("value") or "").strip()


def truncate_for_api(s: str, max_chars: int = 2000) -> str:
    """Truncate with ellipsis for API call to stay within token limits."""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 50].rstrip() + "\n\n[... truncated for classification ...]"


def classify_related_unrelated(x_msgs: list, y_content: str, o_content: str) -> str | None:
    """
    Call GPT-4o-mini to classify: is the follow-up o related to the response y?
    Returns "RELATED", "UNRELATED", or None on API error.
    """
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai and set OPENAI_API_KEY")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    # Build short context for the model (avoid token overflow)
    x_preview = "\n".join(
        f"{m.get('role', 'user')}: {truncate_for_api(content(m), 800)}" for m in x_msgs[-3:]
    )
    y_short = truncate_for_api(y_content, 1200)
    o_short = truncate_for_api(o_content, 800)

    system = (
        "You are a strict binary classifier. Your only job is to output exactly one word: "
        "RELATED or UNRELATED.\n\n"
        "RELATED: The user's follow-up message (o) is clearly about the assistant's previous "
        "response (y). Examples: o corrects y, asks for clarification about y, asks to expand "
        "or change y, references specific content in y, or continues the same task/topic as y.\n\n"
        "UNRELATED: The follow-up (o) is about something else. The user has switched topic, "
        "asked a new question that does not refer to y, or o could have been asked without "
        "ever seeing y. If o does not mention or depend on the content of y, answer UNRELATED."
    )
    user = (
        "Conversation context (recent messages):\n"
        f"{x_preview}\n\n"
        "Assistant's response (y):\n"
        f"{y_short}\n\n"
        "User's follow-up message (o):\n"
        f"{o_short}\n\n"
        "Is the follow-up (o) related to the assistant's response (y)? Answer with exactly one word: RELATED or UNRELATED."
    )

    client = openai.OpenAI(api_key=api_key)
    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            text = (completion.choices[0].message.content or "").strip().upper()
            # Check UNRELATED first so we don't match it as RELATED
            if "UNRELATED" in text:
                return "UNRELATED"
            if "RELATED" in text:
                return "RELATED"
            # Fallback: first word
            first = (text.split() or [""])[0]
            if first in ("RELATED", "UNRELATED"):
                return first
            return None
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
    return None


def _classify_one(record: dict) -> tuple[dict, str | None]:
    """Worker: (record, label) with label from API, or (record, None) if skip/error."""
    y_c = content(record["y"]) if isinstance(record["y"], dict) else str(record.get("y", ""))
    o_c = content(record["o"]) if isinstance(record["o"], dict) else str(record.get("o", ""))
    if not y_c or not o_c:
        return (record, None)
    label = classify_related_unrelated(record["x"], y_c, o_c)
    return (record, label)


def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def to_probe_item(record: dict, category: str, probe_id: str) -> dict:
    """Convert a WildChat tuple record to probe set format: id, category, x, y (str), o (str)."""
    x = record["x"]
    y = record["y"]
    o = record["o"]
    return {
        "id": probe_id,
        "category": category,
        "x": x,
        "y": content(y) if isinstance(y, dict) else str(y),
        "o": content(o) if isinstance(o, dict) else str(o),
    }


def main():
    parser = argparse.ArgumentParser(description="Build 500 corrective + 500 unrelated probe set via GPT-4o-mini.")
    parser.add_argument(
        "--filtered",
        type=Path,
        default=REPO_ROOT / "datasets" / "wildchat" / "filtered_tuples.jsonl",
        help="Path to filtered_tuples.jsonl (related pool)",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=REPO_ROOT / "datasets" / "wildchat" / "processed_tuples.jsonl",
        help="Path to processed_tuples.jsonl (for neg intersection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "probe_set.json",
        help="Output path for probe_set.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-unrelated", type=int, default=500)
    parser.add_argument("--max-corrective", type=int, default=500)
    parser.add_argument("--workers", type=int, default=50, help="Concurrent threads for API calls")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading filtered set and records...")
    filtered_keys, related_candidates = load_filtered_keys_and_records(args.filtered)
    print(f"  Filtered: {len(filtered_keys)} unique (conv_id, turn), {len(related_candidates)} records")

    print("Loading neg intersection (processed \\ filtered)...")
    unrelated_candidates = load_neg_intersection(args.processed, filtered_keys)
    print(f"  Unrelated candidates: {len(unrelated_candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(unrelated_candidates)
    rng.shuffle(related_candidates)

    # Drop records with empty y/o so workers don't waste API calls
    def _has_y_o(rec: dict) -> bool:
        y = rec.get("y")
        o = rec.get("o")
        y_str = content(y) if isinstance(y, dict) else str(y or "").strip()
        o_str = content(o) if isinstance(o, dict) else str(o or "").strip()
        return bool(y_str and o_str)

    unrelated_candidates = [r for r in unrelated_candidates if _has_y_o(r)]
    related_candidates = [r for r in related_candidates if _has_y_o(r)]

    probe = []
    n_unrelated = 0
    n_corrective = 0

    # First: fill unrelated from neg intersection (model must say UNRELATED), 50 concurrent
    print(f"Classifying unrelated pool (need model label UNRELATED, {args.workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(unrelated_candidates), desc="Unrelated", unit="cand")
        for chunk in _chunks(unrelated_candidates, args.workers):
            if n_unrelated >= args.max_unrelated:
                break
            futures = [executor.submit(_classify_one, rec) for rec in chunk]
            for future in as_completed(futures):
                rec, label = future.result()
                if label == "UNRELATED" and n_unrelated < args.max_unrelated:
                    probe.append(to_probe_item(rec, "unrelated", f"unrelated-{n_unrelated}"))
                    n_unrelated += 1
                pbar.update(1)
                pbar.set_postfix(accepted=n_unrelated, refresh=True)
            if n_unrelated >= args.max_unrelated:
                break
        pbar.close()

    # Second: fill corrective from filtered (model must say RELATED), 50 concurrent
    print(f"Classifying corrective pool (need model label RELATED, {args.workers} workers)...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(related_candidates), desc="Corrective", unit="cand")
        for chunk in _chunks(related_candidates, args.workers):
            if n_corrective >= args.max_corrective:
                break
            futures = [executor.submit(_classify_one, rec) for rec in chunk]
            for future in as_completed(futures):
                rec, label = future.result()
                if label == "RELATED" and n_corrective < args.max_corrective:
                    probe.append(to_probe_item(rec, "corrective", f"corrective-{n_corrective}"))
                    n_corrective += 1
                pbar.update(1)
                pbar.set_postfix(accepted=n_corrective, refresh=True)
            if n_corrective >= args.max_corrective:
                break
        pbar.close()

    print(f"Collected: {n_unrelated} unrelated, {n_corrective} corrective")

    with open(args.output, "w") as f:
        json.dump(probe, f, indent=2)

    # Summary
    by_cat = {}
    len_y = []
    len_o = []
    for p in probe:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
        len_y.append(len(p["y"]))
        len_o.append(len(p["o"]))
    n = len(probe)
    print("Summary:")
    print(f"  Category counts: {by_cat}")
    print(f"  Mean response length (chars): {sum(len_y) / n:.1f}" if n else "  (no samples)")
    print(f"  Mean follow-up length (chars): {sum(len_o) / n:.1f}" if n else "  (no samples)")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
