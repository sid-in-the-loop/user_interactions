#!/usr/bin/env python3
"""
Clean y_star field in ystar JSONL files:
  - Strip <improved response> / **Improved Response:** markers and everything before them
  - Drop examples where y_star is unrecoverable (preamble with no marker)
  - Works for any ystar_thinking_*.jsonl file

Usage:
  python scripts/fkl/clean_ystar.py \
    --input datasets/wildfeedback/qwen3_8b/ystar_thinking_full.jsonl \
    --output datasets/wildfeedback/qwen3_8b/ystar_thinking_full_clean.jsonl
"""

import argparse
import json
import re
from pathlib import Path

MARKER_RE = re.compile(
    r'(?:\*\*Improved [Rr]esponse:\*\*|<improved response>|Improved Response:)\s*\n?',
    re.IGNORECASE,
)

META_PREFIXES = (
    "The follow-up",
    "The user's follow-up",
    "Based on the follow-up",
    "Looking at the follow-up",
    "From the follow-up",
    "The follow up",
)

MIN_TOKENS = 10  # drop if cleaned response is too short


def clean_ystar(text: str) -> str | None:
    """Return cleaned y_star, or None if unrecoverable."""
    m = MARKER_RE.search(text)
    if m:
        cleaned = text[m.end():].strip()
        return cleaned if len(cleaned.split()) >= MIN_TOKENS else None
    # No marker — check if it starts with preamble (unrecoverable)
    if any(text.startswith(p) for p in META_PREFIXES):
        return None
    # Already a direct answer
    return text.strip() or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    kept, dropped = [], 0
    for r in rows:
        cleaned = clean_ystar(r.get("y_star", ""))
        if cleaned is None:
            dropped += 1
            continue
        r = {**r, "y_star": cleaned}
        kept.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Input:   {len(rows)}")
    print(f"Kept:    {len(kept)}  ({len(kept)/len(rows)*100:.1f}%)")
    print(f"Dropped: {dropped}  ({dropped/len(rows)*100:.1f}%)")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
