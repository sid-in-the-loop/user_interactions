#!/usr/bin/env python3
"""
Convert wildfeedback interactions.jsonl to (x, y, o) tuple format for filter_wildchat_tuples
and generate_ystar_fkl. Reads prompt/completion/user_response (from/value) and outputs
x (list of {role, content}), y ({role, content}), o ({role, content}), conversation_id, turn_index.
"""
import argparse
import json
import sys
from pathlib import Path


def from_to_role(f: str) -> str:
    return "user" if f == "human" else "assistant"


def convert_row(row: dict) -> dict:
    prompt = row.get("prompt", [])
    x = [
        {"role": from_to_role(m.get("from", "human")), "content": m.get("value", "")}
        for m in prompt
    ]
    comp = row.get("completion") or {}
    y = {"role": "assistant", "content": comp.get("value", "") if isinstance(comp, dict) else str(comp)}
    ur = row.get("user_response") or {}
    o = {"role": "user", "content": ur.get("value", "") if isinstance(ur, dict) else str(ur)}
    return {
        "id": row.get("id"),
        "conversation_id": row.get("original_conv_id", row.get("id")),
        "turn_index": row.get("turn_id", 0),
        "x": x,
        "y": y,
        "o": o,
    }


def main():
    p = argparse.ArgumentParser(description="Convert wildfeedback interactions to (x,y,o) tuples")
    p.add_argument("--input", type=Path, default=Path("datasets/wildfeedback/interactions.jsonl"), help="wildfeedback interactions.jsonl")
    p.add_argument("--output", type=Path, default=None, help="Output JSONL (default: same dir as input, tuples.jsonl)")
    args = p.parse_args()
    args.input = args.input.resolve()
    if args.output is None:
        args.output = args.input.parent / "tuples.jsonl"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input) as f, open(args.output, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} tuples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
