"""Convert wildchat to unified TAC schema. Samples 1000 held-out examples."""

import argparse
import json
import random
from pathlib import Path


def format_conversation(x):
    """Render multi-turn x as 'Role: content' text (matches repo convention)."""
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "") or ""
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def msg_content(v):
    if isinstance(v, dict):
        return v.get("content", "") or ""
    if isinstance(v, list) and v:
        return v[-1].get("content", "") or ""
    return str(v) if v else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/u/ssredharan/user_interactions/datasets/wildchat/filtered_tuples.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/u/ssredharan/user_interactions/experiments/tac_winrates/data/wildchat_unified.jsonl",
    )
    ap.add_argument("--n", type=int, default=0,
                    help="0 = no cap; take all valid rows.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {args.input}")

    random.Random(args.seed).shuffle(rows)

    out = []
    cap = args.n if args.n > 0 else len(rows) + 1
    for r in rows:
        if len(out) >= cap:
            break
        x_raw = r.get("x")
        y_raw = r.get("y")
        o_raw = r.get("o")
        if not x_raw or not y_raw or not o_raw:
            continue
        x_str = format_conversation(x_raw) if isinstance(x_raw, list) else str(x_raw)
        y_str = msg_content(y_raw)
        o_str = msg_content(o_raw)
        if not x_str.strip() or not y_str.strip() or not o_str.strip():
            continue
        cid = r.get("conversation_id", "")
        tidx = r.get("turn_index", 0)
        out.append({
            "id": f"wildchat_{cid}_{tidx}",
            "dataset": "wildchat",
            "x": x_str,
            "y": y_str,
            "o": o_str,
            "ground_truth": None,
            "eval_type": "judge",
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
