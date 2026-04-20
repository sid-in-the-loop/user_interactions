"""Convert WebInstruct-CFT to unified TAC schema.

Reads from the official HF parquet (50k) by default. Falls back to the older
truncated JSON file on disk if pointed there via --input.
"""

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_PARQUET = (
    "/work/hdd/bgtw/ssredharan/models/hub/"
    "datasets--TIGER-Lab--WebInstruct-CFT/snapshots/"
    "66826614531a7281e727d260d993d90e13e0bbcf/"
    "WebInstruct-CFT-50K/train-00000-of-00001.parquet"
)

QS_PATTERN = re.compile(r"^\s*Question:\s*(.*?)\n\s*Solution:\s*(.*)$", re.DOTALL)


def parse_input(inp: str):
    m = QS_PATTERN.match(inp)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def load_parquet(path: str):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return t.to_pylist()


def stream_top_level_objects(text: str, max_items: int):
    """Legacy streaming parser for the truncated JSON variant."""
    i = text.find("[")
    if i < 0:
        return
    i += 1
    n = len(text)
    depth = 0
    in_str = False
    esc = False
    start = None
    count = 0
    while i < n:
        c = text[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i + 1]
                    try:
                        yield json.loads(chunk)
                        count += 1
                        if count >= max_items:
                            return
                    except Exception:
                        pass
                    start = None
        i += 1


def load_rows(path: str, pool_cap: int):
    p = Path(path)
    if p.suffix == ".parquet":
        return load_parquet(str(p))
    # legacy truncated-JSON path
    with open(p) as f:
        text = f.read()
    return list(stream_top_level_objects(text, pool_cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_PARQUET)
    ap.add_argument(
        "--output",
        default="/u/ssredharan/user_interactions/experiments/tac_winrates/data/webinstruct_unified.jsonl",
    )
    ap.add_argument("--n", type=int, default=0,
                    help="0 = no cap; take all valid rows.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pool", type=int, default=200000,
        help="Max objects to scan from legacy JSON before sampling. Ignored for parquet.",
    )
    args = ap.parse_args()

    print(f"reading {args.input} ...", flush=True)
    raw = load_rows(args.input, pool_cap=args.pool)
    print(f"loaded {len(raw)} raw rows", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(raw)

    cap = args.n if args.n > 0 else len(raw) + 1
    out = []
    skipped = 0
    for r in raw:
        if len(out) >= cap:
            break
        inp = r.get("input", "") or ""
        crit = r.get("output", "") or ""
        question, solution = parse_input(inp)
        if not question or not solution or not crit.strip():
            skipped += 1
            continue
        out.append({
            "id": f"webinstruct_{len(out)}",
            "dataset": "webinstruct_cft",
            "x": question,
            "y": solution,
            "o": crit,
            "ground_truth": None,
            "eval_type": "judge",
        })

    print(f"kept {len(out)}, skipped {skipped} (bad input format)", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
