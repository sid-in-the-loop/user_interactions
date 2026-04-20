"""Evaluate generations: GPT-4o-mini judge (wildchat, webinstruct) or math_verify
(polaris). Writes one raw-results CSV row per comparison × ordering.

Spec: position-bias correction means each comparison is run twice with swapped
orderings; aggregation (separate script) counts as "A wins" only if A wins both.

Comparisons:
  y_star_vs_y_base   at prefix in {0, 30, 70, 100}   — ALL datasets
  y_star_vs_y        at prefix in {0, 30, 70, 100}   — wildchat, webinstruct only

Resumable: if the output CSV exists, already-logged (id, comparison, prefix, order)
tuples are skipped.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.math_scorer import MathVerifyScorer


PREFIX_PCTS = [0, 30, 70, 100]


# ─── IO helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_keys(csv_path):
    """Return set of (id, comparison, prefix_pct, order) already written."""
    keys = set()
    if not Path(csv_path).exists():
        return keys
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            keys.add((row["id"], row["comparison"],
                      int(row["prefix_pct"]), row["order"]))
    return keys


# ─── Judge (OpenAI) ──────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an impartial judge. You will be given a user's request and two "
    "candidate responses labeled [A] and [B]. Decide which response is a "
    "better answer to the request. Consider correctness, completeness, and "
    "whether the response addresses what the user actually asked. "
    "Respond with exactly one token: A, B, or tie."
)

JUDGE_USER = """\
User request:
{x}

[A]
{a}

[B]
{b}

Which response is better? Output exactly one token: A, B, or tie."""


def judge_once(client, model, x, a, b, retries=3):
    """Return 'A', 'B', or 'tie'."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER.format(x=x, a=a, b=b)},
                ],
                temperature=0.0,
                max_tokens=4,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            if text.startswith("A"):
                return "A"
            if text.startswith("B"):
                return "B"
            return "tie"
        except Exception as e:
            if attempt == retries - 1:
                print(f"judge error after retries: {e}", flush=True)
                return "tie"
            time.sleep(2 ** attempt)


# ─── Verifier (math_verify) ──────────────────────────────────────────────────

def verify_once(scorer, gt, a, b):
    """Return 'A' / 'B' / 'tie' — A wins iff A correct AND B wrong."""
    a_ok = scorer.score(a, gt) >= 0.5
    b_ok = scorer.score(b, gt) >= 0.5
    if a_ok and not b_ok:
        return "A"
    if b_ok and not a_ok:
        return "B"
    return "tie"


# ─── Main ────────────────────────────────────────────────────────────────────

def build_comparisons(unified_row, gen_row):
    """Yield (comparison_name, prefix_pct, left_text, right_text, gt)."""
    include_y = unified_row.get("y") not in (None, "")
    skip_y_cmp = unified_row["dataset"] == "polaris"
    gt = unified_row.get("ground_truth")
    for pct in PREFIX_PCTS:
        ys = gen_row.get(f"y_star_{pct}")
        if ys is None:
            continue
        yb = gen_row.get("y_base")
        if yb is not None:
            yield ("y_star_vs_y_base", pct, ys, yb, gt)
        if include_y and not skip_y_cmp:
            yield ("y_star_vs_y", pct, ys, unified_row["y"], gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True, help="Unified jsonl.")
    ap.add_argument("--generations", required=True, help="Output of generate.py.")
    ap.add_argument("--output_csv", required=True, help="Raw results CSV.")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    unified = {r["id"]: r for r in load_jsonl(args.unified)}
    gens = {r["id"]: r for r in load_jsonl(args.generations)}
    ids = [i for i in unified if i in gens]
    if args.limit:
        ids = ids[: args.limit]
    print(f"eval on {len(ids)} rows (unified={len(unified)} gens={len(gens)})", flush=True)

    first_row = unified[ids[0]]
    eval_type = first_row["eval_type"]
    dataset = first_row["dataset"]
    print(f"dataset={dataset} eval_type={eval_type}", flush=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_keys(args.output_csv)
    print(f"already logged: {len(existing)} rows", flush=True)

    new_file = not Path(args.output_csv).exists()
    out_fp = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(out_fp, fieldnames=[
        "id", "dataset", "comparison", "prefix_pct", "order",
        "verdict", "ground_truth",
    ])
    if new_file:
        writer.writeheader()

    # Build task list: (id, comparison, pct, order, left, right, gt)
    # order='ystar_first' => A=ystar, B=comparator; order='ystar_second' => swapped.
    tasks = []
    for rid in ids:
        u = unified[rid]
        g = gens[rid]
        for cmp_name, pct, ystar, comparator, gt in build_comparisons(u, g):
            for order in ("ystar_first", "ystar_second"):
                key = (rid, cmp_name, pct, order)
                if key in existing:
                    continue
                if order == "ystar_first":
                    a, b = ystar, comparator
                else:
                    a, b = comparator, ystar
                tasks.append((rid, cmp_name, pct, order, a, b, gt))

    print(f"pending tasks: {len(tasks)}", flush=True)
    if not tasks:
        out_fp.close()
        return

    if eval_type == "verifier":
        scorer = MathVerifyScorer()
        for i, (rid, cmp_name, pct, order, a, b, gt) in enumerate(tasks):
            verdict = verify_once(scorer, gt, a, b)
            writer.writerow({
                "id": rid, "dataset": dataset, "comparison": cmp_name,
                "prefix_pct": pct, "order": order,
                "verdict": verdict, "ground_truth": gt,
            })
            if (i + 1) % 500 == 0:
                out_fp.flush()
                print(f"[{i+1}/{len(tasks)}]", flush=True)
        out_fp.close()
        print(f"done -> {args.output_csv}", flush=True)
        return

    # eval_type == "judge"
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in env", flush=True)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    def run_task(t):
        rid, cmp_name, pct, order, a, b, gt = t
        u = unified[rid]
        verdict = judge_once(client, args.judge_model, u["x"], a, b)
        return (rid, cmp_name, pct, order, verdict, gt)

    done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_task, t) for t in tasks]
        for fut in as_completed(futures):
            rid, cmp_name, pct, order, verdict, gt = fut.result()
            writer.writerow({
                "id": rid, "dataset": dataset, "comparison": cmp_name,
                "prefix_pct": pct, "order": order,
                "verdict": verdict, "ground_truth": gt,
            })
            done += 1
            if done % 200 == 0:
                out_fp.flush()
                rate = done / max(1e-6, time.time() - start)
                print(f"[{done}/{len(tasks)}] {rate:.1f} calls/s", flush=True)

    out_fp.close()
    print(f"done -> {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
