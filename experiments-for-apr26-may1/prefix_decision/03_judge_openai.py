"""GPT-4o-mini judge: free-form A/B output, both orderings. Resumable CSV."""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from tqdm import tqdm


JUDGE_SYSTEM = (
    "You are an impartial judge of math solutions. You will be given a problem "
    "and two candidate solutions labeled [A] and [B]. Decide which solution is "
    "better. Consider correctness, completeness, and clarity. "
    "Respond with exactly one token: A or B."
)

JUDGE_USER = """\
Problem:
{x}

[A]
{a}

[B]
{b}

Which solution is better? Output exactly one token: A or B."""


def judge_once(client, model, x, a, b, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER.format(x=x, a=a, b=b)},
                ],
                temperature=0.0,
                max_tokens=2,
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
    return "tie"


def build_jobs(gens_row):
    """Three comparisons per row: no_y / full / seeded vs base, two orderings each."""
    rid = gens_row["id"]
    x = gens_row["x"]
    yb = gens_row["y_base"]
    for comp, ystar in (("no_y_vs_base",   gens_row["y_star_no_y"]),
                        ("full_vs_base",   gens_row["y_star_full"]),
                        ("seeded_vs_base", gens_row["y_star_seeded"])):
        yield (rid, comp, "AB"), x, ystar, yb
        yield (rid, comp, "BA"), x, yb, ystar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if Path(args.output).exists():
        with open(args.output) as f:
            r = csv.DictReader(f)
            for row in r:
                seen.add((row["id"], row["comparison"], row["order"]))
        print(f"resume: {len(seen)} verdicts already in {args.output}", flush=True)

    jobs = []
    n_rows = 0
    with open(args.gens) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n_rows += 1
            for key, x, a, b in build_jobs(d):
                if key in seen:
                    continue
                jobs.append((key, x, a, b))
            if args.limit and n_rows >= args.limit:
                break
    print(f"rows: {n_rows}, jobs to run: {len(jobs)}", flush=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set", flush=True)
        sys.exit(1)
    client = OpenAI()

    write_header = not Path(args.output).exists() or Path(args.output).stat().st_size == 0
    out_f = open(args.output, "a", buffering=1, newline="")
    w = csv.writer(out_f)
    if write_header:
        w.writerow(["id", "comparison", "order", "verdict"])

    n_done = 0
    pbar = tqdm(total=len(jobs), desc="gpt4o-mini", smoothing=0.05,
                mininterval=2.0, dynamic_ncols=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(judge_once, client, args.model, x, a, b): key
                for (key, x, a, b) in jobs
            }
            for fut in as_completed(futs):
                key = futs[fut]
                rid, comp, order = key
                try:
                    verdict = fut.result()
                except Exception as e:
                    pbar.write(f"worker error: {e}")
                    pbar.update(1)
                    continue
                w.writerow([rid, comp, order, verdict])
                n_done += 1
                pbar.update(1)
    finally:
        pbar.close()
        out_f.close()
    print(f"done. {n_done} verdicts written to {args.output}")


if __name__ == "__main__":
    main()
