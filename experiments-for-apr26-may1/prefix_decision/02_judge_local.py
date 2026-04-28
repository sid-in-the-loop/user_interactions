"""Student-judge (Qwen2.5-Math-7B base) via vLLM logprob comparison.

For every (id, comparison ∈ {full_vs_base, prefix_vs_base}, order ∈ {AB, BA}),
build a judge prompt ending in "The better response is letter ", request 1 token
with logprobs=20, then compare logprob("A") vs logprob("B") at that position.
Verdict ∈ {A, B}. Resumable CSV.
"""

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI
from tqdm import tqdm


JUDGE_PROMPT = """\
You are evaluating which of two solutions to a math problem is better.

Problem:
{x}

Response A:
{a}

Response B:
{b}

Which response is better, A or B? Consider correctness, completeness, and clarity.
The better response is letter """


def _norm_token(t: str) -> str:
    """Normalize a token returned by vLLM logprobs to a letter we care about."""
    if t is None:
        return ""
    s = t.replace("Ġ", "").replace("▁", "").strip()
    return s.upper()


def logsumexp(values):
    if not values:
        return -float("inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def judge_once(client, model, x, a, b, retries=4):
    """Return (verdict, logp_A, logp_B). verdict ∈ {'A','B'}."""
    prompt = JUDGE_PROMPT.format(x=x, a=a, b=b)
    for attempt in range(retries):
        try:
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=20,
                n=1,
            )
            choice = resp.choices[0]
            lp_obj = choice.logprobs
            top = lp_obj.top_logprobs[0] if lp_obj and lp_obj.top_logprobs else {}
            a_lps, b_lps = [], []
            for tok, lp in top.items():
                norm = _norm_token(tok)
                if norm == "A":
                    a_lps.append(lp)
                elif norm == "B":
                    b_lps.append(lp)
            la = logsumexp(a_lps)
            lb = logsumexp(b_lps)
            verdict = "A" if la >= lb else "B"
            return verdict, la, lb
        except Exception as e:
            if attempt == retries - 1:
                print(f"judge error after retries: {e}", flush=True)
                return "tie", float("-inf"), float("-inf")
            time.sleep(2 ** attempt)
    return "tie", float("-inf"), float("-inf")


def build_jobs(gens_row):
    """Yield (key, x, a, b) for each judge call we need to run for this row.
    key = (id, comparison, order). Three comparisons: no_y / full / seeded vs base.
    """
    rid = gens_row["id"]
    x = gens_row["x"]
    yb = gens_row["y_base"]
    for comp, ystar in (("no_y_vs_base",   gens_row["y_star_no_y"]),
                        ("full_vs_base",   gens_row["y_star_full"]),
                        ("seeded_vs_base", gens_row["y_star_seeded"])):
        # AB: A=y_star, B=y_base. BA: A=y_base, B=y_star.
        yield (rid, comp, "AB"), x, ystar, yb
        yield (rid, comp, "BA"), x, yb, ystar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vllm_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--workers", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Resume: keys already in CSV
    seen = set()
    if Path(args.output).exists():
        with open(args.output) as f:
            r = csv.DictReader(f)
            for row in r:
                seen.add((row["id"], row["comparison"], row["order"]))
        print(f"resume: {len(seen)} verdicts already in {args.output}", flush=True)

    # Build job list
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

    http_client = httpx.Client(
        limits=httpx.Limits(
            max_connections=args.workers + 16,
            max_keepalive_connections=args.workers,
        ),
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    )
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY", http_client=http_client)

    write_header = not Path(args.output).exists() or Path(args.output).stat().st_size == 0
    out_f = open(args.output, "a", buffering=1, newline="")
    w = csv.writer(out_f)
    if write_header:
        w.writerow(["id", "comparison", "order", "verdict", "logp_A", "logp_B"])

    n_done = 0
    pbar = tqdm(total=len(jobs), desc="judge calls", smoothing=0.05,
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
                    verdict, la, lb = fut.result()
                except Exception as e:
                    pbar.write(f"worker error: {e}")
                    pbar.update(1)
                    continue
                w.writerow([rid, comp, order, verdict, f"{la:.4f}", f"{lb:.4f}"])
                n_done += 1
                pbar.update(1)
    finally:
        pbar.close()
        out_f.close()
    print(f"done. {n_done} verdicts written to {args.output}")


if __name__ == "__main__":
    main()
