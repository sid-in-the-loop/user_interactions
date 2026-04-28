"""Generate three y_star variants for every WebInstruct row via vLLM
(OpenAI-compatible /v1/completions). Resumable: skips ids already in output.

Variants:
  V1 y_star_no_y    : teacher | (x, o)              — no y in context
  V2 y_star_full    : teacher | (x, y, o)           — full y in context
  V3 y_star_seeded  : teacher | (x, y, o) + seed    — seed = y[:7 tokens]
                      → y_star_seeded = seed + completion

y_base = y (the dataset's pre-generated sample from the same model).

Job-flattened: each completion is its own pool task and ALL completions are
in flight against vLLM simultaneously. httpx pool is sized to match worker
count so the OpenAI client doesn't bottleneck before vLLM does.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI
from tqdm import tqdm

from common.prefix import get_tokenizer, make_seed


VARIANTS = ("no_y", "full", "seeded")


def prompt_full(x, y, o):
    return (
        f"Problem:\n{x}\n\n"
        f"A student attempted the following solution:\n{y}\n\n"
        f"Critique of the attempt:\n{o}\n\n"
        f"Using the critique, write a corrected, complete solution to the problem.\n\n"
        f"Solution:\n"
    )


def prompt_no_y(x, o):
    return (
        f"Problem:\n{x}\n\n"
        f"Feedback on a previous attempt at this problem:\n{o}\n\n"
        f"Write a complete, correct solution to the problem.\n\n"
        f"Solution:\n"
    )


def complete(client, model, prompt, max_tokens, temperature, top_p,
             tokenizer=None, max_ctx=4096, min_gen=64, retries=4):
    """Clamp max_tokens to fit the model context. Returns "" if the prompt alone
    leaves less than min_gen tokens of room (oversized prompt, ~10/50k rows)."""
    if tokenizer is not None:
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        room = max_ctx - prompt_len - 8
        if room < min_gen:
            return ""
        max_tokens = min(max_tokens, room)
    for attempt in range(retries):
        try:
            resp = client.completions.create(
                model=model, prompt=prompt,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=1,
            )
            return resp.choices[0].text
        except Exception as e:
            if attempt == retries - 1:
                print(f"completion error after retries: {e}", flush=True)
                return ""
            time.sleep(2 ** attempt)
    return ""


def build_jobs_for_row(row, tokenizer, seed_tokens):
    """Return ((rid, 'no_y', prompt), (rid, 'full', prompt), (rid, 'seeded', prompt), seed_text)."""
    rid, x, y, o = row["id"], row["x"], row["y"], row["o"]
    seed = make_seed(y, tokenizer, n_tokens=seed_tokens)
    p_full = prompt_full(x, y, o)
    return [
        (rid, "no_y",   prompt_no_y(x, o)),
        (rid, "full",   p_full),
        (rid, "seeded", p_full + seed),
    ], seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vllm_url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--seed_tokens", type=int, default=7)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    try:
                        seen.add(json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"resume: {len(seen)} ids already in {args.output}", flush=True)

    rows = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] in seen:
                continue
            rows.append(d)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"to process: {len(rows)} rows ({3 * len(rows)} completions)", flush=True)

    if not rows:
        return

    tokenizer = get_tokenizer(args.model)

    # Sized httpx pool — must be >= workers or we'll bottleneck on connections,
    # not on vLLM.
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_connections=args.workers + 16,
            max_keepalive_connections=args.workers,
        ),
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0),
    )
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY", http_client=http_client)

    # Flatten: build all (id, variant, prompt) jobs at once + remember seeds.
    seeds = {}
    jobs = []
    rows_by_id = {r["id"]: r for r in rows}
    for r in rows:
        row_jobs, seed = build_jobs_for_row(r, tokenizer, args.seed_tokens)
        seeds[r["id"]] = seed
        jobs.extend(row_jobs)
    print(f"submitting {len(jobs)} completions with {args.workers} workers "
          f"(httpx pool={args.workers + 16}, vLLM batch ceiling = max-num-seqs)...",
          flush=True)

    # Per-row buffer: write a row only after all 3 of its variants finish.
    buf = {r["id"]: {"no_y": None, "full": None, "seeded": None,
                     "remaining": len(VARIANTS)} for r in rows}

    n_rows_done = 0
    out_f = open(args.output, "a", buffering=1)

    pbar = tqdm(total=len(jobs), desc="completions", smoothing=0.05,
                mininterval=2.0, dynamic_ncols=True)
    pbar_rows = tqdm(total=len(rows), desc="rows", position=1,
                     smoothing=0.05, mininterval=2.0, dynamic_ncols=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(complete, client, args.model, prompt,
                          args.max_tokens, args.temperature, args.top_p,
                          tokenizer, args.max_ctx): (rid, variant)
                for (rid, variant, prompt) in jobs
            }
            for fut in as_completed(futs):
                rid, variant = futs[fut]
                try:
                    text = fut.result()
                except Exception as e:
                    pbar.write(f"worker error: {e}")
                    text = ""

                buf[rid][variant] = text
                buf[rid]["remaining"] -= 1
                pbar.update(1)

                if buf[rid]["remaining"] == 0:
                    r = rows_by_id[rid]
                    seed = seeds[rid]
                    out = {
                        "id": rid,
                        "x": r["x"], "y": r["y"], "o": r["o"],
                        "y_star_no_y":      buf[rid]["no_y"],
                        "y_star_full":      buf[rid]["full"],
                        "y_star_seeded":    seed + (buf[rid]["seeded"] or ""),
                        "y_star_seed_text": seed,
                        "y_base":           r["y"],
                    }
                    out_f.write(json.dumps(out) + "\n")
                    n_rows_done += 1
                    pbar_rows.update(1)
                    del buf[rid]
    finally:
        pbar.close()
        pbar_rows.close()
        out_f.close()
    print(f"done. {n_rows_done} rows written to {args.output}")


if __name__ == "__main__":
    main()
