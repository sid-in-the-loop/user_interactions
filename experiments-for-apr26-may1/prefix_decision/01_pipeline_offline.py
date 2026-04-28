"""Offline vLLM pipeline for WebInstruct prefix-decision: generation + student
judge in a single model load. No HTTP server, no client thread pool.

Stage A — generation:
  Build prompts for 3 y_star variants per row (no_y, full, seeded).
  Single llm.generate() call. Write 01_generations.jsonl.

Stage B — student judge (logprob-based):
  Build prompts for 3 comparisons × 2 orderings = 6 judge calls per row.
  Single llm.generate() call with max_tokens=1, logprobs=20.
  Parse logprob("A") vs logprob("B"). Write 02_verdicts_student.csv.

Resumability: stage B reads stage A's output. If the script crashes between
stages, restart skips stage A (output exists) and only redoes stage B.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common.prefix import make_seed


# ─── Generation prompts ──────────────────────────────────────────────────────

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


# ─── Judge prompt (mirrors 02_judge_local.py) ────────────────────────────────

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


def _norm_token(t):
    if t is None:
        return ""
    s = t.replace("Ġ", "").replace("▁", "").strip()
    return s.upper()


def _logsumexp(values):
    if not values:
        return -float("inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def parse_judge_logprobs(out):
    """vLLM offline RequestOutput → (verdict, logp_A, logp_B)."""
    completion = out.outputs[0]
    lps = completion.logprobs
    if not lps:
        return "tie", float("-inf"), float("-inf")
    first = lps[0]  # dict[int → Logprob] for the (only) generated token
    a_lps, b_lps = [], []
    for tok_id, lp in first.items():
        norm = _norm_token(getattr(lp, "decoded_token", None))
        if norm == "A":
            a_lps.append(lp.logprob)
        elif norm == "B":
            b_lps.append(lp.logprob)
    la = _logsumexp(a_lps)
    lb = _logsumexp(b_lps)
    return ("A" if la >= lb else "B"), la, lb


# ─── Stage A: generation ─────────────────────────────────────────────────────

def stage_generation(args, llm, tokenizer):
    """Build prompts for all 3 variants, single llm.generate(), write jsonl."""
    out_path = Path(args.gens)
    if out_path.exists() and args.skip_existing_gens:
        print(f"[stage A] {out_path} exists, skipping generation", flush=True)
        return

    rows = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[stage A] loaded {len(rows)} rows from {args.input}", flush=True)

    # Per-prompt budget: max_ctx - max_tokens - 8 (matches 01_generate.py logic).
    # We pre-tokenize to skip prompts that are too long; for those rows the
    # corresponding y_star is "".
    max_input = args.max_ctx - args.max_tokens - 8

    prompts = []        # (row_idx, variant, prompt_str)
    seeds = {}
    skipped = {"no_y": 0, "full": 0, "seeded": 0}

    for idx, r in enumerate(rows):
        x, y, o = r["x"], r["y"], r["o"]
        seed = make_seed(y, tokenizer, n_tokens=args.seed_tokens)
        seeds[idx] = seed

        for variant, p in (
            ("no_y",   prompt_no_y(x, o)),
            ("full",   prompt_full(x, y, o)),
            ("seeded", prompt_full(x, y, o) + seed),
        ):
            n_tok = len(tokenizer.encode(p, add_special_tokens=False))
            if n_tok > max_input:
                skipped[variant] += 1
                continue
            prompts.append((idx, variant, p))

    print(f"[stage A] built {len(prompts)} prompts ({len(rows)}×3 - skipped: {skipped})",
          flush=True)

    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(f"[stage A] sampling: temp={args.temperature} top_p={args.top_p} "
          f"max_tokens={args.max_tokens}", flush=True)

    t0 = time.time()
    outs = llm.generate([p for _, _, p in prompts], sampling, use_tqdm=True)
    print(f"[stage A] gen done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Reassemble per-row
    per_row = {idx: {"no_y": "", "full": "", "seeded": ""} for idx in range(len(rows))}
    for (idx, variant, _), out in zip(prompts, outs):
        per_row[idx][variant] = out.outputs[0].text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for idx, r in enumerate(rows):
            seed = seeds[idx]
            seeded_text = per_row[idx]["seeded"]
            rec = {
                "id": r["id"],
                "x": r["x"], "y": r["y"], "o": r["o"],
                "y_star_no_y":      per_row[idx]["no_y"],
                "y_star_full":      per_row[idx]["full"],
                "y_star_seeded":    (seed + seeded_text) if seeded_text else "",
                "y_star_seed_text": seed,
                "y_base":           r["y"],
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[stage A] wrote {len(rows)} rows → {out_path}", flush=True)


# ─── Stage B: student judge ──────────────────────────────────────────────────

def _truncate_ab_to_fit(prompt_template, x, a, b, tokenizer, max_input):
    """Build prompt = template.format(x, a, b); if it exceeds max_input,
    proportionally clip a and b until it fits. Returns (prompt_text, ok)
    where ok=False means even clipping a and b to ~50 tokens each isn't enough
    (the x alone is too long)."""
    prompt = prompt_template.format(x=x, a=a, b=b)
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n <= max_input:
        return prompt, True

    # Token-wise clip a and b. Compute fixed overhead = scaffold + x.
    scaffold_only = prompt_template.format(x=x, a="", b="")
    overhead = len(tokenizer.encode(scaffold_only, add_special_tokens=False))
    budget_for_ab = max_input - overhead
    if budget_for_ab < 100:
        return prompt, False  # x alone too long; skip this row

    a_ids = tokenizer.encode(a, add_special_tokens=False)
    b_ids = tokenizer.encode(b, add_special_tokens=False)
    half = max(50, budget_for_ab // 2 - 8)
    a_trunc = tokenizer.decode(a_ids[:min(len(a_ids), half)],
                                skip_special_tokens=True) if a_ids else ""
    b_trunc = tokenizer.decode(b_ids[:min(len(b_ids), half)],
                                skip_special_tokens=True) if b_ids else ""
    prompt = prompt_template.format(x=x, a=a_trunc, b=b_trunc)
    n = len(tokenizer.encode(prompt, add_special_tokens=False))
    if n <= max_input:
        return prompt, True
    return prompt, False


def stage_judge(args, llm):
    """Read 01_generations.jsonl, build judge prompts (CPU-side truncated to
    fit the model context), single llm.generate() with logprobs=20, parse,
    write 02_verdicts_student.csv."""
    out_path = Path(args.student_csv)
    tokenizer = llm.get_tokenizer()
    max_input = args.max_ctx - 1 - 8  # judge max_tokens=1, +8 safety margin

    seen = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            r = csv.DictReader(f)
            for row in r:
                seen.add((row["id"], row["comparison"], row["order"]))
        print(f"[stage B] resume: {len(seen)} verdicts already in {out_path}",
              flush=True)

    rows = []
    with open(args.gens) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[stage B] loaded {len(rows)} gens", flush=True)

    # Build (key, prompt) jobs: 6 per row, with CPU-side truncation.
    jobs = []
    n_skipped_oversize = 0
    n_truncated = 0
    pbar_build = tqdm(total=len(rows), desc="[stage B] building prompts",
                      mininterval=2.0, dynamic_ncols=True)
    for d in rows:
        rid = d["id"]
        x = d["x"]
        yb = d["y_base"]
        for comp, ystar in (
            ("no_y_vs_base",   d["y_star_no_y"]),
            ("full_vs_base",   d["y_star_full"]),
            ("seeded_vs_base", d["y_star_seeded"]),
        ):
            if not ystar or not yb:
                continue
            for order, a, b in (("AB", ystar, yb), ("BA", yb, ystar)):
                key = (rid, comp, order)
                if key in seen:
                    continue
                # Quick check: try as-is; only truncate if needed (saves
                # one tokenization per row in the common case).
                full = JUDGE_PROMPT.format(x=x, a=a, b=b)
                n = len(tokenizer.encode(full, add_special_tokens=False))
                if n <= max_input:
                    jobs.append((key, full))
                    continue
                prompt, ok = _truncate_ab_to_fit(JUDGE_PROMPT, x, a, b,
                                                  tokenizer, max_input)
                if not ok:
                    n_skipped_oversize += 1
                    continue
                n_truncated += 1
                jobs.append((key, prompt))
        pbar_build.update(1)
    pbar_build.close()
    print(f"[stage B] {len(jobs)} judge prompts to run "
          f"(truncated={n_truncated}, skipped_oversize={n_skipped_oversize})",
          flush=True)

    if not jobs:
        return

    sampling = SamplingParams(
        n=1, temperature=0.0, max_tokens=1, logprobs=20,
    )
    t0 = time.time()
    outs = llm.generate([p for _, p in jobs], sampling, use_tqdm=True)
    print(f"[stage B] judge done in {(time.time()-t0)/60:.1f} min", flush=True)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", buffering=1, newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["id", "comparison", "order", "verdict", "logp_A", "logp_B"])

    for (key, _), out in zip(jobs, outs):
        rid, comp, order = key
        verdict, la, lb = parse_judge_logprobs(out)
        w.writerow([rid, comp, order, verdict, f"{la:.4f}", f"{lb:.4f}"])
    f.close()
    print(f"[stage B] wrote verdicts → {out_path}", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
        default="experiments/tac_winrates/data/webinstruct_unified.jsonl")
    ap.add_argument("--gens",
        default="experiments-for-apr26-may1/prefix_decision/data/01_generations.jsonl")
    ap.add_argument("--student_csv",
        default="experiments-for-apr26-may1/prefix_decision/data/02_verdicts_student.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    ap.add_argument("--max_ctx",        type=int,   default=4096)
    ap.add_argument("--max_tokens",     type=int,   default=1500)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top_p",          type=float, default=0.95)
    ap.add_argument("--seed_tokens",    type=int,   default=7)
    ap.add_argument("--max_num_seqs",   type=int,   default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--limit",          type=int,   default=0)
    ap.add_argument("--skip_existing_gens", action="store_true",
        help="If 01_generations.jsonl exists, skip stage A.")
    ap.add_argument("--only_stage", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    print(f"Loading {args.model} (bf16, max_ctx={args.max_ctx}, "
          f"max_num_seqs={args.max_num_seqs}, gpu_mem={args.gpu_memory_utilization})...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_ctx,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    if args.only_stage in ("A", "both"):
        stage_generation(args, llm, tokenizer)
    if args.only_stage in ("B", "both"):
        stage_judge(args, llm)


if __name__ == "__main__":
    main()
