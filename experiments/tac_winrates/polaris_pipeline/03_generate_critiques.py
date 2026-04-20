"""Stage 3: critic generates the critique o (Qwen3-14B non-thinking by default).

For each stage-2 survivor:
  - Pick one failed attempt from stage-1 attempts (random) as y.
  - Prompt the critic with the template from the spec.
  - Greedy decode.

Model is Qwen3-14B-Instruct non-thinking (swapped from Polaris-4B per spec
fallback). The critique we need is a short, focused explanation that a 4B
student can learn from — not another long reasoning trace.

strip_think() is retained so --enable_thinking remains usable if we want to
A/B the reasoning variant later.

Also runs a 5-problem sanity check (writes sanity_samples.txt) before scaling.

Output: unified schema JSONL ready for the generation phase.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


CRITIQUE_TEMPLATE = """\
Problem: {x}
A student attempted this problem with the following solution:
{y}
This solution is incorrect.
Write a brief critique explaining:
1. Where the student's reasoning goes wrong
2. What approach would work instead
Do not give the final answer directly.
Keep your critique under 300 tokens."""


def strip_think(text: str) -> str:
    """Remove <think>...</think> block(s) from a reasoning-model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle unterminated or partial think blocks
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def pick_failed_attempt(attempts, rng):
    failed = [a for a in attempts if not a.get("correct", False)]
    if not failed:
        return None
    return rng.choice(failed)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="experiments/tac_winrates/data/polaris_stage2_solvable.jsonl",
    )
    ap.add_argument(
        "--output",
        default="experiments/tac_winrates/data/polaris_unified.jsonl",
    )
    ap.add_argument(
        "--sanity_output",
        default="experiments/tac_winrates/data/polaris_sanity_samples.txt",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Default is NON-thinking; pass flag to re-enable.")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--target_n", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (smoke-test helper).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"stage-2 survivors (after limit): {len(rows)}", flush=True)

    # Attach a failed attempt (y) to each surviving row.
    prepared = []
    for r in rows:
        y = pick_failed_attempt(r["attempts"], rng)
        if y is None:
            continue  # shouldn't happen since pass_rate <= 0.4
        prepared.append({**r, "y": y})
    print(f"prepared (with failed y): {len(prepared)}", flush=True)

    if args.target_n and len(prepared) > args.target_n:
        prepared = prepared[: args.target_n]
        print(f"truncated to target_n={args.target_n}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    def make_prompt(r):
        user_msg = CRITIQUE_TEMPLATE.format(x=r["x"], y=r["y"])
        msgs = [{"role": "user", "content": user_msg}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    prompts = [make_prompt(r) for r in prepared]

    # Sanity check: generate first 5 and write to sanity_output for manual review.
    sanity_n = min(5, len(prompts))
    print(f"sanity check: generating {sanity_n} critiques first", flush=True)
    sanity_sampling = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    sanity_res = llm.generate(prompts[:sanity_n], sanity_sampling)

    Path(args.sanity_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.sanity_output, "w") as f:
        for r, res in zip(prepared[:sanity_n], sanity_res):
            raw = res.outputs[0].text
            crit = strip_think(raw)
            f.write("=" * 80 + "\n")
            f.write(f"id={r['id']} gt={r['ground_truth']}\n")
            f.write(f"--- problem ---\n{r['x']}\n")
            f.write(f"--- failed y ---\n{r['y']}\n")
            f.write(f"--- raw critique (with think) ---\n{raw}\n")
            f.write(f"--- stripped critique ---\n{crit}\n\n")
    print(f"sanity samples written -> {args.sanity_output}", flush=True)

    # Full run
    print(f"generating critiques for all {len(prompts)} problems", flush=True)
    full_sampling = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    results = llm.generate(prompts, full_sampling)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.output, "w") as fout:
        for r, res in zip(prepared, results):
            raw = res.outputs[0].text
            crit = strip_think(raw)
            # Hard token cap on the critique (spec: under 300)
            # We don't retokenize here; downstream schema just stores text.
            if not crit.strip():
                continue
            fout.write(json.dumps({
                "id": r["id"],
                "dataset": "polaris",
                "x": r["x"],
                "y": r["y"],
                "o": crit,
                "ground_truth": r["ground_truth"],
                "eval_type": "verifier",
                "difficulty": r.get("difficulty"),
                "pass_rate": r.get("pass_rate"),
                "critic_model": args.model,
                "critique_raw": raw,
            }, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
