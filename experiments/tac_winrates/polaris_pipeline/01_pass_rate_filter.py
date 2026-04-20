"""Stage 1 of POLARIS construction: pass-rate filter with Qwen3-4B.

For each problem in the pool, sample N=8 solutions at T=0.7 with Qwen3-4B.
Verify each against ground_truth via math_verify. Keep problems whose pass-rate
lies in [0.1, 0.4] (hard-for-4B). Record all 8 attempts so we can later pick a
failed one as y.

Output JSONL rows:
  {id, x, ground_truth, difficulty, pass_rate, attempts: [{text, correct}, ...]}
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from common.math_scorer import MathVerifyScorer


SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Solve the problem step by step, then give the final answer inside \\boxed{}."
)


def build_prompt(tokenizer, problem: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/tac_winrates/data/polaris_pool.jsonl")
    ap.add_argument(
        "--output",
        default="experiments/tac_winrates/data/polaris_stage1_passrate.jsonl",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--pass_rate_min", type=float, default=0.1)
    ap.add_argument("--pass_rate_max", type=float, default=0.4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    with open(args.input) as f:
        pool = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        pool = pool[: args.limit]
    print(f"pool size: {len(pool)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    prompts = [build_prompt(tokenizer, ex["x"]) for ex in pool]
    print(f"generating {len(prompts)} prompts x N={args.n_samples}", flush=True)
    results = llm.generate(prompts, sampling)

    scorer = MathVerifyScorer()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_total = 0
    with open(args.output, "w") as fout:
        for ex, res in zip(pool, results):
            gt = ex["ground_truth"]
            attempts = []
            n_correct = 0
            for out in res.outputs:
                txt = out.text
                correct = scorer.score(txt, gt) >= 0.5
                if correct:
                    n_correct += 1
                attempts.append({"text": txt, "correct": correct})
            pass_rate = n_correct / max(1, len(attempts))
            n_total += 1
            keep = args.pass_rate_min <= pass_rate <= args.pass_rate_max
            if keep:
                n_kept += 1
                fout.write(json.dumps({
                    "id": ex["id"],
                    "x": ex["x"],
                    "ground_truth": gt,
                    "difficulty": ex.get("difficulty"),
                    "pass_rate": pass_rate,
                    "attempts": attempts,
                }, ensure_ascii=False) + "\n")
            if n_total % 50 == 0:
                print(f"[{n_total}/{len(pool)}] kept={n_kept} last_pr={pass_rate:.2f}", flush=True)

    print(f"done. kept {n_kept}/{n_total} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
