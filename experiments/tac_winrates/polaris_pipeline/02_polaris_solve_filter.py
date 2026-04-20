"""Stage 2: critic solvability filter (Qwen3-14B non-thinking by default).

Per spec, the critic model is swapped from Polaris-4B to Qwen3-14B-Instruct
(non-thinking) because Polaris-4B's <think> blocks blow past reasonable token
budgets during smoke testing. Non-thinking gives us direct answers in <=4k
tokens and avoids the "<think> never closes" failure mode.

For each stage-1 survivor, run the critic greedy. Keep only problems where it
gets the correct answer (math_verify against ground_truth).

Output JSONL rows (stage-1 fields preserved + critic solution):
  {id, x, ground_truth, difficulty, pass_rate, attempts,
   critic_solution, critic_correct}
"""

import argparse
import json
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


def build_prompt(tokenizer, problem: str, enable_thinking: bool) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Tokenizer doesn't accept enable_thinking kwarg.
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="experiments/tac_winrates/data/polaris_stage1_passrate.jsonl",
    )
    ap.add_argument(
        "--output",
        default="experiments/tac_winrates/data/polaris_stage2_solvable.jsonl",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Default is NON-thinking; pass flag to re-enable.")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--keep_wrong", action="store_true",
                    help="For debugging: keep all, don't filter.")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"stage-1 survivors (after limit): {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    prompts = [build_prompt(tokenizer, r["x"], args.enable_thinking) for r in rows]
    print(f"generating greedy for {len(prompts)} problems (model={args.model} "
          f"thinking={args.enable_thinking})", flush=True)
    results = llm.generate(prompts, sampling)

    scorer = MathVerifyScorer()

    n_kept = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for row, res in zip(rows, results):
            sol = res.outputs[0].text
            ok = scorer.score(sol, row["ground_truth"]) >= 0.5
            if args.keep_wrong or ok:
                n_kept += 1
                fout.write(json.dumps({
                    **row,
                    "critic_solution": sol,
                    "critic_correct": bool(ok),
                    "critic_model": args.model,
                }, ensure_ascii=False) + "\n")

    print(f"kept {n_kept}/{len(rows)} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
