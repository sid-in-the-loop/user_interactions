"""Generate y_base and y_star_{0,30,70,100} for one unified dataset.

Greedy (T=0) throughout. Dataset-agnostic — input is a unified jsonl
({id, dataset, x, y, o, ground_truth, eval_type}).

Output jsonl: {id, dataset, y_base, y_star_0, y_star_30, y_star_70, y_star_100}.

Prefix is token-based via the Qwen3-4B tokenizer. y_star_0 means o-only (no y
prefix); y_star_100 means the full y (sanity/oracle).
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


PREFIX_PCTS = [0, 30, 70, 100]

YSTAR_TEMPLATE = """\
{x}

Here is a partial attempt at the solution:
{y_prefix}

Feedback on the attempt:
{o}

Now provide a complete response."""


def token_prefix(tokenizer, y: str, pct: int) -> str:
    """Tokenize y, take first pct% tokens, detokenize."""
    if pct <= 0 or not y:
        return ""
    ids = tokenizer.encode(y, add_special_tokens=False)
    if not ids:
        return ""
    n = max(1, (len(ids) * pct) // 100)
    return tokenizer.decode(ids[:n], skip_special_tokens=True)


def load_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, kind: str, row: dict, pct: int, enable_thinking: bool) -> str:
    """kind in {ybase, ystar}; pct ignored for ybase."""
    if kind == "ybase":
        user_content = row["x"]
    else:
        y_prefix = token_prefix(tokenizer, row.get("y") or "", pct)
        user_content = YSTAR_TEMPLATE.format(
            x=row["x"], y_prefix=y_prefix, o=row.get("o") or ""
        )
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Unified jsonl.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--prompt_truncate_margin", type=int, default=256,
                    help="Safety margin below max_model_len-max_tokens for prompt length.")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Default non-thinking; set to enable Qwen3's <think>.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--only_xo", action="store_true",
                    help="Phase 2 mode: produce only y_base + y_star_0 (teacher sees only x+o). "
                         "Skips y_star_{30,70,100}. Cuts compute by 60%%.")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} rows from {args.input}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    pcts_to_run = [0] if args.only_xo else PREFIX_PCTS

    # Build every prompt up front. Each entry is (row_idx, key, prompt_text).
    all_prompts = []
    for i, r in enumerate(rows):
        all_prompts.append((i, "y_base",
                            build_prompt(tokenizer, "ybase", r, 0, args.enable_thinking)))
        for pct in pcts_to_run:
            all_prompts.append((i, f"y_star_{pct}",
                                build_prompt(tokenizer, "ystar", r, pct, args.enable_thinking)))
    print(f"built {len(all_prompts)} prompts (={len(rows)} rows x {1 + len(pcts_to_run)})",
          flush=True)

    # Recursive doubling: process everything that fits in the current
    # max_model_len tier, then retry the overflow at 2x. Qwen3-4B's native
    # cap is 32768; beyond that would need rope_scaling and rarely pays off.
    MAX_TIER = 32768
    per_row = [dict(id=r["id"], dataset=r["dataset"]) for r in rows]

    current_max = args.max_model_len
    remaining = all_prompts

    while remaining:
        input_cap = current_max - args.max_tokens - args.prompt_truncate_margin
        fits, overflow = [], []
        for ri, key, p in remaining:
            tok_len = len(tokenizer.encode(p))
            if tok_len <= input_cap:
                fits.append((ri, key, p))
            else:
                overflow.append((ri, key, p, tok_len))
        print(f"tier max_model_len={current_max}: fits={len(fits)} overflow={len(overflow)}",
              flush=True)

        if fits:
            llm = LLM(
                model=args.model,
                dtype="bfloat16",
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=current_max,
                max_num_seqs=args.max_num_seqs,
                trust_remote_code=True,
            )
            sampling = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
            outs = llm.generate([p for _, _, p in fits], sampling)
            for (ri, key, _), out in zip(fits, outs):
                per_row[ri][key] = out.outputs[0].text
            # Release the engine so we can reload at a larger context.
            del llm
            import gc; gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        if not overflow:
            break
        next_max = current_max * 2
        if next_max > MAX_TIER:
            print(f"WARNING: {len(overflow)} prompts exceed MAX_TIER={MAX_TIER}; "
                  f"dropping them.", flush=True)
            break
        current_max = next_max
        remaining = [(ri, key, p) for ri, key, p, _ in overflow]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_complete = sum(
        1 for rec in per_row
        if "y_base" in rec and all(f"y_star_{p}" in rec for p in pcts_to_run)
    )
    with open(args.output, "w") as f:
        for rec in per_row:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(per_row)} rows (fully-populated={n_complete}) -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
