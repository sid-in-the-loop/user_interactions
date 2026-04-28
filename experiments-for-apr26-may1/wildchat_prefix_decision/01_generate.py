"""Generate y_base + 3 y_star variants for WildChat filtered_tuples (33,920 rows)
via offline vLLM (Qwen3-4B), single llm.generate() call.

Variants per row:
  y_base                  : student | x       (no critique, no y)
  cond_xo                 : teacher | (x, o)
  cond_xyo                : teacher | (x, y, o)
  cond_xyo_ystart         : teacher | (x, y, o) + first 7 tokens of y forced as
                            generation start  → y_star = seed + completion

Prompts use the natural multi-turn structure of x/y/o (parsed from the Python
repr strings in filtered_tuples.jsonl), passed through Qwen3-4B's chat template
with enable_thinking=False. Long prompts are truncated CPU-side by clipping the
longest message's content, before any GPU work.
"""

import argparse
import ast
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_msg_field(raw):
    """Field is a Python repr of a dict or list of dicts. Parse safely."""
    if isinstance(raw, (dict, list)):
        return raw
    return ast.literal_eval(raw)


def make_seed_text(y_content: str, tokenizer, n_tokens: int) -> str:
    if not y_content:
        return ""
    ids = tokenizer.encode(y_content, add_special_tokens=False)
    n = min(n_tokens, len(ids))
    if n == 0:
        return ""
    return tokenizer.decode(ids[:n], skip_special_tokens=True)


def chat_prompt(tok, messages, suffix=""):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + suffix


def fit_prompt(tok, messages, max_input, suffix=""):
    """Build the chat-templated prompt; if too long, clip the longest message's
    content to fit. Returns the final prompt text."""
    prompt = chat_prompt(tok, messages, suffix)
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_input:
        return prompt
    over = len(ids) - max_input
    longest_i = max(range(len(messages)),
                    key=lambda i: len(messages[i].get("content", "") or ""))
    msg = messages[longest_i]
    cids = tok.encode(msg.get("content", "") or "", add_special_tokens=False)
    new_len = max(50, len(cids) - over - 64)
    new_messages = list(messages)
    new_messages[longest_i] = {**msg,
                               "content": tok.decode(cids[:new_len], skip_special_tokens=True)}
    return chat_prompt(tok, new_messages, suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="datasets/wildchat/filtered_tuples.jsonl")
    ap.add_argument("--output", default="experiments-for-apr26-may1/wildchat_prefix_decision/"
                                        "data/wildchat_filtered_qwen3_4b_4variants_generations.jsonl")
    ap.add_argument("--model",  default="Qwen/Qwen3-4B")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_num_seqs", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--seed_tokens", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input) as f:
        # Count total lines first (cheap) so tqdm gets a real total
        try:
            with open(args.input) as ff:
                total_lines = sum(1 for _ in ff)
        except Exception:
            total_lines = None
        for line in tqdm(f, total=total_lines, desc="loading rows",
                          mininterval=1.0, dynamic_ncols=True):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} rows from {args.input}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_input = args.max_model_len - args.max_tokens - 64

    parsed = []
    n_parse_fail = 0
    for r in tqdm(rows, desc="parsing x/y/o", mininterval=1.0, dynamic_ncols=True):
        try:
            x_list = parse_msg_field(r["x"])
            y_dict = parse_msg_field(r["y"])
            o_dict = parse_msg_field(r["o"])
        except Exception:
            n_parse_fail += 1
            continue
        if not isinstance(x_list, list):
            x_list = [x_list]
        eid = f"{r['conversation_id']}_{r['turn_index']}"
        parsed.append({"example_id": eid, "x": x_list, "y": y_dict, "o": o_dict})
    if n_parse_fail:
        print(f"WARNING: {n_parse_fail} rows failed to parse, skipped", flush=True)
    print(f"parsed {len(parsed)} rows", flush=True)

    # Build all prompts CPU-side. Each entry: (row_idx, variant, prompt_text).
    all_prompts = []
    seeds = {}  # idx -> seed text
    for idx, p in enumerate(tqdm(parsed, desc="building prompts (4× per row)",
                                  mininterval=1.0, dynamic_ncols=True)):
        x, y, o = p["x"], p["y"], p["o"]
        all_prompts.append((idx, "y_base",          fit_prompt(tok, x,            max_input)))
        all_prompts.append((idx, "cond_xo",         fit_prompt(tok, x + [o],      max_input)))
        all_prompts.append((idx, "cond_xyo",        fit_prompt(tok, x + [y, o],   max_input)))
        seed = make_seed_text(y.get("content", "") or "", tok, args.seed_tokens)
        seeds[idx] = seed
        all_prompts.append((idx, "cond_xyo_ystart", fit_prompt(tok, x + [y, o],   max_input, suffix=seed)))

    print(f"built {len(all_prompts)} prompts ({len(parsed)} rows × 4)", flush=True)

    # Quick prompt-length stats so we can see if truncation kicked in.
    plens = [len(tok.encode(p, add_special_tokens=False)) for _, _, p in all_prompts[:5000]]
    if plens:
        plens.sort()
        print(f"prompt token length (first 5000 sampled): "
              f"min={plens[0]} median={plens[len(plens)//2]} "
              f"p99={plens[int(len(plens)*0.99)]} max={plens[-1]}",
              flush=True)

    print(f"Loading {args.model} (bf16, gpu_mem={args.gpu_memory_utilization}, "
          f"max_num_seqs={args.max_num_seqs}, max_model_len={args.max_model_len}) ...",
          flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )

    sampling = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"sampling: temperature={args.temperature}, max_tokens={args.max_tokens}, n=1",
          flush=True)

    t0 = time.time()
    outs = llm.generate([p for _, _, p in all_prompts], sampling, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"generation done in {elapsed/60:.1f} min "
          f"({len(all_prompts)/elapsed:.1f} prompts/s)", flush=True)

    # Reassemble per-row
    per_row = {idx: {} for idx in range(len(parsed))}
    for (idx, variant, _), out in tqdm(list(zip(all_prompts, outs)),
                                        desc="reassembling outputs",
                                        mininterval=1.0, dynamic_ncols=True):
        text = out.outputs[0].text
        if variant == "cond_xyo_ystart":
            per_row[idx][variant] = seeds[idx] + text
            per_row[idx]["cond_xyo_ystart_seed_text"] = seeds[idx]
        else:
            per_row[idx][variant] = text

    with open(args.output, "w") as f:
        for idx, p in enumerate(tqdm(parsed, desc="writing jsonl",
                                      mininterval=1.0, dynamic_ncols=True)):
            rec = {
                "example_id": p["example_id"],
                "x": p["x"],
                "y": p["y"],
                "o": p["o"],
                "y_base":                          per_row[idx].get("y_base", ""),
                "y_star_cond_xo":                  per_row[idx].get("cond_xo", ""),
                "y_star_cond_xyo":                 per_row[idx].get("cond_xyo", ""),
                "y_star_cond_xyo_ystart":          per_row[idx].get("cond_xyo_ystart", ""),
                "y_star_cond_xyo_ystart_seed_text": per_row[idx].get("cond_xyo_ystart_seed_text", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(parsed)} rows → {args.output}")


if __name__ == "__main__":
    main()
