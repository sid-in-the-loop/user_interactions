"""
Generate y_base and y* for OLMo-3-7B-Instruct on WildFeedback.

    y_base = OLMo(x)           -- student sees only conversation history
    y*     = OLMo(x, y, o)     -- teacher sees history + GPT-4 response + feedback

No thinking mode — OLMo doesn't support it. Runs y_base first (single vLLM
instance), saves to disk, then y*, saves to disk. Prints a summary (sample
count, avg response length in tokens) after each pass. Sample indices are
stored in each output record as 'sample_index'.

Output files (in --output_dir):
    ybase_olmo.jsonl   -- fields: sample_index, conversation_id, turn_index, x, y, o, y_base
    ystar_olmo.jsonl   -- fields: sample_index, conversation_id, turn_index, x, y, o, y_star

Usage:
    python scripts/eval/generate_olmo.py \\
        --input  datasets/wildfeedback/tuples.jsonl \\
        --output_dir datasets/wildfeedback/olmo_3_7b \\
        --model  allenai/OLMo-2-7B-Instruct \\
        --tp_size 4
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ── System prompts ─────────────────────────────────────────────────────────────

# y_base: OLMo responds to x (conversation so far) as a fresh assistant turn.
SYSTEM_YBASE = "You are a helpful assistant. Respond directly and helpfully to the user's request."

# y* teacher prompts — three variants, selected via --teacher-prompt
# original: the broken prompt (no instruction on what to do with y/o)
SYSTEM_YSTAR_ORIGINAL = (
    "You are a helpful assistant. "
    "Given the conversation history, provide a helpful and direct response."
)

# A: minimal — tells OLMo to use o as a hint and re-answer x
SYSTEM_YSTAR_A = (
    "You are a helpful assistant. "
    "A previous response to the user's request was given, followed by the user's follow-up. "
    "Use the follow-up as a hint about what was missing or wrong in the previous response. "
    "Re-answer the original request only. "
    "Do not respond to the follow-up directly."
)

# B: explicit — spells out the three parts and forbids addressing y or o
SYSTEM_YSTAR_B = (
    "You are a helpful assistant. "
    "Below you will see: (1) a conversation, (2) a previous assistant response, "
    "(3) a user follow-up. Your task is ONLY to re-answer the original user request "
    "— better than the previous response, informed by what the follow-up reveals was missing. "
    "Do not address the follow-up directly. "
    "Do not acknowledge the previous response. "
    "Just provide an improved answer."
)

# xo-family: (x, o) only — y never shown. The hindsight block is injected
# inline into the last user turn of x. System is minimal; instruction is inline.
SYSTEM_YSTAR_XO = "You are a helpful assistant."

# Hindsight injection templates for all xo-family variants.
# {o} is substituted with the o message content at runtime.
# A double newline is prepended when injecting into the last user turn.
HINDSIGHT_TEMPLATES = {
    # Paper Table 1 template (current best, 38.8% vs y_base)
    "xo":    "[hindsight context] The following is a future user message.\n"
             "Use this to guide your answer to the user prompt: {o}",
    # Explicit revision instruction
    "xo_B":  "A user will follow up with: {o}\n"
             "Given this follow-up, provide a better answer to the original request. "
             "Focus on what the follow-up reveals was missing or wrong. "
             "Do not address the follow-up directly.",
    # Minimal, no framing
    "xo_C":  "Note: {o}",
    # Socratic — model reasons before answering
    "xo_D":  "The user responded with: {o}\n"
             "Before answering, consider what this reveals about what was missing from a "
             "good response. Then provide an improved answer to the original request only.",
}

YSTAR_SYSTEMS = {
    "original": SYSTEM_YSTAR_ORIGINAL,
    "A":        SYSTEM_YSTAR_A,
    "B":        SYSTEM_YSTAR_B,
    "xo":       SYSTEM_YSTAR_XO,
    "xo_B":     SYSTEM_YSTAR_XO,
    "xo_C":     SYSTEM_YSTAR_XO,
    "xo_D":     SYSTEM_YSTAR_XO,
}


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Prompt building ────────────────────────────────────────────────────────────

def build_messages_ybase(item: dict, system: str) -> list:
    """OLMo sees only x (conversation history up to but not including y/o)."""
    msgs = [{"role": "system", "content": system}]
    for turn in item.get("x", []):
        msgs.append({"role": turn["role"], "content": turn["content"]})
    return msgs


def build_messages_ystar(item: dict, system: str) -> list:
    """OLMo sees full hindsight context: x + y (GPT-4 answer) + o (user feedback)."""
    msgs = [{"role": "system", "content": system}]
    msgs += list(item["x"])
    msgs.append(item["y"])   # GPT-4 response: {role: "assistant", content: ...}
    msgs.append(item["o"])   # user follow-up: {role: "user",      content: ...}
    return msgs


def make_xo_builder(template: str):
    """
    Factory for xo-family prompt builders.
    Returns a build_fn(item, system) → messages list that injects
    `template.format(o=o_content)` into the last user turn of x.
    y is never shown; the instruction lives in the injected text.
    """
    def builder(item: dict, system: str) -> list:
        x = list(item.get("x", []))
        o_content = item["o"].get("content", "") if isinstance(item["o"], dict) else str(item["o"])
        injection = "\n\n" + template.format(o=o_content)
        msgs = [{"role": "system", "content": system}]
        for i, turn in enumerate(x):
            if i == len(x) - 1 and turn.get("role") == "user":
                msgs.append({"role": "user", "content": turn["content"] + injection})
            else:
                msgs.append({"role": turn["role"], "content": turn["content"]})
        return msgs
    return builder


# ── Truncation ─────────────────────────────────────────────────────────────────

def _token_count(tokenizer, messages: list) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        return len(ids) if ids is not None else 0
    except Exception:
        parts = [m.get("content", "") for m in messages if isinstance(m, dict)]
        return len(tokenizer.encode(" ".join(parts), add_special_tokens=True))


def _truncate(item: dict, build_fn, system: str, max_input_tokens: int, tokenizer) -> list:
    """Drop earliest turns from x until the prompt fits max_input_tokens."""
    x = item.get("x", [])
    for start in range(len(x)):
        trimmed = {**item, "x": x[start:]}
        msgs = build_fn(trimmed, system)
        if _token_count(tokenizer, msgs) <= max_input_tokens:
            return msgs
    # Fallback: keep only the last turn
    last = {**item, "x": x[-1:] if x else []}
    return build_fn(last, system)


def preprocess_all(
    data: list,
    build_fn,
    system: str,
    tokenizer,
    max_input_tokens: int,
    max_workers: int = 32,
) -> list:
    """Parallel truncation before vLLM init to avoid CPU bottleneck during gen."""
    messages_list = [None] * len(data)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_truncate, item, build_fn, system, max_input_tokens, tokenizer): i
            for i, item in enumerate(data)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  Truncating"):
            messages_list[futures[fut]] = fut.result()
    return messages_list


# ── Generation ─────────────────────────────────────────────────────────────────

def run_generation(
    data: list,
    all_messages: list,
    output_field: str,   # "y_base" or "y_star"
    llm: LLM,
    sampling_params: SamplingParams,
    min_tokens: int,
    tokenizer,
) -> list:
    """
    Single batched vLLM chat call. Returns output records with sample_index.
    No enable_thinking — OLMo doesn't support it.
    """
    t0 = time.perf_counter()
    outputs = llm.chat(all_messages, sampling_params=sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - t0
    n = len(data)
    print(f"  Generated {n} samples in {elapsed:.1f}s  ({n / elapsed:.1f} samples/sec)", flush=True)

    results = []
    lengths = []
    for idx, (item, output) in enumerate(zip(data, outputs)):
        text = output.outputs[0].text.strip()
        if not text:
            continue
        token_len = len(tokenizer.encode(text, add_special_tokens=False))
        if token_len < min_tokens:
            continue
        record = {
            "sample_index":    item.get("sample_index", idx),
            "conversation_id": item["conversation_id"],
            "turn_index":      item["turn_index"],
            "x":               item["x"],
            "y":               item["y"],
            "o":               item.get("o"),
            output_field:      text,
        }
        results.append(record)
        lengths.append(token_len)

    return results, lengths


def print_summary(tag: str, results: list, lengths: list) -> None:
    n = len(results)
    avg_len = sum(lengths) / n if n else 0
    print(f"\n{'─'*50}")
    print(f"  {tag}")
    print(f"  Samples saved : {n}")
    print(f"  Avg length    : {avg_len:.1f} tokens")
    print(f"{'─'*50}\n", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate y_base and y* for OLMo on WildFeedback (no thinking mode)."
    )
    parser.add_argument(
        "--input",
        default="datasets/wildfeedback/tuples.jsonl",
        help="Input JSONL with (conversation_id, turn_index, x, y, o) fields",
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/wildfeedback/olmo_3_7b",
        help="Directory to write ybase_olmo.jsonl and ystar_olmo.jsonl",
    )
    parser.add_argument(
        "--target",
        choices=["ybase", "ystar", "both"],
        default="both",
        help="Which responses to generate (default: both — ybase first, then ystar)",
    )
    parser.add_argument(
        "--teacher-prompt",
        choices=["original", "A", "B", "xo", "xo_B", "xo_C", "xo_D"],
        default="original",
        help="Teacher system prompt variant for y* (original|A|B|xo|xo_B|xo_C|xo_D). Ignored for ybase.",
    )
    parser.add_argument("--model", default="allenai/OLMo-3-7B-Instruct-SFT")
    parser.add_argument("--max_tokens",  type=int,   default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tp_size",     type=int,   default=1,   help="Tensor parallel size")
    parser.add_argument("--gpu_util",    type=float, default=0.95)
    parser.add_argument("--max_num_seqs",type=int,   default=512, help="vLLM continuous batching width")
    parser.add_argument("--max_model_len",type=int,  default=4096)
    parser.add_argument("--preprocess_workers", type=int, default=32)
    parser.add_argument("--min_tokens",  type=int,   default=10,  help="Drop outputs shorter than this")
    parser.add_argument(
        "--ids-file",
        default=None,
        help="JSON list of {conversation_id, turn_index} — restrict to these rows only",
    )
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    out_base = out_dir / "ybase_olmo.jsonl"
    # ystar filename encodes the prompt variant so outputs don't collide
    variant  = args.teacher_prompt  # "original", "A", "B", or "xo"
    out_star = out_dir / (f"ystar_olmo_{variant}.jsonl" if variant != "original" else "ystar_olmo.jsonl")

    # Load data
    data = load_jsonl(args.input)
    if args.ids_file:
        with open(args.ids_file) as f:
            fixed_ids = {(r["conversation_id"], r.get("turn_index")) for r in json.load(f)}
        data = [r for r in data if (r["conversation_id"], r.get("turn_index")) in fixed_ids]
        print(f"Loaded {len(data)} samples (filtered by {args.ids_file})", flush=True)
    else:
        print(f"Loaded {len(data)} samples from {args.input}", flush=True)

    # Stamp sample_index on each row (position in filtered list)
    for i, item in enumerate(data):
        item["sample_index"] = i

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_input_tokens = max(256, args.max_model_len - args.max_tokens - 64)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # ── Preprocess (truncate) on CPU before loading vLLM ──────────────────────
    msgs_base, msgs_star = None, None
    if args.target in ("ybase", "both"):
        print("Preprocessing y_base messages (truncate to fit context) ...", flush=True)
        msgs_base = preprocess_all(
            data, build_messages_ybase, SYSTEM_YBASE,
            tokenizer, max_input_tokens, args.preprocess_workers,
        )
    if args.target in ("ystar", "both"):
        system_ystar  = YSTAR_SYSTEMS[args.teacher_prompt]
        if args.teacher_prompt in HINDSIGHT_TEMPLATES:
            build_fn_star = make_xo_builder(HINDSIGHT_TEMPLATES[args.teacher_prompt])
        else:
            build_fn_star = build_messages_ystar
        print(f"Preprocessing y* messages (prompt={args.teacher_prompt}) ...", flush=True)
        msgs_star = preprocess_all(
            data, build_fn_star, system_ystar,
            tokenizer, max_input_tokens, args.preprocess_workers,
        )

    # ── Init vLLM (single instance for both passes) ────────────────────────────
    print(f"\nInitializing vLLM: {args.model}  (tp={args.tp_size}, max_num_seqs={args.max_num_seqs})", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )

    # ── y_base pass ────────────────────────────────────────────────────────────
    if args.target in ("ybase", "both"):
        print(f"\n{'='*50}\n  Pass 1 / y_base  →  {out_base}\n{'='*50}", flush=True)
        results_base, lengths_base = run_generation(
            data, msgs_base, "y_base", llm, sampling_params, args.min_tokens, tokenizer
        )
        save_jsonl(results_base, str(out_base))
        print_summary("y_base summary", results_base, lengths_base)

    # ── y* pass ────────────────────────────────────────────────────────────────
    if args.target in ("ystar", "both"):
        print(f"\n{'='*50}\n  Pass 2 / y*      →  {out_star}\n{'='*50}", flush=True)
        results_star, lengths_star = run_generation(
            data, msgs_star, "y_star", llm, sampling_params, args.min_tokens, tokenizer
        )
        save_jsonl(results_star, str(out_star))
        print_summary("y* summary", results_star, lengths_star)

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
