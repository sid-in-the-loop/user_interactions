"""
Generate y_star (hindsight) or y_base (baseline) responses.

  --target ystar : model sees x + y (original GPT-4) + o (feedback) — hindsight context
  --target ybase : model sees x only — clean baseline

  --mode A    : nonthinking  (enable_thinking=False)
  --mode B    : thinking     (enable_thinking=True)
  --mode both : run A then B

Output files written to --output_dir:
  ystar_nonthinking.jsonl / ystar_thinking.jsonl
  ybase_nonthinking.jsonl / ybase_thinking.jsonl

Examples:
  # Generate hindsight responses (nonthinking) for wildchat 4B
  python scripts/fkl/generate_responses.py \\
      --target ystar --mode A \\
      --input datasets/wildchat/filtered_tuples.jsonl \\
      --output_dir datasets/wildchat/qwen3_4b \\
      --model Qwen/Qwen3-4B --tp_size 4

  # Generate baseline responses (both modes) for wildfeedback 8B
  python scripts/fkl/generate_responses.py \\
      --target ybase --mode both \\
      --input datasets/wildfeedback/tuples.jsonl \\
      --output_dir datasets/wildfeedback/qwen3_8b \\
      --model Qwen/Qwen3-8B --tp_size 4
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ── System prompts ────────────────────────────────────────────────────────────

# ystar nonthinking: explicitly suppress <think> blocks since the context
# already contains the GPT-4 response + feedback; we want a clean rewrite.
SYSTEM_YSTAR_NOTHINK = (
    "You are a helpful assistant. "
    "Respond concisely and directly. "
    "Do not output internal reasoning or <think> blocks."
)

# ybase nonthinking: plain instruction-following baseline
SYSTEM_YBASE_NOTHINK = (
    "You are a helpful assistant. Respond directly and helpfully to the user's request."
)

# thinking mode (same for both targets)
SYSTEM_THINKING = (
    "You are a helpful assistant capable of careful reasoning. "
    "Think through the user's request, then provide your best response."
)

# ── Feedback-style prompts (--prompt_style feedback) ──────────────────────────
# Formats x as plain text without y in context (matches generate_ystar_fkl.py).

SYSTEM_FEEDBACK_NOTHINK = (
    "You are a helpful assistant. Given a conversation and a follow-up message from the user, "
    "respond directly and concisely to the original request, taking the follow-up into account. "
    "Do not explain your reasoning. Output only your revised response."
)

SYSTEM_FEEDBACK_THINK = (
    "You are a helpful assistant capable of careful reasoning. Given a conversation and a "
    "follow-up message, first think through what the follow-up reveals about what was wrong "
    "or missing in the original response, then provide an improved response."
)

USER_FEEDBACK_NOTHINK_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

Given the above follow-up, provide an improved response to the original request."""

USER_FEEDBACK_THINK_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

Think carefully about what the follow-up reveals, then provide an improved response."""


# ── I/O helpers ───────────────────────────────────────────────────────────────

def format_conversation(x: list) -> str:
    """Format a list of {role, content} messages as 'Role: content' text."""
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def get_o_content(o) -> str:
    if isinstance(o, dict):
        return o.get("content", "") or ""
    return str(o) if o else ""


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Per-example cache (enables resumable generation for long runs) ─────────────

def load_cache(path: str) -> dict:
    cache = {}
    if not Path(path).exists():
        return cache
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["content"]
    return cache


def append_cache(path: str, key: str, content: str):
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")


def cache_key(item: dict) -> str:
    return f"{item['conversation_id']}_{item['turn_index']}"


# ── Text processing ───────────────────────────────────────────────────────────

def extract_response(text: str) -> tuple[str, str]:
    """Split raw output into (think_trace, response). Works for both modes."""
    think_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    think_trace = "\n\n".join(p.strip() for p in think_parts) if think_parts else ""
    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in response:
        response = response.split("<think>")[0]
    return think_trace, response.strip()


# ── Prompt building ───────────────────────────────────────────────────────────

def build_messages(item: dict, target: str, system: str, user_template: str = None) -> list:
    msgs = [{"role": "system", "content": system}]
    if user_template is not None and target == "ystar":
        # Feedback style: text template without y in context
        user_content = user_template.format(
            x=format_conversation(item["x"]),
            o=get_o_content(item.get("o") or {}),
        )
        msgs.append({"role": "user", "content": user_content})
    elif target == "ystar":
        # Hindsight: full context — x (conversation) + y (GPT-4 answer) + o (feedback)
        msgs += list(item["x"]) + [item["y"]] + [item["o"]]
    else:
        # Baseline: x only (no y, no o)
        for turn in item.get("x", []):
            msgs.append({"role": turn["role"], "content": turn["content"]})
    return msgs


def _token_count_messages(tokenizer, messages: list) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        return len(ids) if ids is not None else 0
    except Exception:
        parts = [m.get("content", "") for m in messages if isinstance(m, dict)]
        return len(tokenizer.encode(" ".join(parts), add_special_tokens=True))


def _truncate_messages(item: dict, target: str, system: str, max_input_tokens: int,
                       tokenizer, user_template: str = None) -> list:
    """Drop earliest turns from x until the prompt fits within max_input_tokens."""
    x = item.get("x", [])
    for start in range(len(x)):
        trimmed = {**item, "x": x[start:]}
        msgs = build_messages(trimmed, target, system, user_template)
        if _token_count_messages(tokenizer, msgs) <= max_input_tokens:
            return msgs
    last = {**item, "x": x[-1:] if x else []}
    return build_messages(last, target, system, user_template)


def preprocess_all(data: list, target: str, system: str, tokenizer, max_input_tokens: int,
                   max_workers: int = 32, user_template: str = None) -> list:
    messages_list = [None] * len(data)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_truncate_messages, item, target, system, max_input_tokens,
                      tokenizer, user_template): i
            for i, item in enumerate(data)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocess"):
            messages_list[futures[fut]] = fut.result()
    return messages_list


# ── Generation ────────────────────────────────────────────────────────────────

def run_generation(
    data: list,
    all_messages: list,
    mode: str,           # "A" or "B"
    target: str,         # "ystar" or "ybase"
    llm: LLM,
    sampling_params: SamplingParams,
    chat_kwargs: dict,
    min_tokens: int,
    tokenizer,
    cache_path: str = None,
) -> list:
    cache = load_cache(cache_path) if cache_path else {}

    to_generate = [(i, item) for i, item in enumerate(data) if cache_key(item) not in cache]
    already_cached = len(data) - len(to_generate)
    print(f"  To generate: {len(to_generate)}  |  Cached: {already_cached}")

    if to_generate:
        gen_messages = [all_messages[i] for i, _ in to_generate]
        t0 = time.perf_counter()
        outputs = llm.chat(gen_messages, sampling_params=sampling_params,
                           chat_template_kwargs=chat_kwargs, use_tqdm=True)
        elapsed = time.perf_counter() - t0
        print(f"  Generated {len(to_generate)} samples in {elapsed:.1f}s ({len(to_generate)/elapsed:.1f}/s)")

        for (_, item), output in zip(to_generate, outputs):
            _, content = extract_response(output.outputs[0].text)
            cache[cache_key(item)] = content
            if cache_path:
                append_cache(cache_path, cache_key(item), content)

    field = "y_star" if target == "ystar" else "y_base"
    results = []
    for item in data:
        ck = cache_key(item)
        if ck not in cache:
            continue
        content = cache[ck]
        if not content or len(tokenizer.encode(content, add_special_tokens=False)) < min_tokens:
            continue
        record = {
            "conversation_id": item["conversation_id"],
            "turn_index": item["turn_index"],
            "x": item["x"],
            "y": item["y"],
            "o": item.get("o"),
            field: content,
            "mode": "thinking" if mode == "B" else "nonthinking",
        }
        results.append(record)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate y_star (hindsight) or y_base (baseline) responses via vLLM."
    )
    parser.add_argument("--target", choices=["ystar", "ybase"], required=True,
                        help="ystar: context=x+y+o  |  ybase: context=x only")
    parser.add_argument("--prompt_style", choices=["hindsight", "feedback"], default="hindsight",
                        help="hindsight: x+y+o as chat messages (default)  |  "
                             "feedback: text template with x+o only, no y (matches generate_ystar_fkl.py)")
    parser.add_argument("--input", required=True,
                        help="Input JSONL with (conversation_id, turn_index, x, y, o) fields")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write output files")
    parser.add_argument("--mode", choices=["A", "B", "both"], default="A",
                        help="A=nonthinking, B=thinking, both=run A then B")
    parser.add_argument("--ids-file", default=None,
                        help="JSON list of {conversation_id, turn_index} — restrict to these rows")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_util", type=float, default=0.95)
    parser.add_argument("--max_num_seqs", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--preprocess_workers", type=int, default=32)
    parser.add_argument("--min_tokens", type=int, default=10,
                        help="Drop outputs shorter than this many tokens")
    parser.add_argument("--cache", action="store_true",
                        help="Enable per-example .cache file for resumable generation")
    args = parser.parse_args()

    prefix = args.target  # "ystar" or "ybase"
    out_dir = Path(args.output_dir)
    out_non   = out_dir / f"{prefix}_nonthinking.jsonl"
    out_think = out_dir / f"{prefix}_thinking.jsonl"
    cache_non   = str(out_non)   + ".cache" if args.cache else None
    cache_think = str(out_think) + ".cache" if args.cache else None

    # Load and optionally filter data
    data = load_jsonl(args.input)
    if args.ids_file:
        with open(args.ids_file) as f:
            fixed_ids = {(r["conversation_id"], r.get("turn_index")) for r in json.load(f)}
        data = [r for r in data if (r["conversation_id"], r.get("turn_index")) in fixed_ids]
        print(f"Loaded {len(data)} samples (filtered by --ids-file)")
    else:
        print(f"Loaded {len(data)} samples from {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_input_tokens = max(256, args.max_model_len - args.max_tokens - 64)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # System prompts + user templates
    if args.prompt_style == "feedback" and args.target == "ystar":
        sys_a = SYSTEM_FEEDBACK_NOTHINK
        sys_b = SYSTEM_FEEDBACK_THINK
        tmpl_a = USER_FEEDBACK_NOTHINK_TEMPLATE
        tmpl_b = USER_FEEDBACK_THINK_TEMPLATE
    else:
        sys_a = SYSTEM_YSTAR_NOTHINK if args.target == "ystar" else SYSTEM_YBASE_NOTHINK
        sys_b = SYSTEM_THINKING
        tmpl_a = tmpl_b = None  # hindsight / ybase: use chat-message format

    # Preprocess (truncate to fit context window)
    msgs_a, msgs_b = None, None
    if args.mode in ("A", "both"):
        print("Preprocessing mode A (nonthinking)...")
        msgs_a = preprocess_all(data, args.target, sys_a, tokenizer, max_input_tokens,
                                args.preprocess_workers, user_template=tmpl_a)
    if args.mode in ("B", "both"):
        print("Preprocessing mode B (thinking)...")
        msgs_b = preprocess_all(data, args.target, sys_b, tokenizer, max_input_tokens,
                                args.preprocess_workers, user_template=tmpl_b)

    print(f"Initializing vLLM: {args.model} (tp={args.tp_size}, max_num_seqs={args.max_num_seqs})...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )

    if args.mode in ("A", "both"):
        print(f"\nMode A (nonthinking) → {out_non}")
        results = run_generation(data, msgs_a, "A", args.target, llm, sampling_params,
                                 {"enable_thinking": False}, args.min_tokens, tokenizer, cache_non)
        save_jsonl(results, str(out_non))
        print(f"  Saved {len(results)} examples → {out_non}")

    if args.mode in ("B", "both"):
        print(f"\nMode B (thinking) → {out_think}")
        results = run_generation(data, msgs_b, "B", args.target, llm, sampling_params,
                                 {"enable_thinking": True}, args.min_tokens, tokenizer, cache_think)
        save_jsonl(results, str(out_think))
        print(f"  Saved {len(results)} examples → {out_think}")

    print("\nDone.")


if __name__ == "__main__":
    main()
