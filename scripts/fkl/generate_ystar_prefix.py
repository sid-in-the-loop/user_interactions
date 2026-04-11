"""
Generate y* under three prefix conditions for offline RL research.

For each (x, y, o) sample in the BEST tier dataset, generate y* under:
  Condition 1 — prefix30:  model sees x, o, and first 30% of y tokens.
  Condition 2 — noprefix:  model sees only x and o (no reference).
  Condition 3 — full:      model sees x, o, and the complete y.

Model: any HF model served via vLLM (default Qwen/Qwen3-4B).

Output JSONL fields: conversation_id, turn_index, x, y, o,
                     y_star_prefix30, y_star_noprefix, y_star_full, prefix_used.
Only rows valid for ALL three conditions are kept.

Usage:
  python scripts/fkl/generate_ystar_prefix.py \
      --input  datasets/wildfeedback/filtered_BEST.jsonl \
      --output datasets/wildfeedback/ystar_prefix_best.jsonl \
      --model  Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — one system + user template per condition
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PREFIX30 = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and the beginning of a reference response. Study the "
    "partial response and the feedback carefully, then generate a complete, "
    "high-quality improved response from scratch. Do not simply continue from "
    "where the partial response ends — write a full response addressing the "
    "user's original request. Output only your response."
)

USER_PREFIX30_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<partial reference response (first 30% of tokens)>
{prefix}

Given the conversation, feedback, and the partial reference response above, \
generate a complete improved response from scratch."""

SYSTEM_NOPREFIX = (
    "You are a helpful assistant. Given a conversation and a follow-up message "
    "from the user, respond directly and concisely to the original request, "
    "taking the follow-up into account. Do not explain your reasoning. Output "
    "only your revised response."
)

USER_NOPREFIX_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

Given the above follow-up, provide an improved response to the original request."""

SYSTEM_FULL = (
    "You are a helpful assistant. You will be shown a conversation, the user's "
    "follow-up feedback, and a complete reference response. Study the reference "
    "response and the feedback carefully, then generate a new, improved response "
    "from scratch. Do not copy the reference response — write your own improved "
    "version that addresses the user's original request. Output only your response."
)

USER_FULL_TEMPLATE = """\
<conversation history>
{x}

<user follow-up>
{o}

<complete reference response>
{prefix}

Given the conversation, feedback, and the complete reference response above, \
generate an improved response from scratch."""


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Field extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_y_content(y) -> str:
    if isinstance(y, dict):
        return y.get("content", "") or ""
    return str(y) if y else ""


def get_o_content(o) -> str:
    if isinstance(o, dict):
        return o.get("content", "") or ""
    return str(o) if o else ""


def format_conversation(x: list[dict]) -> str:
    parts = []
    for msg in x:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Client-mode helpers (--server_url_file)
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_url_file(path: str, timeout: int = 1200) -> str:
    """Poll until the server URL file appears; return the URL."""
    print(f"Waiting for server URL file: {path}", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        p = Path(path)
        if p.exists():
            url = p.read_text().strip()
            if url:
                print(f"Server URL: {url}", flush=True)
                return url
        time.sleep(5)
    raise TimeoutError(f"Server URL file not found after {timeout}s: {path}")


async def _generate_one(
    client: "AsyncOpenAI",
    semaphore: asyncio.Semaphore,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[API error] {e}", file=sys.stderr)
            return ""


async def generate_client_batch(
    all_messages: list[list[dict]],
    client: "AsyncOpenAI",
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
    label: str,
) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _generate_one(client, semaphore, model_name, msgs, max_tokens, temperature)
        for msgs in all_messages
    ]
    results = await atqdm.gather(*tasks, desc=label)
    return list(results)


# ─────────────────────────────────────────────────────────────────────────────
# Prefix extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_prefix(y_content: str, prefix_frac: float, tokenizer) -> str:
    """Return first prefix_frac of y_content's tokens decoded back to text."""
    if not y_content.strip():
        return ""
    token_ids = tokenizer.encode(y_content, add_special_tokens=False)
    n_keep = max(1, int(len(token_ids) * prefix_frac))
    if n_keep >= len(token_ids):
        return y_content
    return tokenizer.decode(token_ids[:n_keep], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────────────────────────

def build_messages_prefix30(item: dict, prefix: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PREFIX30},
        {"role": "user",   "content": USER_PREFIX30_TEMPLATE.format(
            x=format_conversation(item["x"]),
            o=get_o_content(item["o"]),
            prefix=prefix,
        )},
    ]


def build_messages_noprefix(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_NOPREFIX},
        {"role": "user",   "content": USER_NOPREFIX_TEMPLATE.format(
            x=format_conversation(item["x"]),
            o=get_o_content(item["o"]),
        )},
    ]


def build_messages_full(item: dict, full_y: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_FULL},
        {"role": "user",   "content": USER_FULL_TEMPLATE.format(
            x=format_conversation(item["x"]),
            o=get_o_content(item["o"]),
            prefix=full_y,
        )},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt truncation
# ─────────────────────────────────────────────────────────────────────────────

def _token_count(tokenizer, messages: list[dict]) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        return len(ids) if ids is not None else 0
    except Exception:
        parts = [m.get("content", "") for m in messages if isinstance(m, dict)]
        return len(tokenizer.encode(" ".join(parts), add_special_tokens=True))


def truncate_to_fit(messages: list[dict], max_input_tokens: int, tokenizer) -> list[dict]:
    if _token_count(tokenizer, messages) <= max_input_tokens:
        return messages
    user_idx = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if user_idx is None:
        return messages
    content = messages[user_idx]["content"]
    for trim in [0.1, 0.25, 0.5, 0.75]:
        short = content[int(len(content) * trim):]
        trimmed = [
            m if i != user_idx else {"role": "user", "content": short}
            for i, m in enumerate(messages)
        ]
        if _token_count(tokenizer, trimmed) <= max_input_tokens:
            return trimmed
    fallback = content[-2000:] if len(content) > 2000 else content
    return [
        m if i != user_idx else {"role": "user", "content": fallback}
        for i, m in enumerate(messages)
    ]


def preprocess_parallel(
    items: list[dict],
    build_fn,               # callable(idx, item) -> messages
    max_input_tokens: int,
    tokenizer,
    desc: str,
    max_workers: int = 32,
) -> list[list[dict]]:
    results = [None] * len(items)

    def process_one(args):
        idx, item = args
        msgs = build_fn(idx, item)
        return idx, truncate_to_fit(msgs, max_input_tokens, tokenizer)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, (i, item)): i for i, item in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, msgs = fut.result()
            results[idx] = msgs
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(
    all_messages: list[list[dict]],
    llm: LLM,
    sampling_params: SamplingParams,
    chat_template_kwargs: dict,
    label: str,
) -> list[str]:
    t0 = time.perf_counter()
    outputs = llm.chat(
        all_messages,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs,
        use_tqdm=True,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"{label}: {len(all_messages)} samples in {elapsed:.1f}s "
        f"→ {len(all_messages)/elapsed:.1f} samples/sec",
        flush=True,
    )
    return [out.outputs[0].text if out.outputs else "" for out in outputs]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate y* under three prefix conditions (prefix30 / noprefix / full)."
    )
    parser.add_argument("--input",   default="datasets/wildfeedback/filtered_BEST.jsonl")
    parser.add_argument("--output",  default="datasets/wildfeedback/ystar_prefix_best.jsonl")
    parser.add_argument("--model",   default="Qwen/Qwen3-4B")
    parser.add_argument("--prefix_frac", type=float, default=0.30,
                        help="Fraction of y tokens for the partial prefix condition (default 0.30)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode (Qwen3 only). Think traces are stripped from y*.")
    parser.add_argument("--max_tokens",          type=int,   default=1024)
    parser.add_argument("--temperature",         type=float, default=1.0)
    parser.add_argument("--tp_size",             type=int,   default=1)
    parser.add_argument("--gpu_util",            type=float, default=0.92)
    parser.add_argument("--max_num_seqs",        type=int,   default=256)
    parser.add_argument("--max_model_len",       type=int,   default=8192)
    parser.add_argument("--min_tokens",          type=int,   default=10)
    parser.add_argument("--preprocess_workers",  type=int,   default=32)
    # Client mode: connect to a running vLLM OpenAI-compatible server instead of
    # loading the model directly (replaces generate_ystar_client.py).
    parser.add_argument("--server_url_file", default=None,
                        help="Path to file containing the server URL http://host:port. "
                             "Polls until the file appears (timeout: --server_wait_timeout).")
    parser.add_argument("--server_url", default=None,
                        help="Direct server URL override (skips waiting for --server_url_file).")
    parser.add_argument("--max_concurrent", type=int, default=256,
                        help="Max parallel requests in client mode (default 256).")
    parser.add_argument("--server_wait_timeout", type=int, default=1200,
                        help="Seconds to wait for --server_url_file to appear (default 20 min).")
    args = parser.parse_args()

    client_mode = bool(args.server_url_file or args.server_url)

    # ── Load ─────────────────────────────────────────────────────────────────
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} samples from {args.input}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    buffer = 64
    max_input_tokens = max(256, args.max_model_len - args.max_tokens - buffer)

    # ── Extract prefixes (parallel) ───────────────────────────────────────────
    print(f"Extracting {args.prefix_frac*100:.0f}% prefixes and full y ...")
    prefixes30  = [None] * len(data)
    full_ys     = [None] * len(data)

    def extract_one(args_tuple):
        idx, item = args_tuple
        y_content = get_y_content(item["y"])
        p30  = extract_prefix(y_content, args.prefix_frac, tokenizer)
        full = y_content
        return idx, p30, full

    with ThreadPoolExecutor(max_workers=args.preprocess_workers) as ex:
        futures = {ex.submit(extract_one, (i, item)): i for i, item in enumerate(data)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Prefix extraction"):
            idx, p30, full = fut.result()
            prefixes30[idx] = p30
            full_ys[idx]    = full

    # Filter samples with empty y
    valid_mask    = [bool(prefixes30[i]) for i in range(len(data))]
    data_v        = [item for item, ok in zip(data, valid_mask) if ok]
    prefixes30_v  = [p    for p,   ok in zip(prefixes30, valid_mask) if ok]
    full_ys_v     = [f    for f,   ok in zip(full_ys,    valid_mask) if ok]
    n_skip = len(data) - len(data_v)
    if n_skip:
        print(f"Skipped {n_skip} samples with empty y")
    print(f"Proceeding with {len(data_v)} samples")

    # ── Build prompts (parallel) ──────────────────────────────────────────────
    print("Preprocessing prefix30 prompts ...")
    msgs_prefix30 = preprocess_parallel(
        data_v,
        lambda idx, item: build_messages_prefix30(item, prefixes30_v[idx]),
        max_input_tokens, tokenizer, "prefix30 prompts", args.preprocess_workers,
    )

    print("Preprocessing noprefix prompts ...")
    msgs_noprefix = preprocess_parallel(
        data_v,
        lambda idx, item: build_messages_noprefix(item),
        max_input_tokens, tokenizer, "noprefix prompts", args.preprocess_workers,
    )

    print("Preprocessing full prompts ...")
    msgs_full = preprocess_parallel(
        data_v,
        lambda idx, item: build_messages_full(item, full_ys_v[idx]),
        max_input_tokens, tokenizer, "full prompts", args.preprocess_workers,
    )

    # ── Generate ─────────────────────────────────────────────────────────────
    out_stem = args.output.replace(".jsonl", "")
    ckpt_prefix30 = out_stem + ".ckpt_prefix30.json"
    ckpt_noprefix = out_stem + ".ckpt_noprefix.json"
    ckpt_full     = out_stem + ".ckpt_full.json"

    def load_ckpt(path):
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            print(f"  [resume] Loaded {len(d)} results from {path}")
            return d
        return None

    def save_ckpt(results, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)

    if client_mode:
        # ── Client mode: async OpenAI-compatible server ───────────────────────
        if AsyncOpenAI is None:
            print("error: openai package required for client mode. pip install openai",
                  file=sys.stderr)
            sys.exit(1)
        server_url = args.server_url or wait_for_url_file(
            args.server_url_file, args.server_wait_timeout
        )
        client = AsyncOpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="placeholder")
        model_name = args.model

        print("\n── Condition 1: prefix30 (client) ──")
        raw_prefix30 = load_ckpt(ckpt_prefix30)
        if raw_prefix30 is None:
            raw_prefix30 = asyncio.run(generate_client_batch(
                msgs_prefix30, client, model_name,
                args.max_tokens, args.temperature, args.max_concurrent, "prefix30",
            ))
            save_ckpt(raw_prefix30, ckpt_prefix30)

        print("\n── Condition 2: noprefix (client) ──")
        raw_noprefix = load_ckpt(ckpt_noprefix)
        if raw_noprefix is None:
            raw_noprefix = asyncio.run(generate_client_batch(
                msgs_noprefix, client, model_name,
                args.max_tokens, args.temperature, args.max_concurrent, "noprefix",
            ))
            save_ckpt(raw_noprefix, ckpt_noprefix)

        print("\n── Condition 3: full (client) ──")
        raw_full = load_ckpt(ckpt_full)
        if raw_full is None:
            raw_full = asyncio.run(generate_client_batch(
                msgs_full, client, model_name,
                args.max_tokens, args.temperature, args.max_concurrent, "full",
            ))
            save_ckpt(raw_full, ckpt_full)

    else:
        # ── Direct vLLM mode ─────────────────────────────────────────────────
        if LLM is None:
            print("error: vllm package not installed. Use --server_url_file for client mode.",
                  file=sys.stderr)
            sys.exit(1)

        print(f"Initializing vLLM (model={args.model}, tp={args.tp_size}) ...")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=args.gpu_util,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            skip_special_tokens=True,
        )
        chat_kwargs = {"enable_thinking": args.enable_thinking}

        print("\n── Condition 1: prefix30 ──")
        raw_prefix30 = load_ckpt(ckpt_prefix30)
        if raw_prefix30 is None:
            raw_prefix30 = generate_batch(msgs_prefix30, llm, sampling_params, chat_kwargs, "prefix30")
            save_ckpt(raw_prefix30, ckpt_prefix30)

        print("\n── Condition 2: noprefix ──")
        raw_noprefix = load_ckpt(ckpt_noprefix)
        if raw_noprefix is None:
            raw_noprefix = generate_batch(msgs_noprefix, llm, sampling_params, chat_kwargs, "noprefix")
            save_ckpt(raw_noprefix, ckpt_noprefix)

        print("\n── Condition 3: full ──")
        raw_full = load_ckpt(ckpt_full)
        if raw_full is None:
            raw_full = generate_batch(msgs_full, llm, sampling_params, chat_kwargs, "full")
            save_ckpt(raw_full, ckpt_full)

    # ── Post-process ─────────────────────────────────────────────────────────
    results = []
    skip_counts = {"prefix30": 0, "noprefix": 0, "full": 0}

    for item, p30, fy, rp, rn, rf in zip(
        data_v, prefixes30_v, full_ys_v, raw_prefix30, raw_noprefix, raw_full
    ):
        ystar_p  = strip_think_blocks(rp)
        ystar_np = strip_think_blocks(rn)
        ystar_f  = strip_think_blocks(rf)

        toks_p  = len(tokenizer.encode(ystar_p,  add_special_tokens=False))
        toks_np = len(tokenizer.encode(ystar_np, add_special_tokens=False))
        toks_f  = len(tokenizer.encode(ystar_f,  add_special_tokens=False))

        if toks_p  < args.min_tokens: skip_counts["prefix30"] += 1; continue
        if toks_np < args.min_tokens: skip_counts["noprefix"] += 1; continue
        if toks_f  < args.min_tokens: skip_counts["full"]     += 1; continue

        results.append({
            "conversation_id":  item["conversation_id"],
            "turn_index":       item["turn_index"],
            "x":                item["x"],
            "y":                item["y"],
            "o":                item["o"],
            "y_star_prefix30":  ystar_p,
            "y_star_noprefix":  ystar_np,
            "y_star_full":      ystar_f,
            "prefix_used":      p30,
        })

    for cond, n in skip_counts.items():
        if n:
            print(f"Skipped {n} samples: {cond} y* too short (<{args.min_tokens} tokens)")

    save_jsonl(results, args.output)
    print(f"\nSaved {len(results)} combined tuples → {args.output}")

    # Clean up checkpoints now that the final file is written
    for ckpt in [ckpt_prefix30, ckpt_noprefix, ckpt_full]:
        if Path(ckpt).exists():
            Path(ckpt).unlink()


if __name__ == "__main__":
    main()
