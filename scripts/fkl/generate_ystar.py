#!/usr/bin/env python3
"""
Phase 1 — Generate y* using vLLM across all 4 GPUs.

Run with:
    python generate_y_star.py \
        --input  datasets/wildchat/filtered_tuples.jsonl \
        --output datasets/wildchat/y_star.jsonl \
        --model  Qwen/Qwen3-4B

vLLM uses tensor_parallel_size=4 internally — no torchrun needed here.
With 4x48GB GPUs and Qwen3-4B, this will saturate GPU throughput
and process ~34k samples in minutes.
"""

import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_cache(path: str) -> dict:
    """Load already-generated outputs so we can resume."""
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cache[item["key"]] = item["y_star_content"]
    return cache


def append_cache(path: str, key: str, content: str):
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "y_star_content": content}, ensure_ascii=False) + "\n")


def cache_key(item: dict) -> str:
    return f"{item['conversation_id']}_{item['turn_index']}"


def build_hindsight_messages(item: dict) -> list:
    """Teacher sees: system + x + y (GPT-4) + o (follow-up)."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Respond concisely and directly. "
                "Do not output internal reasoning or <think> blocks."
            ),
        }
    ] + list(item["x"]) + [item["y"]] + [item["o"]]


def strip_think_blocks(text: str) -> str:
    """Strip <think>...</think>. Handle truncated open tags."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,             help="Path to filtered_tuples.jsonl")
    parser.add_argument("--output",      required=True,             help="Path to write y_star.jsonl")
    parser.add_argument("--cache",       default=None,              help="Cache file for resuming (default: <output>.cache)")
    parser.add_argument("--model",       default="Qwen/Qwen3-4B")
    parser.add_argument("--max_tokens",  type=int,   default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu_util",    type=float, default=0.92,  help="vLLM GPU memory utilization per GPU")
    args = parser.parse_args()

    cache_file = args.cache or (args.output + ".cache")

    # ── Load data & cache ────────────────────────────────────────────
    data = load_jsonl(args.input)
    cache = load_cache(cache_file)
    print(f"Loaded {len(data)} samples. Cache hits: {len(cache)}")

    # Split into cached and to-generate
    to_generate = [(i, item) for i, item in enumerate(data) if cache_key(item) not in cache]
    print(f"To generate: {len(to_generate)}  |  Already cached: {len(data) - len(to_generate)}")

    # ── vLLM setup — all 4 GPUs together ────────────────────────────
    if to_generate:
        print(f"Initializing vLLM with tensor_parallel_size=4 ...")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=4,        # use all 4 GPUs as one engine
            gpu_memory_utilization=args.gpu_util,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8192,            # Increase to handle long prompts
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            skip_special_tokens=True,
        )

        # Build all prompts at once — vLLM handles batching internally
        print("Building prompts ...")
        prompts = []
        # Max prompt tokens = 8192 - max_new_tokens
        max_p = 8192 - args.max_tokens
        for _, item in to_generate:
            text = tokenizer.apply_chat_template(
                build_hindsight_messages(item),
                tokenize=False,
                add_generation_prompt=True,
            )
            # Truncate from the left (recent history is more important)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_p:
                tokens = tokens[-max_p:]
                text = tokenizer.decode(tokens)
            prompts.append(text)

        # Generate — vLLM will use continuous batching across all prompts
        print(f"Generating y* for {len(prompts)} samples ...")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        # Write to cache immediately
        for (orig_idx, item), output in zip(to_generate, outputs):
            content = strip_think_blocks(output.outputs[0].text)
            cache[cache_key(item)] = content
            append_cache(cache_file, cache_key(item), content)

        del llm
        print("vLLM generation done.")

    # ── Assemble final output ────────────────────────────────────────
    print("Assembling output ...")
    results = []
    skipped = 0
    for item in tqdm(data):
        ck = cache_key(item)
        if ck not in cache:
            skipped += 1
            continue
        out = dict(item)
        out["y_star"] = {"role": "assistant", "content": cache[ck]}
        results.append(out)

    save_jsonl(results, args.output)
    print(f"Saved {len(results)} samples to {args.output}  (skipped {skipped})")


if __name__ == "__main__":
    main()