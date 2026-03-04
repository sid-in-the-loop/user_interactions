#!/usr/bin/env python3
"""
Generate model answers for AlpacaEval 2.0 using vLLM across 2 GPUs.
Fixed: 
  - Patches vLLM 0.16+ bug for absolute local paths.
  - Optimized for AlpacaEval 2.0 requirements (instruction/output keys).
"""

import os
# ── MUST be set before any vLLM import ──────────────────────────────────────
os.environ["VLLM_DISABLE_COMPILE"] = "1"
os.environ["VLLM_COMPILE_LEVEL"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


import argparse
import json
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_json(data: List[Dict[str, Any]], path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", default="alpaca_eval_data/alpaca_eval_prompts.jsonl")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7) # AlpacaEval standard
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use.")
    args = parser.parse_args()

    questions = load_jsonl(args.input_file)
    print(f"Loaded {len(questions)} prompts from {args.input_file}")

    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists. Skipping.")
        return

    model_path = os.path.abspath(args.model_path) if os.path.isdir(args.model_path) else args.model_path
    
    print(f"Initializing vLLM for {args.model_name} (TP={args.tensor_parallel_size}, Forced Eager)...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=True,
    )

    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    print("Formatting prompts...")
    prompts: List[str] = []
    for q in questions:
        messages = [{"role": "user", "content": q["instruction"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"Generating {len(prompts)} answers...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    results: List[Dict[str, Any]] = []
    for q, output in zip(questions, outputs):
        results.append({
            "instruction": q["instruction"],
            "dataset": q["dataset"],
            "output": output.outputs[0].text.strip(),
            "generator": args.model_name
        })

    save_json(results, args.output_file)
    print(f"Done! Saved outputs to {args.output_file}")
    del llm


if __name__ == "__main__":
    main()
