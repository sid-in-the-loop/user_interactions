#!/usr/bin/env python3
"""
Phase 1: Generate model outputs for all benchmarks across all checkpoints.

Uses vLLM with LoRA hot-swapping — loads base model once, swaps adapters per checkpoint.
Resumable: skips benchmarks where outputs.json already exists.

Usage:
  python scripts/eval/generate_all.py \
      --method_dir /projects/bgtw/ssredharan/checkpoints/jsd_p30 \
      --output_root eval_results/jsd_p30 \
      --benchmarks all

  # Specific benchmarks only:
  python scripts/eval/generate_all.py \
      --method_dir checkpoints/jsd_p30 \
      --output_root eval_results/jsd_p30 \
      --benchmarks alpaca_eval math500 reasoning_gym
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Add parent to path for benchmark imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.eval.benchmarks import BENCHMARKS, ALL_BENCHMARK_NAMES


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def discover_checkpoints(method_dir: Path) -> list[tuple[str, Path]]:
    """Find all LoRA checkpoint dirs, sorted by step number."""
    entries = []
    for p in method_dir.iterdir():
        if p.is_dir() and (p / "adapter_config.json").exists():
            entries.append((p.name, p))

    def sort_key(e):
        name = e[0]
        if name == "final":
            return 999999
        m = re.search(r"\d+", name)
        return int(m.group()) if m else 0

    return sorted(entries, key=sort_key)


def _truncate_long_prompts(prompts, tokenizer, max_input_tokens):
    """Truncate any prompts that would exceed model context."""
    truncated = 0
    for i, msgs in enumerate(prompts):
        # Estimate token count from last user message
        text = msgs[-1]["content"] if msgs else ""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_input_tokens:
            msgs[-1]["content"] = tokenizer.decode(tokens[:max_input_tokens], skip_special_tokens=True)
            truncated += 1
    if truncated:
        print(f"      Truncated {truncated} long prompts to fit context window")
    return prompts


def generate_single_turn(llm, benchmark, lora_req, output_path, max_retries=2, min_len=10):
    """Generate outputs for a single-turn benchmark. Retries empty/short outputs."""
    prompts = benchmark.format_prompts()
    params = benchmark.sampling_params()

    # Truncate prompts that exceed model context (leave room for output tokens)
    max_input = llm.llm_engine.model_config.max_model_len - params.max_tokens - 256
    prompts = _truncate_long_prompts(prompts, llm.get_tokenizer(), max_input)

    outputs = llm.chat(
        prompts, params,
        lora_request=lora_req,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )

    # Strip think blocks from all outputs
    for o in outputs:
        o.outputs[0].text = strip_think(o.outputs[0].text)

    # Retry empty/short outputs
    for retry in range(max_retries):
        short_indices = [i for i, o in enumerate(outputs) if len(o.outputs[0].text.strip()) < min_len]
        if not short_indices:
            break
        print(f"      Retry {retry+1}: regenerating {len(short_indices)} short outputs...")
        retry_prompts = [prompts[i] for i in short_indices]
        retry_outputs = llm.chat(
            retry_prompts, params,
            lora_request=lora_req,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=True,
        )
        for idx, ro in zip(short_indices, retry_outputs):
            new_text = strip_think(ro.outputs[0].text)
            if len(new_text.strip()) > len(outputs[idx].outputs[0].text.strip()):
                outputs[idx].outputs[0].text = new_text

    benchmark.save_outputs(outputs, output_path)
    return len(prompts)


def generate_multi_turn(llm, benchmark, lora_req, output_path):
    """Generate outputs for MT-Bench (2-turn)."""
    params = benchmark.sampling_params()

    # Turn 1
    print("      Turn 1...")
    t1_prompts = benchmark.format_turn1_prompts()
    t1_outputs = llm.chat(
        t1_prompts, params,
        lora_request=lora_req,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    t1_texts = [strip_think(o.outputs[0].text) for o in t1_outputs]

    # Turn 2
    print("      Turn 2...")
    t2_prompts = benchmark.format_turn2_prompts(t1_texts)
    t2_outputs = llm.chat(
        t2_prompts, params,
        lora_request=lora_req,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    t2_texts = [strip_think(o.outputs[0].text) for o in t2_outputs]

    benchmark.save_multiturn_outputs(t1_texts, t2_texts, output_path)
    return len(t1_prompts) * 2


def main():
    parser = argparse.ArgumentParser(description="Generate model outputs for all benchmarks")
    parser.add_argument("--method_dir", required=True, help="Path to checkpoint dir (e.g. checkpoints/jsd_p30)")
    parser.add_argument("--output_root", required=True, help="Output root (e.g. eval_results/jsd_p30)")
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmarks", nargs="+", default=["all"],
                        help="Which benchmarks to run (default: all)")
    parser.add_argument("--gpu_util", type=float, default=0.92)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    output_root = Path(args.output_root)

    # Resolve benchmarks
    if "all" in args.benchmarks:
        benchmark_names = ALL_BENCHMARK_NAMES
    else:
        benchmark_names = args.benchmarks

    # Initialize benchmarks and load data
    print("Loading benchmark data...")
    benchmarks = []
    for name in benchmark_names:
        if name not in BENCHMARKS:
            print(f"WARNING: Unknown benchmark '{name}', skipping")
            continue
        try:
            b = BENCHMARKS[name]()
            b.load_data()
            benchmarks.append(b)
        except Exception as e:
            print(f"WARNING: Failed to load {name}: {e}")

    if not benchmarks:
        print("No benchmarks loaded!")
        sys.exit(1)

    print(f"\nBenchmarks: {[b.name for b in benchmarks]}")

    # Discover checkpoints
    checkpoints = discover_checkpoints(method_dir)
    print(f"Found {len(checkpoints)} checkpoints in {method_dir}")

    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)

    # Initialize vLLM with LoRA support
    print(f"\nInitializing vLLM ({args.base_model}, enable_lora=True)...")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("vLLM ready.\n")

    # Generate for each checkpoint × benchmark
    total_generated = 0
    for ckpt_idx, (ckpt_name, ckpt_path) in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"  Checkpoint {ckpt_idx+1}/{len(checkpoints)}: {ckpt_name}")
        print(f"{'='*60}")

        lora_req = LoRARequest("adapter", ckpt_idx + 1, str(ckpt_path))

        for benchmark in benchmarks:
            output_path = output_root / ckpt_name / benchmark.name / "outputs.json"

            if output_path.exists():
                print(f"    [{benchmark.name}] SKIP (already exists)")
                continue

            print(f"    [{benchmark.name}] Generating...")

            try:
                if benchmark.needs_multi_turn():
                    n = generate_multi_turn(llm, benchmark, lora_req, output_path)
                else:
                    n = generate_single_turn(llm, benchmark, lora_req, output_path)
                total_generated += n
                print(f"    [{benchmark.name}] Done ({n} outputs saved)")
            except Exception as e:
                print(f"    [{benchmark.name}] ERROR: {e}")
                # Don't crash — continue to next benchmark
                continue

    print(f"\n{'='*60}")
    print(f"Generation complete. {total_generated} total outputs across "
          f"{len(checkpoints)} checkpoints × {len(benchmarks)} benchmarks.")
    print(f"Results in {output_root}/")


if __name__ == "__main__":
    main()
