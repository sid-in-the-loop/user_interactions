#!/usr/bin/env python3
"""
Evaluate a LoRA checkpoint on AlpacaEval: generate responses + judge with local LLM.

1. Load base model + LoRA adapter
2. Generate responses to AlpacaEval prompts (805 samples)
3. Judge pairwise vs GPT-4 turbo reference using vLLM server (Qwen3-14B)

Usage:
  python scripts/eval/eval_checkpoint.py \
      --checkpoint checkpoints/jsd_p30/step-10 \
      --judge-url http://gh140:8002 \
      --output-dir eval_results/jsd_p30/step-10
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import requests


# ── AlpacaEval data ──────────────────────────────────────────────────────────

def load_alpaca_eval(reference_path="alpaca_eval_data/gpt4_turbo_reference.json"):
    """Load AlpacaEval prompts + GPT-4 turbo references."""
    with open(reference_path) as f:
        data = json.load(f)
    return data


# ── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=2048):
    """Generate responses to a list of instruction strings."""
    model.eval()
    device = next(model.parameters()).device
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]

        # Build chat messages
        batch_messages = []
        for instruction in batch_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            batch_messages.append(messages)

        # Tokenize with chat template
        batch_texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        # Left-pad for generation
        tokenizer.padding_side = "left"
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048, add_special_tokens=False).to(device)

        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            chat_template_kwargs={"enable_thinking": False},
        )

        # Decode only generated part
        for j, output in enumerate(outputs):
            generated = output[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            # Strip think blocks if any
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            responses.append(text)

    tokenizer.padding_side = "right"  # restore
    return responses


# ── Judging ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a strict, impartial evaluator comparing two AI assistant responses to the same user request. Judge based on: correctness, relevance, completeness, and quality of the response relative to what the user asked. Do NOT favor longer responses automatically. Do NOT consider formatting unless it meaningfully affects clarity.
Output exactly one character: A, B, or C where C means tie or too close to call."""

JUDGE_USER_TEMPLATE = """User request:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response better addresses the user's request? Output only A, B, or C."""


def parse_judge_output(text):
    if not text:
        return None
    m = re.search(r"\b([ABC])\b", text.strip().upper())
    return m.group(1) if m else None


def call_judge(judge_url, model_name, instruction, response_a, response_b):
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                instruction=instruction, response_a=response_a, response_b=response_b
            )},
        ],
        "max_tokens": 3,
        "temperature": 0,
    }
    # For local vLLM (Qwen3), disable thinking
    if "qwen" in model_name.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(4):
        try:
            resp = requests.post(f"{judge_url}/v1/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = parse_judge_output(content)
            if result is not None:
                return result
            # Got response but no A/B/C — retry
        except Exception as e:
            if attempt == 3:
                return None


def judge_all(judge_url, judge_model, instructions, model_responses, reference_responses, max_workers=8):
    """Run pairwise judging with position-bias removal."""
    wins, losses, ties = 0, 0, 0
    results = []

    def _judge(idx):
        inst = instructions[idx]
        model_resp = model_responses[idx]
        ref_resp = reference_responses[idx]

        # Call 1: A=model, B=reference
        r1 = call_judge(judge_url, judge_model, inst, model_resp, ref_resp)
        # Call 2: A=reference, B=model (flipped)
        r2 = call_judge(judge_url, judge_model, inst, ref_resp, model_resp)

        # Position-bias removal
        if r1 is None or r2 is None or r1 == "C" or r2 == "C":
            winner = "tie"
        elif r1 == "A" and r2 == "B":
            winner = "model"
        elif r1 == "B" and r2 == "A":
            winner = "reference"
        else:
            winner = "tie"

        return idx, winner, r1, r2

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_judge, i): i for i in range(len(instructions))}
        for f in tqdm(as_completed(futures), total=len(instructions), desc="Judging"):
            idx, winner, r1, r2 = f.result()
            if winner == "model":
                wins += 1
            elif winner == "reference":
                losses += 1
            else:
                ties += 1
            results.append({"index": idx, "winner": winner, "call1": r1, "call2": r2})

    n = len(instructions)
    win_rate = wins / n * 100
    wr_no_ties = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_len = sum(len(r) for r in model_responses) / n

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "win_rate_no_ties": wr_no_ties,
        "avg_length": avg_len,
        "n": n,
        "results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--reference", default="alpaca_eval_data/gpt4_turbo_reference.json")
    parser.add_argument("--judge-url", required=True, help="vLLM judge server URL")
    parser.add_argument("--judge-model", default=None, help="Model name (auto-detect)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    # Auto-detect judge model
    if args.judge_model is None:
        try:
            resp = requests.get(f"{args.judge_url}/v1/models")
            args.judge_model = resp.json()["data"][0]["id"]
        except Exception:
            args.judge_model = "Qwen/Qwen3-14B"

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Judge: {args.judge_model} @ {args.judge_url}")

    # Load AlpacaEval
    reference_data = load_alpaca_eval(args.reference)
    instructions = [d["instruction"] for d in reference_data]
    reference_responses = [d["output"] for d in reference_data]
    print(f"AlpacaEval: {len(instructions)} prompts")

    # Load model + LoRA
    print(f"Loading {args.base_model} + LoRA from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.to("cuda").eval()
    print("Model loaded.")

    # Generate
    print("Generating responses...")
    model_responses = generate_responses(
        model, tokenizer, instructions,
        batch_size=args.gen_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Save model outputs
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = [
        {"instruction": inst, "output": resp, "generator": os.path.basename(args.checkpoint)}
        for inst, resp in zip(instructions, model_responses)
    ]
    with open(os.path.join(args.output_dir, "model_outputs.json"), "w") as f:
        json.dump(outputs, f, indent=2)

    # Judge
    print("Judging with 14B...")
    eval_result = judge_all(
        args.judge_url, args.judge_model,
        instructions, model_responses, reference_responses,
        max_workers=args.judge_workers,
    )

    # Save scores
    scores = {
        "checkpoint": args.checkpoint,
        "judge": args.judge_model,
        "win_rate": eval_result["win_rate"],
        "win_rate_no_ties": eval_result["win_rate_no_ties"],
        "wins": eval_result["wins"],
        "losses": eval_result["losses"],
        "ties": eval_result["ties"],
        "avg_length": eval_result["avg_length"],
        "n": eval_result["n"],
    }
    with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    with open(os.path.join(args.output_dir, "judge_results.json"), "w") as f:
        json.dump(eval_result["results"], f, indent=2)

    print(f"\n{'='*50}")
    print(f"Win rate: {eval_result['win_rate']:.1f}% (inc ties)")
    print(f"Win rate: {eval_result['win_rate_no_ties']:.1f}% (exc ties)")
    print(f"Wins: {eval_result['wins']}, Losses: {eval_result['losses']}, Ties: {eval_result['ties']}")
    print(f"Avg length: {eval_result['avg_length']:.0f}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
