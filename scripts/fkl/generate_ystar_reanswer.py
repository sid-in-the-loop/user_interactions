"""
Generate y_star_reanswer: model sees x + y + o, then is explicitly asked to
re-answer the original question in x using the feedback as a hint.

This produces a fair comparison against y_base (x only):
  - y_star_reanswer: given (x, y, o), re-answer x
  - y_base:          given x,          answer x
  - judge asks: which better addresses x?

Contrast with the original y*: model responds to o (the follow-up), which
is a different target than x.

Usage:
  python scripts/fkl/generate_ystar_reanswer.py \
    --input datasets/wildfeedback/tuples.jsonl \
    --ids-file eval_results/winrate_sample_500_seed42.json \
    --output_dir datasets/wildfeedback/qwen3_8b \
    --model Qwen/Qwen3-8B \
    --mode A \
    --max_tokens 1024 --tp_size 1 --gpu_util 0.95
"""

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

SYSTEM = "You are a helpful assistant."

REDIRECT_TURN = (
    "Thank you for the feedback. "
    "Now please provide an improved response to my original question."
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_messages(item, enable_thinking):
    """
    system
    + x (original conversation)
    + y (original assistant response)
    + o (user feedback)
    + redirect user turn ("now re-answer my original question")
    """
    msgs = [{"role": "system", "content": SYSTEM}]
    for turn in item["x"]:
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": item["y"]["role"], "content": item["y"]["content"]})
    msgs.append({"role": item["o"]["role"], "content": item["o"]["content"]})
    msgs.append({"role": "user", "content": REDIRECT_TURN})
    return msgs


def extract_response(text):
    """Strip any <think>...</think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["A", "B", "both"], default="A",
                        help="A=nonthinking, B=thinking")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_util", type=float, default=0.95)
    parser.add_argument("--max_num_seqs", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--min_tokens", type=int, default=10)
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = {(r["conversation_id"], r.get("turn_index")) for r in json.load(f)}
        data = [r for r in data if (r["conversation_id"], r.get("turn_index")) in ids]
        print(f"Filtered to {len(data)} samples via --ids-file")
    else:
        print(f"Loaded {len(data)} samples")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_input_tokens = max(256, args.max_model_len - args.max_tokens - 64)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    print(f"Initializing vLLM (tp={args.tp_size})...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
    )

    def run_mode(mode_char):
        enable_thinking = (mode_char == "B")
        chat_kwargs = {"enable_thinking": enable_thinking}
        suffix = "thinking" if enable_thinking else "nonthinking"
        out_path = Path(args.output_dir) / f"ystar_reanswer_{suffix}.jsonl"

        print(f"\nMode {mode_char} ({suffix}): building {len(data)} prompts...")
        all_msgs = []
        for item in data:
            msgs = build_messages(item, enable_thinking)
            # truncate from left if too long (keep recent context)
            try:
                ids_tok = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True
                )
                if len(ids_tok) > max_input_tokens:
                    # trim x turns from the front, keep y+o+redirect
                    x_turns = [m for m in msgs if m["role"] != "system"]
                    kept = [msgs[0]]  # system
                    # always keep last 4 turns: last x turn, y, o, redirect
                    kept += x_turns[-4:]
                    msgs = kept
            except Exception:
                pass
            all_msgs.append(msgs)

        t0 = time.perf_counter()
        outputs = llm.chat(all_msgs, sampling_params=sampling_params,
                           chat_template_kwargs=chat_kwargs, use_tqdm=True)
        elapsed = time.perf_counter() - t0
        print(f"Generated {len(data)} in {elapsed:.1f}s")

        results = []
        for item, output in zip(data, outputs):
            raw = output.outputs[0].text
            response = extract_response(raw)
            if not response or len(tokenizer.encode(response, add_special_tokens=False)) < args.min_tokens:
                continue
            results.append({
                "conversation_id": item["conversation_id"],
                "turn_index": item.get("turn_index"),
                "x": item["x"],
                "y": item["y"],
                "o": item.get("o"),
                "y_star": response,
                "mode": mode_char,
            })

        save_jsonl(results, str(out_path))
        print(f"Saved {len(results)} → {out_path}")

    if args.mode in ("A", "both"):
        run_mode("A")
    if args.mode in ("B", "both"):
        run_mode("B")

    print("Done.")


if __name__ == "__main__":
    main()
