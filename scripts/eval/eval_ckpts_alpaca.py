#!/usr/bin/env python3
"""
Evaluate all LoRA checkpoints under a given directory using AlpacaEval-style judging.

- Loads base model once in vLLM, swaps LoRA per checkpoint
- Async judging with semaphore(N) concurrent Gemini calls, max_tokens=16
- Skips checkpoints that already have results
- Results saved to eval_results/alpaca/<run>/<step>/scores.json (incrementally)

Usage:
    export GOOGLE_API_KEY=...

    # Single run:
    python scripts/eval/eval_ckpts_alpaca.py checkpoints/qwen3_8b_sft_p30_wfbest/

    # Everything:
    python scripts/eval/eval_ckpts_alpaca.py checkpoints/
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from google import genai as google_genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROMPTS_PATH   = "arena-hard-auto/arena-hard-auto/alpaca_eval_data/alpaca_eval_prompts.jsonl"
REFERENCE_PATH = "alpaca_eval_data/gpt4_turbo_reference.json"
RESULTS_ROOT   = Path("eval_results/alpaca")
GEMINI_MODEL     = "gemini-2.5-flash"
GPT4O_MINI_MODEL = "gpt-4o-mini-2024-07-18"

JUDGE_PROMPT = """\
You are comparing two AI assistant responses to a user instruction.

[Instruction]
{instruction}

[Response 1]
{output_1}

[Response 2]
{output_2}

Which response is better? Output only "1", "2", or "tie"."""


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def find_checkpoints(root: Path) -> list[tuple[str, Path, Path]]:
    """
    Returns list of (key, adapter_dir, result_dir) sorted by run then step.
    key  = "run_name/step_name"
    """
    entries = []
    for p in sorted(root.rglob("adapter_config.json")):
        adapter_dir = p.parent
        parts = adapter_dir.parts
        key = f"{parts[-2]}/{parts[-1]}"
        result_dir = RESULTS_ROOT / parts[-2] / parts[-1]
        entries.append((key, adapter_dir, result_dir))

    def sort_key(e):
        key = e[0]
        run, step = key.rsplit("/", 1)
        n = 999999 if step == "final" else int(re.search(r"\d+", step).group())
        return (run, n)

    return sorted(entries, key=sort_key)


def already_done(result_dir: Path) -> bool:
    return (result_dir / "scores.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Generation (vLLM)
# ─────────────────────────────────────────────────────────────────────────────

def generate_outputs_base(llm, prompts, max_tokens: int, out_path: Path):
    if out_path.exists():
        print(f"    outputs cached, loading ...")
        return load_json(str(out_path))

    params = SamplingParams(temperature=1.0, max_tokens=max_tokens, skip_special_tokens=True)
    msgs   = [[{"role": "system", "content": ""},
               {"role": "user",   "content": q["instruction"]}] for q in prompts]
    outputs = llm.chat(msgs, params,
                       chat_template_kwargs={"enable_thinking": False},
                       use_tqdm=True)

    records = []
    for q, o in zip(prompts, outputs):
        text = re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.DOTALL).strip()
        records.append({"instruction": q["instruction"], "output": text,
                        "generator": "base", "dataset": q.get("dataset", "")})
    save_json(records, out_path)
    return records


def generate_outputs(llm, prompts, adapter_dir: Path, tag: str, max_tokens: int, out_path: Path):
    partial_path = out_path.with_suffix(".partial.json")

    if out_path.exists():
        print(f"    outputs cached, loading ...")
        return load_json(str(out_path))

    # Resume from partial if killed mid-run
    done = {}
    if partial_path.exists():
        for r in load_json(str(partial_path)):
            done[r["instruction"]] = r
        print(f"    resuming from partial ({len(done)}/{len(prompts)} done)")

    remaining = [q for q in prompts if q["instruction"] not in done]

    if remaining:
        lora_req = LoRARequest("adapter", 1, str(adapter_dir))
        params   = SamplingParams(temperature=1.0, max_tokens=max_tokens, skip_special_tokens=True)
        msgs = [[{"role": "system", "content": ""},
                 {"role": "user",   "content": q["instruction"]}]
                for q in remaining]
        outputs = llm.chat(msgs, params, lora_request=lora_req,
                           chat_template_kwargs={"enable_thinking": False},
                           use_tqdm=True)

        for q, o in zip(remaining, outputs):
            text = o.outputs[0].text
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            done[q["instruction"]] = {"instruction": q["instruction"], "output": text,
                                      "generator": tag, "dataset": q.get("dataset", "")}

        # Save partial after generation completes (safe point)
        save_json(list(done.values()), partial_path)

    # All done — write final file and remove partial
    records = [done[q["instruction"]] for q in prompts]  # preserve original order
    save_json(records, out_path)
    partial_path.unlink(missing_ok=True)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Async judging
# ─────────────────────────────────────────────────────────────────────────────

async def judge_one_gemini(client, instruction, out1, out2, sem, max_tokens):
    prompt = JUDGE_PROMPT.format(instruction=instruction, output_1=out1, output_2=out2)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                text = resp.text.strip().lower()
                if "1" in text and "2" not in text: return "1"
                if "2" in text and "1" not in text: return "2"
                return "tie"
            except Exception as e:
                if attempt == 2:
                    print(f"      [WARN] judge call failed: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def judge_one_gpt4omini(client, instruction, out1, out2, sem, max_tokens):
    prompt = JUDGE_PROMPT.format(instruction=instruction, output_1=out1, output_2=out2)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=GPT4O_MINI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("empty response")
                text = content.strip().lower()
                if "1" in text and "2" not in text: return "1"
                if "2" in text and "1" not in text: return "2"
                return "tie"
            except Exception as e:
                if attempt == 2:
                    print(f"      [WARN] judge call failed: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)
    return None


async def judge_all(our_outputs, ref_outputs, concurrency: int, max_tokens: int, judge: str = "gemini"):
    """
    AB + BA ordering for position bias removal.
    Returns (win_rate, lc_win_rate, n_wins, n_losses, n_ties).
    """
    sem = asyncio.Semaphore(concurrency)

    if judge == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        client  = google_genai.Client(api_key=api_key)
        judge_fn = judge_one_gemini
    else:  # gpt4omini
        api_key = os.environ.get("OPENAI_API_KEY")
        client  = AsyncOpenAI(api_key=api_key)
        judge_fn = judge_one_gpt4omini

    tasks_ab, tasks_ba = [], []
    for ours, ref in zip(our_outputs, ref_outputs):
        instr = ours["instruction"]
        tasks_ab.append(judge_fn(client, instr, ours["output"], ref["output"], sem, max_tokens))
        tasks_ba.append(judge_fn(client, instr, ref["output"], ours["output"], sem, max_tokens))

    print(f"    Running {len(tasks_ab)*2} judge calls (concurrency={concurrency}, max_tokens={max_tokens}) ...")
    results_ab = await asyncio.gather(*tasks_ab)
    results_ba = await asyncio.gather(*tasks_ba)

    wins, losses, ties = 0, 0, 0
    annotations = []

    for ours, ref, r_ab, r_ba in zip(our_outputs, ref_outputs, results_ab, results_ba):
        # AB: ours=1 → win if "1". BA: ours=2 → win if "2"
        votes = 0
        if r_ab == "1": votes += 1
        if r_ba == "2": votes += 1

        if   votes == 2: wins   += 1; outcome = 1
        elif votes == 0: losses += 1; outcome = 0
        else:            ties   += 1; outcome = 0.5

        annotations.append({"outcome": outcome,
                             "our_len": len(ours["output"]),
                             "ref_len": len(ref["output"])})

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
    lc_win_rate = _lc_win_rate(annotations)

    return win_rate, lc_win_rate, wins, losses, ties


def _lc_win_rate(annotations) -> float:
    """
    Logistic regression of win ~ log(our_len / ref_len).
    Evaluate at ratio=1 (log=0) → length-controlled win rate.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        X = np.array([np.log(max(a["our_len"], 1) / max(a["ref_len"], 1))
                      for a in annotations]).reshape(-1, 1)
        y = np.array([a["outcome"] for a in annotations])
        # Binarize: treat 0.5 (tie) as loss for simplicity
        y_bin = (y >= 1.0).astype(int)
        clf = LogisticRegression(max_iter=1000).fit(X, y_bin)
        lc_wr = clf.predict_proba([[0.0]])[0][1] * 100
        return round(lc_wr, 2)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def select_ckpts(ckpts, max_ckpts):
    """Select up to max_ckpts evenly-spaced checkpoints (always includes first and last)."""
    if max_ckpts <= 0 or len(ckpts) <= max_ckpts:
        return ckpts
    # Always include first and last; distribute the rest evenly
    indices = sorted(set(
        [0, len(ckpts) - 1] +
        [round(i * (len(ckpts) - 1) / (max_ckpts - 1)) for i in range(max_ckpts)]
    ))
    return [ckpts[i] for i in indices[:max_ckpts]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir",           help="Directory to scan for checkpoints")
    ap.add_argument("--base_model",       default="Qwen/Qwen3-8B")
    ap.add_argument("--prompts",          default=PROMPTS_PATH)
    ap.add_argument("--reference",        default=REFERENCE_PATH)
    ap.add_argument("--max_gen_tokens",   type=int, default=1024, help="Max tokens for generation")
    ap.add_argument("--max_judge_tokens", type=int, default=16,   help="Max tokens per judge API call")
    ap.add_argument("--concurrency",      type=int, default=200,  help="Concurrent judge API calls")
    ap.add_argument("--gpu_util",         type=float, default=0.92)
    ap.add_argument("--judge",            default="gemini", choices=["gemini", "gpt4omini"],
                    help="Which judge to use (default: gemini)")
    ap.add_argument("--results_root",     default=None,
                    help="Override results root dir (default: eval_results/alpaca)")
    ap.add_argument("--outputs_root",     default=None,
                    help="Directory to reuse cached model_outputs.json from (e.g. eval_results/alpaca)")
    ap.add_argument("--max_ckpts",        type=int, default=0,
                    help="Max checkpoints to eval per run, evenly spaced (0=all)")
    ap.add_argument("--eval_base",        action="store_true",
                    help="Also evaluate the base model (no LoRA)")
    args = ap.parse_args()

    global RESULTS_ROOT
    if args.results_root:
        RESULTS_ROOT = Path(args.results_root)

    if args.judge == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("Error: GOOGLE_API_KEY not set")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Error: OPENAI_API_KEY not set")

    for p in [args.prompts, args.reference]:
        if not Path(p).exists():
            sys.exit(f"Error: {p} not found")

    prompts   = load_jsonl(args.prompts)
    reference = load_json(args.reference)
    assert len(prompts) == len(reference), "prompts/reference length mismatch"
    print(f"Loaded {len(prompts)} prompts  |  reference: {reference[0]['generator']}")

    all_ckpts = find_checkpoints(Path(args.ckpt_dir))

    # Group by run, apply max_ckpts per run, then flatten
    if args.max_ckpts > 0:
        from itertools import groupby
        grouped = {}
        for k, a, r in all_ckpts:
            run = k.rsplit("/", 1)[0]
            grouped.setdefault(run, []).append((k, a, r))
        ckpts = []
        for run, entries in grouped.items():
            ckpts.extend(select_ckpts(entries, args.max_ckpts))
    else:
        ckpts = all_ckpts

    todo  = [(k, a, r) for k, a, r in ckpts if not already_done(r)]
    print(f"Judge             : {args.judge}")
    print(f"Results root      : {RESULTS_ROOT}")
    print(f"Checkpoints found : {len(ckpts)}")
    print(f"Already done      : {len(ckpts) - len(todo)}")
    print(f"To evaluate       : {len(todo)}\n")

    # Check if all outputs can be satisfied from outputs_root (skip vLLM load)
    def needs_generation(key, result_dir):
        out_path = result_dir / "model_outputs.json"
        if out_path.exists():
            return False
        if args.outputs_root:
            alt_path = Path(args.outputs_root) / key / "model_outputs.json"
            if alt_path.exists():
                return False
        return True

    base_result_dir = RESULTS_ROOT / "base"
    base_needs_gen = args.eval_base and not already_done(base_result_dir) and not (
        args.outputs_root and (Path(args.outputs_root) / "base" / "model_outputs.json").exists()
    ) and not (base_result_dir / "model_outputs.json").exists()

    needs_vllm = base_needs_gen or any(needs_generation(k, r) for k, _, r in todo)

    if needs_vllm:
        print(f"Loading {args.base_model} into vLLM ...")
        llm = LLM(
            model=args.base_model,
            enable_lora=True, max_lora_rank=64,
            gpu_memory_utilization=args.gpu_util,
            dtype="bfloat16", max_model_len=8192,
            max_num_seqs=512, trust_remote_code=True,
            disable_log_stats=True,
        )
        print("Model ready.\n")
    else:
        llm = None
        print("All outputs cached — skipping vLLM load.\n")

    # ── Base model eval ──────────────────────────────────────────────────────
    if args.eval_base:
        base_result_dir = RESULTS_ROOT / "base"
        if already_done(base_result_dir):
            print("Base model already evaluated, skipping.\n")
        else:
            print("[base] Evaluating base model (no LoRA) ...")
            out_path = base_result_dir / "model_outputs.json"
            base_result_dir.mkdir(parents=True, exist_ok=True)
            alt_base = Path(args.outputs_root) / "base" / "model_outputs.json" if args.outputs_root else None
            if alt_base and alt_base.exists():
                print(f"    base outputs found in outputs_root, loading ...")
                our_outputs = load_json(str(alt_base))
            else:
                our_outputs = generate_outputs_base(llm, prompts, args.max_gen_tokens, out_path)
            wr, lc_wr, wins, losses, ties = asyncio.run(
                judge_all(our_outputs, reference, args.concurrency, args.max_judge_tokens, args.judge))
            scores = {"run": "base", "step": 0, "step_name": "base",
                      "win_rate": round(wr, 2), "lc_win_rate": lc_wr,
                      "wins": wins, "losses": losses, "ties": ties}
            save_json(scores, base_result_dir / "scores.json")
            print(f"    win_rate={wr:.1f}%  lc_win_rate={lc_wr}%  ({wins}W/{losses}L/{ties}T)\n")

    for i, (key, adapter_dir, result_dir) in enumerate(todo):
        print(f"[{i+1}/{len(todo)}] {key}")

        out_path = result_dir / "model_outputs.json"

        # Check alternate outputs_root for pre-generated outputs (avoids re-running vLLM)
        if args.outputs_root and not out_path.exists():
            alt_path = Path(args.outputs_root) / key / "model_outputs.json"
            if alt_path.exists():
                print(f"    outputs found in outputs_root, loading ...")
                our_outputs = load_json(str(alt_path))
            else:
                our_outputs = generate_outputs(llm, prompts, adapter_dir,
                                               tag=key, max_tokens=args.max_gen_tokens,
                                               out_path=out_path)
        else:
            our_outputs = generate_outputs(llm, prompts, adapter_dir,
                                            tag=key, max_tokens=args.max_gen_tokens,
                                            out_path=out_path)

        win_rate, lc_win_rate, wins, losses, ties = asyncio.run(
            judge_all(our_outputs, reference,
                      concurrency=args.concurrency,
                      max_tokens=args.max_judge_tokens)
        )

        step_name = adapter_dir.name
        step_num  = 999999 if step_name == "final" else int(re.search(r"\d+", step_name).group())

        scores = {
            "run":          adapter_dir.parts[-2],
            "step":         step_num,
            "step_name":    step_name,
            "win_rate":     round(win_rate, 2),
            "lc_win_rate":  lc_win_rate,
            "wins":         wins,
            "losses":       losses,
            "ties":         ties,
        }
        save_json(scores, result_dir / "scores.json")
        print(f"    win_rate={win_rate:.1f}%  lc_win_rate={lc_win_rate}%  "
              f"({wins}W / {losses}L / {ties}T)\n")

    print(f"All done. Results in {RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
