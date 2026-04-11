#!/usr/bin/env python3
"""
MMLU-Pro local eval with vLLM (CoT prompt style from TIGER-AI-Lab/MMLU-Pro).

Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Paper: MMLU-Pro (NeurIPS 24).

Does not depend on cloning MMLU-Pro; uses the same prompt template and answer
extraction as evaluate_from_local.py, with chunked generation to avoid OOM.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any

os.environ.setdefault("VLLM_DISABLE_COMPILE", "1")

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CHOICES = list("ABCDEFGHIJKLMNOP")

INITIAL_PROMPT = """The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.

"""


def preprocess_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        opts = [o for o in row["options"] if o != "N/A"]
        row = dict(row)
        row["options"] = opts
        out.append(row)
    return out


def select_by_category(df: list[dict], subject: str) -> list[dict]:
    return [x for x in df if x["category"] == subject]


def format_cot_example(example: dict, including_answer: bool) -> str:
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += "{}. {}\n".format(CHOICES[i], opt)
    if including_answer:
        cot = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df: list[dict], curr: dict, k: int) -> str:
    prompt = INITIAL_PROMPT.replace("{$}", curr["category"])
    subject = curr["category"]
    val_sub = select_by_category(val_df, subject)[:k]
    for ex in val_sub:
        prompt += format_cot_example(ex, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text: str) -> str | None:
    m = re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if m:
        return m.group(1).upper()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", "-m", required=True, help="HF id or local path to merged model")
    ap.add_argument("--output_dir", "-o", required=True, help="e.g. eval_results/mmlu_pro/my_run")
    ap.add_argument("--ntrain", "-k", type=int, default=5, help="few-shot examples per subject (from val)")
    ap.add_argument(
        "--subjects",
        type=str,
        default="all",
        help='Comma-separated subject substrings, or "all"',
    )
    ap.add_argument("--gen_batch_size", type=int, default=32, help="vLLM prompts per forward batch")
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--gpu_util", type=float, default=0.92)
    ap.add_argument("--max_questions_per_subject", type=int, default=0, help="0 = full test split")
    ap.add_argument("--tensor_parallel_size", "--tp", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run_log.txt")

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Loading TIGER-Lab/MMLU-Pro …")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = preprocess_rows(list(ds["test"]))
    val_df = preprocess_rows(list(ds["validation"]))

    subjects = sorted({x["category"] for x in test_df})
    if args.subjects.strip().lower() != "all":
        want = [s.strip() for s in args.subjects.split(",") if s.strip()]
        subjects = [
            s
            for s in subjects
            if any(w.replace(" ", "_") in s.replace(" ", "_") for w in want)
        ]
    if not subjects:
        log("No subjects selected.")
        sys.exit(1)
    log(f"Subjects ({len(subjects)}): {subjects}")

    model_path = os.path.abspath(args.model_path) if os.path.isdir(args.model_path) else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"vLLM load: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        max_num_seqs=min(args.gen_batch_size, 256),
    )
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["Question:"],
    )

    summary_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for subject in subjects:
        test_sub = select_by_category(test_df, subject)
        val_sub = select_by_category(val_df, subject)
        if args.max_questions_per_subject > 0:
            test_sub = test_sub[: args.max_questions_per_subject]
        if not test_sub:
            continue

        prompts: list[str] = []
        for curr in test_sub:
            k = args.ntrain
            prompt = None
            while k >= 0:
                prompt = generate_cot_prompt(val_sub, curr, k)
                ntok = len(tokenizer.encode(prompt))
                if ntok < args.max_model_len - args.max_new_tokens - 64:
                    break
                k -= 1
            if k < 0:
                prompt = generate_cot_prompt(val_sub, curr, 0)
            prompts.append(prompt)

        preds: list[str | None] = []
        responses: list[str] = []
        for i in tqdm(
            range(0, len(prompts), args.gen_batch_size),
            desc=subject[:40],
        ):
            chunk = prompts[i : i + args.gen_batch_size]
            outs = llm.generate(chunk, sp)
            for o in outs:
                text = o.outputs[0].text
                responses.append(text)
                preds.append(extract_answer(text))

        corr = wrong = 0
        results: list[dict] = []
        for j, curr in enumerate(test_sub):
            pred = preds[j]
            gold = curr["answer"]
            if pred is None or pred != gold:
                wrong += 1
            else:
                corr += 1
            row = dict(curr)
            row["pred"] = preds[j]
            row["model_output"] = responses[j]
            results.append(row)

        acc = corr / max(1, corr + wrong)
        log(f"{subject}: acc={acc:.4f} ({corr}/{corr + wrong})")
        summary_rows.append({"subject": subject, "accuracy": acc, "correct": corr, "total": corr + wrong})
        with open(
            os.path.join(args.output_dir, f"{subject.replace('/', '_')}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=0)

    total_c = sum(r["correct"] for r in summary_rows)
    total_n = sum(r["total"] for r in summary_rows)
    overall = total_c / max(1, total_n)
    log(f"OVERALL accuracy: {overall:.4f} ({total_c}/{total_n}) elapsed_s={time.time() - t0:.0f}")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"per_subject": summary_rows, "overall_accuracy": overall, "model_path": model_path},
            f,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "accuracy", "correct", "total"])
        w.writeheader()
        w.writerows(summary_rows)
        w.writerow({"subject": "OVERALL", "accuracy": overall, "correct": total_c, "total": total_n})


if __name__ == "__main__":
    main()
