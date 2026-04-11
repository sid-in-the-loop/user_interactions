#!/usr/bin/env python3
"""
AlpacaEval judge using gpt-oss-20b (harmony format).
Extracts the logprob of the answer token (m/M) which appears
after <|channel|>final<|message|> in the harmony response format.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) "
    "based on the quality of their responses to a given instruction. This process will be used to create a "
    "leaderboard reflecting the most accurate and human-preferred answers."
)

def make_prompt(instruction, output_1, output_2):
    instr = instruction.replace('"', '\\"')
    o1 = output_1.replace('"', '\\"')
    o2 = output_2.replace('"', '\\"')
    return (
        "I require a leaderboard for various large language models. I'll provide you with prompts given to these"
        " models and their corresponding outputs. Your task is to assess these responses, and select the model"
        " that produces the best output from a human perspective.\n\n"
        "## Instruction\n\n"
        "{\n"
        f'    "instruction": "{instr}",\n'
        "}\n\n"
        "## Model Outputs\n\n"
        "Here are the unordered outputs from the models. Each output is associated with a specific model,"
        " identified by a unique model identifier.\n\n"
        "{\n"
        "    {\n"
        '        "model_identifier": "m",\n'
        f'        "output": "{o1}"\n'
        "    },\n"
        "    {\n"
        '        "model_identifier": "M",\n'
        f'        "output": "{o2}"\n'
        "    }\n"
        "}\n\n"
        "## Task\n\n"
        "Evaluate the models based on the quality and relevance of their outputs, and select the model that"
        " generated the best output. Answer by providing the model identifier of the best model. We will use"
        " your output as the name of the best model, so make sure your output only contains one of the following"
        " model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.\n\n"
        "## Best Model Identifier"
    )


def get_preference(client, model, instruction, output_1, output_2, max_tokens=512, retries=3):
    """
    Returns soft preference in [1, 2]:
      ~1.0 = output_2 (our model) wins
      ~2.0 = output_1 (reference) wins
      None = failed
    """
    prompt = make_prompt(instruction, output_1, output_2)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=1.0,
                logprobs=True,
                top_logprobs=5,
            )
            tokens = resp.choices[0].logprobs.content

            # Find token after <|channel|>final<|message|>
            for i, tok in enumerate(tokens):
                if (tok.token == "<|channel|>"
                        and i + 3 < len(tokens)
                        and tokens[i + 1].token == "final"
                        and tokens[i + 2].token == "<|message|>"):
                    answer_tok = tokens[i + 3]
                    lp = {t.token: t.logprob for t in answer_tok.top_logprobs}
                    lp_m = lp.get("m", -100.0)
                    lp_M = lp.get("M", -100.0)
                    prob_m = math.exp(lp_m)
                    prob_M = math.exp(lp_M)
                    total = prob_m + prob_M
                    if total == 0:
                        return None
                    return 1.0 + prob_M / total

            return None

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed: {e}")
                return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--api_base", default="http://localhost:8004/v1")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    with open(args.model_outputs) as f:
        model_outputs = json.load(f)
    print(f"Loaded {len(model_outputs)} examples")

    # Get reference outputs from existing gpt4_turbo annotations
    model_outputs_dir = Path(args.model_outputs).parent
    for candidate in ["weighted_alpaca_eval_gpt4_turbo_new", "weighted_alpaca_eval_gpt4_turbo"]:
        ref_path = model_outputs_dir / candidate / "annotations.json"
        if ref_path.exists():
            break
    else:
        raise FileNotFoundError("Could not find gpt4_turbo annotations to use as reference outputs.")

    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_map = {r["instruction"]: r["output_1"] for r in ref_data}
    print(f"Loaded {len(ref_map)} reference outputs from {ref_path}")

    pairs = []
    for ex in model_outputs:
        ref_out = ref_map.get(ex["instruction"])
        if ref_out is None:
            continue
        pairs.append({
            "instruction": ex["instruction"],
            "dataset": ex.get("dataset", ""),
            "output_1": ref_out,
            "output_2": ex["output"],
            "generator_1": "gpt4_1106_preview",
            "generator_2": ex.get("generator", "model"),
            "preference": None,
        })
    print(f"Matched {len(pairs)} pairs")

    def annotate(idx_pair):
        idx, pair = idx_pair
        pref = get_preference(
            client, args.model,
            pair["instruction"], pair["output_1"], pair["output_2"],
            max_tokens=args.max_tokens,
        )
        return idx, pref

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(annotate, (i, p)): i for i, p in enumerate(pairs)}
        for fut in tqdm(as_completed(futures), total=len(pairs), desc="Annotating"):
            idx, pref = fut.result()
            pairs[idx]["preference"] = pref

    out_dir = Path(args.output_path) / "weighted_alpaca_eval_gpt_oss_20b"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(pairs, f, indent=2)

    prefs = [p["preference"] for p in pairs]
    valid = [p for p in prefs if p is not None]
    null_count = len(prefs) - len(valid)
    win_rate = sum(1 for p in valid if p < 1.5) / len(valid) * 100 if valid else 0
    avg_len = sum(len(p["output_2"]) for p in pairs) / len(pairs) if pairs else 0

    print(f"\n=== Results ===")
    print(f"Total: {len(pairs)}, Valid: {len(valid)}, Null: {null_count}")
    print(f"Raw win rate (our model vs gpt4_1106): {win_rate:.2f}%")

    generator = pairs[0]["generator_2"] if pairs else "model"
    with open(out_dir / "leaderboard.csv", "w") as f:
        f.write(",win_rate,n_total,avg_length\n")
        f.write(f"{generator},{win_rate:.4f},{len(valid)},{avg_len:.0f}\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
