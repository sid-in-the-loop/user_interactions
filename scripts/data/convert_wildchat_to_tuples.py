"""
Convert wildchat feedback JSONL (rubric-filtered) to the {x, y, o} tuple format
used by generate_ystar_prefix.py and train_lora.py.

Input fields used:
    prompt          – list of {from: human/gpt, value: ...}  (conversation history)
    completion      – {from: gpt, value: ...}                (bad assistant response)
    user_response   – {from: human, value: ...}              (user critique)

Output fields:
    conversation_id, turn_index,
    x  – list of {role: user/assistant, content: ...}
    y  – {role: assistant, content: ...}
    o  – {role: user, content: ...}
    critique_score, dissatisfied_score, improvable_score, avg_log_prob  (kept for filtering)

Usage:
    python scripts/data/convert_wildchat_to_tuples.py \\
        --input  datasets/wildchat/qwen3_4b/wildchat_interactions_1m_feedback_rubric_Qwen-Qwen3-8B_excl_bot0pct_top10000.jsonl \\
        --output datasets/wildchat/tuples_wildchat_qwen3_8b.jsonl
"""

import argparse
import json
from pathlib import Path


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def convert_msg(msg: dict) -> dict:
    return {
        "role":    ROLE_MAP.get(msg["from"], msg["from"]),
        "content": msg["value"],
    }


def convert_record(r: dict) -> dict | None:
    prompt = r.get("prompt")
    completion = r.get("completion")
    user_response = r.get("user_response")

    if not prompt or not completion or not user_response:
        return None
    if not completion.get("value", "").strip():
        return None
    if not user_response.get("value", "").strip():
        return None

    x = [convert_msg(m) for m in prompt]
    y = {"role": "assistant", "content": completion["value"]}
    o = {"role": "user",      "content": user_response["value"]}

    return {
        "conversation_id":    r.get("id", r.get("conversation_id")),
        "turn_index":         r.get("turn_id", 0),
        "x":                  x,
        "y":                  y,
        "o":                  o,
        # keep scores for downstream filtering / curriculum
        "critique_score":     r.get("critique_score"),
        "dissatisfied_score": r.get("dissatisfied_score"),
        "improvable_score":   r.get("improvable_score"),
        "avg_log_prob":       r.get("avg_log_prob"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/wildchat/qwen3_4b/wildchat_interactions_1m_feedback_rubric_Qwen-Qwen3-8B_excl_bot0pct_top10000.jsonl",
    )
    parser.add_argument(
        "--output",
        default="datasets/wildchat/tuples_wildchat_qwen3_8b.jsonl",
    )
    parser.add_argument(
        "--min_critique_score",  type=int, default=0,
        help="Keep only records with critique_score >= N (0 = keep all)",
    )
    parser.add_argument(
        "--min_improvable_score", type=int, default=0,
        help="Keep only records with improvable_score >= N (0 = keep all)",
    )
    args = parser.parse_args()

    records, skipped, filtered = [], 0, 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            rec = convert_record(raw)
            if rec is None:
                skipped += 1
                continue

            if args.min_critique_score  and (rec["critique_score"]  or 0) < args.min_critique_score:
                filtered += 1
                continue
            if args.min_improvable_score and (rec["improvable_score"] or 0) < args.min_improvable_score:
                filtered += 1
                continue

            records.append(rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written : {len(records):,} tuples → {args.output}")
    if skipped:
        print(f"Skipped : {skipped:,} (parse errors / missing fields)")
    if filtered:
        print(f"Filtered: {filtered:,} (below score threshold)")


if __name__ == "__main__":
    main()
