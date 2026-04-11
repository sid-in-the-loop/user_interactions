#!/usr/bin/env python3
"""
Fill eval_results/benchmark_manifest.csv from finished benchmark runs.

Reads:
  - Arena-Hard: arena-hard-auto/data/arena-hard-v2.0/model_judgment/<judge>/*.jsonl
    Categories: hard_prompt, creative_writing. Score = bootstrap mean win rate (%), same spirit as show_result.py.
  - AlpacaEval: alpaca_eval_data/results/<model_id>/**/leaderboard.csv (win_rate column)
  - MMLU-Pro: eval_results/mmlu_pro/<model_id>/summary.json (overall_accuracy → %)

IFEval: not automated in this repo — leaves blank; set status note.

Usage:
  python scripts/eval/fill_benchmark_manifest.py
  python scripts/eval/fill_benchmark_manifest.py --manifest path/to.csv --write
  # default is dry-run (prints table); add --write to update CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Need pandas: pip install pandas", file=sys.stderr)
    sys.exit(1)


REPO = Path(__file__).resolve().parents[2]
ARENA_BENCH = "arena-hard-v2.0"
WEIGHT = 3

LABEL_TO_SCORE = {
    "A>B": [1],
    "A>>B": [1] * WEIGHT,
    "A=B": [0.5],
    "A<<B": [0] * WEIGHT,
    "A<B": [0],
    "B>A": [0],
    "B>>A": [0] * WEIGHT,
    "B=A": [0.5],
    "B<<A": [1] * WEIGHT,
    "B<A": [1],
}


def load_arena_judgments(judge_dir: Path) -> pd.DataFrame | None:
    if not judge_dir.is_dir():
        return None
    files = list(judge_dir.glob("*.jsonl"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_json(f, lines=True))
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def arena_category_score(df: pd.DataFrame, model_id: str, category: str) -> float | None:
    sub = df[(df["model"] == model_id) & (df["category"] == category)].reset_index(drop=True)
    if sub.empty:
        return None

    def row_mean(games):
        try:
            g0, g1 = games[0], games[1]
            if g0 is None or g1 is None or g0.get("score") is None or g1.get("score") is None:
                return None
            s1 = LABEL_TO_SCORE.get(g1["score"], [])
            s0 = LABEL_TO_SCORE.get(g0["score"], [])
            if not s1 or not s0:
                return None
            combined = s1 + [1 - x for x in s0]
            return sum(combined) / len(combined)
        except Exception:
            return None

    sub = sub.copy()
    sub["_m"] = sub["games"].map(row_mean)
    sub = sub.dropna(subset=["_m"])
    if sub.empty:
        return None
    # quick bootstrap mean (100 resamples) like show_result
    import numpy as np

    vals = sub["_m"].values
    rng = np.random.default_rng(42)
    boots = [vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(100)]
    return round(float(np.mean(boots)) * 100, 2)


def find_arena_judge_dir() -> Path | None:
    base = REPO / "arena-hard-auto" / "data" / ARENA_BENCH / "model_judgment"
    for name in ("gpt-4o-mini-2024-07-18", "gpt-4o-mini", "gpt-4.1"):
        p = base / name
        if p.is_dir() and list(p.glob("*.jsonl")):
            return p
    return None


def alpaca_win_rate(model_id: str) -> float | None:
    target = REPO / "alpaca_eval_data" / "results" / model_id / "weighted_alpaca_eval_gpt-4o-mini-2024-07-18" / "leaderboard.csv"
    if not target.is_file():
        return None
    with open(target, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None
    for row in rows:
        first_col = row.get("", row.get(next(iter(row), ""), ""))
        if first_col == model_id:
            try:
                return round(float(row["length_controlled_winrate"]), 4)
            except (KeyError, ValueError):
                return None
    try:
        return round(float(rows[0]["length_controlled_winrate"]), 4)
    except (KeyError, ValueError, IndexError):
        return None


def mmlu_overall(model_id: str) -> float | None:
    p = REPO / "eval_results" / "mmlu_pro" / model_id / "summary.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return round(float(d.get("overall_accuracy", 0)) * 100, 2)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "eval_results" / "benchmark_manifest.csv"))
    ap.add_argument("--write", action="store_true", help="Write updated CSV")
    args = ap.parse_args()

    judge_dir = find_arena_judge_dir()
    arena_df = load_arena_judgments(judge_dir) if judge_dir else None
    judge_name = judge_dir.name if judge_dir else "none"

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

    for row in rows:
        mid = row.get("model_id", "").strip()
        if not mid:
            continue
        ah = ac = None
        if arena_df is not None:
            ah = arena_category_score(arena_df, mid, "hard_prompt")
            ac = arena_category_score(arena_df, mid, "creative_writing")
        alp = alpaca_win_rate(mid)
        mmlu = mmlu_overall(mid)

        parts = []
        if ah is not None:
            row["arena_hard_prompt"] = str(ah)
            parts.append("arena")
        if ac is not None:
            row["arena_creative_writing"] = str(ac)
        if alp is not None:
            row["alpaca_eval2_weighted_turbo"] = str(alp)
            parts.append("alpaca")
        if mmlu is not None:
            row["mmlu_pro_5shot_cot"] = str(mmlu)
            parts.append("mmlu")
        # Column name says weighted_turbo; current jobs use gpt-4o-mini — still store win_rate here.
        row["ifeval_prompt_loose"] = row.get("ifeval_prompt_loose") or ""
        if parts:
            row["status"] = "ok:" + "+".join(parts)
        else:
            row["status"] = row.get("status") or "pending"

    # Print summary
    print(f"Arena judge dir: {judge_name} ({judge_dir})")
    for row in rows:
        print(
            row["model_id"],
            row.get("arena_hard_prompt", ""),
            row.get("arena_creative_writing", ""),
            row.get("alpaca_eval2_weighted_turbo", ""),
            row.get("mmlu_pro_5shot_cot", ""),
            row.get("status", ""),
        )

    if args.write:
        with open(args.manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.manifest}")
    else:
        print("\nDry-run only. Re-run with --write to update the CSV.")


if __name__ == "__main__":
    main()
