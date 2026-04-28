"""Build the 6 WebInstruct training datasets from Prompt 0 outputs.

Inputs:
  - .../prefix_decision/data/01_generations.jsonl
       columns: id, x, y, o, y_star_no_y, y_star_full, y_star_seeded, y_base
  - .../prefix_decision/data/02_verdicts_student.csv
  - .../prefix_decision/data/03_verdicts_gpt4o_mini.csv

Per-conditioning (3 variants), resolve student-judge winner (both-orderings),
build wins/loses splits, downsample to equal size, write jsonl. The 3 WI
internal variants map to the standardized cond_* names used by WildChat:

    no_y    →  cond_xo            (teacher conditioned on (x, o), no y)
    full    →  cond_xyo           (teacher conditioned on (x, y, o))
    seeded  →  cond_xyo_ystart    (teacher conditioned on (x, y, o) + first
                                   7 tokens of y forced into generation)

Output: 6 files in --out_dir, matching what the training launcher expects.
  teacher_wins_cond_xo.jsonl,           teacher_loses_cond_xo.jsonl
  teacher_wins_cond_xyo.jsonl,          teacher_loses_cond_xyo.jsonl
  teacher_wins_cond_xyo_ystart.jsonl,   teacher_loses_cond_xyo_ystart.jsonl
  + downsample_indices.json
  + stats.txt
  + manifest.json    (lists all 6 files with row counts)

Each row schema:
  {id, x, y, o, y_star, y_base, conditioning,
   student_verdict, gpt4o_mini_verdict, agreement}
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# WI internal variant name → standardized conditioning name (matches WC)
VARIANT_TO_COND = {
    "no_y":   "cond_xo",
    "full":   "cond_xyo",
    "seeded": "cond_xyo_ystart",
}


def load_verdicts(path, target_comparison):
    """Return {id: {'AB': verdict, 'BA': verdict}} filtered to one comparison."""
    out = defaultdict(dict)
    if not Path(path).exists():
        print(f"warning: {path} not found", file=sys.stderr)
        return out
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["comparison"] != target_comparison:
                continue
            out[row["id"]][row["order"]] = row["verdict"]
    return out


def resolve(verdict_pair):
    if not verdict_pair:
        return None
    ab = verdict_pair.get("AB")
    ba = verdict_pair.get("BA")
    if ab == "A" and ba == "B":
        return "y_star"
    if ab == "B" and ba == "A":
        return "y_base"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens",
        default="experiments-for-apr26-may1/prefix_decision/data/01_generations.jsonl")
    ap.add_argument("--student_csv",
        default="experiments-for-apr26-may1/prefix_decision/data/02_verdicts_student.csv")
    ap.add_argument("--openai_csv",
        default="experiments-for-apr26-may1/prefix_decision/data/03_verdicts_gpt4o_mini.csv")
    ap.add_argument("--out_dir",
        default="experiments-for-apr26-may1/build_datasets/data/webinstruct")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gens once
    gens_rows = []
    with open(args.gens) as f:
        for line in f:
            if line.strip():
                gens_rows.append(json.loads(line))
    print(f"loaded {len(gens_rows)} generations from {args.gens}")

    # Pre-load student/openai verdicts for all 3 comparisons
    student_per_cond = {v: load_verdicts(args.student_csv, f"{v}_vs_base")
                        for v in VARIANT_TO_COND.keys()}
    openai_per_cond  = {v: load_verdicts(args.openai_csv,  f"{v}_vs_base")
                        for v in VARIANT_TO_COND.keys()}
    for v in VARIANT_TO_COND:
        print(f"  {v}: student verdict-pairs={len(student_per_cond[v])}, "
              f"gpt4o={len(openai_per_cond[v])}")

    splits_info = {}
    manifest = {}

    for variant, cond_name in VARIANT_TO_COND.items():
        print(f"\n=== Variant: {variant} → output cond name: {cond_name} ===")
        student = student_per_cond[variant]
        openai_ = openai_per_cond[variant]
        y_star_field = f"y_star_{variant}"

        # Walk gens, build per-row records for THIS variant
        records = []
        n_skip_empty = 0
        n_skip_no_verdict = 0
        for d in gens_rows:
            rid = d["id"]
            y_star = d.get(y_star_field, "")
            y_base = d.get("y_base", d.get("y", ""))
            if not y_star or not y_base:
                n_skip_empty += 1
                continue
            s_winner = resolve(student.get(rid))
            o_winner = resolve(openai_.get(rid))
            if s_winner is None and o_winner is None:
                n_skip_no_verdict += 1
                continue
            agreement = (s_winner is not None and o_winner is not None
                         and s_winner == o_winner)
            records.append({
                "id": rid,
                "x": d["x"], "y": d["y"], "o": d["o"],
                "y_star": y_star,
                "y_base": y_base,
                "conditioning":       cond_name,
                "student_verdict":    s_winner,
                "gpt4o_mini_verdict": o_winner,
                "agreement":          agreement,
            })
        print(f"  records: {len(records)}  (skipped: empty_y_star={n_skip_empty}, "
              f"no_verdict={n_skip_no_verdict})")

        # Counts
        s_counts = defaultdict(int)
        o_counts = defaultdict(int)
        for r in records:
            s_counts[r["student_verdict"]]    += 1
            o_counts[r["gpt4o_mini_verdict"]] += 1
        print(f"  student verdicts:    {dict(s_counts)}")
        print(f"  gpt4o_mini verdicts: {dict(o_counts)}")

        # Agreement
        both = [r for r in records
                if r["student_verdict"] is not None and r["gpt4o_mini_verdict"] is not None]
        agree = sum(1 for r in both if r["agreement"])
        agree_rate = agree / max(1, len(both))
        print(f"  agreement (both decided): {agree}/{len(both)} = {agree_rate:.4f}")

        # Split by student verdict (primary judge for training-set construction)
        wins  = [r for r in records if r["student_verdict"] == "y_star"]
        loses = [r for r in records if r["student_verdict"] == "y_base"]
        n = min(len(wins), len(loses))
        print(f"  raw wins/losses: {len(wins)} / {len(loses)}  → downsample to {n} each")

        # Reproducible sample
        wins_sorted  = sorted(wins,  key=lambda r: r["id"])
        loses_sorted = sorted(loses, key=lambda r: r["id"])
        rng_w = random.Random(args.seed + (10 * list(VARIANT_TO_COND).index(variant)) + 1)
        rng_l = random.Random(args.seed + (10 * list(VARIANT_TO_COND).index(variant)) + 2)
        rng_w.shuffle(wins_sorted)
        rng_l.shuffle(loses_sorted)
        wins_take  = wins_sorted[:n]
        loses_take = loses_sorted[:n]
        # Final on-disk order: by id
        wins_take.sort(key=lambda r: r["id"])
        loses_take.sort(key=lambda r: r["id"])

        wins_path  = out_dir / f"teacher_wins_{cond_name}.jsonl"
        loses_path = out_dir / f"teacher_loses_{cond_name}.jsonl"
        with open(wins_path, "w") as f:
            for r in wins_take:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(loses_path, "w") as f:
            for r in loses_take:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  → {wins_path}")
        print(f"  → {loses_path}")

        splits_info[cond_name] = {
            "wi_internal_variant": variant,
            "raw_wins": len(wins),
            "raw_losses": len(loses),
            "downsampled_n": n,
            "agreement_rate": agree_rate,
            "agreement_decided_pairs": len(both),
            "agreement_count": agree,
            "wins_ids":  [r["id"] for r in wins_take],
            "loses_ids": [r["id"] for r in loses_take],
        }
        manifest[cond_name] = {
            "wins_file":  str(wins_path),
            "loses_file": str(loses_path),
            "n_each":     n,
        }

    # Save indices file (reproducibility)
    indices_path = out_dir / "downsample_indices.json"
    with open(indices_path, "w") as f:
        json.dump({"seed": args.seed,
                   "variant_to_cond": VARIANT_TO_COND,
                   "splits": splits_info}, f, indent=2)
    print(f"\nindices: {indices_path}")

    # Manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}")

    # Stats summary
    lines = ["=== WebInstruct training-dataset build ===", ""]
    lines.append(f"input gens: {args.gens}")
    lines.append(f"input student verdicts: {args.student_csv}")
    lines.append(f"input gpt4o_mini verdicts: {args.openai_csv}")
    lines.append(f"output dir: {out_dir}")
    lines.append("")
    lines.append("Per-conditioning splits (split key = student judge winner):")
    lines.append(f"{'cond':<18} {'wi_internal':<10} {'raw_wins':>8} {'raw_loses':>9} "
                 f"{'downsampled':>11} {'agreement':>10}")
    for cond_name, info in splits_info.items():
        lines.append(f"{cond_name:<18} {info['wi_internal_variant']:<10} "
                     f"{info['raw_wins']:>8} {info['raw_losses']:>9} "
                     f"{info['downsampled_n']:>11} {info['agreement_rate']:>10.4f}")
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n")
    print(f"stats: {out_dir / 'stats.txt'}")


if __name__ == "__main__":
    main()
