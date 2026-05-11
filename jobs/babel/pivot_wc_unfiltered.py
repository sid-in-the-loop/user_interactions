"""Pivot wildchat_filtered_qwen3_4b_4variants_generations.jsonl into per-condition
training files matching the schema used by training scripts.

Source row has all 3 conditioning variants. We emit two files:
  unfiltered_cond_xyo.jsonl       (target = y_star_cond_xyo)
  unfiltered_cond_xyo_ystart.jsonl (target = y_star_cond_xyo_ystart)
33,920 rows each.

Output schema (matches teacher_*.jsonl):
  example_id, x, y, o, y_star, y_base, conditioning
"""
import json, os
SRC = "/home/ssmurali/demo2teacher_data/wildchat_filtered_qwen3_4b_4variants_generations.jsonl"
OUT = "/home/ssmurali/demo2teacher_data/wildchat_unfiltered"
os.makedirs(OUT, exist_ok=True)

# (cond_label_in_filename, key_for_y_star)
CONDS = [
    ("cond_xyo",        "y_star_cond_xyo"),
    ("cond_xyo_ystart", "y_star_cond_xyo_ystart"),
]

writers = {c: open(f"{OUT}/unfiltered_{c}.jsonl", "w") for c, _ in CONDS}

n = 0
with open(SRC) as fp:
    for line in fp:
        r = json.loads(line)
        for cond, ystar_key in CONDS:
            ys = r.get(ystar_key)
            if ys is None: continue
            row = {
                "example_id":   r["example_id"],
                "x":            r["x"],
                "y":            r["y"],
                "o":            r["o"],
                "y_base":       r["y_base"],
                "y_star":       ys,
                "conditioning": cond,
            }
            writers[cond].write(json.dumps(row) + "\n")
        n += 1

for w in writers.values(): w.close()
for cond, _ in CONDS:
    p = f"{OUT}/unfiltered_{cond}.jsonl"
    cnt = sum(1 for _ in open(p))
    print(f"  {cond:<22}  {cnt:>6} rows  → {p}")
print(f"\nDone. Source rows: {n}")
