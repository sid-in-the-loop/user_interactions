"""Convert mixture jsonl into the two formats our training scripts expect.

Inputs (per mixture file, e.g. data/mixtures/mix_wildchat_teacher_xo_w100.jsonl):
  {id, dataset, x (str), y_star, y_base, verdict, prefix_used, ground_truth, source_o}

Outputs (written to data/training_inputs/):
  <basename>_for_methods.jsonl   # train_methods.py --objective {sft,jsd,dpo}
      {id, x:[msg], y_star_prefix30: <y_star>, y: <y_base>}
      (field names chosen to match train_methods.py's choices= restriction.)

  <basename>_for_sdpo.jsonl      # train_sdpo_lora.py (offline SDPO = RKL)
      {id, prompt:[msg], user_response: {"content": <source_o>},
       completion: {"content": <y_star>}}

No filtering: y_star is always chosen, y_base always rejected, irrespective of
verdict. Verdict is retained as metadata ("verdict" field on the output) for
audit, but not used for sampling here.
"""

import argparse
import json
from pathlib import Path


def wrap_x_as_messages(x_str: str):
    """Our unified schema stores x as a single string (for wildchat,
    multi-turn is already serialized as "Role: content"). Wrap as one
    user-turn message for training scripts that expect list[dict]."""
    return [{"role": "user", "content": x_str}]


def convert_mixture(mix_path: Path, out_dir: Path):
    base = mix_path.stem  # e.g. mix_wildchat_teacher_xo_w100
    out_methods = out_dir / f"{base}_for_methods.jsonl"
    out_sdpo = out_dir / f"{base}_for_sdpo.jsonl"

    n_in, n_out = 0, 0
    with open(mix_path) as fin, \
         open(out_methods, "w") as fm, \
         open(out_sdpo, "w") as fs:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            n_in += 1
            x_str = r.get("x") or ""
            y_star = r.get("y_star") or ""
            y_base = r.get("y_base") or ""
            o = r.get("source_o") or ""
            if not (x_str.strip() and y_star.strip()):
                continue
            # for SFT/JSD we don't need y, but we always include it so the
            # same file feeds DPO too
            if not y_base.strip():
                continue

            x_msgs = wrap_x_as_messages(x_str)

            fm.write(json.dumps({
                "id": r.get("id"),
                "x": x_msgs,
                "y_star_prefix30": y_star,   # train_methods.py field (choices=restricted)
                "y": y_base,
                "verdict": r.get("verdict"),
            }, ensure_ascii=False) + "\n")

            fs.write(json.dumps({
                "id": r.get("id"),
                "prompt": x_msgs,
                "user_response": {"content": o},
                "completion": {"content": y_star},
                "verdict": r.get("verdict"),
            }, ensure_ascii=False) + "\n")

            n_out += 1

    print(f"{base}: in={n_in} out={n_out}  -> {out_methods.name}, {out_sdpo.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixtures_dir",
                    default="experiments/tac_winrates/data/mixtures")
    ap.add_argument("--out_dir",
                    default="experiments/tac_winrates/data/training_inputs")
    ap.add_argument("--pattern", default="mix_*.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixture_paths = sorted(Path(args.mixtures_dir).glob(args.pattern))
    if not mixture_paths:
        print(f"no mixtures found in {args.mixtures_dir} matching {args.pattern}",
              flush=True)
        return

    for p in mixture_paths:
        convert_mixture(p, out_dir)
    print(f"\ntraining inputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
