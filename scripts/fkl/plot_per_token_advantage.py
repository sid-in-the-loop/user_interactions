#!/usr/bin/env python3
"""
Plot per-token KL (advantage signal) for one or more samples from a *_signal.jsonl file.
Usage:
  # Single sample:
  python scripts/fkl/plot_per_token_advantage.py --signal_file ... --sample_id unrelated-0
  # Multiple (e.g. 3 unrelated + 3 related):
  python scripts/fkl/plot_per_token_advantage.py --signal_file ... --sample_ids "unrelated-0 unrelated-1 unrelated-2 corrective-0 corrective-1 corrective-2" --output_dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_id(sid_in: str) -> str:
    if sid_in.strip().startswith("related-"):
        return "corrective-" + sid_in.strip().split("-", 1)[1]
    return sid_in.strip()


def _plot_one(samples: list, sample_id: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sid_resolved = _resolve_id(sample_id)
    sample = next((s for s in samples if s.get("id") == sid_resolved), None)
    if sample is None:
        raise SystemExit(f"Sample id '{sample_id}' not found.")
    kls = sample.get("per_token_kl") or []
    if not kls:
        raise SystemExit(f"Sample {sample_id} has empty per_token_kl")
    sid = sample.get("id", "?")
    cat = sample.get("category", "?")

    fig, ax = plt.subplots(figsize=(12, 3))
    x = np.arange(len(kls))
    ax.fill_between(x, 0, kls, alpha=0.6, step="post", color="steelblue")
    ax.step(x, kls, where="post", color="steelblue", linewidth=0.8)
    ax.set_xlabel("Token position $i$")
    ax.set_ylabel("Per-token KL")
    ax.set_title(f"Per-token KL(π(·|x,o,y_{{<i}}) ‖ π(·|x,y_{{<i}})) — {sid} ({cat})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0, len(kls))
    ax.set_ylim(0, None)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-token advantage (KL) for one or more samples.")
    parser.add_argument("--signal_file", type=Path, required=True, help="Path to *_signal.jsonl")
    parser.add_argument("--sample_id", type=str, default=None, help="Single sample id (e.g. unrelated-0)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Space-separated ids (e.g. 'unrelated-0 unrelated-1 corrective-0')")
    parser.add_argument("--sample_idx", type=int, default=None, help="0-based line index (single sample only)")
    parser.add_argument("--output", type=Path, default=None, help="Output plot path (single sample)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for multiple samples")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise SystemExit("pip install matplotlib numpy")

    samples = []
    with open(args.signal_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    if args.sample_ids is not None:
        ids = [s.strip() for s in args.sample_ids.split() if s.strip()]
        out_dir = args.output_dir or (REPO_ROOT / "results")
        for sid in ids:
            out_path = out_dir / f"per_token_{_resolve_id(sid).replace('-', '_')}.png"
            _plot_one(samples, sid, out_path)
        return

    if args.sample_id is not None:
        sid_in = args.sample_id.strip()
        sid_resolved = _resolve_id(sid_in)
        sample = next((s for s in samples if s.get("id") == sid_resolved), None)
        if sample is None:
            raise SystemExit(f"Sample id '{args.sample_id}' not found in {args.signal_file}. (Related samples use id 'corrective-N'.)")
    elif args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= len(samples):
            raise SystemExit(f"sample_idx {args.sample_idx} out of range [0, {len(samples)-1}]")
        sample = samples[args.sample_idx]
    else:
        raise SystemExit("Provide --sample_id, --sample_ids, or --sample_idx")

    out = args.output
    if out is None:
        out = REPO_ROOT / "results" / f"per_token_{sample.get('id', '?').replace('-', '_')}.png"
    _plot_one(samples, sample.get("id", ""), out)


if __name__ == "__main__":
    main()
