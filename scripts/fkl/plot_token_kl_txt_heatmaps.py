#!/usr/bin/env python3
"""
Plot heatmaps from token_kl_txt outputs.

Input layout (produced by measure_fkl_signal.py):
  <token_root>/<model_dir>/<sample_id>.txt

Each txt has:
  sample_id=...
  category=...
  token_index<TAB>token_id<TAB>token_str<TAB>kl[<TAB>logprob_delta]
  ...

Outputs:
  - Per-model heatmap over samples (rows=samples, cols=token positions)
  - Optional per-sample heatmap over models (rows=models, cols=token positions)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


@dataclass
class SampleSeries:
    sample_id: str
    category: str
    values: List[float]


def parse_txt_file(path: Path, metric: str) -> SampleSeries:
    sample_id = ""
    category = ""
    values: List[float] = []
    col_idx = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("sample_id="):
                sample_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("category="):
                category = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if parts and parts[0] == "token_index":
                try:
                    col_idx = parts.index(metric)
                except ValueError:
                    # Backward compatibility with older logs that had only KL.
                    if metric == "kl" and "kl" in parts:
                        col_idx = parts.index("kl")
                    else:
                        raise ValueError(f"{path}: metric '{metric}' not found in header: {parts}")
                continue

            if col_idx is None:
                continue
            if len(parts) <= col_idx:
                continue
            try:
                values.append(float(parts[col_idx]))
            except ValueError:
                continue

    if not sample_id:
        sample_id = path.stem
    return SampleSeries(sample_id=sample_id, category=category, values=values)


def pad_to_grid(series: List[SampleSeries], max_len: int | None = None) -> Tuple[np.ndarray, List[str], List[str]]:
    ids = [s.sample_id for s in series]
    cats = [s.category for s in series]
    observed_max = max((len(s.values) for s in series), default=0)
    width = observed_max if max_len is None else min(max_len, observed_max)
    grid = np.zeros((len(series), width), dtype=np.float32)
    for i, s in enumerate(series):
        arr = np.asarray(s.values, dtype=np.float32)
        L = min(len(arr), width)
        if L > 0:
            grid[i, :L] = arr[:L]
    return grid, ids, cats


def plot_per_model_heatmap(
    model_name: str,
    series: List[SampleSeries],
    metric: str,
    out_file: Path,
    max_len: int | None,
) -> None:
    grid, ids, cats = pad_to_grid(series, max_len=max_len)
    if grid.size == 0:
        return

    fig_h = max(3.5, 0.45 * len(series))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    if metric == "logprob_delta":
        vmax = float(np.percentile(np.abs(grid), 99)) if np.any(grid) else 1.0
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = "RdBu"
    else:
        vmax = float(np.percentile(grid, 99)) if np.any(grid) else 1.0
        vmax = max(vmax, 1e-6)
        norm = None
        cmap = "Blues"

    im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, vmin=None if norm else 0.0, vmax=None if norm else vmax)
    ax.set_title(f"{model_name} - {metric} heatmap")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Sample")
    ax.set_yticks(np.arange(len(ids)))
    y_labels = [f"{sid} ({cat})" if cat else sid for sid, cat in zip(ids, cats)]
    ax.set_yticklabels(y_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(metric)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_cross_model(
    sample_id: str,
    by_model: Dict[str, SampleSeries],
    metric: str,
    out_file: Path,
    max_len: int | None,
) -> None:
    model_names = sorted(by_model.keys())
    ordered = [by_model[m] for m in model_names]
    grid, _, _ = pad_to_grid(ordered, max_len=max_len)
    if grid.size == 0:
        return

    fig_h = max(3.0, 0.45 * len(model_names))
    fig, ax = plt.subplots(figsize=(14, fig_h))

    if metric == "logprob_delta":
        vmax = float(np.percentile(np.abs(grid), 99)) if np.any(grid) else 1.0
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = "RdBu"
    else:
        vmax = float(np.percentile(grid, 99)) if np.any(grid) else 1.0
        vmax = max(vmax, 1e-6)
        norm = None
        cmap = "Blues"

    im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, vmin=None if norm else 0.0, vmax=None if norm else vmax)
    ax.set_title(f"{sample_id} - {metric} across models")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Model")
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(metric)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot heatmaps from token_kl_txt directories.")
    p.add_argument(
        "--token_root",
        type=Path,
        default=Path("results/qual_token_kl_seed42/token_kl_txt"),
        help="Root directory containing per-model txt directories.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/qual_token_kl_seed42/heatmaps"),
        help="Output directory for generated heatmaps.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="logprob_delta",
        choices=["logprob_delta", "kl"],
        help="Column to visualize from txt files.",
    )
    p.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Max token positions to plot (use <=0 for full length).",
    )
    p.add_argument(
        "--skip_cross_model",
        action="store_true",
        help="If set, skip per-sample cross-model heatmaps.",
    )
    args = p.parse_args()

    token_root = args.token_root.resolve()
    out_dir = args.out_dir.resolve()
    max_len = None if args.max_len <= 0 else args.max_len

    if not token_root.exists():
        raise SystemExit(f"token_root does not exist: {token_root}")

    model_dirs = sorted([p for p in token_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise SystemExit(f"No model directories found under {token_root}")

    # model_name -> list[SampleSeries]
    all_series: Dict[str, List[SampleSeries]] = {}
    # sample_id -> model_name -> SampleSeries
    cross: Dict[str, Dict[str, SampleSeries]] = {}

    for model_dir in model_dirs:
        txt_files = sorted(model_dir.glob("*.txt"))
        if not txt_files:
            continue
        model_name = model_dir.name
        series: List[SampleSeries] = []
        for fp in txt_files:
            s = parse_txt_file(fp, metric=args.metric)
            series.append(s)
            cross.setdefault(s.sample_id, {})[model_name] = s
        all_series[model_name] = series

        per_model_out = out_dir / "per_model" / f"{model_name}_{args.metric}.png"
        plot_per_model_heatmap(model_name, series, args.metric, per_model_out, max_len=max_len)
        print(f"Saved {per_model_out}")

    if not args.skip_cross_model:
        for sample_id, by_model in sorted(cross.items()):
            out = out_dir / "per_sample_cross_model" / f"{sample_id}_{args.metric}.png"
            plot_per_sample_cross_model(sample_id, by_model, args.metric, out, max_len=max_len)
            print(f"Saved {out}")


if __name__ == "__main__":
    main()

