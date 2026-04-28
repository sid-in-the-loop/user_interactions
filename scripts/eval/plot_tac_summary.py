"""
Produce 4 summary plots for the TAC-winrates matched-N eval:
  A. Winrate-tier sweep per dataset × benchmark (line plot, lines=methods)
  B. Method comparison bar chart per benchmark
  C. Training dynamics: small multiples (run × benchmark)
  D. Delta-vs-SFT per method

Reads scores from eval_results/tac/<run>/<method>/<step>/<bench>/scores.json.
JSD is excluded (broken at LR=1e-4, retrained — awaiting re-eval).
Writes PDFs to plots/tac_summary/.
"""

import json, re, collections, pathlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path("/u/ssredharan/user_interactions/eval_results/tac")
OUT  = pathlib.Path("/u/ssredharan/user_interactions/plots/tac_summary")
OUT.mkdir(parents=True, exist_ok=True)

BENCHES = ["alpaca_eval", "aime"]
METHODS = ["sft", "fkl", "dpo", "rkl"]  # JSD excluded (all 0.0 from LR=1e-4 divergence)
WINRATES = [20, 50, 100]

METHOD_COLOR = {"sft":"#888", "fkl":"#1f77b4", "dpo":"#2ca02c", "rkl":"#d62728"}
BENCH_LABEL = {"alpaca_eval":"AlpacaEval WR (vs GPT-4 turbo) %", "aime":"AIME acc %"}

PRIMARY = {
    "alpaca_eval": lambda s: s.get("win_rate_no_ties", s.get("win_rate")),
    "aime":        lambda s: s.get("accuracy"),
}

def parse_run(name):
    # e.g. wildchat_w100_n8550 -> (wildchat, 100)
    m = re.match(r"(wildchat|webinstruct)_w(\d+)_n\d+", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)

def step_int(step_name):
    if step_name in ("final", "final_model"): return 10**9
    m = re.search(r"(\d+)", step_name)
    return int(m.group(1)) if m else 0

# --- Collect scores -----------------------------------------------------------
# R[dataset][winrate][method][bench] = list[(step_int, score)]
R = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list))))

for p in ROOT.rglob("scores.json"):
    parts = p.relative_to(ROOT).parts  # run, method, step, bench, scores.json
    if len(parts) != 5: continue
    run, method, step, bench, _ = parts
    if method not in METHODS: continue
    if bench not in BENCHES: continue
    ds, w = parse_run(run)
    if ds is None: continue
    with open(p) as f: s = json.load(f)
    v = PRIMARY[bench](s)
    if v is None: continue
    R[ds][w][method][bench].append((step_int(step), float(v)))

def final_score(ds, w, method, bench):
    """Pick 'final' if present, else highest step."""
    pts = sorted(R[ds][w][method][bench])
    if not pts: return None
    # final marker is step_int = 10**9
    for si, v in pts[::-1]:
        if si == 10**9: return v
    return pts[-1][1]

# ============================================================================
# A. Winrate-tier sweep: 2 cols (dataset) × 2 rows (benchmark)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for row, bench in enumerate(BENCHES):
    for col, ds in enumerate(["wildchat", "webinstruct"]):
        ax = axes[row][col]
        for method in METHODS:
            ys = [final_score(ds, w, method, bench) for w in WINRATES]
            xs = [w for w, y in zip(WINRATES, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not ys: continue
            ax.plot(xs, ys, "o-", color=METHOD_COLOR[method], label=method.upper(),
                    linewidth=2, markersize=8)
        ax.set_xticks(WINRATES)
        ax.set_xlabel("winrate filter (w%)")
        ax.set_ylabel(BENCH_LABEL[bench])
        ax.set_title(f"{ds} — {bench}")
        ax.grid(alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(loc="best", fontsize=9)
fig.suptitle("A. Winrate-tier sweep (matched-N) — final-ckpt score", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "A_winrate_sweep.pdf"); fig.savefig(OUT / "A_winrate_sweep.png", dpi=150)
plt.close(fig)
print(f"[A] wrote {OUT/'A_winrate_sweep.pdf'}")

# ============================================================================
# B. Method comparison bars: per benchmark, x = (dataset, winrate), grouped bars
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
settings = [(ds, w) for ds in ["wildchat","webinstruct"] for w in WINRATES]
labels = [f"{ds}\nw{w}" for ds,w in settings]
for ax, bench in zip(axes, BENCHES):
    xs = np.arange(len(settings))
    width = 0.2
    for i, method in enumerate(METHODS):
        ys = [final_score(ds, w, method, bench) or 0 for ds,w in settings]
        ax.bar(xs + (i - 1.5)*width, ys, width,
               label=method.upper(), color=METHOD_COLOR[method])
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(BENCH_LABEL[bench])
    ax.set_title(bench)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=4)
fig.suptitle("B. Method comparison — final-ckpt score per (dataset, winrate)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "B_method_bars.pdf"); fig.savefig(OUT / "B_method_bars.png", dpi=150)
plt.close(fig)
print(f"[B] wrote {OUT/'B_method_bars.pdf'}")

# ============================================================================
# C. Training dynamics: small multiples, row = benchmark, col = run
# ============================================================================
runs = [(ds, w) for ds in ["wildchat","webinstruct"] for w in WINRATES]
fig, axes = plt.subplots(len(BENCHES), len(runs),
                         figsize=(3.2*len(runs), 3.3*len(BENCHES)),
                         sharey="row")
for i, bench in enumerate(BENCHES):
    for j, (ds, w) in enumerate(runs):
        ax = axes[i][j] if len(BENCHES) > 1 else axes[j]
        for method in METHODS:
            pts = sorted(R[ds][w][method][bench])
            if not pts: continue
            # Normalize x: intermediate steps as-is, "final" mapped to max+1
            xs_raw = [s for s,_ in pts]
            max_real = max(s for s in xs_raw if s < 10**9) if any(s<10**9 for s in xs_raw) else 1
            xs = [(max_real*1.1 if s==10**9 else s) for s in xs_raw]
            ys = [v for _, v in pts]
            ax.plot(xs, ys, "o-", color=METHOD_COLOR[method], label=method.upper(),
                    linewidth=1.5, markersize=5)
        ax.set_title(f"{ds} w{w}", fontsize=9)
        ax.set_xlabel("step")
        if j == 0: ax.set_ylabel(BENCH_LABEL[bench], fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(loc="best", fontsize=8)
fig.suptitle("C. Training dynamics per run × benchmark (final ckpt at right)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "C_training_dynamics.pdf"); fig.savefig(OUT / "C_training_dynamics.png", dpi=150)
plt.close(fig)
print(f"[C] wrote {OUT/'C_training_dynamics.pdf'}")

# ============================================================================
# D. Delta-vs-SFT: bars of (method - sft) per (dataset, winrate, benchmark)
# ============================================================================
non_sft = [m for m in METHODS if m != "sft"]
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, bench in zip(axes, BENCHES):
    xs = np.arange(len(settings))
    width = 0.25
    for i, method in enumerate(non_sft):
        deltas = []
        for ds, w in settings:
            s_sft = final_score(ds, w, "sft", bench)
            s_m   = final_score(ds, w, method, bench)
            deltas.append((s_m - s_sft) if (s_sft is not None and s_m is not None) else 0)
        ax.bar(xs + (i - 1)*width, deltas, width,
               label=method.upper(), color=METHOD_COLOR[method])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Δ {bench} vs SFT")
    ax.set_title(f"{bench}: method − SFT")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
fig.suptitle("D. Δ-vs-SFT per method — positive = beats SFT", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "D_delta_vs_sft.pdf"); fig.savefig(OUT / "D_delta_vs_sft.png", dpi=150)
plt.close(fig)
print(f"[D] wrote {OUT/'D_delta_vs_sft.pdf'}")

print(f"\nAll plots in: {OUT}")
