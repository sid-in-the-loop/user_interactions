"""
2x4 grid: rows = benchmark (alpaca_eval, aime), cols = method (sft, dpo, fkl, rkl).
Each subplot has 6 lines for (dataset, winrate) combos — wildchat w20/w50/w100
and webinstruct w20/w50/w100. X = training progress (step / max_step). Y = score.
JSD skipped (broken).
"""

import json, re, collections, pathlib
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/u/ssredharan/user_interactions/eval_results/tac")
OUT  = pathlib.Path("/u/ssredharan/user_interactions/plots/tac_summary")
OUT.mkdir(parents=True, exist_ok=True)

BENCHES = ["alpaca_eval", "aime", "writingbench", "arena_hard"]
METHODS = ["sft", "dpo", "fkl", "rkl"]
# Winrate filter → student-vs-teacher regime (filter = min teacher winrate):
#   w100 = teacher always won   → student < teacher (hardest for student)
#   w50  = 50/50                → student = teacher
#   w20  = teacher loses often  → student > teacher (easy for student)
SETTINGS = [
    ("wildchat",    100, "#08306b", "wildchat: student < teacher"),
    ("wildchat",    50,  "#4292c6", "wildchat: student = teacher"),
    ("wildchat",    20,  "#9ecae1", "wildchat: student > teacher"),
    ("webinstruct", 100, "#7f2704", "webinstruct: student < teacher"),
    ("webinstruct", 50,  "#f16913", "webinstruct: student = teacher"),
    ("webinstruct", 20,  "#fdae6b", "webinstruct: student > teacher"),
]
BENCH_LABEL = {
    "alpaca_eval":  "AlpacaEval WR (vs GPT-4 turbo) %",
    "aime":         "AIME acc %",
    "writingbench": "WritingBench score (1–10, 200 subset)",
    "arena_hard":   "Arena-Hard WR (vs GPT-4.1) %",
}

# Base Qwen3-4B (no LoRA) reference scores — dashed horizontal line per panel.
BASE_SCORE = {
    "alpaca_eval":  67.07,
    "aime":         18.33,
    "writingbench": 8.03,   # base Qwen3-4B, same 200-query subset + rubric
    "arena_hard":   16.76,  # base Qwen3-4B vs GPT-4.1 (win_rate_no_ties %)
}
PRIMARY = {
    "alpaca_eval":  lambda s: s.get("win_rate_no_ties", s.get("win_rate")),
    "aime":         lambda s: s.get("accuracy"),
    "writingbench": lambda s: s.get("overall_score"),
    "arena_hard":   lambda s: s.get("win_rate_no_ties", s.get("win_rate")),
}

def parse_run(name):
    m = re.match(r"(wildchat|webinstruct)_w(\d+)_n\d+", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)

def step_int(step_name):
    if step_name in ("final", "final_model"): return None  # mark as final
    m = re.search(r"(\d+)", step_name)
    return int(m.group(1)) if m else 0

# R[ds][w][method][bench] = list[(step_int_or_None, score)]
R = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list))))

for p in ROOT.rglob("scores.json"):
    parts = p.relative_to(ROOT).parts
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

# --- Plot ---------------------------------------------------------------------
fig, axes = plt.subplots(len(BENCHES), len(METHODS), figsize=(18, 4*len(BENCHES)), sharey="row")

for r, bench in enumerate(BENCHES):
    for c, method in enumerate(METHODS):
        ax = axes[r][c]
        base = BASE_SCORE.get(bench)
        for ds, w, color, label in SETTINGS:
            pts = R[ds][w][method][bench]
            if not pts: continue
            # Rank-based positioning: K points evenly spaced 1/K .. K/K along (0,1].
            # Prepend (0, base) so every line visually starts from the base-model score.
            steps = sorted(set(s for s,_ in pts if s is not None))
            order = steps + [None]
            by_step = {s: v for s, v in pts}
            ranks = [(i+1)/len(order) for i in range(len(order))]
            xs, ys = ([0.0], [base]) if base is not None else ([], [])
            for s, x in zip(order, ranks):
                if s in by_step:
                    xs.append(x); ys.append(by_step[s])
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=1.8, markersize=5)

        # Base Qwen3-4B reference line (dashed, grey) — no legend entry
        base = BASE_SCORE.get(bench)
        if base is not None:
            ax.axhline(base, linestyle="--", color="black", linewidth=1.2, alpha=0.6)

        ax.set_title(f"{method.upper()}", fontsize=12, weight="bold")
        ax.set_xlim(0, 1.05)
        ax.grid(alpha=0.3)
        if r == len(BENCHES)-1:
            ax.set_xlabel("training progress")
        if c == 0:
            ax.set_ylabel(BENCH_LABEL[bench])
# Shared legend on the right outside the plot area
handles, labels = axes[0][0].get_legend_handles_labels()
fig.tight_layout(rect=[0, 0, 0.83, 1.0])
if handles:
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.84, 0.5),
               fontsize=10, frameon=False)
fig.savefig(OUT / "per_method_dynamics.pdf", bbox_inches="tight")
fig.savefig(OUT / "per_method_dynamics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'per_method_dynamics.pdf'}")
print(f"wrote {OUT/'per_method_dynamics.png'}")
