#!/usr/bin/env python3
"""Bar chart: Qwen3-4B vs Qwen3-8B y* winrates on WildFeedback (NeurIPS style)."""
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
import matplotlib.pyplot as plt
import numpy as np

configs = ["Thinking\nFull", "Thinking\nBest", "Nonthinking\nFull", "Nonthinking\nBest"]

# w/(w+l)
wr_4b = [52.3, 64.3, 21.5, 24.6]
wr_8b = [65.7, 71.5, 22.3, 25.3]

x = np.arange(len(configs))
w = 0.30

fig, ax = plt.subplots(figsize=(6, 3.8))

# Muted academic palette
c4b = "#5B8EC9"   # steel blue
c8b = "#D45B5B"   # muted red

bars_4b = ax.bar(x - w/2, wr_4b, w, label="Qwen3-4B", color=c4b,
                 edgecolor="black", linewidth=0.4, zorder=3)
bars_8b = ax.bar(x + w/2, wr_8b, w, label="Qwen3-8B", color=c8b,
                 edgecolor="black", linewidth=0.4, zorder=3)

# Value labels
for bars in [bars_4b, bars_8b]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.1f}",
                ha="center", va="bottom", fontsize=8.5)

ax.set_ylabel("Win Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.axhline(50, color="grey", linestyle="--", linewidth=0.6, alpha=0.6, zorder=1)
ax.set_ylim(0, 82)
ax.set_xlim(-0.55, len(configs) - 0.45)

# Grid
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Spines
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)

ax.legend(frameon=True, fancybox=False, edgecolor="black",
          framealpha=1.0, loc="upper right")

plt.tight_layout()
out = "data/winrate_results/4b_vs_8b_winrate_wf.png"
fig.savefig(out, dpi=300)
print(f"Saved → {out}")
plt.close()
