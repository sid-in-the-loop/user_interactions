"""NeurIPS-quality figure generation. Five plots, dummy data.

Each plot's data lives in a small block at the top of its function so you can
swap real numbers in later without touching style code.

Outputs land in ./plots/ relative to this file (both .pdf and .png at 300 dpi).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator


# ─── Output directory ────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Global rcParams ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           12,
    "savefig.dpi":         300,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "axes.labelsize":      14,
    "legend.fontsize":     11,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "lightgrey",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})


# ─── Palette ─────────────────────────────────────────────────────────────────

NAVY        = "#1a3a6b"
CYAN        = "#00b4d8"
TEAL        = "#2ec4b6"
STEEL_BLUE  = "#4a90d9"
LIGHT_CYAN  = "#90e0ef"
MUTED_TEAL  = "#52b788"
LIGHT_GREY  = "#adb5bd"
RED         = "#e63946"
DARK_TEAL   = "#177b6e"  # darker variant for "Demonstrator (y)"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="grey")
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)


def save_fig(fig, basename):
    pdf_path = OUT_DIR / f"{basename}.pdf"
    png_path = OUT_DIR / f"{basename}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


# ─── Plot 1: Judge agreement density ─────────────────────────────────────────

def plot1_judge_agreement():
    rng = np.random.default_rng(0)

    # Hexbin data: 5000 points clustered along the diagonal
    diag = rng.uniform(0.0, 1.0, 5000)
    nx = rng.normal(0.0, 0.12, 5000)
    ny = rng.normal(0.0, 0.12, 5000)
    x = np.clip(diag + nx, 0.0, 1.0)
    y = np.clip(diag + ny, 0.0, 1.0)

    # 200 triangle scatter points (60% red = math_verify correct, 40% blue)
    n_tri = 200
    tri_x = np.clip(rng.normal(0.78, 0.10, n_tri), 0.0, 1.0)
    tri_y = np.clip(rng.normal(0.78, 0.10, n_tri), 0.0, 1.0)
    correct = rng.random(n_tri) < 0.6

    fig, ax = plt.subplots(figsize=(10, 8))
    style_axes(ax)

    hb = ax.hexbin(x, y, gridsize=40, cmap="Blues", mincnt=1, norm=LogNorm())
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("density", fontsize=12)
    cb.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=6))

    # Diagonal y=x
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1.0, alpha=0.7,
            zorder=2)
    ax.text(0.55, 0.55, "perfect agreement", color="grey", fontsize=10,
            style="italic", rotation=45, ha="center", va="bottom", zorder=3)

    # Crosshairs at 0.5
    ax.axhline(0.5, linestyle="--", color="lightgrey", lw=0.8, alpha=0.7,
               zorder=1)
    ax.axvline(0.5, linestyle="--", color="lightgrey", lw=0.8, alpha=0.7,
               zorder=1)

    # Quadrant labels
    qstyle = dict(color="grey", fontsize=9, style="italic", ha="center",
                  va="center", zorder=3)
    ax.text(0.78, 0.965, "Both prefer $y^*$", **qstyle)
    ax.text(0.22, 0.04,  "Both prefer $y$", **qstyle)
    ax.text(0.22, 0.965,
            "Student prefers $y^*$,\nGPT-4o-mini prefers $y$", **qstyle)
    ax.text(0.78, 0.04,
            "GPT-4o-mini prefers $y^*$,\nStudent prefers $y$", **qstyle)

    # Triangles
    ax.scatter(tri_x[correct], tri_y[correct],
               marker="^", s=60, c=RED, alpha=0.7, zorder=5,
               edgecolor="white", linewidth=0.5,
               label="WebInstruct: math_verify correct")
    ax.scatter(tri_x[~correct], tri_y[~correct],
               marker="^", s=60, c=NAVY, alpha=0.7, zorder=5,
               edgecolor="white", linewidth=0.5,
               label="WebInstruct: math_verify incorrect")

    ax.set_xlabel(r"GPT-4o-mini preference for $y^*$")
    ax.set_ylabel(r"Student preference for $y^*$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="lightgrey")

    fig.text(0.5, 0.01,
             "Figure 1: Joint preference density across student and GPT-4o-mini "
             "judges. WebInstruct triangles colored by math_verify outcome.",
             ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "plot1_judge_agreement")


# ─── Plot 2: Teacher quality vs KL-to-base plane ─────────────────────────────

def plot2_teacher_kl_quality():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    style_axes(ax)

    # Iso-quality contour background
    X = np.linspace(0.0, 1.4, 240)
    Y = np.linspace(0.3, 0.9, 200)
    XX, YY = np.meshgrid(X, Y)
    Z = YY / (1.0 + np.exp(3.0 * (XX - 0.6)))
    cmap = mpl.cm.get_cmap("Blues_r") if hasattr(mpl.cm, "get_cmap") else plt.colormaps["Blues_r"]
    ax.contourf(XX, YY, Z, levels=10, cmap="Blues_r", alpha=0.6)

    # Anchor points: (label, x, y, color, size)
    pts = [
        ("Base Policy",                 0.02, 0.30, LIGHT_GREY,  120),
        ("Demonstrator ($y$)",          1.22, 0.82, DARK_TEAL,   180),
        ("Teacher: $(x, o)$",           0.75, 0.71, STEEL_BLUE,  150),
        ("Teacher: $(x, y, o)$",        0.42, 0.74, CYAN,        150),
        ("Teacher: $(x, y, o, y^*[:5])$", 0.27, 0.80, LIGHT_CYAN, 150),
    ]
    label_offsets = [(8, -16), (8, 8), (8, 8), (8, 8), (-8, 10)]
    for (label, px, py, color, size), (dx, dy) in zip(pts, label_offsets):
        ax.scatter(px, py, s=size, c=color, edgecolor="black", lw=0.8,
                   zorder=6)
        ax.annotate(label, xy=(px, py), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11, zorder=7)

    # Curved arrows: Demonstrator → (x,o) → (x,y,o) → (x,y,o,y*[:5])
    chain = [
        ((1.22, 0.82), (0.75, 0.71)),
        ((0.75, 0.71), (0.42, 0.74)),
        ((0.42, 0.74), (0.27, 0.80)),
    ]
    for (xt, yt), (xh, yh) in chain:
        ax.annotate("", xy=(xh, yh), xytext=(xt, yt),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="white",
                                    connectionstyle="arc3,rad=-0.2"),
                    zorder=4)
    ax.text(0.7, 0.86, "stronger conditioning", color="white",
            fontsize=11, style="italic", ha="center", va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc=NAVY, alpha=0.65,
                      ec="none"))

    ax.set_xlabel(r"$\mathrm{KL}(\pi_{\mathrm{teacher}} \,\|\, \pi_{\mathrm{base}})$")
    ax.set_ylabel(r"Winrate of $y^*$ over $y_{\mathrm{base}}$")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0.3, 0.9)

    fig.text(0.5, 0.01,
             "Figure 2: Teacher quality in the (KL, winrate) plane. "
             "Stronger conditioning lowers KL while preserving winrate.",
             ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "plot2_teacher_kl_quality")


# ─── Plot 3: Checkpoint trajectory in policy space ───────────────────────────

def plot3_checkpoint_trajectory():
    fig, ax = plt.subplots(figsize=(11, 8))
    style_axes(ax)

    # Concentric Gaussian background centered at (0.05, 0.46)
    X = np.linspace(0.0, 1.8, 240)
    Y = np.linspace(0.30, 0.80, 200)
    XX, YY = np.meshgrid(X, Y)
    cx, cy = 0.05, 0.46
    sx, sy = 0.55, 0.18
    Z = np.exp(-(((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))
    ax.contourf(XX, YY, Z, levels=6, cmap="Blues", alpha=0.3)

    # Manual iso-KL labels (decorative; not exact contour values)
    iso_labels = [(0.30, 0.66, "0.3"), (0.60, 0.66, "0.6"),
                  (0.90, 0.66, "0.9"), (1.20, 0.66, "1.2"),
                  (1.50, 0.66, "1.5")]
    for px, py, lbl in iso_labels:
        ax.text(px, py, lbl, color="grey", fontsize=9, style="italic",
                ha="center", va="center")

    # Anchor points
    ax.scatter(0.04, 0.46, s=120, c=LIGHT_GREY, edgecolor="black", lw=0.8,
               zorder=7)
    ax.annotate("Base Policy", xy=(0.04, 0.46), xytext=(8, -16),
                textcoords="offset points", fontsize=11, zorder=8)
    ax.scatter(0.38, 0.76, s=140, c=TEAL, edgecolor="black", lw=0.8, zorder=7)
    ax.annotate("Teacher", xy=(0.38, 0.76), xytext=(10, 6),
                textcoords="offset points", fontsize=11, zorder=8)

    # Trajectories: (label, points, color, linestyle)
    trajs = [
        ("teacher_wins + FKL",
         [(0.05, 0.46), (0.12, 0.54), (0.21, 0.60), (0.28, 0.66),
          (0.33, 0.69), (0.38, 0.73)],
         NAVY, "solid"),
        ("teacher_wins + SFT",
         [(0.05, 0.46), (0.22, 0.50), (0.45, 0.55), (0.65, 0.59),
          (0.82, 0.62), (1.05, 0.65)],
         CYAN, "solid"),
        ("teacher_loses + FKL",
         [(0.05, 0.46), (0.18, 0.46), (0.35, 0.46), (0.52, 0.47),
          (0.70, 0.47), (0.88, 0.47)],
         RED, "solid"),
        ("teacher_loses + SFT",
         [(0.05, 0.46), (0.28, 0.44), (0.55, 0.43), (0.85, 0.43),
          (1.15, 0.43), (1.55, 0.42)],
         LIGHT_GREY, "dashed"),
    ]
    for label, points, color, ls in trajs:
        # arrows between consecutive
        for i in range(len(points) - 1):
            (x1, y1), (x2, y2) = points[i], points[i + 1]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", lw=2.0, color=color,
                                        linestyle=ls), zorder=4)
        # dots + numbers
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=55, c=color, edgecolor="white", lw=0.8,
                   zorder=6, label=label)
        for i, (px, py) in enumerate(points, start=1):
            ax.annotate(str(i), xy=(px, py), xytext=(0, 9),
                        textcoords="offset points", fontsize=9, ha="center",
                        color=color)

    ax.set_xlabel(r"$\mathrm{KL}(\pi_{\mathrm{ckpt}} \,\|\, \pi_{\mathrm{base}})$")
    ax.set_ylabel("Task performance")
    ax.set_xlim(0, 1.8)
    ax.set_ylim(0.30, 0.80)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="lightgrey")

    fig.text(0.5, 0.01,
             "Figure 3: Per-checkpoint trajectories in the (KL, performance) "
             "plane. Numbers index training steps 1–6.",
             ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "plot3_checkpoint_trajectory")


# ─── Plot 4: Displacement arrows in (mean, variance) space ───────────────────

def plot4_displacement_arrows():
    fig, ax = plt.subplots(figsize=(11, 8))
    style_axes(ax)

    ax.axvline(0.0, linestyle="--", color="lightgrey", lw=0.8, zorder=1)
    ax.axhline(0.0, linestyle="--", color="lightgrey", lw=0.8, zorder=1)

    # Each entry: (label, marker, color, family, tail, head)
    arrows = [
        ("WI cond_xyo wins",          "o", NAVY,       "WI", (0.88,  0.79), (0.26, 0.28)),
        ("WC cond_xyo wins",          "^", STEEL_BLUE, "WC", (0.76,  0.71), (0.24, 0.26)),
        ("WI cond_xyo_ystart wins",   "o", CYAN,       "WI", (1.06,  0.54), (0.18, 0.20)),
        ("WC cond_xyo_ystart wins",   "^", LIGHT_CYAN, "WC", (0.93,  0.49), (0.17, 0.19)),
        ("WI cond_xo wins",           "o", TEAL,       "WI", (0.37,  0.39), (0.13, 0.18)),
        ("WC cond_xo wins",           "^", MUTED_TEAL, "WC", (0.31,  0.35), (0.12, 0.17)),
        ("teacher_loses (all)",       "o", LIGHT_GREY, "WI", (0.05,  0.06), (0.04, 0.05)),
        ("SDPO original",             "X", RED,        "X",  (-0.27, 0.33), (-0.10, 0.27)),
    ]
    for label, marker, color, family, (xt, yt), (xh, yh) in arrows:
        ax.annotate("", xy=(xh, yh), xytext=(xt, yt),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=color),
                    zorder=3)
        # Tail (filled)
        ax.scatter(xt, yt, marker=marker, s=110, c=color,
                   edgecolor="black", lw=0.8, zorder=5)
        # Head (open)
        ax.scatter(xh, yh, marker=marker, s=80, facecolor="white",
                   edgecolor=color, lw=1.5, zorder=5)
        # Label near tail
        label_color = "grey" if "loses" in label else color
        label_size = 10 if "loses" in label else 10
        ax.annotate(label, xy=(xt, yt), xytext=(8, 6),
                    textcoords="offset points", fontsize=label_size,
                    color=label_color, zorder=6)

    # Annotation boxes
    box_kw = dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85,
                  ec="lightgrey")
    ax.text(1.1, 0.92, "Long arrows = strong\nteacher signal absorbed",
            fontsize=10, ha="center", va="center", bbox=box_kw, zorder=7)
    ax.text(-0.32, 0.78, "Short arrows =\nno teachable signal",
            fontsize=10, ha="left", va="center", bbox=box_kw, zorder=7)

    ax.set_xlabel("Advantage mean")
    ax.set_ylabel("Advantage variance")
    ax.set_xlim(-0.5, 1.3)
    ax.set_ylim(0.0, 1.0)

    # Legend by marker shape (family)
    shape_handles = [
        Line2D([], [], marker="o", color="grey", linestyle="", markersize=10,
               markeredgecolor="black", markeredgewidth=0.5,
               label="WebInstruct (●)"),
        Line2D([], [], marker="^", color="grey", linestyle="", markersize=10,
               markeredgecolor="black", markeredgewidth=0.5,
               label="WildChat (▲)"),
        Line2D([], [], marker="X", color=RED, linestyle="", markersize=10,
               markeredgecolor="black", markeredgewidth=0.5,
               label="SDPO original (✕)"),
        Line2D([], [], marker="o", color="white", linestyle="", markersize=10,
               markeredgecolor="grey", markeredgewidth=1.2,
               label="open marker = converged (head)"),
    ]
    ax.legend(handles=shape_handles, loc="upper left", framealpha=0.9,
              edgecolor="lightgrey")

    fig.text(0.5, 0.01,
             "Figure 4: Displacement of advantage statistics from "
             "initialization (filled) to convergence (open).",
             ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "plot4_displacement_arrows")


# ─── Plot 5: Bubble chart in (KL, advantage) space ───────────────────────────

def plot5_bubble_chart():
    fig, ax = plt.subplots(figsize=(11, 8))
    style_axes(ax)

    # Reference lines
    ax.axhline(0.0, linestyle="--", color="lightgrey", lw=0.8, zorder=1)
    ax.text(1.45, 0.02, "zero advantage", fontsize=9, style="italic",
            color="grey", ha="right", va="bottom")
    ax.plot([0.0, 1.5], [0.0, 1.5], linestyle="--", color="lightgrey", lw=0.8,
            zorder=1)
    ax.text(1.18, 1.16, "iso-SNR=1", fontsize=9, style="italic",
            color="grey", rotation=45, ha="center")

    # Bubbles: (label, x, y, size, color, conditioning_family)
    bubbles = [
        ("cond_xyo WI wins",          0.68,  0.88,  800, NAVY,        "cond_xyo"),
        ("cond_xyo WC wins",          0.65,  0.76,  700, STEEL_BLUE,  "cond_xyo"),
        ("cond_xyo_ystart WI wins",   0.45,  1.06,  500, CYAN,        "cond_xyo_ystart"),
        ("cond_xyo_ystart WC wins",   0.43,  0.93,  450, LIGHT_CYAN,  "cond_xyo_ystart"),
        ("cond_xo WI wins",           0.95,  0.37,  300, TEAL,        "cond_xo"),
        ("cond_xo WC wins",           0.90,  0.30,  280, MUTED_TEAL,  "cond_xo"),
        ("teacher_loses (all)",       0.52,  0.04,  100, LIGHT_GREY,  "teacher_loses"),
        ("SDPO original",             1.25, -0.32,  150, RED,         "sdpo_original"),
    ]
    label_offsets = {
        "cond_xyo WI wins":          (10, 12),
        "cond_xyo WC wins":          (10, -16),
        "cond_xyo_ystart WI wins":   (-10, 12),
        "cond_xyo_ystart WC wins":   (-10, -22),
        "cond_xo WI wins":           (10, 8),
        "cond_xo WC wins":           (10, -16),
        "teacher_loses (all)":       (10, 8),
        "SDPO original":             (10, -16),
    }
    for label, px, py, size, color, _ in bubbles:
        ax.scatter(px, py, s=size, c=color, edgecolor="black", lw=0.8,
                   alpha=0.85, zorder=4)
        dx, dy = label_offsets.get(label, (10, 8))
        ha = "right" if dx < 0 else "left"
        ax.annotate(label, xy=(px, py), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10, ha=ha,
                    zorder=5)

    # Annotation arrows
    box_kw = dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9,
                  ec="lightgrey")
    ax.annotate("Low KL, high advantage —\nideal teacher",
                xy=(0.45, 1.06), xytext=(0.05, 0.80),
                fontsize=10, ha="left", bbox=box_kw,
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.0),
                zorder=6)
    ax.annotate("High KL, negative advantage —\nSDPO regime",
                xy=(1.25, -0.32), xytext=(0.55, -0.25),
                fontsize=10, ha="left", bbox=box_kw,
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.0),
                zorder=6)

    ax.set_xlabel(r"$\mathrm{KL}(\pi_{\mathrm{teacher}} \,\|\, \pi_{\mathrm{base}})$")
    ax.set_ylabel("Advantage mean at init")
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(-0.4, 1.2)

    # Two-part legend
    color_handles = [
        Line2D([], [], marker="o", color=NAVY,        linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="cond_xyo"),
        Line2D([], [], marker="o", color=CYAN,        linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="cond_xyo_ystart"),
        Line2D([], [], marker="o", color=TEAL,        linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="cond_xo"),
        Line2D([], [], marker="o", color=LIGHT_GREY,  linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="teacher_loses"),
        Line2D([], [], marker="o", color=RED,         linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="SDPO original"),
    ]
    size_handles = [
        Line2D([], [], marker="o", color="grey", linestyle="",
               markersize=6,  markeredgecolor="black", markeredgewidth=0.5,
               label="low variance"),
        Line2D([], [], marker="o", color="grey", linestyle="",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="medium variance"),
        Line2D([], [], marker="o", color="grey", linestyle="",
               markersize=15, markeredgecolor="black", markeredgewidth=0.5,
               label="high variance"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right",
                     framealpha=0.9, edgecolor="lightgrey",
                     title="conditioning")
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, loc="lower right",
              framealpha=0.9, edgecolor="lightgrey",
              title="bubble size = advantage variance")

    fig.text(0.5, 0.01,
             "Figure 5: Teacher quality in the (KL, advantage) plane. "
             "Bubble size encodes advantage variance.",
             ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(fig, "plot5_bubble_chart")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}\n")
    plot1_judge_agreement()
    print()
    plot2_teacher_kl_quality()
    print()
    plot3_checkpoint_trajectory()
    print()
    plot4_displacement_arrows()
    print()
    plot5_bubble_chart()
    print("\nAll five figures saved.")
