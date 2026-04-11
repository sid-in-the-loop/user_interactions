#!/usr/bin/env python3
"""
NeurIPS-style evaluation plots — WildFeedback SFT results (Mar 26 2026).
Outputs 7 figures (6 + optional W3 when reanswer data exists) to docs/figures/.

Run:
    python scripts/eval/plot_results_mar26.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUT = Path('/home/ssmurali/user_interactions/docs/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Serif',
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.titleweight':   'bold',
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.frameon':     False,
    'legend.borderpad':   0.2,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.7,
    'axes.grid':          True,
    'axes.axisbelow':     True,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.5,
    'grid.color':         '#888888',
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth':    1.2,
    'patch.linewidth':    0.7,
})

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE       = '#2166ac'   # thinking, 8B
BLUE_LIGHT = '#74add1'   # thinking, 4B  or lighter variant
RED        = '#d6604d'   # nonthinking
RED_LIGHT  = '#f4a582'   # nonthinking 4B
GREEN      = '#4dac26'   # extended training
GRAY       = '#878787'   # base model
GRAY_LIGHT = '#bbbbbb'

# ── Data ──────────────────────────────────────────────────────────────────────

# Alpaca eval LC winrate (%) — gpt-4o-mini judge, all consistent
ALPACA = {
    'base_8b':        25.6,
    'think_best':     45.1,
    'think_best_ext': 34.9,
    'think_full':     31.4,
    'nothink_best':   11.3,
    'nothink_full':    6.7,
    'base_4b':        19.1,
}

# Arena-hard v2.0 win% vs GPT-4.1
ARENA = {
    'base_8b':        7.6,
    'think_best':     8.7,
    'think_best_ext': 11.2,
    'think_full':     9.9,
    'nothink_best':   1.3,
    'nothink_full':   0.7,
}

# WF winrate net% (y* vs GPT-4), n=500, position-bias removed
WF = {
    '8b_think_best':    44.8,
    '8b_think_full':    26.2,
    '8b_nothink_best': -39.9,
    '8b_nothink_full': -44.2,
    '4b_think_best':    40.3,
    '4b_think_full':    13.7,
    '4b_nothink_best': -27.9,
    '4b_nothink_full': -38.9,
}

# Paired ablation comparisons (first minus second, net winrate)
ABLATIONS = {
    'Think vs. No-think': 49.2,
    'Best vs. Full':       0.0,
    '8B vs. 4B':           5.9,
}

# Reanswer hindsight results (filled in when available)
REANSWER_NOTHINK_PATH = Path(
    '/home/ssmurali/user_interactions/data/winrate_results/'
    'qwen3_8b/reanswer_vs_ybase_nothink/winrate_summary.txt'
)
REANSWER_THINK_PATH = Path(
    '/home/ssmurali/user_interactions/data/winrate_results/'
    'qwen3_8b/reanswer_vs_ybase_think/winrate_summary.txt'
)


def parse_net_winrate(path):
    """Read net winrate % from a winrate_summary.txt file."""
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    net = parts[4].replace('%', '').replace('+', '').strip()
                    val = float(net)
                    return val
                except (ValueError, IndexError):
                    continue
    return None


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    print(f'  → {path}')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: "Does WF SFT work?" — three-panel overview
# ════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    models = ['base_8b', 'think_full', 'think_best']
    labels = ['Base\n(Qwen3-8B)', 'WF-Think\n(full)', 'WF-Think\n(best)']
    colors = [GRAY, BLUE_LIGHT, BLUE]

    alpaca_vals = [ALPACA[m] for m in models]
    arena_vals  = [ARENA[m]  for m in models]
    wf_vals     = [WF[f'8b_{m}'] if m != 'base_8b' else 0.0 for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.38)

    panels = [
        (axes[0], alpaca_vals, 'Alpaca Eval LC Win Rate (%)', 55),
        (axes[1], arena_vals,  'Arena-Hard Win Rate (%)',     14),
        (axes[2], wf_vals,     'WF Net Win Rate (%)',         55),
    ]

    x = np.arange(len(models))
    bw = 0.52

    for ax, vals, ylabel, ylim in panels:
        bars = ax.bar(x, vals, width=bw, color=colors,
                      edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(bottom=0, top=ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.grid(axis='y', zorder=0)
        ax.set_axisbelow(True)
        # value labels
        for bar, v in zip(bars, vals):
            if v != 0:
                va = 'bottom' if v >= 0 else 'top'
                offset = 0.8 if v >= 0 else -0.8
                ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                        f'{v:.1f}', ha='center', va=va, fontsize=7, color='#333333')

    axes[0].set_title('(a)', loc='left', fontsize=8, color='#555555')
    axes[1].set_title('(b)', loc='left', fontsize=8, color='#555555')
    axes[2].set_title('(c)', loc='left', fontsize=8, color='#555555')

    save(fig, 'fig1_overview.pdf')
    save(fig, 'fig1_overview.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: "Thinking mode is decisive" — 4 conditions, WF winrate
# ════════════════════════════════════════════════════════════════════════════
def fig2_think_vs_nothink():
    labels = ['Think\n(best)', 'Think\n(full)', 'No-think\n(best)', 'No-think\n(full)']
    vals   = [WF['8b_think_best'], WF['8b_think_full'],
              WF['8b_nothink_best'], WF['8b_nothink_full']]
    colors = [BLUE, BLUE_LIGHT, RED, RED_LIGHT]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(labels))
    bw = 0.55

    bars = ax.bar(x, vals, width=bw, color=colors,
                  edgecolor='white', linewidth=0.5, zorder=3)
    ax.axhline(0, color='#444444', linewidth=0.9, linestyle='--', zorder=2)

    for bar, v in zip(bars, vals):
        va = 'bottom' if v >= 0 else 'top'
        offset = 1.2 if v >= 0 else -1.2
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f'{v:+.1f}%', ha='center', va=va, fontsize=7.5, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Net Win Rate vs. GPT-4 (%)', labelpad=4)
    ax.set_ylim(-55, 58)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    legend_handles = [
        mpatches.Patch(facecolor=BLUE,      label='Thinking (8B)'),
        mpatches.Patch(facecolor=RED,       label='No-think (8B)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7.5)
    ax.grid(axis='y', zorder=0)

    save(fig, 'fig2_think_vs_nothink.pdf')
    save(fig, 'fig2_think_vs_nothink.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: "Best vs. full — benchmark trade-off"
# ════════════════════════════════════════════════════════════════════════════
def fig3_best_vs_full():
    models  = ['base_8b', 'think_full', 'think_best', 'think_best_ext']
    labels  = ['Base', 'Full', 'Best', 'Best+Ext']
    colors  = [GRAY, BLUE_LIGHT, BLUE, GREEN]
    alpaca  = [ALPACA[m] for m in models]
    arena   = [ARENA[m]  for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(wspace=0.38)

    x  = np.arange(len(models))
    bw = 0.52

    for ax, vals, ylabel, ylim in [
        (axes[0], alpaca, 'Alpaca Eval LC Win Rate (%)', 52),
        (axes[1], arena,  'Arena-Hard Win Rate (%)',     13),
    ]:
        bars = ax.bar(x, vals, width=bw, color=colors,
                      edgecolor='white', linewidth=0.5, zorder=3)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10 if ylabel.startswith('Alpaca') else 5))
        ax.grid(axis='y', zorder=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#333333')

    axes[0].set_title('(a)', loc='left', fontsize=8, color='#555555')
    axes[1].set_title('(b)', loc='left', fontsize=8, color='#555555')

    legend_handles = [
        mpatches.Patch(facecolor=GRAY,       label='Base'),
        mpatches.Patch(facecolor=BLUE_LIGHT, label='Think / Full'),
        mpatches.Patch(facecolor=BLUE,       label='Think / Best'),
        mpatches.Patch(facecolor=GREEN,      label='Think / Best+Ext'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.08),
               frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

    save(fig, 'fig3_best_vs_full.pdf')
    save(fig, 'fig3_best_vs_full.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure 4: "Scale: 4B vs. 8B" — WF winrate, think conditions
# ════════════════════════════════════════════════════════════════════════════
def fig4_scale():
    # grouped bars: best / full   × 4B / 8B
    groups  = ['Best', 'Full']
    vals_8b = [WF['8b_think_best'], WF['8b_think_full']]
    vals_4b = [WF['4b_think_best'], WF['4b_think_full']]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x  = np.arange(len(groups))
    bw = 0.32
    offset = bw / 2 + 0.03

    b8 = ax.bar(x - offset, vals_8b, width=bw, color=BLUE,
                edgecolor='white', linewidth=0.5, label='8B', zorder=3)
    b4 = ax.bar(x + offset, vals_4b, width=bw, color=BLUE_LIGHT,
                edgecolor='white', linewidth=0.5, label='4B', zorder=3)

    ax.axhline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=2)

    for bars in [b8, b4]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                    f'{v:+.1f}%', ha='center', va='bottom', fontsize=7.5, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel('Net Win Rate vs. GPT-4 (%)', labelpad=4)
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', zorder=0)

    save(fig, 'fig4_scale.pdf')
    save(fig, 'fig4_scale.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure W1: "All conditions — full WF winrate matrix"
# ════════════════════════════════════════════════════════════════════════════
def figW1_all_conditions():
    entries = [
        ('8B Think Best',    WF['8b_think_best'],    BLUE,       'o'),
        ('8B Think Full',    WF['8b_think_full'],    BLUE_LIGHT, 'o'),
        ('4B Think Best',    WF['4b_think_best'],    BLUE,       's'),
        ('4B Think Full',    WF['4b_think_full'],    BLUE_LIGHT, 's'),
        ('8B No-think Best', WF['8b_nothink_best'],  RED,        'o'),
        ('8B No-think Full', WF['8b_nothink_full'],  RED_LIGHT,  'o'),
        ('4B No-think Best', WF['4b_nothink_best'],  RED,        's'),
        ('4B No-think Full', WF['4b_nothink_full'],  RED_LIGHT,  's'),
    ]
    # sort descending
    entries.sort(key=lambda e: e[1], reverse=True)

    labels = [e[0] for e in entries]
    vals   = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    marks  = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    y = np.arange(len(entries))

    ax.axvline(0, color='#444444', linewidth=0.9, linestyle='--', zorder=2)

    for i, (v, c, m) in enumerate(zip(vals, colors, marks)):
        ax.plot([0, v], [i, i], color=c, linewidth=1.5, zorder=3, alpha=0.8)
        ax.scatter([v], [i], color=c, marker=m, s=55, zorder=4,
                   edgecolors='white', linewidths=0.5)
        ax.text(v + (1.5 if v >= 0 else -1.5), i,
                f'{v:+.1f}%', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=7, color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Net Win Rate vs. GPT-4 (%)', labelpad=4)
    ax.set_xlim(-60, 62)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis='x', zorder=0)
    ax.invert_yaxis()

    legend_handles = [
        plt.Line2D([0], [0], color=BLUE,      marker='o', linewidth=1.5, markersize=5, label='Think, 8B'),
        plt.Line2D([0], [0], color=BLUE,      marker='s', linewidth=1.5, markersize=5, label='Think, 4B'),
        plt.Line2D([0], [0], color=BLUE_LIGHT,marker='o', linewidth=1.5, markersize=5, label='Think/Full, 8B'),
        plt.Line2D([0], [0], color=RED,        marker='o', linewidth=1.5, markersize=5, label='No-think, 8B'),
        plt.Line2D([0], [0], color=RED,        marker='s', linewidth=1.5, markersize=5, label='No-think, 4B'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='lower right',
              handlelength=1.8)

    save(fig, 'figW1_all_conditions.pdf')
    save(fig, 'figW1_all_conditions.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure W2: "What drives the gain?" — ablation decomposition
# ════════════════════════════════════════════════════════════════════════════
def figW2_ablations():
    labels = list(ABLATIONS.keys())
    vals   = list(ABLATIONS.values())
    colors = [BLUE if v > 5 else (GREEN if v > 0 else GRAY) for v in vals]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y  = np.arange(len(labels))
    bw = 0.45

    bars = ax.barh(y, vals, height=bw, color=colors,
                   edgecolor='white', linewidth=0.5, zorder=3)
    ax.axvline(0, color='#444444', linewidth=0.8, linestyle='--', zorder=2)

    for bar, v in zip(bars, vals):
        offset = 0.8 if v >= 0 else -0.8
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f'{v:+.1f}%', va='center', ha=ha,
                fontsize=8, color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Net Win Rate Difference (%)', labelpad=4)
    ax.set_xlim(-10, 62)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.grid(axis='x', zorder=0)
    ax.invert_yaxis()

    save(fig, 'figW2_ablations.pdf')
    save(fig, 'figW2_ablations.png')


# ════════════════════════════════════════════════════════════════════════════
# Figure W3: "Does hindsight help at inference?" — reanswer vs y_base
# (only rendered when result files exist)
# ════════════════════════════════════════════════════════════════════════════
def figW3_reanswer():
    if not (REANSWER_NOTHINK_PATH.exists() and REANSWER_THINK_PATH.exists()):
        print('  ⚠  Reanswer results not yet available — skipping W3')
        return

    net_nothink = parse_net_winrate(REANSWER_NOTHINK_PATH)
    net_think   = parse_net_winrate(REANSWER_THINK_PATH)

    if net_nothink is None or net_think is None:
        print('  ⚠  Could not parse reanswer results — skipping W3')
        return

    # Also include the old (flawed) comparisons for context
    labels = [
        'No-think\n(old prompt)',
        'No-think\n(re-answer x)',
        'Think\n(old prompt)',
        'Think\n(re-answer x)',
    ]
    vals   = [
        WF.get('8b_nothink_ystar_vs_ybase', -18.9),   # old
        net_nothink,                                    # new
        WF.get('8b_think_ystar_vs_ybase',   -54.7),   # old
        net_think,                                      # new
    ]
    colors = [RED_LIGHT, RED, BLUE_LIGHT, BLUE]
    hatches = ['///', '', '///', '']

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x  = np.arange(len(labels))
    bw = 0.52

    bars = ax.bar(x, vals, width=bw, color=colors,
                  hatch=hatches, edgecolor='white', linewidth=0.5, zorder=3)
    ax.axhline(0, color='#444444', linewidth=0.9, linestyle='--', zorder=2)

    for bar, v in zip(bars, vals):
        va = 'bottom' if v >= 0 else 'top'
        offset = 0.8 if v >= 0 else -0.8
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f'{v:+.1f}%', ha='center', va=va, fontsize=7.5, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Net Win Rate vs. y_base (%)', labelpad=4)
    ax.grid(axis='y', zorder=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    legend_handles = [
        mpatches.Patch(facecolor=GRAY_LIGHT, hatch='///', label='Old prompt (responds to o)'),
        mpatches.Patch(facecolor=GRAY_LIGHT, hatch='',    label='Re-answer x with hint'),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc='upper right')

    save(fig, 'figW3_reanswer.pdf')
    save(fig, 'figW3_reanswer.png')


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Rendering figures...')
    fig1_overview()
    fig2_think_vs_nothink()
    fig3_best_vs_full()
    fig4_scale()
    figW1_all_conditions()
    figW2_ablations()
    figW3_reanswer()
    print(f'\nDone. Figures saved to {OUT}')
