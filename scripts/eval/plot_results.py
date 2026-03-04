import matplotlib.pyplot as plt
import numpy as np

# Data
benchmarks = ['Alpaca Eval 2.0', 'ArenaHard-v2', 'ArenaHard-v2']
metrics = ['(LC Winrate)', '(Creative Writing)', '(Hard Prompt)']

# Data points based on user correction and latest logs
# Alpaca: base=19.7, best FKL (baseline_sft)=23.8
# Creative Writing: base=25.7, best FKL (fkl_mid)=23.7
# Hard Prompt: base=2.9, best FKL (baseline_sft)=7.9
base_values = [19.7, 25.7, 2.9]
fkl_values = [23.8, 23.7, 7.9]
gains = [f - b for b, f in zip(base_values, fkl_values)]

# Colors matching the requested styles (light/dark pairs)
# Blue, Red, Green
colors = [
    ('#AED6F1', '#2E86C1'),  # Blue
    ('#F5B7B1', '#C0392B'),  # Red
    ('#ABEBC6', '#27AE60')   # Green
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

width = 0.4  # Slightly wider bars
x = [0, 0.45]  # Positions for the two bars, very close together

for i in range(3):
    ax = axes[i]
    
    # Plot bars
    ax.bar(x[0], base_values[i], width, color=colors[i][0], alpha=0.9)
    ax.bar(x[1], fkl_values[i], width, color=colors[i][1], alpha=0.9)
    
    # Header values (bold)
    y_max = max(base_values[i], fkl_values[i])
    headroom = y_max * 0.15
    ax.set_ylim(0, y_max + headroom)
    
    # Add percentage labels on top (bold)
    ax.text(x[0], base_values[i] + headroom*0.1, f'{base_values[i]:.1f}%', ha='center', fontweight='bold', fontsize=11)
    ax.text(x[1], fkl_values[i] + headroom*0.1, f'{fkl_values[i]:.1f}%', ha='center', fontweight='bold', fontsize=11)
    
    # Add gain label inside the FKL bar (bold, white)
    gain_text = f'{gains[i]:+.1f}%'
    # Put gain in the middle of the FKL bar
    ax.text(x[1], fkl_values[i] / 2, gain_text, ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # Titles (not bold as per user request "dont include bold text in the promopt")
    ax.set_title(f'{benchmarks[i]}\n{metrics[i]}', fontsize=11, pad=10)
    
    # Bottom labels (not bold)
    ax.set_xticks(x)
    ax.set_xticklabels(['Qwen3-4B base', 'FKL-4B'], fontsize=10)
    
    # Y-axis ticks
    ax.set_yticks([0, y_max])
    ax.set_yticklabels(['0', f'{int(y_max)}%'], fontsize=9)
    
    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)

plt.tight_layout()
plt.savefig('/home/ssmurali/user_interactions/results_plot.png', dpi=300, bbox_inches='tight')
plt.close()
