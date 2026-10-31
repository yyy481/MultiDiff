"""
Figure B: Stacked bar chart of uncertainty decomposition.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

properties = ['QED', 'SA', 'LogP', 'Affinity']
sigma_ens = np.array([0.0531, 0.2023, 0.3232, 0.3569])
sigma_ale = np.array([0.0757, 0.1515, 0.2462, 0.3829])
sigma_epi = np.array([0.0947, 0.3920, 0.6732, 1.5689])
sigma_total = np.array([0.1368, 0.4752, 0.8024, 1.6638])

variances = np.array([sigma_ens**2, sigma_ale**2, sigma_epi**2])
total_var = variances.sum(axis=0)
contrib_pct = variances / total_var * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(properties))
colors = ['#2a5784', '#dd8452', '#c44e52']
labels = [r'$\sigma_{ensemble}$', r'$\sigma_{aleatoric}$', r'$\sigma_{epistemic}$']

# ============ Left ============
bottom = np.zeros(len(properties))
bar_width = 0.55
for vals, c, lbl in zip([sigma_ens, sigma_ale, sigma_epi], colors, labels):
    ax1.bar(x, vals, bar_width, bottom=bottom, color=c, edgecolor='black',
            linewidth=0.5, alpha=0.85, label=lbl)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.05:
            ax1.text(j, b + v / 2, f'{v:.3f}', ha='center', va='center',
                     fontsize=9, color='white', fontweight='bold')
    bottom += vals

star_x_offset = bar_width / 2 + 0.05
for xi, st in zip(x, sigma_total):
    ax1.scatter(xi + star_x_offset, st, color='gold', s=110, marker='*',
                edgecolor='black', linewidth=1.0, zorder=10)
    ax1.annotate(f'{st:.3f}',
                 xy=(xi + star_x_offset, st),
                 xytext=(10, 0),
                 textcoords='offset points',
                 ha='left', va='center',
                 fontsize=9, color='black', fontweight='bold')

ax1.scatter([], [], color='gold', s=110, marker='*',
            edgecolor='black', linewidth=1.0,
            label=r'$\sigma_{total}$ (quadratic fusion)')

ax1.set_xticks(x)
ax1.set_xticklabels(properties, fontsize=11)
ax1.set_ylabel('Uncertainty (σ)', fontsize=12)
ax1.set_title('(a) Uncertainty Components', fontsize=12)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.grid(True, axis='y', alpha=0.25, linestyle='--')

# === 关键修改：4 边框全显示 ===
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color('black')

ax1.set_ylim(0, max(sigma_total) * 1.18)
ax1.set_xlim(-0.6, len(properties) - 0.1)

# ============ Right ============
bottom = np.zeros(len(properties))
for pct, c, lbl in zip(contrib_pct, colors, labels):
    ax2.bar(x, pct, bar_width, bottom=bottom, color=c, edgecolor='black',
            linewidth=0.5, alpha=0.85, label=lbl)
    for j, (p, b) in enumerate(zip(pct, bottom)):
        if p > 5:
            ax2.text(j, b + p / 2, f'{p:.1f}%', ha='center', va='center',
                     fontsize=10, color='white', fontweight='bold')
    bottom += pct

ax2.set_xticks(x)
ax2.set_xticklabels(properties, fontsize=11)
ax2.set_ylabel('Variance Contribution (%)', fontsize=12)
ax2.set_title('(b) Variance Decomposition', fontsize=12)
ax2.set_ylim(0, 105)
ax2.grid(True, axis='y', alpha=0.25, linestyle='--')

# === 关键修改：4 边框全显示 ===
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color('black')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figB_uncertainty_decomposition.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figB_uncertainty_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Saved figB with full borders")