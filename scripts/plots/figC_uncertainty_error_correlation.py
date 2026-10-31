"""
Figure C: Grouped bar chart of uncertainty-error correlation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'

properties = ['QED', 'SA', 'LogP', 'Affinity']
corr_ens   = np.array([0.2218, 0.1452, 0.4129, 0.1262])
corr_ale   = np.array([0.5302, 0.1194, 0.5573, 0.0354])
corr_total = np.array([0.5374, 0.1959, 0.5799, 0.1660])

x = np.arange(len(properties))
width = 0.27

fig, ax = plt.subplots(figsize=(11, 5.5))

bars1 = ax.bar(x - width, corr_ens,   width, label=r'corr($\sigma_{ensemble}$, err)',
                color='#2a5784', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x,         corr_ale,   width, label=r'corr($\sigma_{aleatoric}$, err)',
                color='#dd8452', edgecolor='black', linewidth=0.5, alpha=0.85)
bars3 = ax.bar(x + width, corr_total, width, label=r'corr($\sigma_{total}$, err)',
                color='#c44e52', edgecolor='black', linewidth=0.5, alpha=0.85)

for bars in (bars1, bars2, bars3):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.3f}', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8.5)

for i, (e, t) in enumerate(zip(corr_ens, corr_total)):
    if t > e:
        improve = (t - e) / e * 100
        ax.annotate(f'↑{improve:.0f}%',
                    xy=(x[i] + width, t),
                    xytext=(0, 25), textcoords='offset points',
                    ha='center', fontsize=10, color='darkgreen',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(properties, fontsize=12)
ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
ax.set_title('Uncertainty-Error Correlation: Hybrid vs. Single Uncertainty', fontsize=12)
ax.set_ylim(0, max(corr_total.max(), corr_ale.max()) * 1.30)

ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, axis='y', alpha=0.25, linestyle='--')

# === 关键修改：4 边框全显示 ===
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.0)
    spine.set_color('black')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/figC_uncertainty_error_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figC_uncertainty_error_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved figC with full borders")