"""
Figure A2: 4-panel multi-scale graph statistics (publication-quality, framed).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.edgecolor'] = '#222'

NPZ = 'outputs/3ztx_multiscale.npz'
OUT = 'figures/figA2_multiscale_stats4.pdf'

# === LOAD ===
g = dict(np.load(NPZ, allow_pickle=False))
coords = g['coords']
is_lig = g['is_ligand'].astype(bool)

layer_edges = {L: g[f'l{L}_edge_index'].T for L in [1, 2, 3, 4]}
layer_dists = {L: g[f'l{L}_distance'] for L in [1, 2, 3, 4]}
layer_weights = {L: g[f'l{L}_weight'] for L in [1, 2, 3, 4]}
feat = {L: g[f'feat_layer{L}'] for L in [1, 2, 3, 4]}

# Classic ColorBrewer-style palette: distinguishable + print-friendly
COLORS = {
    1: '#4878D0',  # blue (covalent)
    2: '#EE854A',  # orange (h-bond)
    3: '#6ACC64',  # green (hydrophobic)
    4: '#D65F5F',  # red (long-range)
}
LAYER_NAMES = {
    1: 'L1 Covalent',
    2: 'L2 H-bond',
    3: 'L3 Hydrophobic',
    4: 'L4 Long-range',
}
DISTANCE_THRESHOLDS = [2.0, 3.5, 5.0]

# === FIGURE ===
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
((ax_a, ax_b), (ax_c, ax_d)) = axes

def style_frame(ax):
    """Full frame on all 4 sides."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#222')

# ========== (a) Distance distribution ==========
bins = np.linspace(0, 8.5, 90)
for L in [1, 2, 3, 4]:
    d = layer_dists[L]
    if len(d) == 0: continue
    ax_a.hist(d, bins=bins, color=COLORS[L], alpha=0.78,
              edgecolor='white', linewidth=0.3,
              label=f'{LAYER_NAMES[L]} (n={len(d)})')

for th in DISTANCE_THRESHOLDS:
    ax_a.axvline(th, color='#555', ls=':', lw=0.8, alpha=0.5)

ax_a.set_xlabel('Distance (Å)')
ax_a.set_ylabel('Edge count')
ax_a.set_title('(a) Edge Distance Distribution per Layer')
ax_a.legend(loc='upper left', fontsize=9, framealpha=0.95,
            edgecolor='#888')
ax_a.grid(True, axis='y', alpha=0.20, linestyle='--')
ax_a.set_xlim(0, 8.5)
style_frame(ax_a)

# ========== (b) Weight distribution ==========
bins_w = np.linspace(0, 1.05, 50)
for L in [1, 2, 3, 4]:
    w = layer_weights[L]
    if len(w) == 0: continue
    ax_b.hist(w, bins=bins_w, color=COLORS[L], alpha=0.65,
              edgecolor='white', linewidth=0.3,
              label=LAYER_NAMES[L])

ax_b.set_xlabel(r'Interaction weight  $w = \exp(-\Delta E / E_0)$')
ax_b.set_ylabel('Edge count')
ax_b.set_title('(b) Edge Weight Distribution')
ax_b.legend(loc='upper left', fontsize=9, framealpha=0.95,
            edgecolor='#888')
ax_b.grid(True, axis='y', alpha=0.20, linestyle='--')
ax_b.set_xlim(0, 1.05)
style_frame(ax_b)

# ========== (c) Stacked edge counts: L-L / L-P / P-P ==========
counts = {L: {'ll': 0, 'lp': 0, 'pp': 0} for L in [1, 2, 3, 4]}
for L in [1, 2, 3, 4]:
    edges = layer_edges[L]
    if len(edges) == 0: continue
    src_lig = is_lig[edges[:, 0]]
    dst_lig = is_lig[edges[:, 1]]
    counts[L]['ll'] = int((src_lig & dst_lig).sum())
    counts[L]['lp'] = int((src_lig ^ dst_lig).sum())
    counts[L]['pp'] = int((~src_lig & ~dst_lig).sum())

x = np.arange(4)
width = 0.55
ll_vals = np.array([counts[L]['ll'] for L in [1, 2, 3, 4]])
lp_vals = np.array([counts[L]['lp'] for L in [1, 2, 3, 4]])
pp_vals = np.array([counts[L]['pp'] for L in [1, 2, 3, 4]])

ax_c.bar(x, ll_vals, width, color='#4878D0', alpha=0.88,
         edgecolor='#222', linewidth=0.6, label='Ligand-Ligand')
ax_c.bar(x, lp_vals, width, bottom=ll_vals, color='#EE854A', alpha=0.88,
         edgecolor='#222', linewidth=0.6, label='Ligand-Protein')
ax_c.bar(x, pp_vals, width, bottom=ll_vals + lp_vals,
         color='#B0B8C4', alpha=0.88,
         edgecolor='#222', linewidth=0.6, label='Protein-Protein')

totals = ll_vals + lp_vals + pp_vals
for xi, t in zip(x, totals):
    ax_c.annotate(f'{int(t)}', xy=(xi, t), xytext=(0, 4),
                  textcoords='offset points', ha='center',
                  fontsize=10, fontweight='bold')

ax_c.set_xticks(x)
ax_c.set_xticklabels([f'L{L}\n{LAYER_NAMES[L].split()[1]}' for L in [1, 2, 3, 4]],
                     fontsize=10)
ax_c.set_ylabel('Edge count')
ax_c.set_title('(c) Edge Composition by Interaction Type')
ax_c.legend(loc='upper left', fontsize=9, framealpha=0.95,
            edgecolor='#888')
ax_c.grid(True, axis='y', alpha=0.20, linestyle='--')
ax_c.set_ylim(0, totals.max() * 1.15)
style_frame(ax_c)

# ========== (d) Scale-specific node features ==========
LIG_COLOR  = '#4878D0'
PROT_COLOR = '#EE854A'

positions, data_list, colors_list = [], [], []
for i, L in enumerate([1, 2, 3, 4]):
    f = feat[L]
    lig_vals = f[is_lig]
    pro_vals = f[~is_lig]
    if len(lig_vals) > 0:
        positions.append(i * 2.5)
        data_list.append(lig_vals)
        colors_list.append(LIG_COLOR)
    if len(pro_vals) > 0:
        positions.append(i * 2.5 + 0.9)
        data_list.append(pro_vals)
        colors_list.append(PROT_COLOR)

bp = ax_d.boxplot(data_list, positions=positions, widths=0.7,
                  patch_artist=True, showfliers=True,
                  medianprops=dict(color='white', linewidth=1.5),
                  flierprops=dict(marker='o', markersize=3,
                                  markerfacecolor='#888',
                                  markeredgecolor='#888', alpha=0.6),
                  whiskerprops=dict(color='#222', linewidth=0.9),
                  capprops=dict(color='#222', linewidth=0.9),
                  boxprops=dict(linewidth=0.7))
for box, color in zip(bp['boxes'], colors_list):
    box.set(facecolor=color, edgecolor='#222', alpha=0.88, linewidth=0.7)

center_positions = [i * 2.5 + 0.45 for i in range(4)]
labels_d = ['L1\nbond rule', 'L2\nh-bond participation',
            'L3\nhydrophobic', 'L4\npocket env']
ax_d.set_xticks(center_positions)
ax_d.set_xticklabels(labels_d, fontsize=10)
ax_d.set_ylabel('Feature value')
ax_d.set_title('(d) Scale-Specific Node Features (Ligand vs Protein)')
ax_d.grid(True, axis='y', alpha=0.20, linestyle='--')

from matplotlib.patches import Patch
legend_d = [
    Patch(facecolor=LIG_COLOR, edgecolor='#222', alpha=0.88, label='Ligand atoms'),
    Patch(facecolor=PROT_COLOR, edgecolor='#222', alpha=0.88, label='Protein atoms'),
]
ax_d.legend(handles=legend_d, loc='upper right', fontsize=9, framealpha=0.95,
            edgecolor='#888')
style_frame(ax_d)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.savefig(OUT.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUT}")
