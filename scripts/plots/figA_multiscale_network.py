"""
Figure A: 3-panel network visualization of multi-scale graph on 3ztx.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

mpl.rcParams['font.family'] = 'DejaVu Sans'

NPZ = 'outputs/3ztx_multiscale.npz'   # 用没有 P-P 边的版本
OUT = 'figures/figA_multiscale_network.pdf'

# === LOAD ===
g = dict(np.load(NPZ, allow_pickle=False))
coords = g['coords']
is_lig = g['is_ligand'].astype(bool)
N = coords.shape[0]


def pca_2d(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


pos2d = pca_2d(coords)

layer_edges = {L: g[f'l{L}_edge_index'].T for L in [1, 2, 3, 4]}
for L in [1, 2, 3, 4]:
    print(f'L{L}: {len(layer_edges[L])} edges')

# === Convex hull for pocket boundary ===
prot_pos = pos2d[~is_lig]
try:
    hull = ConvexHull(prot_pos)
    hull_closed = np.vstack([prot_pos[hull.vertices], prot_pos[hull.vertices][0]])
except Exception:
    hull_closed = None

# === Figure ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

PANELS = [
    {'title': '(a) L1 + L2:  Covalent + H-bond',
     'layers': [1, 2], 'colors': ['#1d3557', '#c44e52'], 'lws': [2.2, 1.5]},
    {'title': '(b) L3:  Hydrophobic Contact',
     'layers': [3], 'colors': ['#dd8452'], 'lws': [1.0]},
    {'title': '(c) L4:  Long-range Polar / Electrostatic',
     'layers': [4], 'colors': ['#55a868'], 'lws': [0.8]},
]

LIG_COLOR   = '#1d3557'
PROT_COLOR  = '#7d8ca0'
PROT_EDGE   = '#3d4a5c'
POCKET_FILL = '#e8eaef'
LIG_SIZE    = 50
PROT_SIZE   = 22

for ax, panel in zip(axes, PANELS):
    # 口袋背景
    if hull_closed is not None:
        ax.fill(hull_closed[:, 0], hull_closed[:, 1],
                color=POCKET_FILL, alpha=0.9, zorder=0,
                edgecolor='#b0b8c4', linewidth=0.8, linestyle='--')

    # 边
    for L, color, lw in zip(panel['layers'], panel['colors'], panel['lws']):
        edges = layer_edges[L]
        if len(edges) == 0: continue
        if L in (3, 4):
            mask = is_lig[edges[:, 0]] | is_lig[edges[:, 1]]
            edges = edges[mask]
        if len(edges) > 250:
            idx = np.random.RandomState(42).choice(len(edges), 250, replace=False)
            edges = edges[idx]
        segs = pos2d[edges]
        lc = LineCollection(segs, colors=color, linewidths=lw,
                            alpha=0.55, zorder=1)
        ax.add_collection(lc)

    # 蛋白原子
    ax.scatter(pos2d[~is_lig, 0], pos2d[~is_lig, 1],
               c=PROT_COLOR, s=PROT_SIZE,
               edgecolor=PROT_EDGE, linewidth=0.4,
               alpha=0.7, zorder=2)

    # 配体原子
    ax.scatter(pos2d[is_lig, 0], pos2d[is_lig, 1],
               c=LIG_COLOR, s=LIG_SIZE,
               edgecolor='black', linewidth=0.8,
               alpha=1.0, zorder=3)

    ax.set_title(panel['title'], fontsize=12, pad=12)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.margins(0.05)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=LIG_COLOR,
           markeredgecolor='black', markersize=9, markeredgewidth=0.8,
           label='Ligand atom'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PROT_COLOR,
           markeredgecolor=PROT_EDGE, markersize=7, markeredgewidth=0.4,
           label='Protein atom'),
    Patch(facecolor=POCKET_FILL, edgecolor='#b0b8c4', linewidth=0.8,
          linestyle='--', label='Pocket region'),
    Line2D([0], [0], color='#1d3557', lw=2.2, label='L1 Covalent'),
    Line2D([0], [0], color='#c44e52', lw=1.5, label='L2 H-bond'),
    Line2D([0], [0], color='#dd8452', lw=1.0, label='L3 Hydrophobic'),
    Line2D([0], [0], color='#55a868', lw=0.8, label='L4 Long-range'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=7,
           bbox_to_anchor=(0.5, -0.04), fontsize=10, framealpha=0.95)

plt.suptitle('Multi-scale Molecular Graph on 3ztx Pocket Complex',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

os.makedirs('figures', exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight')
plt.savefig(OUT.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUT}")
