"""
Multiscale Edge Builder (lightweight, for MultiDiff integration)
==============================================================
On-the-fly builds L2/L3/L4 edges from protein_pos + ligand_pos.
L1 (covalent) is already in the original halfedge data, so we skip it.

This module is called inside FeatureComplex.__call__() during data loading.
It uses numpy for CPU-side computation (DataLoader workers are CPU).

Design decisions:
  - L3/L4: skip protein-protein edges (per your decision)
  - L2: keep all (protein backbone H-bonds serve as pocket context)
  - All features are scalars (E(3)-equivariant compatible)
  - Edge weights use empirical energy formulas from build_multiscale_graph.py
"""

import numpy as np
import torch
from rdkit import Chem

# ======================================================================
# Constants (same as build_multiscale_graph.py)
# ======================================================================

HBOND_DIST_IDEAL = 2.9
HBOND_ENERGY_MAX = 25.0
HYDROPHOBIC_ENERGY_MAX = 10.0
COULOMB_ENERGY_MAX = 50.0

# Simplified atomic LogP contribution for hydrophobic scoring
_LOGP_CONTRIB = {6: 0.8, 16: 0.7, 15: 0.5, 17: 0.6, 35: 0.7, 53: 0.8, 9: 0.4, 7: 0.2, 8: 0.1, 1: 0.3}

# H-bond donor/acceptor by atomic number
_DONOR_Z = {7, 8}  # N, O can be donors (if they have H neighbors)
_ACCEPTOR_Z = {7, 8, 9}  # N, O, F can be acceptors


def _is_hbond_donor_simple(z: int, num_hs: int) -> bool:
    """Simplified donor check: N or O with at least one H."""
    return z in _DONOR_Z and num_hs > 0


def _is_hbond_acceptor_simple(z: int) -> bool:
    """Simplified acceptor check: N, O, or F."""
    return z in _ACCEPTOR_Z


def build_multiscale_edges_for_complex(
        protein_pos: np.ndarray,  # [N_pro, 3]
        protein_element: np.ndarray,  # [N_pro] atomic numbers
        ligand_pos: np.ndarray,  # [N_lig, 3]
        ligand_element: np.ndarray,  # [N_lig] atomic numbers
        ligand_num_hs: np.ndarray = None,  # [N_lig] number of H neighbors per ligand atom
) -> dict:
    """
    Build L2/L3/L4 multiscale edges for a protein-ligand complex.

    All returned edge indices are in the COMBINED coordinate system:
      indices 0..N_lig-1 = ligand atoms
      indices N_lig..N_lig+N_pro-1 = protein atoms
    This matches MultiDiff's compose() convention (ligand first, then protein...
    actually MultiDiff does protein first then ligand in compose(), so we need
    to be careful 鈥?see note below).

    NOTE on index convention:
      MultiDiff's compose() puts PROTEIN first, LIGAND second in the combined graph.
      But the edge indices we build here are for ligand-internal and ligand-protein
      edges. We return them in LIGAND-LOCAL indices (0..N_lig-1) for ligand atoms.
      The model.py code will handle the offset when composing.

      For L2 protein-protein edges, we use PROTEIN-LOCAL indices (0..N_pro-1).

    Returns dict with keys:
      l2_edge_index: [2, E2]  (ligand-local for lig atoms, protein-local for pro atoms)
      l2_edge_weight: [E2]
      l2_edge_src_type: [E2]  0=lig-lig, 1=lig-pro, 2=pro-pro
      l3_edge_index, l3_edge_weight, l3_edge_src_type
      l4_edge_index, l4_edge_weight, l4_edge_src_type
    """
    N_lig = len(ligand_pos)
    N_pro = len(protein_pos)

    # Combine positions for distance computation
    all_pos = np.concatenate([ligand_pos, protein_pos], axis=0)  # [N_total, 3]
    all_z = np.concatenate([ligand_element, protein_element], axis=0)
    N = len(all_pos)
    is_lig = np.zeros(N, dtype=bool)
    is_lig[:N_lig] = True

    # Precompute per-atom properties
    logp_score = np.array([_LOGP_CONTRIB.get(int(z), 0.3) for z in all_z])
    is_hydrophobic = logp_score >= 0.5

    # For H-bond: ligand atoms use ligand_num_hs, protein atoms assume potential donors
    # (PDB proteins: backbone N-H is donor, backbone C=O oxygen is acceptor)
    is_donor = np.zeros(N, dtype=bool)
    is_acceptor = np.zeros(N, dtype=bool)

    # Ligand atoms
    if ligand_num_hs is not None:
        for i in range(N_lig):
            is_donor[i] = _is_hbond_donor_simple(int(all_z[i]), int(ligand_num_hs[i]))
            is_acceptor[i] = _is_hbond_acceptor_simple(int(all_z[i]))
    else:
        # Fallback: just use element type
        for i in range(N_lig):
            is_donor[i] = int(all_z[i]) in _DONOR_Z
            is_acceptor[i] = _is_hbond_acceptor_simple(int(all_z[i]))

    # Protein atoms: N is donor, O is acceptor (simplified)
    for i in range(N_lig, N):
        z = int(all_z[i])
        if z == 7:  # N
            is_donor[i] = True
            is_acceptor[i] = True  # N can be both
        elif z == 8:  # O
            is_acceptor[i] = True
        elif z == 9:  # F
            is_acceptor[i] = True

    # Polar atoms for L4
    polar = np.zeros(N, dtype=bool)
    for i in range(N):
        z = int(all_z[i])
        if z in (7, 8):
            polar[i] = True
        elif z in (16, 15):
            polar[i] = True

    # ====== Build edges ======
    # We use a distance matrix approach for efficiency
    # For N < 1000, this is fast enough; for larger systems, use scipy.spatial.KDTree

    # Pairwise distances
    diff = all_pos[:, None, :] - all_pos[None, :, :]  # [N, N, 3]
    dist_mat = np.linalg.norm(diff, axis=-1)  # [N, N]

    results = {}

    # ------ L2: H-bond layer (2.0 < d <= 3.5) ------
    l2_ei, l2_w, l2_st = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = dist_mat[i, j]
            if not (2.0 < d <= 3.5):
                continue
            is_hbond = (is_donor[i] and is_acceptor[j]) or (is_donor[j] and is_acceptor[i])
            if not is_hbond:
                continue
            E = HBOND_ENERGY_MAX * float(np.exp(-((d - HBOND_DIST_IDEAL) ** 2) / (2 * 0.4 ** 2)))
            w = float(np.exp(-(HBOND_ENERGY_MAX - E) / HBOND_ENERGY_MAX))
            # Source type
            if is_lig[i] and is_lig[j]:
                st = 0  # lig-lig
            elif is_lig[i] or is_lig[j]:
                st = 1  # lig-pro
            else:
                st = 2  # pro-pro
            # Bidirectional
            for a, b in [(i, j), (j, i)]:
                l2_ei.append([a, b])
                l2_w.append(w)
                l2_st.append(st)

    results['l2_edge_index'] = np.array(l2_ei, dtype=np.int64).reshape(-1, 2).T if l2_ei else np.zeros((2, 0),
                                                                                                       dtype=np.int64)
    results['l2_edge_weight'] = np.array(l2_w, dtype=np.float32) if l2_w else np.zeros(0, dtype=np.float32)
    results['l2_edge_src_type'] = np.array(l2_st, dtype=np.int64) if l2_st else np.zeros(0, dtype=np.int64)

    # ------ L3: Hydrophobic layer (3.5 < d <= 5.0) ------
    # Skip protein-protein edges
    l3_ei, l3_w, l3_st = [], [], []
    for i in range(N):
        if not is_hydrophobic[i]:
            continue
        for j in range(i + 1, N):
            if not is_hydrophobic[j]:
                continue
            # Skip pro-pro
            if (not is_lig[i]) and (not is_lig[j]):
                continue
            d = dist_mat[i, j]
            if not (3.5 < d <= 5.0):
                continue
            lp = logp_score[i] * logp_score[j]
            E = HYDROPHOBIC_ENERGY_MAX * lp * float(np.exp(-((d - 4.0) ** 2) / (2 * 0.5 ** 2)))
            w = float(np.exp(-(HYDROPHOBIC_ENERGY_MAX - E) / HYDROPHOBIC_ENERGY_MAX))
            if is_lig[i] and is_lig[j]:
                st = 0
            else:
                st = 1
            for a, b in [(i, j), (j, i)]:
                l3_ei.append([a, b])
                l3_w.append(w)
                l3_st.append(st)

    results['l3_edge_index'] = np.array(l3_ei, dtype=np.int64).reshape(-1, 2).T if l3_ei else np.zeros((2, 0),
                                                                                                       dtype=np.int64)
    results['l3_edge_weight'] = np.array(l3_w, dtype=np.float32) if l3_w else np.zeros(0, dtype=np.float32)
    results['l3_edge_src_type'] = np.array(l3_st, dtype=np.int64) if l3_st else np.zeros(0, dtype=np.int64)

    # ------ L4: Long-range layer (5.0 < d <= 8.0) ------
    # Skip protein-protein edges
    l4_ei, l4_w, l4_st = [], [], []
    for i in range(N):
        if not polar[i]:
            continue
        for j in range(i + 1, N):
            if not polar[j]:
                continue
            if (not is_lig[i]) and (not is_lig[j]):
                continue
            d = dist_mat[i, j]
            if not (5.0 < d <= 8.0):
                continue
            # Polar-polar energy
            p_i = 0.6 if int(all_z[i]) in (7, 8) else 0.4
            p_j = 0.6 if int(all_z[j]) in (7, 8) else 0.4
            E = 0.3 * COULOMB_ENERGY_MAX * p_i * p_j / d
            if E < 0.5:
                continue
            w = float(np.exp(-(COULOMB_ENERGY_MAX - E) / COULOMB_ENERGY_MAX))
            if is_lig[i] and is_lig[j]:
                st = 0
            else:
                st = 1
            for a, b in [(i, j), (j, i)]:
                l4_ei.append([a, b])
                l4_w.append(w)
                l4_st.append(st)

    results['l4_edge_index'] = np.array(l4_ei, dtype=np.int64).reshape(-1, 2).T if l4_ei else np.zeros((2, 0),
                                                                                                       dtype=np.int64)
    results['l4_edge_weight'] = np.array(l4_w, dtype=np.float32) if l4_w else np.zeros(0, dtype=np.float32)
    results['l4_edge_src_type'] = np.array(l4_st, dtype=np.int64) if l4_st else np.zeros(0, dtype=np.int64)

    return results