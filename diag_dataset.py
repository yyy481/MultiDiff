"""
Diagnostic: run dataset preprocessing on JUST ONE sample, with full traceback.

Usage:
    cd <project root>
    python diag_dataset.py
"""

import os
import sys
import pickle
import traceback

sys.path.append('.')

# Adjust if your config has different paths
DATA_PATH = 'data/PDBbind_v2020_pocket10'
INDEX_PATH = os.path.join(DATA_PATH, 'index.pkl')

print("=" * 70)
print("DATASET PREPROCESSING DIAGNOSTIC")
print("=" * 70)

# 1. Check that index.pkl exists and is loadable
print(f"\n[1] Loading index from {INDEX_PATH} ...")
if not os.path.exists(INDEX_PATH):
    print(f"  ❌ index.pkl NOT FOUND at {INDEX_PATH}")
    print(f"  → Check config.data.path in your train.yml")
    sys.exit(1)
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
print(f"  ✓ Loaded {len(index)} entries")

# 2. Look at the first non-None entry
print("\n[2] First few index entries:")
for i, entry in enumerate(index[:3]):
    print(f"  Entry {i}: {entry[:4]}  (showing first 4 fields)")

# Find first non-null pocket_fn
first_real = None
for i, entry in enumerate(index):
    pocket_fn = entry[0]
    if pocket_fn is not None:
        first_real = (i, entry)
        break

if first_real is None:
    print("  ❌ No non-None pocket_fn entries! index.pkl is corrupt.")
    sys.exit(1)

i, (pocket_fn, _, ligand_fn, _, logp, tpsa, sa, qed, aff, _) = first_real
print(f"\n[3] First real sample (idx {i}):")
print(f"     pocket_fn  = {pocket_fn}")
print(f"     ligand_fn  = {ligand_fn}")

# 3. Check that the actual files exist
pocket_path = os.path.join(DATA_PATH, pocket_fn)
ligand_path = os.path.join(DATA_PATH, ligand_fn)
print(f"\n[4] Checking file existence:")
print(f"     pocket exists: {os.path.exists(pocket_path)}  → {pocket_path}")
print(f"     ligand exists: {os.path.exists(ligand_path)}  → {ligand_path}")

if not os.path.exists(pocket_path):
    print(f"\n  ❌ POCKET FILE MISSING. This is why every sample fails to preprocess.")
    print(f"     Likely cause: data.path in train.yml doesn't match where pockets actually are.")
    print(f"     Listing first 5 entries in {DATA_PATH}:")
    if os.path.exists(DATA_PATH):
        for n in os.listdir(DATA_PATH)[:5]:
            print(f"       {n}")
    sys.exit(1)

# 4. Try the parsing pipeline with FULL traceback
print(f"\n[5] Attempting to parse sample {i} (with full traceback on failure)...")
try:
    from utils.data import PDBProtein, parse_drug3d_mol
    print("     ✓ imported PDBProtein, parse_drug3d_mol")

    print("     → calling PDBProtein(...).to_dict_atom() ...")
    pocket_dict = PDBProtein(pocket_path).to_dict_atom()
    print(f"        ✓ got pocket_dict with keys: {list(pocket_dict.keys())}")
    print(f"        n_atoms = {len(pocket_dict['element'])}")

    print("     → calling parse_drug3d_mol(...) ...")
    ligand_dict = parse_drug3d_mol(ligand_path)
    print(f"        ✓ got ligand_dict with keys: {list(ligand_dict.keys())}")
    print(f"        n_atoms = {ligand_dict['num_atoms']}, n_bonds = {ligand_dict['num_bonds']}")

    print("     → wrapping in ProteinLigandData ...")
    from utils.dataset import ProteinLigandData, to_torch_dict
    data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pocket_dict),
        ligand_dict=to_torch_dict(ligand_dict),
    )
    print(f"        ✓ built ProteinLigandData")

    print("\n" + "=" * 70)
    print("✓✓✓  ALL OK — preprocessing should work for this sample.")
    print("=" * 70)
    print("If train.py still skips everything, something is wrong with the")
    print("dataset.py being USED (not the one in /mnt/.../uploads). Check:")
    print("  ls -la utils/dataset.py")
    print("  python -c 'from utils import dataset; print(dataset.__file__)'")
    print("  rm -rf utils/__pycache__")
except Exception as e:
    print()
    print("=" * 70)
    print(f"❌ FAILED with {type(e).__name__}: {e}")
    print("=" * 70)
    print("\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("THIS is the real error. Fix this and preprocessing will work.")
    print("=" * 70)