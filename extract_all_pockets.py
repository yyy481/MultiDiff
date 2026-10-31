"""
Batch-extract 10A pockets for all PDBbind complexes.

Reads:
    data/PDBbind_v2020/{pdbid}/{pdbid}_protein.pdb
    data/PDBbind_v2020/{pdbid}/{pdbid}_ligand.sdf

Writes:
    data/PDBbind_v2020_pocket10/{pdbid}/{pdbid}_pocket_10.pdb
    data/PDBbind_v2020_pocket10/{pdbid}/{pdbid}_ligand.sdf  (copied)
    data/PDBbind_v2020_pocket10/index.pkl  (re-uses existing one if present)

Single-process by default; pass --workers N to parallelize.
"""
import os
import sys
import shutil
import argparse
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append('.')
from utils.data import PDBProtein, parse_drug3d_mol


SOURCE_DIR = 'data/PDBbind_v2020'
DEST_DIR = 'data/PDBbind_v2020_pocket10'
RADIUS = 10


def process_one(pdbid):
    """Extract pocket for one complex. Returns (pdbid, status, msg)."""
    src_dir = os.path.join(SOURCE_DIR, pdbid)
    dst_dir = os.path.join(DEST_DIR, pdbid)

    protein_path = os.path.join(src_dir, f'{pdbid}_protein.pdb')
    ligand_src   = os.path.join(src_dir, f'{pdbid}_ligand.sdf')
    pocket_dst   = os.path.join(dst_dir, f'{pdbid}_pocket_10.pdb')
    ligand_dst   = os.path.join(dst_dir, f'{pdbid}_ligand.sdf')

    # Skip if already done
    if os.path.exists(pocket_dst) and os.path.exists(ligand_dst):
        return (pdbid, 'skipped_done', '')

    if not os.path.exists(protein_path):
        return (pdbid, 'missing_protein', protein_path)
    if not os.path.exists(ligand_src):
        return (pdbid, 'missing_ligand', ligand_src)

    try:
        os.makedirs(dst_dir, exist_ok=True)

        # Read protein
        with open(protein_path, 'r') as f:
            protein_block = f.read()
        protein = PDBProtein(protein_block)

        # Read ligand
        ligand = parse_drug3d_mol(ligand_src)

        # Extract pocket residues within RADIUS of ligand
        pdb_pocket_block = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, RADIUS)
        )

        # Write pocket pdb
        with open(pocket_dst, 'w') as f:
            f.write(pdb_pocket_block)

        # Copy ligand sdf
        shutil.copy(ligand_src, ligand_dst)

        return (pdbid, 'ok', '')
    except Exception as e:
        return (pdbid, 'failed', f'{type(e).__name__}: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1, single-process)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N complexes (for quick test)')
    args = parser.parse_args()

    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source dir not found: {SOURCE_DIR}")
        print(f"   Did you run `tar -xzf data/PDBbind_v2020.tar.gz` from project root yet?")
        sys.exit(1)

    # List all complexes
    pdbids = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d)) and not d.startswith('.')
    ])
    print(f"Found {len(pdbids)} complexes in {SOURCE_DIR}")
    if args.limit:
        pdbids = pdbids[:args.limit]
        print(f"Limiting to first {args.limit}")

    os.makedirs(DEST_DIR, exist_ok=True)

    # Copy index.pkl from source if present (some PDBbind tars include it)
    src_index = os.path.join(SOURCE_DIR, 'index.pkl')
    dst_index = os.path.join(DEST_DIR, 'index.pkl')
    if os.path.exists(src_index) and not os.path.exists(dst_index):
        shutil.copy(src_index, dst_index)
        print(f"Copied index.pkl from source")
    elif os.path.exists(dst_index):
        print(f"Re-using existing {dst_index}")
    else:
        print(f"⚠ No index.pkl found — you need to keep the existing one in {DEST_DIR}/")

    # Process
    results = {'ok': 0, 'skipped_done': 0, 'failed': 0,
               'missing_protein': 0, 'missing_ligand': 0}
    failures = []

    if args.workers == 1:
        for pdbid in tqdm(pdbids, desc='Extracting pockets'):
            _, status, msg = process_one(pdbid)
            results[status] = results.get(status, 0) + 1
            if status == 'failed' and len(failures) < 10:
                failures.append((pdbid, msg))
    else:
        with Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(process_one, pdbids),
                          total=len(pdbids), desc='Extracting pockets'):
                pdbid, status, msg = r
                results[status] = results.get(status, 0) + 1
                if status == 'failed' and len(failures) < 10:
                    failures.append((pdbid, msg))

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for k, v in results.items():
        print(f'  {k:20s}: {v}')
    if failures:
        print('\nFirst few failures:')
        for pdbid, msg in failures:
            print(f'  {pdbid}: {msg}')
    print('=' * 60)


if __name__ == '__main__':
    main()