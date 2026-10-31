"""
Train Surrogate Ensemble with Evidential Uncertainty
=====================================================

Each model is trained with dual loss:
  L_total = L_mse (standard prediction) + λ * L_evidential (NIG loss)

Usage:
  python surrogate/train_surrogate.py --device cuda:0
  python surrogate/train_surrogate.py --device cuda:0 --epochs 50 --evi_coeff 0.1
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import QED as QED_module, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from surrogate.model_surrogate import PropertyMPNN, SurrogateEnsemble, EvidentialLoss


# ======================================================================
# SA Score
# ======================================================================

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    def compute_sa(mol):
        raw = sascorer.calculateScore(mol)
        return 1.0 - (raw - 1.0) / 9.0
except:
    def compute_sa(mol):
        return 0.5


def compute_labels(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        qed = QED_module.qed(mol)
        sa = compute_sa(mol)
        logp = Descriptors.MolLogP(mol)
        return {'qed': qed, 'sa': sa, 'logp': logp}
    except:
        return None


# ======================================================================
# Dataset
# ======================================================================

class SurrogateDataset(Dataset):
    def __init__(self, data_path, split_path, split_name='train', max_samples=None):
        import pickle
        import lmdb
        from utils.dataset import ProteinLigandDataset

        self.samples = []
        dataset = ProteinLigandDataset(data_path)
        split = torch.load(split_path)
        indices = split[split_name]
        if max_samples is not None:
            indices = indices[:max_samples]

        print(f"[SurrogateDataset] Loading {len(indices)} samples from {split_name} split...")
        n_skip = 0
        for idx in tqdm(indices, desc=f'Building {split_name} set'):
            try:
                data = dataset.get_ori_data(idx)
                atom_type = data.ligand_element.long()
                pos = data.ligand_pos.float()
                bond_index = data.ligand_bond_index.long()
                bond_type = data.ligand_bond_type.long()

                if len(atom_type) < 3:
                    n_skip += 1
                    continue

                aff = float(data.aff) if hasattr(data, 'aff') and data.aff is not None else 0.0
                qed = float(data.qed) if hasattr(data, 'qed') else 0.5
                sa = float(data.sa) if hasattr(data, 'sa') else 0.5
                logp = float(data.logp) if hasattr(data, 'logp') else 2.0

                atom_type_mapped = torch.clamp(atom_type, 0, 15)

                self.samples.append({
                    'atom_type': atom_type_mapped,
                    'pos': pos,
                    'bond_index': bond_index,
                    'bond_type': bond_type,
                    'labels': torch.tensor([qed, sa, logp, aff], dtype=torch.float32),
                })
            except:
                n_skip += 1
                continue

        print(f"[SurrogateDataset] Loaded {len(self.samples)} samples, skipped {n_skip}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    atom_types, positions, bond_indices, bond_types = [], [], [], []
    labels_list, batch_idx = [], []
    node_offset = 0

    for i, sample in enumerate(batch):
        n_atoms = len(sample['atom_type'])
        atom_types.append(sample['atom_type'])
        positions.append(sample['pos'])
        if sample['bond_index'].numel() > 0:
            bond_indices.append(sample['bond_index'] + node_offset)
            bond_types.append(sample['bond_type'])
        labels_list.append(sample['labels'])
        batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))
        node_offset += n_atoms

    result = {
        'atom_type': torch.cat(atom_types),
        'pos': torch.cat(positions),
        'batch': torch.cat(batch_idx),
        'labels': torch.stack(labels_list),
    }
    if bond_indices:
        result['bond_index'] = torch.cat(bond_indices, dim=1)
        result['bond_type'] = torch.cat(bond_types)
    else:
        result['bond_index'] = torch.zeros(2, 0, dtype=torch.long)
        result['bond_type'] = torch.zeros(0, dtype=torch.long)
    return result


# ======================================================================
# Training one model (with dual loss: MSE + Evidential)
# ======================================================================

def train_one_model(
    model_idx, train_dataset, val_dataset, device, save_dir,
    epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4,
    evi_coeff=0.1, evi_reg=0.01,
):
    print(f"\n{'='*60}")
    print(f"Training Surrogate Model {model_idx}/5 (with Evidential Head)")
    print(f"{'='*60}")

    seed = 42 + model_idx * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PropertyMPNN(
        num_atom_types=16, num_bond_types=5,
        node_dim=128, edge_dim=64, hidden_dim=256,
        num_layers=4, num_properties=4, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    mse_criterion = nn.MSELoss()
    evi_criterion = EvidentialLoss(coeff=evi_reg)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_evi_params = sum(
        p.numel() for n, p in model.named_parameters() if 'evidential' in n
    )
    print(f"  Parameters: {n_params/1e6:.2f}M (evidential heads: {n_evi_params/1e6:.2f}M)")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Epochs: {epochs}, evi_coeff: {evi_coeff}, evi_reg: {evi_reg}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch_data in train_loader:
            atom_type = batch_data['atom_type'].to(device)
            pos = batch_data['pos'].to(device)
            bond_index = batch_data['bond_index'].to(device)
            bond_type = batch_data['bond_type'].to(device)
            batch = batch_data['batch'].to(device)
            labels = batch_data['labels'].to(device)

            optimizer.zero_grad()

            # Dual forward pass
            predictions, evidential = model.forward_evidential(
                atom_type, pos, bond_index, bond_type, batch
            )

            # Loss 1: MSE on standard predictions
            loss_mse = mse_criterion(predictions, labels)

            # Loss 2: Evidential NIG loss (per property)
            loss_evi = 0.0
            for i in range(model.num_properties):
                gamma, nu, alpha, beta = evidential[i]
                target = labels[:, i:i+1]
                loss_evi = loss_evi + evi_criterion(gamma, nu, alpha, beta, target)
            loss_evi = loss_evi / model.num_properties

            # Combined loss
            loss = loss_mse + evi_coeff * loss_evi

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- Validate ----
        model.eval()
        val_losses = []
        all_preds, all_labels = [], []
        all_ale, all_epi = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                atom_type = batch_data['atom_type'].to(device)
                pos = batch_data['pos'].to(device)
                bond_index = batch_data['bond_index'].to(device)
                bond_type = batch_data['bond_type'].to(device)
                batch = batch_data['batch'].to(device)
                labels = batch_data['labels'].to(device)

                predictions, evidential = model.forward_evidential(
                    atom_type, pos, bond_index, bond_type, batch
                )
                loss_mse = mse_criterion(predictions, labels)
                val_losses.append(loss_mse.item())
                all_preds.append(predictions.cpu())
                all_labels.append(labels.cpu())

                # Collect uncertainties for logging
                for i in range(model.num_properties):
                    gamma, nu, alpha, beta = evidential[i]
                    ale, epi = model.evidential_heads[i].compute_uncertainties(nu, alpha, beta)

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        # Per-property MAE
        all_preds_cat = torch.cat(all_preds, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        mae_per_prop = (all_preds_cat - all_labels_cat).abs().mean(dim=0)
        prop_names = ['QED', 'SA', 'LogP', 'Aff']

        if epoch % 5 == 0 or epoch == 1:
            mae_str = ' | '.join([f'{n}: {m:.4f}' for n, m in zip(prop_names, mae_per_prop)])
            print(f"  Epoch {epoch:>3d} | Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | MAE: {mae_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_path = os.path.join(save_dir, f'surrogate_model_{model_idx}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_idx': model_idx,
                'epoch': epoch,
                'val_loss': best_val_loss,
                'seed': seed,
            }, save_path)

    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    return best_val_loss


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Surrogate Ensemble with Evidential Heads')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='data/PDBbind_v2020_pocket10')
    parser.add_argument('--split_path', type=str, default='data/PDBbind_pocket10_split.pt')
    parser.add_argument('--save_dir', type=str, default='surrogate/checkpoints')
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--evi_coeff', type=float, default=0.1,
                        help='Weight of evidential loss (default: 0.1)')
    parser.add_argument('--evi_reg', type=float, default=0.01,
                        help='Evidence regularization coefficient (default: 0.01)')
    parser.add_argument('--max_train_samples', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Surrogate Ensemble Training (with Evidential Heads)")
    print("=" * 60)
    print(f"  Device: {args.device}")
    print(f"  N models: {args.n_models}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Evidential coeff: {args.evi_coeff}")

    # Load dataset
    print("\n[1/3] Loading dataset...")
    train_dataset = SurrogateDataset(
        args.data_path, args.split_path, split_name='train',
        max_samples=args.max_train_samples,
    )
    val_dataset = SurrogateDataset(
        args.data_path, args.split_path, split_name='val',
        max_samples=args.max_train_samples // 5 if args.max_train_samples else None,
    )

    if len(train_dataset) > 0:
        s = train_dataset[0]
        print(f"  Sample 0: atoms={len(s['atom_type'])}, "
              f"labels=[QED={s['labels'][0]:.3f}, SA={s['labels'][1]:.3f}, "
              f"LogP={s['labels'][2]:.3f}, Aff={s['labels'][3]:.3f}]")

    # Train
    print(f"\n[2/3] Training {args.n_models} models...")
    all_val_losses = []
    start_time = time.time()

    for i in range(args.n_models):
        val_loss = train_one_model(
            model_idx=i,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=args.device,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            evi_coeff=args.evi_coeff,
            evi_reg=args.evi_reg,
        )
        all_val_losses.append(val_loss)

    elapsed = time.time() - start_time

    # Save metadata
    print(f"\n[3/3] Saving ensemble metadata...")
    meta_path = os.path.join(args.save_dir, 'ensemble_meta.pt')
    torch.save({
        'n_models': args.n_models,
        'property_names': SurrogateEnsemble.PROPERTY_NAMES,
        'val_losses': all_val_losses,
    }, meta_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    for i, vl in enumerate(all_val_losses):
        print(f"    Model {i}: val_loss = {vl:.4f}")
    print(f"  Mean val loss: {np.mean(all_val_losses):.4f}")


if __name__ == '__main__':
    main()