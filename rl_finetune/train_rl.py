"""
RL-Guided Diffusion Model Fine-tuning (Innovation 4) - v2
==========================================================

Improvements over v1:
  1. KL divergence constraint: prevents catastrophic forgetting by penalizing
     deviation from the pretrained model
  2. Lower uncertainty penalty (lambda_unc=0.5): allows property signal to
     pass through more effectively
  3. Lower learning rate (5e-6): more conservative updates
  4. More frequent checkpointing (save_freq=50): captures the best iteration
  5. Logs KL penalty and base_loss separately for debugging

Usage:
  python rl_finetune/train_rl.py --device cuda:0 --max_iters 200
"""

import os
import sys
import argparse
import time
import copy
import numpy as np

sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_scatter import scatter_sum, scatter_mean

from models.model import MultiDiff
from surrogate.model_surrogate import SurrogateEnsemble
from reward.reward_function import MultiObjectiveReward
from utils.transforms import FeatureComplex, make_data_placeholder
from utils.dataset import get_dataset
from utils.misc import load_config, seed_all, get_logger, get_new_log_dir
from utils.reconstruct import reconstruct_from_generated_with_edges
from utils.evaluation.atom_num_config import CONFIG

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from scipy import spatial
from tqdm import tqdm


def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])

def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)

def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)

def mol_to_graph_for_surrogate(mol_info, device):
    try:
        element = mol_info['element']
        pos = mol_info['atom_pos']
        atom_type = torch.tensor(element, dtype=torch.long, device=device).clamp(0, 15)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
        bond_index = mol_info.get('bond_index', None)
        bond_type = mol_info.get('bond_type', None)
        if bond_index is not None and len(bond_index[0]) > 0:
            edge_index = torch.tensor(bond_index, dtype=torch.long, device=device)
            edge_type = torch.tensor(bond_type, dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_type = torch.zeros(0, dtype=torch.long, device=device)
        return {'atom_type': atom_type, 'pos': pos_t, 'edge_index': edge_index, 'edge_type': edge_type}
    except:
        return None

def batch_mol_graphs(mol_graphs, device):
    atom_types, positions, edge_indices, edge_types, batch_idx = [], [], [], [], []
    offset, valid_idx = 0, []
    for i, g in enumerate(mol_graphs):
        if g is None: continue
        n = len(g['atom_type'])
        atom_types.append(g['atom_type']); positions.append(g['pos'])
        if g['edge_index'].size(1) > 0:
            edge_indices.append(g['edge_index'] + offset)
            edge_types.append(g['edge_type'])
        batch_idx.append(torch.full((n,), len(valid_idx), dtype=torch.long, device=device))
        offset += n; valid_idx.append(i)
    if not valid_idx: return None, []
    result = {'atom_type': torch.cat(atom_types), 'pos': torch.cat(positions), 'batch': torch.cat(batch_idx)}
    if edge_indices:
        result['edge_index'] = torch.cat(edge_indices, dim=1)
        result['edge_type'] = torch.cat(edge_types)
    else:
        result['edge_index'] = torch.zeros(2, 0, dtype=torch.long, device=device)
        result['edge_type'] = torch.zeros(0, dtype=torch.long, device=device)
    return result, valid_idx


class RLFineTuner:
    def __init__(self, model, reward_fn, featurizer, train_data, device='cuda:0',
                 lr=5e-6, reward_temp=5.0, kl_coeff=0.1, max_grad_norm=10.0):
        self.model = model
        self.reward_fn = reward_fn
        self.featurizer = featurizer
        self.train_data = train_data
        self.device = device
        self.reward_temp = reward_temp
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm

        # Frozen copy of original model for KL constraint
        self.original_model = copy.deepcopy(model)
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
        self.scaler = torch.cuda.amp.GradScaler()
        self.batch_lab = torch.tensor([[2.0, 100, 1.0, 0.9, 10.0]], device=device)

        # Load real training data for weighted loss
        from torch_geometric.transforms import Compose as TCompose
        config_train = load_config('configs/train/train.yml')
        featurizer_train = FeatureComplex(config_train.data.transform.ligand_atom_mode, sample=False)
        transform_train = TCompose([featurizer_train])
        dataset, subsets = get_dataset(config=config_train.data, transform=transform_train)
        self.train_data_with_ligand = [subsets['train'][i] for i in range(min(8, len(subsets['train'])))]
        self.featurizer_train = featurizer_train

        self.stats = {'rewards': [], 'losses': [], 'kl_penalties': [],
                      'base_losses': [], 'n_valid': [], 'n_total': []}

    @torch.no_grad()
    def generate_molecules(self, n_mols=4):
        self.model.eval()
        data = self.train_data
        batch = Batch.from_data_list(
            [data.clone() for _ in range(n_mols)], follow_batch=self.featurizer.follow_batch
        ).to(self.device)
        pocket_size = get_pocket_size(batch.protein_pos.cpu().numpy())
        ligand_num_atoms = [int(sample_atom_num(pocket_size)) for _ in range(n_mols)]
        batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=self.device)
        batch_lab = self.batch_lab.expand(n_mols, -1)

        outputs = self.model.sample(
            n_graphs=n_mols, protein_node=batch.protein_atom_feat.float(),
            protein_pos=batch.protein_pos, protein_batch=batch.protein_element_batch,
            ligand_batch=batch_holder['batch_node'], halfedge_index=batch_holder['halfedge_index'],
            halfedge_batch=batch_holder['batch_halfedge'], batch_lab=batch_lab,
            gui_strength=4.0, bond_predictor=None, guidance=None)

        outputs_np = {k: [v.cpu().numpy() for v in val] for k, val in outputs.items()}
        from utils.sample_utils import seperate_outputs
        try:
            output_list = seperate_outputs(
                outputs_np, n_mols, batch_holder['batch_node'].cpu().numpy(),
                batch_holder['halfedge_index'].cpu().numpy(),
                batch_holder['batch_halfedge'].cpu().numpy())
        except:
            return [], []

        mol_infos, smiles_list = [], []
        for output_mol in output_list:
            try:
                mol_info = self.featurizer.decode_output(
                    pred_node=output_mol['pred'][0], pred_pos=output_mol['pred'][1],
                    pred_halfedge=output_mol['pred'][2], halfedge_index=output_mol['halfedge_index'])
                rdmol = reconstruct_from_generated_with_edges(mol_info)
                smi = Chem.MolToSmiles(rdmol)
                if '.' not in smi:
                    mol_info['rdmol'] = rdmol; mol_info['smiles'] = smi
                    mol_infos.append(mol_info); smiles_list.append(smi)
            except: continue
        return mol_infos, smiles_list

    def compute_rewards(self, mol_infos, smiles_list):
        if not mol_infos: return torch.zeros(0, device=self.device)
        mol_graphs = [mol_to_graph_for_surrogate(m, self.device) for m in mol_infos]
        batch_data, valid_idx = batch_mol_graphs(mol_graphs, self.device)
        if batch_data is None: return torch.zeros(0, device=self.device)
        is_valid = torch.ones(len(valid_idx), dtype=torch.bool, device=self.device)
        result = self.reward_fn.compute(
            atom_type=batch_data['atom_type'], pos=batch_data['pos'],
            edge_index=batch_data['edge_index'], bond_type=batch_data['edge_type'],
            batch=batch_data['batch'],
            smiles_list=[smiles_list[i] for i in valid_idx] if smiles_list else None,
            is_valid=is_valid)
        return result['R_final']

    def compute_weighted_loss(self, rewards, n_mols):
        """Reward-weighted loss WITH KL divergence constraint."""
        self.model.train()
        data = self.train_data_with_ligand
        n_use = min(n_mols, len(data))
        batch = Batch.from_data_list(
            [data[i % len(data)].clone() for i in range(n_use)],
            follow_batch=self.featurizer_train.follow_batch,
            exclude_keys=self.featurizer_train.exclude_keys).to(self.device)

        batch_size = batch.num_graphs
        batch_lab = self.batch_lab.expand(batch_size, -1)
        ms_edges = None
        if hasattr(batch, 'ms_l2_edge_index'):
            ms_edges = {}
            for layer in ['l2', 'l3', 'l4']:
                for suffix in ['edge_index', 'edge_weight', 'edge_src_type']:
                    key = f'ms_{layer}_{suffix}'
                    ms_edges[key] = getattr(batch, key).to(self.device)

        # Same noise for both models
        protein_noise = torch.randn_like(batch.protein_pos) * 0.05
        pos_noise = torch.randn_like(batch.ligand_pos) * 0.05

        # Current model loss
        loss_dict, _ = self.model.get_loss(
            protein_node=batch.protein_atom_feat.float(),
            protein_pos=batch.protein_pos + protein_noise,
            protein_batch=batch.protein_element_batch,
            ligand_node=batch.ligand_atom_feat_full,
            ligand_pos=batch.ligand_pos + pos_noise,
            ligand_batch=batch.ligand_element_batch,
            halfedge_type=batch.ligand_halfedge_type,
            halfedge_index=batch.ligand_halfedge_index,
            halfedge_batch=batch.ligand_halfedge_type_batch,
            num_mol=batch.num_graphs, batch_lab=batch_lab, ms_edges=ms_edges)
        base_loss = loss_dict['loss']

        # Original model loss (frozen, for KL constraint)
        with torch.no_grad():
            orig_loss_dict, _ = self.original_model.get_loss(
                protein_node=batch.protein_atom_feat.float(),
                protein_pos=batch.protein_pos + protein_noise,
                protein_batch=batch.protein_element_batch,
                ligand_node=batch.ligand_atom_feat_full,
                ligand_pos=batch.ligand_pos + pos_noise,
                ligand_batch=batch.ligand_element_batch,
                halfedge_type=batch.ligand_halfedge_type,
                halfedge_index=batch.ligand_halfedge_index,
                halfedge_batch=batch.ligand_halfedge_type_batch,
                num_mol=batch.num_graphs, batch_lab=batch_lab, ms_edges=ms_edges)
        orig_loss = orig_loss_dict['loss'].detach()

        # KL penalty (clamped for stability)
        kl_penalty = (base_loss - orig_loss).abs().clamp(max=5.0)
        mean_reward = rewards.mean().detach() if len(rewards) > 0 else torch.tensor(0.0)
        # Exponential scaling: reward_temp amplifies good/bad difference
        reward_weight = torch.exp(self.reward_temp * (mean_reward - 0.5))
        weighted_loss = base_loss * reward_weight + self.kl_coeff * kl_penalty

        return weighted_loss, loss_dict, kl_penalty.item()

    def train_step(self, n_mols=4):
        try:
            mol_infos, smiles_list = self.generate_molecules(n_mols)
        except Exception as e:
            print(f"  [Warning] generate_molecules failed: {e}")
            return {'loss': 0.0, 'reward_mean': 0.0, 'kl_penalty': 0.0,
                    'n_valid': 0, 'n_total': n_mols}
        n_valid, n_total = len(mol_infos), n_mols
        if n_valid == 0:
            return {'loss': 0.0, 'reward_mean': 0.0, 'kl_penalty': 0.0,
                    'n_valid': 0, 'n_total': n_total}

        rewards = self.compute_rewards(mol_infos, smiles_list)
        reward_mean = rewards.mean().item() if len(rewards) > 0 else 0.0
        rewards_cpu = rewards.detach().cpu()
        del mol_infos, smiles_list, rewards
        torch.cuda.empty_cache()

        self.optimizer.zero_grad(set_to_none=True)
        rewards_gpu = rewards_cpu.to(self.device)
        weighted_loss, loss_dict, kl_val = self.compute_weighted_loss(rewards_gpu, n_valid)

        self.scaler.scale(weighted_loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_val = weighted_loss.item()
        base_loss_val = loss_dict['loss'].item()
        del weighted_loss, loss_dict, rewards_gpu
        torch.cuda.empty_cache()

        self.stats['rewards'].append(reward_mean)
        self.stats['losses'].append(loss_val)
        self.stats['kl_penalties'].append(kl_val)
        self.stats['base_losses'].append(base_loss_val)
        self.stats['n_valid'].append(n_valid)
        self.stats['n_total'].append(n_total)

        return {'loss': loss_val, 'base_loss': base_loss_val, 'kl_penalty': kl_val,
                'reward_mean': reward_mean, 'n_valid': n_valid, 'n_total': n_total,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm}


def main():
    parser = argparse.ArgumentParser(description='RL Fine-tuning v2 (KL constraint)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='ablation_results/ms_full/finetuned_final.pt')
    parser.add_argument('--surrogate_dir', type=str, default='surrogate/checkpoints')
    parser.add_argument('--pocket', type=str, default='sample/3ztx_pocket.pdb')
    parser.add_argument('--save_dir', type=str, default='rl_finetune/checkpoints_v3')
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--mols_per_iter', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--reward_temp', type=float, default=5.0)
    parser.add_argument('--kl_coeff', type=float, default=0.01)
    parser.add_argument('--lambda_unc', type=float, default=0.5)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = get_new_log_dir('rl_finetune/logs', prefix='rl_v2')
    logger = get_logger('rl_finetune', log_dir)
    seed_all(2024)

    print("=" * 60)
    print("RL Fine-tuning v2 (KL constraint)")
    print(f"  lr={args.lr}, kl_coeff={args.kl_coeff}, lambda_unc={args.lambda_unc}")
    print("=" * 60)

    logger.info(f'Loading model from {args.checkpoint}...')
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    train_config = ckpt['config']
    featurizer = FeatureComplex(train_config.data.transform.ligand_atom_mode, sample=False)
    model = MultiDiff(config=train_config.model, protein_node_types=featurizer.protein_feat_dim,
                    ligand_node_types=featurizer.atom_feat_dim,
                    num_edge_types=featurizer.bond_feat_dim).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False)
    logger.info(f'Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    logger.info(f'Loading surrogate...')
    ensemble = SurrogateEnsemble.load(args.surrogate_dir, device=args.device,
        num_atom_types=16, num_bond_types=5, node_dim=128, edge_dim=64,
        hidden_dim=256, num_layers=4, num_properties=4, dropout=0.1)

    reward_fn = MultiObjectiveReward(surrogate_ensemble=ensemble, device=args.device,
                                     alpha=0.8, beta=0.2, lambda_unc=args.lambda_unc)

    logger.info(f'Loading pocket: {args.pocket}')
    from utils.data import PDBProtein
    from utils.dataset import to_torch_dict, ProteinLigandData
    pocket_dict = PDBProtein(args.pocket).to_dict_atom()
    ligand_dict = {"element": torch.empty([0], dtype=torch.long),
                   "hybridization": torch.empty([0], dtype=torch.long),
                   "pos": torch.empty([0, 3], dtype=torch.float),
                   "bond_index": torch.empty([2, 0], dtype=torch.long),
                   "bond_type": torch.empty([0], dtype=torch.long),
                   "atom_feature": torch.empty([0, 8], dtype=torch.float)}
    train_data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pocket_dict), ligand_dict=to_torch_dict(ligand_dict))
    featurizer_sample = FeatureComplex(train_config.data.transform.ligand_atom_mode, sample=True)
    train_data = featurizer_sample(train_data)

    finetuner = RLFineTuner(model=model, reward_fn=reward_fn, featurizer=featurizer_sample,
                            train_data=train_data, device=args.device, lr=args.lr,
                            reward_temp=args.reward_temp, kl_coeff=args.kl_coeff)

    logger.info(f'Starting v2: {args.max_iters} iters, {args.mols_per_iter} mols/iter, '
                f'kl={args.kl_coeff}, lambda_unc={args.lambda_unc}, lr={args.lr}')

    best_reward = -float('inf')
    start_time = time.time()

    for it in range(1, args.max_iters + 1):
        result = finetuner.train_step(n_mols=args.mols_per_iter)
        if it % args.log_freq == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(f"[Iter {it}/{args.max_iters}] loss={result['loss']:.4f} "
                        f"base={result.get('base_loss',0):.4f} kl={result.get('kl_penalty',0):.4f} "
                        f"reward={result['reward_mean']:.4f} "
                        f"valid={result['n_valid']}/{result['n_total']} time={elapsed:.1f}min")

        if it % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'rl_iter_{it}.pt')
            torch.save({'model': model.state_dict(), 'config': train_config,
                        'iteration': it, 'reward': result['reward_mean'],
                        'stats': finetuner.stats}, save_path)
            logger.info(f'  Saved: {save_path}')
            if result['reward_mean'] > best_reward and result['n_valid'] > 0:
                best_reward = result['reward_mean']
                torch.save({'model': model.state_dict(), 'config': train_config,
                            'iteration': it, 'reward': best_reward,
                            'stats': finetuner.stats},
                           os.path.join(args.save_dir, 'rl_best.pt'))
                logger.info(f'  New best: {best_reward:.4f} at iter {it}')

    torch.save({'model': model.state_dict(), 'config': train_config,
                'iteration': args.max_iters, 'stats': finetuner.stats},
               os.path.join(args.save_dir, 'rl_final.pt'))

    elapsed = (time.time() - start_time) / 60
    logger.info(f'\nDone in {elapsed:.1f}min. Best reward: {best_reward:.4f}')
    logger.info(f'  Mean reward (last 50): {np.mean(finetuner.stats["rewards"][-50:]):.4f}')
    logger.info(f'  Mean KL (last 50): {np.mean(finetuner.stats["kl_penalties"][-50:]):.4f}')


if __name__ == '__main__':
    main()