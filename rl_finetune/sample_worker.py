"""Independent sampling worker - runs on 1 GPU, saves results to file."""
import os, sys, argparse, torch, numpy as np, pickle
sys.path.append('.')
from models.model import MultiDiff
from utils.transforms import FeatureComplex, make_data_placeholder
from utils.misc import seed_all
from utils.reconstruct import reconstruct_from_generated_with_edges
from utils.evaluation.atom_num_config import CONFIG
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from torch_geometric.data import Batch
from scipy import spatial

def get_pocket_size(pp):
    d = spatial.distance.pdist(pp, metric="euclidean")
    return np.median(np.sort(d)[::-1][:10])

def sample_atom_num(ps):
    bounds = CONFIG["bounds"]
    bi = len(bounds)
    for i in range(len(bounds)):
        if bounds[i] > ps: bi = i; break
    nl, pl = CONFIG["bins"][bi]
    return np.random.choice(nl, p=pl)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--pocket', required=True)
    parser.add_argument('--n_mols', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    seed_all(args.seed + args.gpu_id)
    device = 'cuda:0'
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['config']
    feat = FeatureComplex(cfg.data.transform.ligand_atom_mode, sample=True)
    model = MultiDiff(config=cfg.model, protein_node_types=feat.protein_feat_dim,
                    ligand_node_types=feat.atom_feat_dim,
                    num_edge_types=feat.bond_feat_dim).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    from utils.data import PDBProtein
    from utils.dataset import to_torch_dict, ProteinLigandData
    pd = PDBProtein(args.pocket).to_dict_atom()
    ld = {"element": torch.empty([0], dtype=torch.long),
          "hybridization": torch.empty([0], dtype=torch.long),
          "pos": torch.empty([0, 3], dtype=torch.float),
          "bond_index": torch.empty([2, 0], dtype=torch.long),
          "bond_type": torch.empty([0], dtype=torch.long),
          "atom_feature": torch.empty([0, 8], dtype=torch.float)}
    data = ProteinLigandData.protein_ligand_dicts(
        protein_dict=to_torch_dict(pd), ligand_dict=to_torch_dict(ld))
    data = feat(data)
    n = args.n_mols
    with torch.no_grad():
        batch = Batch.from_data_list([data.clone() for _ in range(n)],
                                     follow_batch=feat.follow_batch).to(device)
        ps = get_pocket_size(batch.protein_pos.cpu().numpy())
        na = [int(sample_atom_num(ps)) for _ in range(n)]
        bh = make_data_placeholder(n_nodes_list=na, device=device)
        bl = torch.tensor([[2.0, 100, 1.0, 0.9, 10.0]], device=device).expand(n, -1)
        outputs = model.sample(
            n_graphs=n, protein_node=batch.protein_atom_feat.float(),
            protein_pos=batch.protein_pos, protein_batch=batch.protein_element_batch,
            ligand_batch=bh['batch_node'], halfedge_index=bh['halfedge_index'],
            halfedge_batch=bh['batch_halfedge'], batch_lab=bl,
            gui_strength=4.0, bond_predictor=None, guidance=None)
    outputs_np = {k: [v.cpu().numpy() for v in val] for k, val in outputs.items()}
    from utils.sample_utils import seperate_outputs
    try:
        output_list = seperate_outputs(
            outputs_np, n, bh['batch_node'].cpu().numpy(),
            bh['halfedge_index'].cpu().numpy(), bh['batch_halfedge'].cpu().numpy())
    except:
        pickle.dump({'mol_infos': [], 'smiles': []}, open(args.output, 'wb'))
        return
    mol_infos, smiles_list = [], []
    for om in output_list:
        try:
            mi = feat.decode_output(pred_node=om['pred'][0], pred_pos=om['pred'][1],
                                    pred_halfedge=om['pred'][2], halfedge_index=om['halfedge_index'])
            rdmol = reconstruct_from_generated_with_edges(mi)
            smi = Chem.MolToSmiles(rdmol)
            if '.' not in smi:
                # Extract hybridization and aromaticity from rdmol for downstream featurization
                hyb_map = {Chem.HybridizationType.SP: 'SP',
                           Chem.HybridizationType.SP2: 'SP2',
                           Chem.HybridizationType.SP3: 'SP3',
                           Chem.HybridizationType.SP3D: 'SP3D',
                           Chem.HybridizationType.SP3D2: 'SP3D2'}
                hyb_list, aro_list = [], []
                for atom in rdmol.GetAtoms():
                    hyb_list.append(hyb_map.get(atom.GetHybridization(), 'SP3'))
                    aro_list.append(atom.GetIsAromatic())
                mi_clean = {k: v for k, v in mi.items() if k != 'rdmol'}
                mi_clean['hybridization_str'] = hyb_list
                mi_clean['is_aromatic'] = aro_list
                mol_infos.append(mi_clean); smiles_list.append(smi)
        except: continue
    pickle.dump({'mol_infos': mol_infos, 'smiles': smiles_list}, open(args.output, 'wb'))
    print(f"[GPU:{args.gpu_id}] Done: {len(mol_infos)}/{n} valid")

if __name__ == '__main__':
    main()
