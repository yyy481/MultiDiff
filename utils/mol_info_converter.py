"""
utils/mol_info_converter.py - Simplified version (compatible with your codebase)

将 model.sample() 生成的分子转回 ProteinLigandData 格式
"""

import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Batch
from copy import deepcopy

from utils.dataset import ProteinLigandData


def mol_info_to_protein_ligand_data(
    mol_info: dict,
    pocket_data: ProteinLigandData,
    transform=None,
    ms_edge_builder=None,
) -> ProteinLigandData:
    """
    将单个生成分子转成 ProteinLigandData

    Args:
        mol_info: dict with 'element', 'pos', 'rdmol'
        pocket_data: pocket from dataset
        transform: not used (for compatibility)
        ms_edge_builder: not used (for compatibility)
    """
    data = ProteinLigandData()

    # 复制 protein 信息
    for key in ['protein_element', 'protein_atom_name', 'protein_atom_to_aa_type',
                'protein_is_backbone', 'protein_pos', 'protein_filename',
                'protein_atom_feature']:
        if hasattr(pocket_data, key):
            setattr(data, key, deepcopy(getattr(pocket_data, key)))

    # Ligand 基础信息
    data.ligand_element = torch.LongTensor(mol_info['element'])
    data.ligand_pos = torch.FloatTensor(mol_info['pos'])

    # 从 rdmol 提取 bond
    rdmol = mol_info['rdmol']
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type_int = int(bond.GetBondTypeAsDouble())
        row += [start, end]
        col += [end, start]
        edge_type += [bond_type_int, bond_type_int]

    if len(row) > 0:
        data.ligand_bond_index = torch.LongTensor([row, col])
        data.ligand_bond_type = torch.LongTensor(edge_type)
    else:
        data.ligand_bond_index = torch.zeros((2, 0), dtype=torch.long)
        data.ligand_bond_type = torch.zeros(0, dtype=torch.long)

    # 简单的 ligand atom features (用 one-hot element)
    num_atoms = len(mol_info['element'])
    data.ligand_atom_feature = torch.zeros(num_atoms, 16)
    for i, elem in enumerate(mol_info['element']):
        if elem < 16:
            data.ligand_atom_feature[i, elem] = 1.0

    return data


def batch_mol_infos_to_protein_ligand_batch(
    mol_infos: list,
    pocket_data: ProteinLigandData,
    transform=None,
    ms_edge_builder=None,
):
    """
    将多个生成分子 batch 成一个 Batch
    """
    data_list = []
    for mol_info in mol_infos:
        if mol_info is None:
            continue
        try:
            data = mol_info_to_protein_ligand_data(mol_info, pocket_data, transform, ms_edge_builder)
            data_list.append(data)
        except Exception as e:
            print(f"Warning: failed to convert mol_info: {e}")
            continue

    if len(data_list) == 0:
        return None

    batch = Batch.from_data_list(data_list)

    # 设置必要的字段名
    if hasattr(batch, 'ligand_atom_feature'):
        batch.ligand_node = batch.ligand_atom_feature
    else:
        batch.ligand_node = batch.ligand_element

    if not hasattr(batch, 'ligand_batch'):
        if hasattr(batch, 'ligand_element_batch'):
            batch.ligand_batch = batch.ligand_element_batch
        else:
            batch.ligand_batch = torch.cat([
                torch.full((data.ligand_element.shape[0],), i, dtype=torch.long)
                for i, data in enumerate(data_list)
            ])

    if hasattr(batch, 'protein_atom_feature'):
        batch.protein_node = batch.protein_atom_feature
    else:
        batch.protein_node = batch.protein_element

    if not hasattr(batch, 'protein_batch'):
        num_protein_atoms = pocket_data.protein_element.shape[0]
        batch.protein_batch = torch.cat([
            torch.full((num_protein_atoms,), i, dtype=torch.long)
            for i in range(len(data_list))
        ])

    # halfedge 处理
    if hasattr(batch, 'ligand_bond_index'):
        batch.halfedge_index = batch.ligand_bond_index
        batch.halfedge_type = batch.ligand_bond_type

        halfedge_batch_list = []
        for i, data in enumerate(data_list):
            if hasattr(data, 'ligand_bond_type'):
                halfedge_batch_list.append(
                    torch.full((data.ligand_bond_type.shape[0],), i, dtype=torch.long)
                )
        if len(halfedge_batch_list) > 0:
            batch.halfedge_batch = torch.cat(halfedge_batch_list)
        else:
            batch.halfedge_batch = torch.zeros(0, dtype=torch.long)
    else:
        batch.halfedge_index = torch.zeros((2, 0), dtype=torch.long)
        batch.halfedge_type = torch.zeros(0, dtype=torch.long)
        batch.halfedge_batch = torch.zeros(0, dtype=torch.long)

    return batch