import argparse
import os
import sys
import json
from datetime import datetime
sys.path.append('.')

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.misc import *
from utils.dataset import get_dataset
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length, eval_all
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(dict, logger):
    for k, v in dict.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def print_ring_ratio(all_ring_sizes, logger):
    sizes_count = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for counter in all_ring_sizes:
        for size in sizes_count.keys():  
            if size in counter:  
                sizes_count[size] += counter[size] 
    total = sum(sizes_count.values())  
    for size in sizes_count.keys():
        ratio = sizes_count[size] / total
        logger.info(f'ring size: {size} ratio: {ratio:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/eval/eval.yml')
    args = parser.parse_args()

    config = load_config(args.config)
    result_path = os.path.join(config.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not config.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(config.sample_path, 'samples_*.pt'))
    if config.eval_num_examples is not None:
        results_fn_list = results_fn_list[:config.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_eval_success = 0
    n_success, n_invalid, n_disconnect = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    ligand_ref_list = []

    all_ligand_list = []
    dataset, subsets = get_dataset(config = config.data)
    train_set, val_set = subsets['train'], subsets['val']
    # for train_data in tqdm(train_set, desc='Get train mols'):
    #     train_ligand_path = os.path.join(config.data.path, train_data.ligand_filename)
    #     train_ligand_rdmol = Chem.SDMolSupplier(train_ligand_path)[0]
    #     all_ligand_list.append(train_ligand_rdmol)
    # for val_data in tqdm(val_set, desc='Get val mols'):
    #     val_ligand_path = os.path.join(config.data.path, val_data.ligand_filename)
    #     val_ligand_rdmol = Chem.SDMolSupplier(val_ligand_path)[0]
    #     all_ligand_list.append(val_ligand_rdmol)
    for train_data in tqdm(train_set, desc='Get train mols'):
        try:
            train_ligand_path = os.path.join(config.data.path, train_data.ligand_filename)
            train_ligand_rdmol = Chem.SDMolSupplier(train_ligand_path)[0]
            if train_ligand_rdmol is not None:
                all_ligand_list.append(train_ligand_rdmol)
        except:
            continue
    for val_data in tqdm(val_set, desc='Get val mols'):
        try:
            val_ligand_path = os.path.join(config.data.path, val_data.ligand_filename)
            val_ligand_rdmol = Chem.SDMolSupplier(val_ligand_path)[0]
            if val_ligand_rdmol is not None:
                all_ligand_list.append(val_ligand_rdmol)
        except:
            continue

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        if config.dataset == 'pdbbind':
            pdbid = r_name.split('/')[-1].split('.')[0].split('_')[1]
            ligand_ref_path = os.path.join(config.protein_root, pdbid, pdbid + '_ligand.sdf')
        elif config.dataset == 'crossdocked':
            ligand_filename = r_name.split('/')[-1].split('.')[0].split('samples_')[1].replace('-', '/')
            ligand_ref_path = os.path.join(config.protein_root, ligand_filename[:-9] + '.sdf')
        r = torch.load(r_name)  # ['element', 'atom_pos', 'rdmol', 'smiles']
        finished_mols = r['finished']
        failed_mols = r['failed']
        n_success += len(finished_mols)
        num_samples += len(finished_mols) + len(failed_mols)
        for mol_info in failed_mols:
            if 'smiles' in mol_info.keys():
                assert '.' in mol_info['smiles']
                n_disconnect += 1
            else:
                n_invalid += 1

        ligand_ref_rdmol = Chem.SDMolSupplier(ligand_ref_path)[0]
        ligand_ref_list.append(ligand_ref_rdmol)

        for sample_idx, mol_info in enumerate(finished_mols):
            pred_atom_type = mol_info['element']
            pred_pos = mol_info['atom_pos']

            # stability check
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            mol = mol_info['rdmol']
            smiles = mol_info['smiles']

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if config.docking_mode == 'qvina':
                    if config.dataset == 'pdbbind':
                        vina_task = QVinaDockingTask.from_generated_mol_pdbbind(
                            mol, pdbid, protein_root=config.protein_root)
                    elif config.dataset == 'crossdocked':
                        vina_task = QVinaDockingTask.from_generated_mol_crossdocked(
                            mol, ligand_filename, protein_root=config.protein_root)
                    vina_results = vina_task.run_sync()
                elif config.docking_mode in ['vina_score', 'vina_dock']:
                    if config.dataset == 'pdbbind':
                        vina_task = VinaDockingTask.from_generated_mol_pdbbind(
                            mol, pdbid, protein_root=config.protein_root)
                    elif config.dataset == 'crossdocked':
                        vina_task = VinaDockingTask.from_generated_mol_crossdocked(
                            mol, ligand_filename, protein_root=config.protein_root)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=config.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=config.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if config.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=config.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except:
                if config.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'pred_pos': pred_pos,
                'chem_results': chem_results,
                'vina': vina_results
            })
    logger.info(f'Evaluate done! {n_success} samples in total.')
    logger.info('Number of evaluated mols: %d' % (len(results)))

    fraction_mol_stable = all_mol_stable / n_success
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_success / num_samples
    fraction_eval = n_eval_success / n_success
    stability_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval
    }
    print_dict(stability_dict, logger)

    validity = (n_success + n_disconnect) / (n_success + n_invalid + n_disconnect)
    connectivity = n_success / (n_success + n_disconnect)
    validity_dict = {'validity': validity, 'connectivity': connectivity}
    print_dict(validity_dict, logger)

    ligand_gen_list = [r['mol'] for r in results]
    sim_dict = scoring_func.get_novelty_and_uniqueness(ligand_gen_list, all_ligand_list)
    diversity = scoring_func.get_diversity(ligand_gen_list)
    sim_dict['diversity'] = diversity
    sim_with_ref = scoring_func.get_sim_with_ref(ligand_gen_list, all_ligand_list)
    sim_dict['sim_with_ref'] = sim_with_ref
    print_dict(sim_dict, logger)


    if config.eval_mode == 'bond_only':
        c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
        c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
        logger.info('JS bond distances of complete mols: ')
        print_dict(c_bond_length_dict, logger)

    elif config.eval_mode == 'all':
        bond_length_dict = eval_all.calc_bonds_JSD(ligand_gen_list, ligand_ref_list)
        angle_dict = eval_all.calc_angles_JSD(ligand_gen_list, ligand_ref_list)
        dihedral_dict = eval_all.calc_dihedrals_JSD(ligand_gen_list, ligand_ref_list)
        logger.info('JS bond distances of complete mols: ')
        print_dict(bond_length_dict, logger)
        logger.info('JS bond angles of complete mols: ')
        print_dict(angle_dict, logger)
        logger.info('JS dihedrals of complete mols: ')
        print_dict(dihedral_dict, logger)

        gen_predicted_rmsd = eval_all.get_predicted_RMSD(ligand_gen_list)
        gen_energy_diff, gen_optimized_rmsd = eval_all.get_optimized_RMSD(ligand_gen_list)
        gen_energy_diff = [diff if diff is not None else 0 for diff in gen_energy_diff]
        logger.info('Generated Mols RMSD with predicte conformations:  Mean: %.3f Median: %.3f' % (np.mean(gen_predicted_rmsd), np.median(gen_predicted_rmsd)))
        logger.info('Generated Mols RMSD before and after RDKit optimization:  Mean: %.3f Median: %.3f' % (np.mean(gen_optimized_rmsd), np.median(gen_optimized_rmsd)))
        logger.info('Generated Mols energy difference before and after RDKit optimization:  Mean: %.3f Median: %.3f' % (np.mean(gen_energy_diff), np.median(gen_energy_diff)))
        if config.save:
            eval_all.plot_rmsd_violin(gen_predicted_rmsd, save_path=os.path.join(result_path, 'predicted_rmsd_violin.png'))
            eval_all.plot_rmsd_violin(gen_optimized_rmsd, save_path=os.path.join(result_path, 'optimized_rmsd_violin.png'))

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if config.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, 'pair_dist_hist.png'))


    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logp = [r['chem_results']['logp'] for r in results]
    lipinski = [r['chem_results']['lipinski'] for r in results]
    tpsa = [r['chem_results']['tpsa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    logger.info('LogP:    Mean: %.3f Median: %.3f' % (np.mean(logp), np.median(logp)))
    logger.info('Lipinski:    Mean: %.3f Median: %.3f' % (np.mean(lipinski), np.median(lipinski)))
    logger.info('TPSA:    Mean: %.3f Median: %.3f' % (np.mean(tpsa), np.median(tpsa)))
    if config.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif config.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        print(f'vina_score_only: {vina_score_only}')
        print(f'vina_min: {vina_min}')
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if config.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if config.save:
        torch.save({
            'stability': stability_dict,
            'validity': validity_dict,
            'similarity': sim_dict,
            'bond_length': all_bond_dist,
            'pair_distance': success_pair_dist,
            'all_results': results
        }, os.path.join(result_path, 'metrics.pt'))

    # ============================================================
    # Auto experiment logging
    # ============================================================
    experiment_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sample_path': config.sample_path,
        'dataset': config.dataset,
        'docking_mode': config.docking_mode,
        'num_evaluated': len(results),
        'num_success': n_success,
        'num_samples': num_samples,
        'metrics': {
            'mol_stable': round(fraction_mol_stable, 4),
            'atm_stable': round(fraction_atm_stable, 4),
            'recon_success': round(fraction_recon, 4),
            'eval_success': round(fraction_eval, 4),
            'validity': round(validity, 4),
            'connectivity': round(connectivity, 4),
            'novelty': round(sim_dict.get('novelty', 0), 4),
            'uniqueness': round(sim_dict.get('uniqueness', 0), 4),
            'diversity': round(diversity, 4),
            'sim_with_ref': round(sim_with_ref, 4),
            'qed_mean': round(float(np.mean(qed)), 4),
            'qed_median': round(float(np.median(qed)), 4),
            'sa_mean': round(float(np.mean(sa)), 4),
            'sa_median': round(float(np.median(sa)), 4),
            'logp_mean': round(float(np.mean(logp)), 4),
            'logp_median': round(float(np.median(logp)), 4),
            'tpsa_mean': round(float(np.mean(tpsa)), 4),
            'tpsa_median': round(float(np.median(tpsa)), 4),
            'lipinski_mean': round(float(np.mean(lipinski)), 4),
        },
    }

    # Add JSD metrics if available
    if config.eval_mode == 'all':
        experiment_record['metrics']['jsd_cc_2a'] = round(float(success_js_metrics.get('JSD_CC_2A', 0)), 4)
        experiment_record['metrics']['jsd_all_12a'] = round(float(success_js_metrics.get('JSD_All_12A', 0)), 4)
        experiment_record['metrics']['atom_type_js'] = round(float(atom_type_js), 4)
        experiment_record['metrics']['rmsd_pred_mean'] = round(float(np.mean(gen_predicted_rmsd)), 4)
        experiment_record['metrics']['rmsd_pred_median'] = round(float(np.median(gen_predicted_rmsd)), 4)
        experiment_record['metrics']['rmsd_opt_mean'] = round(float(np.mean(gen_optimized_rmsd)), 4)
        experiment_record['metrics']['rmsd_opt_median'] = round(float(np.median(gen_optimized_rmsd)), 4)
        experiment_record['metrics']['energy_diff_mean'] = round(float(np.mean(gen_energy_diff)), 4)

    # Add Vina scores
    if config.docking_mode in ['vina_dock', 'vina_score']:
        experiment_record['metrics']['vina_score_mean'] = round(float(np.mean(vina_score_only)), 4)
        experiment_record['metrics']['vina_score_median'] = round(float(np.median(vina_score_only)), 4)
        experiment_record['metrics']['vina_min_mean'] = round(float(np.mean(vina_min)), 4)
        experiment_record['metrics']['vina_min_median'] = round(float(np.median(vina_min)), 4)
        experiment_record['metrics']['vina_scores_all'] = [round(v, 3) for v in vina_score_only]
        experiment_record['metrics']['vina_min_all'] = [round(v, 3) for v in vina_min]
        if config.docking_mode == 'vina_dock':
            experiment_record['metrics']['vina_dock_mean'] = round(float(np.mean(vina_dock)), 4)
            experiment_record['metrics']['vina_dock_median'] = round(float(np.median(vina_dock)), 4)
    elif config.docking_mode == 'qvina':
        experiment_record['metrics']['qvina_mean'] = round(float(np.mean(vina)), 4)
        experiment_record['metrics']['qvina_median'] = round(float(np.median(vina)), 4)

    # Try to extract experiment info from sample_path
    sample_path_parts = config.sample_path.rstrip('/').split('/')
    experiment_record['experiment_name'] = sample_path_parts[-1] if sample_path_parts else 'unknown'

    # Try to read sample config for checkpoint info
    for cfg_name in ['sample_config.yml', '*.yml']:
        import glob as gl
        cfg_files = gl.glob(os.path.join(config.sample_path, '*.yml'))
        if cfg_files:
            try:
                sample_cfg = load_config(cfg_files[0])
                experiment_record['checkpoint'] = getattr(sample_cfg.model, 'checkpoint', 'unknown')
                experiment_record['ms_threshold'] = getattr(sample_cfg.sample, 'ms_apply_threshold', 'N/A')
                experiment_record['use_ms'] = getattr(sample_cfg.sample, 'use_ms_at_inference', 'N/A')
                experiment_record['pocket'] = getattr(sample_cfg.model, 'target', 'unknown')
                experiment_record['seed'] = getattr(sample_cfg.sample, 'seed', 'unknown')
                experiment_record['num_mols_config'] = getattr(sample_cfg.sample, 'num_mols', 'unknown')
                experiment_record['gui_strength'] = getattr(sample_cfg.sample, 'gui_strength', 'unknown')
            except:
                pass
            break

    # Append to global experiment log (JSONL format, one record per line)
    log_file = 'experiment_log.jsonl'
    with open(log_file, 'a') as f:
        f.write(json.dumps(experiment_record, ensure_ascii=False) + '\n')
    logger.info(f'Experiment record appended to {log_file}')

    # Also save to eval_results directory
    record_file = os.path.join(result_path, 'experiment_record.json')
    with open(record_file, 'w') as f:
        json.dump(experiment_record, f, indent=2, ensure_ascii=False)
    logger.info(f'Experiment record saved to {record_file}')

    # Print summary
    logger.info('=' * 60)
    logger.info('EXPERIMENT SUMMARY')
    logger.info(f'  Time:       {experiment_record["timestamp"]}')
    logger.info(f'  Name:       {experiment_record["experiment_name"]}')
    logger.info(f'  Checkpoint: {experiment_record.get("checkpoint", "unknown")}')
    logger.info(f'  MS:         {experiment_record.get("use_ms", "N/A")} (threshold={experiment_record.get("ms_threshold", "N/A")})')
    logger.info(f'  Pocket:     {experiment_record.get("pocket", "unknown")}')
    logger.info(f'  Mols:       {len(results)} evaluated / {n_success} success / {num_samples} total')
    logger.info(f'  Vina Score: {experiment_record["metrics"].get("vina_score_mean", "N/A")}')
    logger.info(f'  QED:        {experiment_record["metrics"]["qed_mean"]}')
    logger.info(f'  SA:         {experiment_record["metrics"]["sa_mean"]}')
    logger.info('=' * 60)


