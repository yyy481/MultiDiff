#!/usr/bin/env python3
"""
export_csv.py
Read experiment_log.jsonl and export to a formatted CSV file.
Usage: python3 scripts/export_csv.py
"""
import json
import csv
import os
import sys

LOG_FILE = 'experiment_log.jsonl'
OUT_FILE = 'experiment_results.csv'

COLUMNS = [
    ('timestamp', 'Time'),
    ('experiment_name', 'Experiment'),
    ('checkpoint', 'Checkpoint'),
    ('use_ms', 'MS Enabled'),
    ('ms_threshold', 'MS Threshold'),
    ('pocket', 'Pocket'),
    ('seed', 'Seed'),
    ('num_mols_config', 'Num Mols'),
    ('num_evaluated', 'Evaluated'),
    ('num_success', 'Success'),
    ('num_samples', 'Total'),
    ('metrics.vina_score_mean', 'Vina Score (mean)'),
    ('metrics.vina_score_median', 'Vina Score (median)'),
    ('metrics.vina_min_mean', 'Vina Min (mean)'),
    ('metrics.vina_min_median', 'Vina Min (median)'),
    ('metrics.qed_mean', 'QED (mean)'),
    ('metrics.qed_median', 'QED (median)'),
    ('metrics.sa_mean', 'SA (mean)'),
    ('metrics.sa_median', 'SA (median)'),
    ('metrics.logp_mean', 'LogP (mean)'),
    ('metrics.logp_median', 'LogP (median)'),
    ('metrics.tpsa_mean', 'TPSA (mean)'),
    ('metrics.tpsa_median', 'TPSA (median)'),
    ('metrics.lipinski_mean', 'Lipinski (mean)'),
    ('metrics.jsd_cc_2a', 'JSD CC 2A'),
    ('metrics.jsd_all_12a', 'JSD All 12A'),
    ('metrics.atom_type_js', 'Atom Type JS'),
    ('metrics.rmsd_pred_mean', 'RMSD Pred (mean)'),
    ('metrics.rmsd_pred_median', 'RMSD Pred (median)'),
    ('metrics.rmsd_opt_mean', 'RMSD Opt (mean)'),
    ('metrics.rmsd_opt_median', 'RMSD Opt (median)'),
    ('metrics.energy_diff_mean', 'Energy Diff (mean)'),
    ('metrics.mol_stable', 'Mol Stability'),
    ('metrics.atm_stable', 'Atm Stability'),
    ('metrics.recon_success', 'Recon Success'),
    ('metrics.eval_success', 'Eval Success'),
    ('metrics.validity', 'Validity'),
    ('metrics.connectivity', 'Connectivity'),
    ('metrics.novelty', 'Novelty'),
    ('metrics.uniqueness', 'Uniqueness'),
    ('metrics.diversity', 'Diversity'),
    ('metrics.sim_with_ref', 'Sim with Ref'),
]


def get_nested(d, key):
    parts = key.split('.')
    val = d
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def main():
    if not os.path.exists(LOG_FILE):
        print(f"No experiment log found at {LOG_FILE}")
        sys.exit(1)

    experiments = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    with open(OUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in COLUMNS])
        for exp in experiments:
            row = []
            for key, _ in COLUMNS:
                val = get_nested(exp, key)
                if val is None:
                    row.append('')
                elif isinstance(val, float):
                    row.append(round(val, 4))
                else:
                    row.append(val)
            writer.writerow(row)

    print(f"Exported {len(experiments)} experiments to {OUT_FILE}")


if __name__ == '__main__':
    main()
