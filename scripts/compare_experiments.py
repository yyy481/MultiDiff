#!/usr/bin/env python3
"""
compare_experiments.py
读取 experiment_log.jsonl，输出所有实验的对比表格
用法：python3 scripts/compare_experiments.py
"""
import json
import os
import sys

LOG_FILE = 'experiment_log.jsonl'

DISPLAY_METRICS = [
    ('vina_score_mean', 'Vina Score', 'lower'),
    ('vina_min_mean', 'Vina Min', 'lower'),
    ('qed_mean', 'QED', 'higher'),
    ('sa_mean', 'SA', 'higher'),
    ('logp_mean', 'LogP', 'info'),
    ('tpsa_mean', 'TPSA', 'info'),
    ('jsd_cc_2a', 'JSD CC', 'lower'),
    ('jsd_all_12a', 'JSD All', 'lower'),
    ('rmsd_pred_mean', 'RMSD pred', 'lower'),
    ('rmsd_opt_mean', 'RMSD opt', 'lower'),
    ('mol_stable', 'Mol Stab', 'higher'),
    ('validity', 'Valid', 'higher'),
    ('diversity', 'Diverse', 'higher'),
]


def load_experiments():
    if not os.path.exists(LOG_FILE):
        print(f"No experiment log found at {LOG_FILE}")
        sys.exit(1)
    experiments = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def print_comparison(experiments):
    if not experiments:
        print("No experiments found.")
        return

    # Header
    name_width = 30
    col_width = 10
    header = f"{'Experiment':<{name_width}} {'Time':<17}"
    for _, label, _ in DISPLAY_METRICS:
        header += f" {label:>{col_width}}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Use first experiment as baseline for comparison
    baseline_metrics = experiments[0].get('metrics', {})

    for i, exp in enumerate(experiments):
        m = exp.get('metrics', {})
        name = exp.get('experiment_name', 'unknown')[:name_width - 1]
        time = exp.get('timestamp', '')[:16]

        row = f"{name:<{name_width}} {time:<17}"
        for key, _, direction in DISPLAY_METRICS:
            val = m.get(key)
            if val is not None:
                # Color coding (ANSI)
                base_val = baseline_metrics.get(key)
                marker = ''
                if i > 0 and base_val is not None:
                    if direction == 'lower' and val < base_val:
                        marker = '+'  # improved
                    elif direction == 'higher' and val > base_val:
                        marker = '+'  # improved
                    elif direction == 'lower' and val > base_val:
                        marker = '-'  # worse
                    elif direction == 'higher' and val < base_val:
                        marker = '-'  # worse
                row += f" {marker}{val:>{col_width - 1}.4f}"
            else:
                row += f" {'—':>{col_width}}"
        print(row)

    print("=" * len(header))
    print(f"\nBaseline: {experiments[0].get('experiment_name', 'first experiment')}")
    print(f"+ = improved vs baseline, - = worse vs baseline")
    print(f"\nTotal experiments: {len(experiments)}")

    # Vina Score progression
    print(f"\nVina Score progression:")
    for exp in experiments:
        m = exp.get('metrics', {})
        vs = m.get('vina_score_mean')
        name = exp.get('experiment_name', 'unknown')[:25]
        time = exp.get('timestamp', '')[:16]
        ckpt = exp.get('checkpoint', 'N/A')
        ms = exp.get('use_ms', 'N/A')
        threshold = exp.get('ms_threshold', 'N/A')

        if vs is not None:
            baseline_vs = experiments[0].get('metrics', {}).get('vina_score_mean')
            if baseline_vs and baseline_vs != 0:
                pct = ((baseline_vs - vs) / abs(baseline_vs)) * 100
                pct_str = f"  (↑{pct:.1f}%)" if pct > 0 else f"  (↓{abs(pct):.1f}%)"
            else:
                pct_str = ""
            print(f"  {time}  {vs:.3f}{pct_str}  {name}")
            print(f"           ckpt={ckpt}  ms={ms}  threshold={threshold}")


if __name__ == '__main__':
    experiments = load_experiments()
    print_comparison(experiments)