#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import subprocess
from itertools import combinations


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def get_mean_accuracy(summary_file):
    if not os.path.exists(summary_file):
        return None
    with open(summary_file, 'r') as f:
        for line in f:
            if line.startswith('mean,'):
                parts = line.strip().split(',')
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def run_experiment(tmp_cfg, stage_dir):
    os.makedirs(stage_dir, exist_ok=True)
    subprocess.run([
        sys.executable, '-m', 'src.training.train_base',
        '--config', tmp_cfg,
        '--out_dir', stage_dir
    ], check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Sweep individual augs and combos above threshold accuracy'
    )
    parser.add_argument('--config', '-c', required=True,
                        help='Path to c.yaml with augment definitions')
    parser.add_argument('--out_dir', required=True,
                        help='Base output directory')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Min mean test accuracy to keep augment')
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    presets = base_cfg['augment']['presets']

    # Flatten all unique augment names
    all_names = []
    for names in presets.values():
        for name in names:
            if name not in all_names:
                all_names.append(name)

    os.makedirs(args.out_dir, exist_ok=True)
    selected = []

    # 1) Individual augment tests
    for aug in all_names:
        print(f"[Individual] Testing augment: {aug}")
        cfg = base_cfg.copy()
        cfg['augment']['presets'] = {'temp': [aug]}
        cfg['augment']['strength'] = 'temp'

        tmp_cfg = f"/tmp/aug_{aug}.yaml"
        write_yaml(cfg, tmp_cfg)

        stage_dir = os.path.join(args.out_dir, f"ind_{aug}")
        run_experiment(tmp_cfg, stage_dir)

        acc = get_mean_accuracy(os.path.join(stage_dir, 'summary.txt'))
        print(f"  Mean accuracy for {aug}: {acc}\n")

        if acc is not None and acc >= args.threshold:
            selected.append(aug)

        os.remove(tmp_cfg)

    if len(selected) < 2:
        print("< 2 augments exceeded threshold; no combos to run.")
        sys.exit(0)

    # # 2) Combo tests (size 2 up to all selected)
    # for r in range(2, len(selected) + 1):
    #     for combo in combinations(selected, r):
    #         combo_name = '_'.join(combo)
    #         print(f"[Combo] Testing combination: {combo_name}")

    #         cfg = base_cfg.copy()
    #         cfg['augment']['presets'] = {'temp': list(combo)}
    #         cfg['augment']['strength'] = 'temp'

    #         tmp_cfg = f"/tmp/aug_{combo_name}.yaml"
    #         write_yaml(cfg, tmp_cfg)

    #         stage_dir = os.path.join(args.out_dir, f"combo_{combo_name}")
    #         run_experiment(tmp_cfg, stage_dir)

    #         os.remove(tmp_cfg)

    print("All sweeps complete.")


if __name__ == '__main__':
    main()
