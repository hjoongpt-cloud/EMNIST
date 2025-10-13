#!/usr/bin/env python3
import os
import sys
import yaml
import subprocess
from itertools import product

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def write_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Sweep conv2_dropout and conv2corr_lambda for D stage')
    parser.add_argument('--base_config', '-b', required=True,
                        help='Path to base YAML config (e.g. configs/base.yaml)')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='Output directory for D-stage experiments')
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    # define sweep values
    dropouts = base_cfg.get('sweep', {}).get('conv2_dropout', [0.0, 0.1, 0.2])
    penalties = base_cfg.get('sweep', {}).get('conv2corr_lambda', [0.0, 0.01, 0.1])

    os.makedirs(args.out_dir, exist_ok=True)

    for d, c in product(dropouts, penalties):
        name = f"dropout{d}_corr{c}".replace('.', 'p')
        print(f"Running D-stage: {name}")
        # merge config
        cfg = dict(base_cfg)
        cfg['conv2_dropout'] = d
        cfg['conv2corr_lambda'] = c
        tmp_cfg = os.path.join('/tmp', f'd_{name}.yaml')
        write_yaml(cfg, tmp_cfg)
        # run experiment
        stage_dir = os.path.join(args.out_dir, name)
        subprocess.run([
            sys.executable, '-m', 'src.training.train_base',
            '--config', tmp_cfg,
            '--out_dir', stage_dir
        ], check=True)
        os.remove(tmp_cfg)
    print('D-stage sweep complete.')
