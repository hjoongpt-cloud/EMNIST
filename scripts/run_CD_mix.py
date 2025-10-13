#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import subprocess

# -----------------------------------------------------------------------------
# Script: run CD-mix experiments
# - For each augment in [add_noise, affine, randaugment]
# - Override conv2_dropout=0.1 and conv2corr_lambda=0.1
# - Run each experiment via src.training.train_base
# Usage: python scripts/run_CD_mix.py \
#            --config configs/c.yaml \
#            --out_dir outputs/stage_CD
# -----------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def write_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run C+D mixed experiments: individual augs with dropout=0.1, corr=0.1'
    )
    parser.add_argument('--config', '-c', required=True,
                        help='path to c.yaml with augment definitions')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='base output directory for CD-mix stage')
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    # list of individual augment names used in C-individual stage
    augs = ['add_noise', 'affine', 'randaugment']

    os.makedirs(args.out_dir, exist_ok=True)

    for aug in augs:
        print(f"\n=== Running CD-mix: aug={aug}, dropout=0.1, corr=0.1 ===")
        # copy base config
        cfg = dict(base_cfg)
        # override augmented preset
        cfg['augment']['presets'] = {'temp': [aug]}
        cfg['augment']['strength'] = 'temp'
        # override D-stage params
        cfg['conv2_dropout'] = 0.1
        cfg['conv2corr_lambda'] = 0.1

        # write tmp config
        tmp_cfg = os.path.join('/tmp', f'cdmix_{aug}.yaml')
        write_yaml(cfg, tmp_cfg)

        # run experiment
        stage_name = f"CDmix_{aug}"
        stage_dir = os.path.join(args.out_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        subprocess.run([
            sys.executable, '-m', 'src.training.train_base',
            '--config', tmp_cfg,
            '--out_dir', stage_dir
        ], check=True)

        # cleanup
        os.remove(tmp_cfg)

    print("\nAll CD-mix experiments completed.")
