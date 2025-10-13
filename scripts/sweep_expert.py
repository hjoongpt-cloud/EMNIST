#!/usr/bin/env python3
import os
import sys
import yaml
import subprocess
import argparse


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


def main():
    parser = argparse.ArgumentParser(
        description='Sweep expert gating variants and OHEM for Stage E')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to base E-stage YAML config (e.yaml)')
    parser.add_argument('--out_dir', '-o', required=True,
                        help='Output directory for Stage E sweeps')
    args = parser.parse_args()

    # Load the base configuration
    base_cfg = load_yaml(args.config)
    #gating_types = ['vanilla', 'load_balance', 'entropy']
    #gating_types = ['load_balance', 'entropy']
    gating_types = ['both']
    ohem_opts = [True]

    os.makedirs(args.out_dir, exist_ok=True)

    # Loop over variants
    for gtype in gating_types:
        for use_ohem in ohem_opts:
            name = f"E_{gtype}" + ("_OHEM" if use_ohem else "")
            print(f"\n=== Running {name} ===")

            # Prepare config for this variant
            cfg = dict(base_cfg)
            cfg['model']['gating']['type'] = gtype
            cfg['model']['ohem']['use_ohem'] = use_ohem

            # Write temporary YAML
            tmp_cfg_path = os.path.join('/tmp', f'e_{gtype}_{"ohem" if use_ohem else "no"}.yaml')
            write_yaml(cfg, tmp_cfg_path)

            # Create output subdir and run
            out_subdir = os.path.join(args.out_dir, name)
            os.makedirs(out_subdir, exist_ok=True)
            subprocess.run([
                sys.executable, '-m', 'src.training.train_expert',
                '--config', tmp_cfg_path,
                '--out_dir', out_subdir
            ], check=True)

            # Cleanup
            os.remove(tmp_cfg_path)

    print("\nStage E sweep complete.")


if __name__ == '__main__':
    main()
