#!/usr/bin/env python3
"""
Run compare_confmats.py for every (variant, seed) pair.

Assumes:
- test_logits.npz (with oracle fields) already exists at:
    {root}/{variant}/seed_{seed}/test_logits.npz
- compare_confmats.py is in PATH or in the same folder as this script.

If the npz is missing *or* lacks oracle fields, optionally re-run dump_logits
with --oracle (set --config and --batch_size). Otherwise it will raise.

Usage
-----
python run_confmat_batch.py \
    --root outputs/stage_E_deep \
    --num_classes 47 \
    --top_percent 10 \
    --variants vanilla_analyze lb_analyze both_analyze entropy_analyze \
    --seeds 42 43 44 \
    --config configs/e.yaml \            # only if you want auto re-dump
    --batch_size 512                     # optional
    --dump_module tools.dump_logits      # dotted path for -m call
    --compare_script tools/compare_confmats.py

Add --parallel to speed up with ProcessPoolExecutor.
"""

import argparse
import os
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

REQUIRED_ORACLE_KEYS = {"logits_all", "best_e", "chosen_e", "ce_best", "ce_chosen"}

def has_oracle_fields(npz_path):
    try:
        d = np.load(npz_path)
        return REQUIRED_ORACLE_KEYS.issubset(d.files)
    except Exception:
        return False

def run(cmd, dry=False):
    print("[CMD]", " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, check=True).returncode

def process_one(root, variant, seed, num_classes, top_percent,
                compare_script, npz_name,
                cfg, dump_module, batch_size, oracle_flag, dry):
    """
    Ensure npz with oracle fields, then run compare_confmats.py
    """
    seed_dir   = os.path.join(root, variant, f"seed_{seed}")
    npz_path   = os.path.join(seed_dir, npz_name)
    out_dir    = os.path.join(seed_dir, "logit_plots")
    ckpt_path  = os.path.join(seed_dir, "model.pt")

    # 1) ensure npz exists & has oracle fields
    need_dump = False
    if not os.path.exists(npz_path):
        need_dump = True
    elif not has_oracle_fields(npz_path):
        need_dump = True

    if need_dump:
        if cfg is None:
            raise FileNotFoundError(f"{npz_path} missing (or no oracle fields) and --config not provided to re-dump.")
        dump_cmd = [
            "python", "-m", dump_module,
            "--config", cfg,
            "--ckpt", ckpt_path,
            "--out", npz_path,
            "--batch_size", str(batch_size)
        ]
        if oracle_flag:
            dump_cmd.append("--oracle")
        run(dump_cmd, dry=dry)

    # 2) run compare_confmats
    cm_cmd = [
        "python", compare_script,
        "--npz", npz_path,
        "--num_classes", str(num_classes),
        "--out_dir", out_dir,
        "--top_percent", str(top_percent)
    ]
    
    run(cm_cmd, dry=dry)
    
    class_cmd = [
    "python", "tools/class_route_analysis.py",
    "--npz", npz_path,
    "--num_classes", str(num_classes),
   "--out_dir", os.path.join(seed_dir, "logit_plots", "class_stats")
    ]
    run(class_cmd, dry=dry)
    
    return (variant, seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage_E_deep")
    ap.add_argument("--variants", nargs="+",
                    default=["vanilla_analyze", "lb_analyze", "both_analyze", "entropy_analyze"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--top_percent", type=float, default=10.0)

    ap.add_argument("--compare_script", default="tools/compare_confmats.py",
                    help="Path to compare_confmats.py")
    ap.add_argument("--npz_name", default="test_logits.npz")

    # For (re)dumping if missing
    ap.add_argument("--config", default=None, help="config for dump_logits (optional)")
    ap.add_argument("--dump_module", default="tools.dump_logits",
                    help="python -m <module> used to dump")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--oracle", action="store_true",
                    help="Pass --oracle to dump_logits (must match your implementation)")

    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    jobs = []
    for v in args.variants:
        for s in args.seeds:
            jobs.append((args.root, v, s, args.num_classes, args.top_percent,
                         args.compare_script, args.npz_name,
                         args.config, args.dump_module, args.batch_size, args.oracle, args.dry_run))


    for j in jobs:
        v, s = j[1], j[2]
        try:
            process_one(*j)
            print(f"[DONE] {v} seed {s}")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL]  {v} seed {s}: {e}")
            break

if __name__ == "__main__":
    main()
