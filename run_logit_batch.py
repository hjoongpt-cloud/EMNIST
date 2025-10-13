#!/usr/bin/env python3
"""
Batch runner for:
1) tools.dump_logits
2) tools.analyze_logits

Runs over variants (vanilla_analyze / lb_analyze / both_analyze / entropy_analyze)
and seeds (42, 43, 44) by default.

Usage
-----
python run_logit_batch.py \
    --root outputs/stage_E_deep \
    --config configs/e.yaml \
    --num_classes 47 \
    --variants vanilla_analyze lb_analyze both_analyze entropy_analyze \
    --seeds 42 43 44 \
    --batch_size 512 \
    --oracle  # (optional) pass through to dump_logits if you added that flag

(Add --parallel to run in parallel with ProcessPoolExecutor)
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_cmd(cmd, dry=False):
    print("[CMD]", " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, check=True).returncode

def run_one(root, cfg, num_classes, variant, seed, batch_size, oracle, dry):
    ckpt = os.path.join(root, variant, f"seed_{seed}", "model.pt")
    out_npz = os.path.join(root, variant, f"seed_{seed}", "test_logits.npz")
    plot_dir = os.path.join(root, variant, f"seed_{seed}", "logit_plots")

    # 1) dump
    dump_cmd = [
        "python", "-m", "tools.dump_logits",
        "--config", cfg,
        "--ckpt",   ckpt,
        "--out",    out_npz,
        "--batch_size", str(batch_size),
    ]
    if oracle:
        dump_cmd.append("--oracle")
    run_cmd(dump_cmd, dry=dry)

    # 2) analyze
    analyze_cmd = [
        "python", "tools/analyze_logits.py",
        out_npz,
        "--num_classes", str(num_classes),
        "--out_dir", plot_dir
    ]
    run_cmd(analyze_cmd, dry=dry)
    return (variant, seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/stage_E_deep")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--variants", nargs="+",
                    default=["vanilla_analyze", "lb_analyze", "both_analyze", "entropy_analyze"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--oracle", action="store_true",
                    help="Pass through to dump_logits if that flag exists.")
    ap.add_argument("--parallel", action="store_true",
                    help="Run in parallel via ProcessPoolExecutor")
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    jobs = []
    for v in args.variants:
        for s in args.seeds:
            jobs.append((args.root, args.config, args.num_classes, v, s,
                         args.batch_size, args.oracle, args.dry_run))

    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(run_one, *j) for j in jobs]
            for fu in as_completed(futures):
                try:
                    v, s = fu.result()
                    print(f"[DONE] {v} seed {s}")
                except Exception as e:
                    print("[ERROR]", e)
    else:
        for j in jobs:
            v, s = j[3], j[4]
            try:
                run_one(*j)
                print(f"[DONE] {v} seed {s}")
            except subprocess.CalledProcessError as e:
                print(f"[FAIL]  {v} seed {s}: {e}")
                break

if __name__ == "__main__":
    main()
