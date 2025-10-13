# =============================================
# FILE: q_run_init_and_pretrain.py
# =============================================
# 1단계(필터 초기화) → 2단계(사전학습+프루닝) 순차 실행 런처

import argparse, os, sys, subprocess
from pathlib import Path


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    # Step-1 args
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--per_image", type=int, default=6)
    ap.add_argument("--q_keep", type=float, default=0.95)
    ap.add_argument("--nms_k", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")

    # Common
    ap.add_argument("--out_root", type=str, default="outputs/Q")

    # Step-2 args
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_gate_l1", type=float, default=1e-3)
    ap.add_argument("--lambda_bal", type=float, default=1e-3)
    ap.add_argument("--use_top1", type=int, default=1)
    ap.add_argument("--auto_prune", type=int, default=1)
    ap.add_argument("--min_keep", type=int, default=32)
    ap.add_argument("--imp_alpha_usage", type=float, default=1.0)
    ap.add_argument("--imp_beta_gate", type=float, default=1.0)
    ap.add_argument("--imp_gamma_l2", type=float, default=1.0)

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- Step-1: init filters -----
    init_dir = out_root / f"filters_init_k{args.k}_p{args.patch}"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_npy = init_dir / f"filters_init_k{args.k}_{args.patch}.npy"

    cmd1 = [sys.executable, "-m", "src_q.q_init_filters_from_patches",
            "--out", str(init_npy),
            "--k", str(args.k), "--patch", str(args.patch),
            "--per_image", str(args.per_image),
            "--q_keep", str(args.q_keep),
            "--nms_k", str(args.nms_k),
            "--max_images", str(args.max_images),
            "--seed", str(args.seed),
            "--data_root", str(args.data_root)]
    run(cmd1)

    # ----- Step-2: pretrain+prune -----
    pre_dir = out_root / f"pretrain_k{args.k}_p{args.patch}"
    pre_dir.mkdir(parents=True, exist_ok=True)

    cmd2 = [sys.executable, "-m", "src_q.q_pretrain_filters",
            "--out_dir", str(pre_dir),
            "--init_filters", str(init_npy),
            "--k_init", str(args.k),
            "--d_model", str(args.d_model),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--lambda_gate_l1", str(args.lambda_gate_l1),
            "--lambda_bal", str(args.lambda_bal),
            "--use_top1", str(args.use_top1),
            "--auto_prune", str(args.auto_prune),
            "--min_keep", str(args.min_keep),
            "--imp_alpha_usage", str(args.imp_alpha_usage),
            "--imp_beta_gate", str(args.imp_beta_gate),
            "--imp_gamma_l2", str(args.imp_gamma_l2),
            "--seed", str(args.seed),
            "--data_root", str(args.data_root)]
    run(cmd2)

    print("\n[done] outputs root:", out_root)

if __name__ == "__main__":
    main()
