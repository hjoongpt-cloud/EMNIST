#!/usr/bin/env python3
import os, argparse, json, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_lines(df, cols, ylabel, out_path):
    plt.figure()
    for c in cols:
        plt.plot(df["epoch"], df[c], label=c)
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_heatmap(mat, title, out_path):
    plt.figure(figsize=(6,5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Expert")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--csv", default="routing_stats.csv")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) epoch-wise csv
    csv_path = os.path.join(args.analysis_dir, args.csv)
    df = pd.read_csv(csv_path)

    # 기본 라인 플롯들
    plot_lines(df, ["misroute_rate"], "misroute_rate", os.path.join(args.out_dir, "misroute_rate.png"))
    plot_lines(df, ["gate_regret_ce"], "gate_regret_ce", os.path.join(args.out_dir, "gate_regret_ce.png"))
    plot_lines(df, ["entropy_mean"], "entropy_mean", os.path.join(args.out_dir, "entropy_mean.png"))
    plot_lines(df, ["grad_router_mean","grad_expert_mean"], "grad_norm", os.path.join(args.out_dir, "grad_norms.png"))

    # usage_e*
    usage_cols = [c for c in df.columns if c.startswith("usage_e")]
    plot_lines(df, usage_cols, "usage", os.path.join(args.out_dir, "usage_per_expert.png"))

    # 2) per-epoch heatmaps (P(expert|class), P(class|expert))
    stats_jsons = sorted(glob.glob(os.path.join(args.analysis_dir, "stats_ep*.json")))
    for p in stats_jsons:
        ep = int(p.split("ep")[-1].split(".")[0])
        with open(p) as f:
            s = json.load(f)
        pe_c = np.array(s["P_expert_given_class"])    # [C,E]
        pc_e = np.array(s["P_class_given_expert"])    # [E,C]
        plot_heatmap(pe_c, f"P(expert|class) ep{ep}", os.path.join(args.out_dir, f"pe_c_ep{ep:03d}.png"))
        plot_heatmap(pc_e, f"P(class|expert) ep{ep}", os.path.join(args.out_dir, f"pc_e_ep{ep:03d}.png"))

if __name__ == "__main__":
    main()
