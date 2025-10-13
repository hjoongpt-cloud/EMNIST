#!/usr/bin/env python3
"""
Class-wise misroute/retarget analysis + router-vs-oracle usage diff.

Requires npz from dump_logits.py --oracle:
  logits, labels, logits_all, best_e, chosen_e, ce_best, ce_chosen, (optional) gate_p

Outputs:
  - class_route_stats.csv / .json  (per-class table)
  - usage_diff.png                 (router chosen_e hist vs oracle best_e hist)
  - misroute_by_class.png          (bar)
  - regret_by_class.png            (bar)
  - P(expert|class) heatmaps (if gate_p present)
"""

import os, argparse, json, csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.npz)
    needed = {"labels","best_e","chosen_e","ce_best","ce_chosen"}
    miss = [k for k in needed if k not in d.files]
    if miss:
        raise ValueError(f"NPZ missing {miss}; re-dump with --oracle")

    labels    = d["labels"]      # [N]
    best_e    = d["best_e"]      # [N]
    chosen_e  = d["chosen_e"]    # [N]
    ce_best   = d["ce_best"]     # [N]
    ce_chosen = d["ce_chosen"]   # [N]
    regret    = ce_chosen - ce_best
    N         = labels.shape[0]
    C         = args.num_classes
    E         = int(best_e.max()+1)

    # class-wise stats
    rows = []
    for c in range(C):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        mis = float((best_e[idx] != chosen_e[idx]).mean())
        r_mean = float(regret[idx].mean())
        r_med  = float(np.median(regret[idx]))
        rows.append({
            "class": c,
            "N": int(idx.size),
            "misroute_rate": mis,
            "regret_mean": r_mean,
            "regret_median": r_med
        })

    # router usage vs oracle usage (global)
    router_hist  = np.bincount(chosen_e, minlength=E) / N
    oracle_hist  = np.bincount(best_e,   minlength=E) / N

    # save stats
    # CSV/JSON
    with open(os.path.join(args.out_dir, "class_route_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.out_dir, "class_route_stats.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # plots
    import pandas as pd
    df = pd.DataFrame(rows)

    plt.figure(figsize=(8,4))
    sns.barplot(data=df, x="class", y="misroute_rate", color="steelblue")
    plt.title("Misroute rate by class")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "misroute_by_class.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    sns.barplot(data=df, x="class", y="regret_mean", color="indianred")
    plt.title("Mean regret (CE_chosen - CE_best) by class")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "regret_by_class.png"))
    plt.close()

    # usage diff plot
    x = np.arange(E)
    width = 0.35
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, router_hist, width, label="router(chosen)")
    plt.bar(x + width/2, oracle_hist, width, label="oracle(best)")
    plt.xticks(x); plt.xlabel("Expert"); plt.ylabel("fraction")
    plt.title("Router vs Oracle expert usage")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "usage_diff.png"))
    plt.close()

    # optional: P(expert|class) heatmap using gate_p (avg prob) and oracle_best freq
    if "gate_p" in d.files:
        gate_p = d["gate_p"]  # [N,E]
        P_e_given_c = np.zeros((C, E))
        for c in range(C):
            idx = (labels == c)
            if idx.sum() > 0:
                P_e_given_c[c] = gate_p[idx].mean(axis=0)
        plt.figure(figsize=(6,5))
        sns.heatmap(P_e_given_c, cmap="Blues", cbar=True)
        plt.xlabel("Expert"); plt.ylabel("Class")
        plt.title("P(expert | class) from router probs")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "P_expert_given_class_router.png"))
        plt.close()

        # oracle version with one-hot best_e
        P_e_given_c_oracle = np.zeros((C, E))
        for c in range(C):
            idx = (labels == c)
            if idx.sum() > 0:
                hist = np.bincount(best_e[idx], minlength=E) / idx.sum()
                P_e_given_c_oracle[c] = hist
        plt.figure(figsize=(6,5))
        sns.heatmap(P_e_given_c_oracle, cmap="Reds", cbar=True)
        plt.xlabel("Expert"); plt.ylabel("Class")
        plt.title("P(expert | class) from oracle best_e")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "P_expert_given_class_oracle.png"))
        plt.close()

    print("Saved to", args.out_dir)

if __name__ == "__main__":
    main()
