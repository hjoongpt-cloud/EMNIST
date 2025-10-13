#!/usr/bin/env python3
import json, glob, csv, os, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", help=".../analysis 폴더")
    ap.add_argument("--out", default="routing_stats.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.analysis_dir, "stats_ep*.json"))):
        ep = int(path.split("ep")[-1].split(".")[0])
        with open(path) as f:
            s = json.load(f)
        row = {
            "epoch": ep,
            "N": s["N"],
            "entropy_mean": s["entropy_mean"],
            "entropy_std": s["entropy_std"],
            "misroute_rate": s["misroute_rate"],
            "gate_regret_ce": s["gate_regret_ce"],
            "grad_router_mean": s["grad_router_mean"],
            "grad_expert_mean": s["grad_expert_mean"],
        }
        for i, v in enumerate(s["usage"]):
            row[f"usage_e{i}"] = v
        for i, v in enumerate(s["top1_share"]):
            row[f"top1share_e{i}"] = v
        rows.append(row)

    out_csv = os.path.join(args.analysis_dir, args.out)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
