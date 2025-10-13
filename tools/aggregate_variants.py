# tools/aggregate_variants.py
#!/usr/bin/env python3
import os, glob, json, csv, argparse
import numpy as np
import pandas as pd

KEY_JSON = "analysis/stats_ep*.json"  # per-epoch json들
KEY_CSV  = "analysis/routing_stats.csv"
RESULTS  = "results.txt"              # test_acc / probe_acc 저장 파일 (분석 코드 기준)
SUMMARY  = "summary.txt"              # train_expert.py 쪽이면 summary.txt일 수도 있음

METRICS_LAST = [
    "entropy_mean","misroute_rate","gate_regret_ce",
    "grad_router_mean","grad_expert_mean"
]
USAGE_PREFIX = "usage_e"  # usage_e0, usage_e1...

def read_results_txt(path):
    d = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k,v = line.strip().split(":")
                d[k.strip()] = float(v)
    return d

def pick_last_row(csv_path):
    df = pd.read_csv(csv_path)
    last = df.iloc[-1].to_dict()
    # usage 관련 파생 지표
    usage_cols = [c for c in df.columns if c.startswith(USAGE_PREFIX)]
    usage_vals = [last[c] for c in usage_cols]
    last["usage_max"] = max(usage_vals)
    last["usage_min"] = min(usage_vals)
    last["usage_var"] = float(np.var(usage_vals))
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="stage_E_deep 폴더 경로")
    ap.add_argument("--out", default="variants_summary.csv")
    args = ap.parse_args()

    rows = []
    # stage_E_deep/<variant>_analyze/seed_XX
    for variant_dir in glob.glob(os.path.join(args.root, "*_analyze")):
        variant = os.path.basename(variant_dir)
        for seed_dir in glob.glob(os.path.join(variant_dir, "seed_*")):
            seed = os.path.basename(seed_dir).split("_")[-1]

            # 1) test_acc / probe_acc
            res_file = os.path.join(seed_dir, "results.txt")
            if not os.path.exists(res_file):
                res_file = os.path.join(seed_dir, "summary.txt")  # 다른 코드일 경우
            res_vals = read_results_txt(res_file)

            # 2) routing_stats.csv (없으면 만들라고 안내)
            csv_path = os.path.join(seed_dir, "analysis", "routing_stats.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{csv_path} not found. Run summarize_analysis.py first.")
            last = pick_last_row(csv_path)

            row = {
                "variant": variant,
                "seed": int(seed),
                "test_acc": res_vals.get("test_acc", np.nan),
                "probe_acc": res_vals.get("probe_acc", np.nan),
            }
            for m in METRICS_LAST:
                row[m] = last[m]
            row["usage_max"] = last["usage_max"]
            row["usage_min"] = last["usage_min"]
            row["usage_var"] = last["usage_var"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.root, args.out), index=False)
    print("Wrote:", os.path.join(args.root, args.out))

    # 평균/표준편차 보고 싶으면 바로 출력
    print("\n=== mean/std by variant ===")
    print(df.groupby("variant").agg(["mean","std"]))

if __name__ == "__main__":
    main()
