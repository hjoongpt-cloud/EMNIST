#!/usr/bin/env python3
"""
Quickly summarize MANY stats_epXXX.json / cosine_epXXX.npy files into a single, readable report:

Usage (single run folder):
    python -m scripts.expert_report \
        --run_dir outputs/stage_E/vanilla_analyze/seed_42 \
        --out_dir reports/vanilla_seed42

Usage (compare multiple runs):
    python -m scripts.expert_report \
        --run_dir outputs/stage_E/vanilla_analyze/seed_42 \
                  outputs/stage_E/entropy_analyze/seed_42 \
        --labels vanilla entropy \
        --out_dir reports/compare_seed42

Artifacts written:
  - summary.csv (per-epoch aggregates)
  - usage_over_epochs.png, entropy_over_epochs.png, grad_norms.png
  - heatmaps for P(expert|class), P(class|expert) of last epoch
  - cosine_heatmap_last.png (expert weight cosine)
  - index.html (lightweight HTML linking all plots)

Requires matplotlib, numpy, pyyaml.
"""
# ==========================
# File: scripts/expert_report.py
# ==========================
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.utils.moe_stat_viz import (
    plot_line, plot_heatmap, save_html_index,
    load_epoch_stats, aggregate_runs
)


def main():
    ap = argparse.ArgumentParser(description='Summarize MoE stats json/npy into plots')
    ap.add_argument('--run_dir', nargs='+', required=True,
                    help='one or more run dirs containing analysis/stats_ep*.json')
    ap.add_argument('--labels', nargs='*', help='optional labels for runs (same length as run_dir)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--last_epoch', type=int, default=None,
                    help='force last epoch index (default = infer from files)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # default labels
    if args.labels is None:
        args.labels = [os.path.basename(rd.rstrip('/')) for rd in args.run_dir]
    assert len(args.labels) == len(args.run_dir), 'labels length mismatch'

    # load all runs
    runs = []
    for rd in args.run_dir:
        stats, cosines = load_epoch_stats(os.path.join(rd, 'analysis'))
        runs.append((rd, stats, cosines))

    # infer last epoch
    if args.last_epoch is None:
        args.last_epoch = max([max(s.keys()) for _, s, _ in runs])

    # aggregate per-run summary
    summary_csv_path = os.path.join(args.out_dir, 'summary.csv')
    aggregate_runs(runs, args.labels, summary_csv_path)

    # plots per-run (usage/entropy/grad norms vs epoch)
    # collect vectors:
    for metric in ['entropy_mean', 'grad_router_mean', 'grad_expert_mean',
                'misroute_rate', 'gate_regret_ce']:
        plot_line(runs, args.labels, metric, out_path=os.path.join(args.out_dir, f'{metric}.png'))

    # usage plot: [E] per epoch may be many lines; just last epoch heatmap and per-epoch mean std line
    # 1) per-epoch mean/std usage (scalar) not stored -> compute
    # 2) last-epoch heatmap of usage and P(expert|class), P(class|expert)
    # We'll plot for each run separately (save name includes label)
    for (label, (rd, stats, cosines)) in zip(args.labels, runs):
        last = stats[args.last_epoch]
        usage = np.array(last['usage'])  # [E]
        # bar chart usage
        plt.figure(figsize=(4,3))
        plt.bar(np.arange(len(usage)), usage)
        plt.xlabel('Expert')
        plt.ylabel('Usage (mean p)')
        plt.title(f'Usage last epoch - {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'usage_last_{label}.png'))
        plt.close()

        # heatmaps
        pe_c = np.array(last['P_expert_given_class'])    # [C,E]
        pc_e = np.array(last['P_class_given_expert'])    # [E,C]
        plot_heatmap(pe_c, 'class', 'expert', f'P(expert|class) {label}',
                     os.path.join(args.out_dir, f'Pexp_given_class_{label}.png'))
        plot_heatmap(pc_e, 'class', 'expert', f'P(class|expert) {label}',
                     os.path.join(args.out_dir, f'Pclass_given_exp_{label}.png'))

        # cosine heatmap (if exists)
        if args.last_epoch in cosines:
            cos = cosines[args.last_epoch]
            plot_heatmap(cos, 'expert', 'expert', f'Cosine(last layer W) {label}',
                         os.path.join(args.out_dir, f'cosine_last_{label}.png'), vmin=-1, vmax=1)

    # make a tiny html index
    save_html_index(args.out_dir)
    print('Report saved to', args.out_dir)


if __name__ == '__main__':
    main()


