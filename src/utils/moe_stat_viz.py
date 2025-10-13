# ==========================
# File: src/utils/moe_stat_viz.py
# ==========================
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_epoch_stats(analysis_dir):
    """Load stats_epXXX.json and cosine_epXXX.npy from a directory.
    Returns:
      stats_dict: {epoch:int -> json_dict}
      cosine_dict: {epoch:int -> np.ndarray}
    """
    stats_dict = {}
    cosine_dict = {}
    if not os.path.exists(analysis_dir):
        return stats_dict, cosine_dict

    for js in sorted(glob.glob(os.path.join(analysis_dir, 'stats_ep*.json'))):
        ep = int(os.path.basename(js).split('ep')[-1].split('.')[0])
        with open(js) as f:
            stats_dict[ep] = json.load(f)
    for npy in sorted(glob.glob(os.path.join(analysis_dir, 'cosine_ep*.npy'))):
        ep = int(os.path.basename(npy).split('ep')[-1].split('.')[0])
        cosine_dict[ep] = np.load(npy)
    return stats_dict, cosine_dict


def aggregate_runs(runs, labels, csv_path):
    """Write a CSV table of per-epoch metrics for multiple runs.
    runs: list of (run_dir, stats_dict, cos_dict)
    """
    # gather epochs union
    epochs = sorted(set().union(*[s.keys() for _, s, _ in runs]))
    # choose metrics to save (must exist in stats json)
    metrics = [
        'entropy_mean', 'entropy_std',
        'grad_router_mean', 'grad_expert_mean',
        'misroute_rate', 'gate_regret_ce'
    ]
    # header
    with open(csv_path, 'w') as f:
        header = ['epoch']
        for label in labels:
            for m in metrics:
                header.append(f'{label}:{m}')
        f.write(','.join(header)+'\n')
        for ep in epochs:
            row = [str(ep)]
            for (_, sdict, _), label in zip(runs, labels):
                if ep in sdict:
                    st = sdict[ep]
                    for m in metrics:
                        row.append(str(st.get(m, '')))
                else:
                    row.extend(['']*len(metrics))
            f.write(','.join(row)+'\n')


def plot_line(runs, labels, metric, out_path):
    plt.figure(figsize=(5,3.2))
    for (rd, stats, _), lab in zip(runs, labels):
        xs = sorted(stats.keys())
        ys = [stats[e].get(metric, np.nan) for e in xs]
        plt.plot(xs, ys, marker='o', label=lab)
    plt.xlabel('epoch'); plt.ylabel(metric)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()


def plot_heatmap(mat, xlabel, ylabel, title, out_path, vmin=None, vmax=None):
    plt.figure(figsize=(5,4))
    im = plt.imshow(mat, aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_html_index(folder):
    imgs = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    csvs = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    html = ["<html><body><h1>MoE Report</h1>"]
    for c in csvs:
        html.append(f'<p><a href="{c}">{c}</a></p>')
    for img in imgs:
        html.append(f'<div><h3>{img}</h3><img src="{img}" style="max-width:700px"></div>')
    html.append('</body></html>')
    with open(os.path.join(folder,'index.html'),'w') as f:
        f.write('\n'.join(html))
