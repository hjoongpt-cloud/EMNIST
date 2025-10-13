# qnext/core/viz.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def save_filters_grid(weight_ko9x9, out_path, ncol=12, pad=2):
    """
    weight_ko9x9: [K, 1, 9, 9] conv1 weight
    """
    W = _to_numpy(weight_ko9x9)
    K = W.shape[0]
    nrow = int(np.ceil(K / ncol))
    fig_w = ncol * 1.0
    fig_h = nrow * 1.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)
    k = 0
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if k < K:
                img = W[k, 0]
                v = np.abs(img).max() + 1e-6
                ax.imshow(img, cmap="gray", vmin=-v, vmax=v)
            k += 1
    plt.tight_layout(pad=pad/10)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_margin_hist(margins_1d, out_path, bins=40, title="true-vs-rival margin"):
    m = _to_numpy(margins_1d).reshape(-1)
    plt.figure(figsize=(5,4))
    plt.hist(m, bins=bins)
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()

def save_slot_cossim_heatmap(M, out_path, vmin=0.0, vmax=1.0, title="slot cos sim (avg)"):
    """
    M: (S,S) numpy or torch cpu tensor
    """
    if M is None:
        return
    if isinstance(M, torch.Tensor):
        M = M.detach().cpu().numpy()
    plt.figure(figsize=(5,4))
    im = plt.imshow(M, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
