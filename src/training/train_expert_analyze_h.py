#!/usr/bin/env python3
"""
Train + Analyze MoE (Stage H) with sparse conv1 activation + reconstruction.
"""
import os
import json
import random
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_h import ExpertModelH
from src.utils.moe_analysis import MoeAnalyzer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch.nn.functional as F

import cv2
from PIL import Image
import numpy as np






def get_activation_contours_from_A_global(A_global, min_area=1):
    if isinstance(A_global, torch.Tensor):
        A_global = A_global.detach().cpu().numpy()
    _, C, H, W = A_global.shape
    mat = A_global[0]
    spatial = np.any(np.abs(mat) > 0, axis=0).astype(np.uint8)
    spatial_uint8 = (spatial * 255).astype(np.uint8)
    contours, _ = cv2.findContours(spatial_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            filtered.append(cnt.squeeze(1))  # (N,2)
    return filtered

def draw_activation_and_receptive_fields(img, A_global, kernel_size, out_path, title=None, min_area=1):
    H_img, W_img = img.shape
    pad = kernel_size // 2

    contours = get_activation_contours_from_A_global(A_global, min_area=min_area)

    if isinstance(A_global, torch.Tensor):
        A_global_np = A_global.detach().cpu().numpy()
    else:
        A_global_np = A_global
    spatial_mask = np.any(np.abs(A_global_np[0]) > 0, axis=0).astype(np.uint8)  # (H,W)

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    # draw contours
    for cnt in contours:
        poly = patches.Polygon(cnt, edgecolor='red', facecolor='none', linewidth=1.5)
        ax.add_patch(poly)

    # draw receptive fields per active spatial location
    for h in range(spatial_mask.shape[0]):
        for w in range(spatial_mask.shape[1]):
            if spatial_mask[h, w] == 0:
                continue
            y0 = h - pad
            x0 = w - pad
            y1 = h + pad
            x1 = w + pad
            y0c = max(0, y0)
            x0c = max(0, x0)
            y1c = min(H_img - 1, y1)
            x1c = min(W_img - 1, x1)
            rect = patches.Rectangle((x0c, y0c), x1c - x0c + 1, y1c - y0c + 1,
                                     linewidth=0.8, edgecolor='yellow', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# reuse training utilities from original training script
from src.training.train_expert_analyze import train_epoch, evaluate, extract_features
import cv2
from PIL import Image

import matplotlib.patches as patches

def get_activation_boxes_from_A_global(A_global, min_area=1):
    """
    A_global: tensor or numpy array shape (1, C, H, W) after conv1+ReLU+global top-k mask
    Returns list of bounding boxes [(x1, y1, x2, y2), ...]
    """
    if isinstance(A_global, torch.Tensor):
        A_global = A_global.detach().cpu().numpy()
    # collapse channel: any nonzero across channels
    _, C, H, W = A_global.shape
    mat = A_global[0]  # (C,H,W)
    spatial = np.any(np.abs(mat) > 0, axis=0).astype(np.uint8)  # (H,W) binary mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spatial, connectivity=8)
    boxes = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes

def draw_activation_boxes(img, boxes, out_path, title=None):
    """
    img: 2D numpy array [H,W] in [0,1]
    boxes: list of (x1,y1,x2,y2)
    """
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    for (x1, y1, x2, y2) in boxes:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
class MedianFilterTransform:
    def __init__(self, k=3):
        self.k = k

    def __call__(self, tensor):
        # tensor: [1,H,W], float in [0,1]
        img = tensor.squeeze(0).numpy()  # H,W
        img_uint8 = (img * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.k)
        denoised = denoised.astype(np.float32) / 255.0
        return torch.from_numpy(denoised).unsqueeze(0)  # [1,H,W]
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def reconstruct_image_from_sparse(x_orig, model_h, device):
    model_h.eval()
    with torch.no_grad():
        _, sparse_weights, _ = model_h.trunk(x_orig.to(device))  # (1,H,W,C)
        conv1_w = model_h.trunk.conv1.weight  # (C,1,K,K)
        _, _, H, W = x_orig.shape
        K = conv1_w.shape[-1]
        pad = K // 2
        recon = torch.zeros_like(x_orig, device=device)

        for h in range(H):
            for w in range(W):
                wts = sparse_weights[0, h, w]  # (C,)
                combined = (wts[:, None, None] * conv1_w[:, 0]).sum(dim=0)  # (K,K)
                h0 = h - pad
                w0 = w - pad
                h_start = max(0, h0)
                w_start = max(0, w0)
                h_end = min(H, h0 + K)
                w_end = min(W, w0 + K)
                fh_start = h_start - h0
                fw_start = w_start - w0
                fh_end = fh_start + (h_end - h_start)
                fw_end = fw_start + (w_end - w_start)
                recon[0, 0, h_start:h_end, w_start:w_end] += combined[fh_start:fh_end, fw_start:fw_end]

        err = (x_orig.to(device) - recon).abs()
    return recon.cpu(), err.cpu()

def visualize_reconstruction(x_orig, recon, err, out_path, title_prefix=""):
    x = x_orig.squeeze().cpu().numpy()
    r = recon.squeeze().cpu().numpy()
    e = err.squeeze().cpu().numpy()
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(x, cmap='gray'); axs[0].set_title(f"{title_prefix} Orig"); axs[0].axis('off')
    axs[1].imshow(r, cmap='gray'); axs[1].set_title(f"{title_prefix} Recon"); axs[1].axis('off')
    axs[2].imshow(e, cmap='hot'); axs[2].set_title(f"{title_prefix} Abs Err"); axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_path); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--no_analysis', action='store_true')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    seeds = cfg.get('seeds', [42,43,44])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary = {'seed': [], 'test_acc': [], 'probe_acc': []}
    
    for seed in seeds:
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        analysis_dir = os.path.join(run_dir, 'analysis'); os.makedirs(analysis_dir, exist_ok=True)

        # Data
        nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})
        #transform = transforms.Compose([transforms.ToTensor(), MedianFilterTransform(k=3), transforms.Normalize(nm['mean'], nm['std'])])
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(nm['mean'], nm['std'])])
        train_ds = EMNIST('data', split='balanced', train=True, download=True, transform=transform)
        test_ds  = EMNIST('data', split='balanced', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.opt.batch_size, shuffle=True, num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.opt.batch_size, shuffle=False, num_workers=4)

        # Model / optimizer
        model = ExpertModelH(cfg).to(device)
        # compatibility for original training utilities
        model.forward_features = model.forward_features
        model.conv2corr_penalty = model.conv2corr_penalty
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.opt.lr)

        analyzer = None if args.no_analysis else MoeAnalyzer(cfg.model.num_experts, cfg.model.num_classes)
        history = {'epoch':[], 'ce':[], 'lb':[], 'ent':[], 'acc':[]}

        # Train
        for ep in range(1, cfg.opt.epochs+1):
            if analyzer: analyzer.begin_epoch()
            ce, lb, ent, acc = train_epoch(model, train_loader, optimizer, device, cfg, analyzer, ep)
            if analyzer:
                stats = analyzer.end_epoch(); analyzer.finalize_epoch(analysis_dir, ep, model)
                with open(os.path.join(analysis_dir, f'stats_ep{ep:03d}.json'),'w') as f: json.dump(stats, f, indent=2)
            history['epoch'].append(ep); history['ce'].append(ce); history['lb'].append(lb); history['ent'].append(ent); history['acc'].append(acc)
            print(f"Seed {seed} Ep {ep:02d} | CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} ACC {acc:.4f}")

        # Eval & probe
        test_acc = evaluate(model, test_loader, device)
        Xtr, ytr = extract_features(model, train_loader, device)
        Xte, yte = extract_features(model, test_loader, device)
        probe_acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)

        summary['seed'].append(seed); summary['test_acc'].append(test_acc); summary['probe_acc'].append(probe_acc)
        with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
            f.write('Seed,TestAcc,ProbeAcc\n')
            for sd, ta, pa in zip(summary['seed'], summary['test_acc'], summary['probe_acc']):
                f.write(f"{sd},{ta:.4f},{pa:.4f}\n")
            f.write(f"mean,{np.mean(summary['test_acc']):.4f},{np.mean(summary['probe_acc']):.4f}\n")
        # Reconstruction for 3 random test images
        # idxs = np.random.choice(len(test_ds), size=3, replace=False)
        # 찾기: inversion이 된 샘플 하나 포함
        # transform 없는 버전 (PIL)으로 반전 여부 검사용
        test_ds_no_transform = EMNIST('data', split='balanced', train=False, download=True, transform=None)

        idxs = np.random.choice(len(test_ds), size=3, replace=False)

        for i, idx in enumerate(idxs):
            x_orig, _ = test_ds[idx]
            x_orig = x_orig.unsqueeze(0)  # (1,1,H,W)

            # 재구성 입력 그대로 사용
            x_for_recon = x_orig  # (1,1,H,W), already transformed

            recon, err = reconstruct_image_from_sparse(x_for_recon, model, device)
            title_prefix = f'IMG{idx}'
            visualize_reconstruction(
                x_for_recon, recon, err,
                os.path.join(run_dir, f'recon_{i}.png'),
                title_prefix=title_prefix
            )
            with open(os.path.join(run_dir, f'recon_{i}_meta.txt'), 'w') as f:
                f.write(f"inverted: False\n")

            # === global top-k activation boxes ===
            with torch.no_grad():
                A = F.relu(model.trunk.conv1(x_orig.to(device)))  # (1,C,H,W)
                total = A.numel()
                k = max(1, int(math.ceil(model.trunk.global_topk_ratio * total)))
                flat = A.abs().view(-1)
                if k >= flat.numel():
                    mask_global = torch.ones_like(A, dtype=torch.bool)
                else:
                    threshold, _ = torch.kthvalue(flat, flat.numel() - k + 1)
                    mask_global = A.abs() >= threshold
                A_global = A * mask_global.to(A.dtype)  # (1,C,H,W)

            orig_img = x_orig.squeeze(0).squeeze(0).cpu().numpy()
            draw_activation_and_receptive_fields(
                orig_img,
                A_global.cpu(),
                kernel_size=9,  # conv1 kernel size
                out_path=os.path.join(run_dir, f'activation_full_{i}.png'),
                title=f'IMG{idx}'
            )
if __name__ == '__main__':
    main()
