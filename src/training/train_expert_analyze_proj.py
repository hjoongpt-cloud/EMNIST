### src/training/train_expert_analyze_proj.py

#!/usr/bin/env python3
"""
Train + Analyze MoE (Stage G) with ExpertModelProj.
"""
import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_proj import ExpertModelProj
from src.utils.moe_analysis import MoeAnalyzer
from sklearn.linear_model import LogisticRegression

# Reuse training utilities from original script
from src.training.train_expert_analyze import train_epoch, evaluate, extract_features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--no_analysis', action='store_true')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    seeds = cfg.get('seeds', [42, 43, 44])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary = {'seed': [], 'test_acc': [], 'probe_acc': []}
    for seed in seeds:
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        analysis_dir = os.path.join(run_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Data loaders
        nm = cfg.get('normalize', {'mean': [0.1307], 'std': [0.3081]})
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(nm['mean'], nm['std'])
        ])
        train_ds = EMNIST('data', split='balanced', train=True, download=True, transform=transform)
        test_ds = EMNIST('data', split='balanced', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=cfg.opt.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=cfg.opt.batch_size, shuffle=False, num_workers=4)

        # Model and optimizer
        model = ExpertModelProj(cfg).to(device)
        model.forward_features = model.trunk
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.opt.lr)

        # Analyzer
        analyzer = None if args.no_analysis else MoeAnalyzer(cfg.model.num_experts, cfg.model.num_classes)
        history = {'epoch': [], 'ce': [], 'lb': [], 'ent': [], 'acc': []}

        # Training loop
        for ep in range(1, cfg.opt.epochs + 1):
            if analyzer:
                analyzer.begin_epoch()
            ce, lb, ent, acc = train_epoch(model, train_loader, optimizer, device, cfg, analyzer, ep)
            if analyzer:
                stats = analyzer.end_epoch()
                analyzer.finalize_epoch(analysis_dir, ep, model)
                with open(os.path.join(analysis_dir, f'stats_ep{ep:03d}.json'), 'w') as f:
                    json.dump(stats, f, indent=2)
            history['epoch'].append(ep)
            history['ce'].append(ce)
            history['lb'].append(lb)
            history['ent'].append(ent)
            history['acc'].append(acc)
            print(f"Seed {seed} Ep {ep:02d} | CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} ACC {acc:.4f}")

        # Evaluation and probing
        test_acc = evaluate(model, test_loader, device)
        Xtr, ytr = extract_features(model, train_loader, device)
        Xte, yte = extract_features(model, test_loader, device)
        probe_acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)

        summary['seed'].append(seed)
        summary['test_acc'].append(test_acc)
        summary['probe_acc'].append(probe_acc)

        # Save results and model
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
        with open(os.path.join(run_dir, 'results.txt'), 'w') as f:
            f.write(f"test_acc: {test_acc:.4f}\nprobe_acc: {probe_acc:.4f}\n")

    # Global summary
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write('Seed,TestAcc,ProbeAcc\n')
        for sd, ta, pa in zip(summary['seed'], summary['test_acc'], summary['probe_acc']):
            f.write(f"{sd},{ta:.4f},{pa:.4f}\n")
        f.write(f"mean,{np.mean(summary['test_acc']):.4f},{np.mean(summary['probe_acc']):.4f}\n")

if __name__ == '__main__':
    main()
