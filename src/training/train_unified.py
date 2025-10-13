#!/usr/bin/env python3
import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

from src.data.augmentations import CenterBoxTransform  # 기존 사용 그대로
from src.models.trunk import Trunk
from src.models.expert_moe import MoELayer, LoadBalanceLoss, EntropyLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms(cfg):
    move_center = CenterBoxTransform()
    mean = cfg.get("normalize", {"mean":[0.1307]})["mean"]
    std  = cfg.get("normalize", {"std":[0.3081]})["std"]
    train_tfm = transforms.Compose([
        move_center,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_tfm = transforms.Compose([
        move_center,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tfm, test_tfm

class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg['model']
        self.use_moe = cfg.get('use_moe', False)
        self.gating_type = mcfg['gating']['type']
        self.num_experts = mcfg['num_experts']
        self.embed_dim   = mcfg['embed_dim']

        # trunk
        self.trunk = Trunk(
            embed_dim=mcfg['embed_dim'],
            num_heads=mcfg['num_heads'],
            conv1_channels=mcfg.get('conv1_channels', 55),
            conv2_dropout=mcfg.get('conv2_dropout', 0.0),
            conv2corr_lambda=mcfg.get('conv2corr_lambda', 0.0)
        )

        # head (for no-MoE path)
        self.head = nn.Sequential(
            nn.Linear(mcfg['embed_dim'], 4 * mcfg['embed_dim']),
            nn.ReLU(),
            nn.Linear(4 * mcfg['embed_dim'], mcfg['embed_dim'])
        )

        # MoE
        self.moe = MoELayer(
            input_dim=mcfg['embed_dim'],
            num_experts=mcfg['num_experts'],
            expert_hidden=mcfg['expert_hidden'],
            expert_out=mcfg['embed_dim'],
            top_k=mcfg.get('top_k', 1)
        )

        # classifier
        self.classifier = nn.Linear(mcfg['embed_dim'], mcfg['num_classes'])

        # gate penalties
        self.lb_loss = LoadBalanceLoss(mcfg['num_experts'])
        self.ent_loss = EntropyLoss()

    def forward(self, x):
        pooled, penalty = self.trunk(x)

        if not self.use_moe or self.num_experts == 1 or self.gating_type == 'vanilla':
            # D 경로
            feats = self.head(pooled)
            logits = self.classifier(feats)
            return logits, None, penalty

        moe_out, p = self.moe(pooled)
        logits = self.classifier(moe_out)
        return logits, p, penalty

    def forward_features(self, x):
        pooled, _ = self.trunk(x)
        if not self.use_moe or self.num_experts == 1 or self.gating_type == 'vanilla':
            return self.head(pooled)
        else:
            moe_out, _ = self.moe(pooled)
            return moe_out

def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_ce = total_lb = total_ent = 0.0
    correct = total = 0

    use_ohem = cfg['model']['ohem'].get('use_ohem', False)
    keep_ratio = cfg['model']['ohem'].get('keep_ratio', 0.5)
    gtype = cfg['model']['gating']['type']

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, p, _ = model(x)

        # per-sample CE
        ce_per = F.cross_entropy(logits, y, reduction='none')

        if use_ohem:
            k = max(1, int(keep_ratio * y.size(0)))
            hard_idx = torch.topk(ce_per, k).indices
            ce = ce_per[hard_idx].mean()
            p_sel = None if p is None else p[hard_idx]
        else:
            ce = ce_per.mean()
            p_sel = p

        # gating penalties
        lb = ent = torch.tensor(0.0, device=device)
        if gtype == 'load_balance':
            lb = cfg['model']['gating']['lambda_load'] * model.lb_loss(p_sel)
        elif gtype == 'entropy':
            ent = cfg['model']['gating']['lambda_ent'] * model.ent_loss(p_sel)

        loss = ce + lb + ent
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_ce += ce.item() * bs
        total_lb += lb.item() * bs
        total_ent += ent.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return total_ce/total, total_lb/total, total_ent/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct/total

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    for x, y in loader:
        x = x.to(device)
        f = model.forward_features(x)
        feats.append(f.cpu().numpy())
        labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--out_dir', '-o', required=True)
    parser.add_argument('--use_moe', action='store_true', help='turn on MoE path')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # override by cli
    cfg['use_moe'] = args.use_moe

    os.makedirs(args.out_dir, exist_ok=True)

    # Deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tfm, test_tfm = build_transforms(cfg)

    train_ds = EMNIST(root=cfg.get('data_root', 'data'),
                      split='balanced', train=True,
                      download=True, transform=train_tfm)
    test_ds = EMNIST(root=cfg.get('data_root', 'data'),
                     split='balanced', train=False,
                     download=True, transform=test_tfm)

    seeds = cfg.get('seeds', [42, 43, 44])
    results = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        set_seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds,
                                  batch_size=cfg['opt']['batch_size'],
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  generator=g)
        test_loader = DataLoader(test_ds,
                                 batch_size=cfg['opt']['batch_size'],
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

        model = UnifiedModel(cfg).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(cfg['opt']['lr']),
            weight_decay=cfg['opt'].get('weight_decay', 0.0)
        )

        for epoch in range(1, cfg['opt']['epochs'] + 1):
            ce, lb, ent, acc = train_epoch(model, train_loader, optimizer, device, cfg)
            print(f"Epoch {epoch:02d} | CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} Acc {acc:.4f}")

        test_acc = evaluate(model, test_loader, device)
        feats_tr, labs_tr = extract_features(model, train_loader, device)
        feats_te, labs_te = extract_features(model, test_loader, device)

        scaler = StandardScaler()
        feats_tr = scaler.fit_transform(feats_tr)
        feats_te = scaler.transform(feats_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(max_iter=1000, solver='saga')
            clf.fit(feats_tr, labs_tr)
        probe_acc = clf.score(feats_te, labs_te)
        print(f"Test Acc {test_acc:.4f} | Probe {probe_acc:.4f}")

        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
        with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
            f.write(f"test_acc: {test_acc:.4f}\nprobe_acc: {probe_acc:.4f}\n")

        results.append(test_acc)

    # summary
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write("Seed,TestAcc\n")
        for sd, acc in zip(seeds, results):
            f.write(f"{sd},{acc:.4f}\n")
        f.write(f"mean,{np.mean(results):.4f}\n")

if __name__ == '__main__':
    main()
