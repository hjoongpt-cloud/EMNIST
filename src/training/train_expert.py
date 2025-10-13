#!/usr/bin/env python3
import os
import random
import math
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.models.expert_moe import MoELayer, LoadBalanceLoss, EntropyLoss
from src.data.augmentations import CenterBoxTransform

print(">>> RUNNING train_expert.py <<<")

# image centering augmentation
move_center = CenterBoxTransform()

# --- Seed setup ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Expert Model ---
class ExpertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg['model']
        # conv→attention trunk
        self.conv1 = nn.Conv2d(1, 55, kernel_size=9)
        self.conv2 = nn.Conv2d(55, D['embed_dim'], kernel_size=3)
        self.dropout2 = nn.Dropout(D.get('conv2_dropout', 0.0))
        self.conv2corr_lambda = cfg.get('conv2corr_lambda', 0.0)
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)
        self.attn = nn.MultiheadAttention(D['embed_dim'], D['num_heads'], batch_first=True)
        # MoE layer
        self.moe = MoELayer(
            input_dim=D['embed_dim'],
            num_experts=D['num_experts'],
            expert_hidden=D['expert_hidden'],
            expert_out=D['embed_dim'],
            top_k=D.get('top_k', 1)
        )
        # classifier
        self.classifier = nn.Linear(D['embed_dim'], cfg['model']['num_classes'])
        # gating penalties
        self.lb_loss = LoadBalanceLoss(D['num_experts'])
        self.ent_loss = EntropyLoss()
        # move-center penalty
    def conv2corr_penalty(self):
        if self.conv2corr_lambda == 0:
            return 0.0
        w = self.conv2.weight
        filters = w.view(w.size(0), -1)
        normed = filters / (filters.norm(dim=1, keepdim=True) + 1e-6)
        corr = normed @ normed.T
        off = corr - torch.eye(corr.size(0), device=corr.device)
        return self.conv2corr_lambda * (off**2).sum()
    def _2d_positional_encoding(self, h, w):
        if self.pos_cache.numel() != h * w * self.conv2.out_channels:
            dim = self.conv2.out_channels
            pe = torch.zeros(h, w, dim, device=self.conv2.weight.device)
            y = torch.arange(h, device=pe.device).unsqueeze(1)
            x = torch.arange(w, device=pe.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2, device=pe.device) *
                            -(math.log(10000.0) / dim))
            pe[..., 0::2] = torch.sin(y * div)
            pe[..., 1::2] = torch.cos(x * div)
            self.pos_cache = pe.view(h * w, dim)
        return self.pos_cache
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        b, c, h, w = x.shape
        penalty = self.conv2corr_penalty()

        seq = x.view(b, x.size(1), -1).permute(0, 2, 1)
        pe = self._2d_positional_encoding(h, w)
        seq = seq + pe.unsqueeze(0)

        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(dim=1)

        # === MoE 우회 처리 ===
        moe_out, p = self.moe(pooled)

        logits = self.classifier(moe_out)
        return logits, p, penalty

# --- Training epoch with OHEM, conditional gating, and move_center penalty ---
def train_epoch(model, loader, criterion, optimizer, device, cfg):
    model.train()
    total_ce = total_lb = total_ent = total_mc = 0.0
    correct = total = 0
    gtype = cfg['model']['gating']['type']

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, p, penalty = model(x)

        # per-sample CE
        ce_per = F.cross_entropy(logits, y, reduction='none')
        ce = ce_per.mean()

        # OHEM selection
        if cfg['model']['ohem'].get('use_ohem', False):
            k = max(1, int(cfg['model']['ohem']['keep_ratio'] * y.size(0)))
            hard_idx = torch.topk(ce_per, k).indices
            p_sel = p[hard_idx]
        else:
            p_sel = p

        # gating penalties
        lb = torch.tensor(0.0, device=device)
        ent = torch.tensor(0.0, device=device)
        if gtype == 'load_balance':
            lb = cfg['model']['gating']['lambda_load'] * model.lb_loss(p_sel)
        elif gtype == 'entropy':
            ent = cfg['model']['gating']['lambda_ent'] * model.ent_loss(p_sel)
        elif gtype == 'both':
            lb = cfg['model']['gating']['lambda_load'] * model.lb_loss(p_sel)
            ent = cfg['model']['gating']['lambda_ent'] * model.ent_loss(p_sel)

        loss = ce + lb + ent
        loss.backward()
        optimizer.step()

        # accumulate metrics
        bs = y.size(0)
        total_ce += ce.item() * bs
        total_lb += lb.item() * bs
        total_ent += ent.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return total_ce/total, total_lb/total, total_ent/total, total_mc/total, correct/total

# --- Evaluation ---
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct/total

# --- Feature extraction ---
def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            x1 = F.relu(model.conv1(x))
            x2 = F.relu(model.conv2(x1))
            b, c, h, w = x2.shape
            seq = x2.view(b, x2.size(1), -1).permute(0, 2, 1)
            pe = model._2d_positional_encoding(h, w)
            seq = seq + pe.unsqueeze(0)
            attn_out, _ = model.attn(seq, seq, seq)
            moe_out, _ = model.moe(attn_out.mean(dim=1))
        
            feats.append(moe_out.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

# --- Main entry ---
def main():
    parser = argparse.ArgumentParser(description='Train ExpertModel sweeps')
    parser.add_argument('--config', '-c', required=True, help='YAML config')
    parser.add_argument('--out_dir', '-o', required=True, help='Output base dir')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds  = cfg.get('seeds', [0, 1, 2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # global summary collector
    all_results = []   # list of (seed, test_acc, probe_acc)

    # optional normalize cfg
    nm = cfg.get("normalize", {"mean": [0.1307], "std": [0.3081]})
    mean, std = nm["mean"], nm["std"]

    for seed in seeds:
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)

        # ====== Transforms (match train_base) ======
        train_transform = transforms.Compose([
            move_center,
            # Uncomment if you really want RandAugment (and cfg has it)
            # transforms.RandAugment(
            #     num_ops=cfg['augment']['params']['randaugment']['num_ops'],
            #     magnitude=cfg['augment']['params']['randaugment']['magnitude']
            # ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            move_center,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # ====== Dataset / Loader ======
        train_ds = EMNIST(root='data', split='balanced', train=True,  download=True, transform=train_transform)
        test_ds  = EMNIST(root='data', split='balanced', train=False, download=True, transform=test_transform)
        train_loader = DataLoader(train_ds, batch_size=cfg['opt']['batch_size'], shuffle=True,  num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=cfg['opt']['batch_size'], shuffle=False, num_workers=4)

        # ====== Model / Optimizer / Criterion ======
        model = ExpertModel(cfg).to(device)

        lr = float(cfg['opt'].get('lr', 1e-3))
        wd = float(cfg['opt'].get('weight_decay', 0.0))
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        # ====== Train ======
        history = {"epoch": [], "ce": [], "lb": [], "ent": [], "mc": [], "acc": []}
        for epoch in range(1, cfg['opt']['epochs'] + 1):
            ce, lb, ent, mc, acc = train_epoch(model, train_loader, criterion, optimizer, device, cfg)
            print(f"Seed {seed} Epoch {epoch:02d} | CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} MC {mc:.4f} Acc {acc:.4f}")
            history["epoch"].append(epoch)
            history["ce"].append(ce)
            history["lb"].append(lb)
            history["ent"].append(ent)
            history["mc"].append(mc)
            history["acc"].append(acc)

        # ====== Eval & Probe ======
        test_acc = evaluate(model, test_loader, device)
        X_tr, y_tr = extract_features(model, train_loader, device)
        X_te, y_te = extract_features(model, test_loader, device)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        probe_acc = clf.score(X_te, y_te)
        print(f"Seed {seed} Test Acc {test_acc:.4f} Probe Acc {probe_acc:.4f}")
        all_results.append((seed, test_acc, probe_acc))

        # ====== Save per-seed artifacts ======
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))

        with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
            f.write(f"test_acc: {test_acc:.4f}\nprobe_acc: {probe_acc:.4f}\n")

        with open(os.path.join(run_dir, 'history.csv'), 'w') as f:
            f.write("epoch,ce,lb,ent,mc,acc\n")
            for e, ce_, lb_, ent_, mc_, acc_ in zip(history["epoch"],
                                                    history["ce"],
                                                    history["lb"],
                                                    history["ent"],
                                                    history["mc"],
                                                    history["acc"]):
                f.write(f"{e},{ce_:.6f},{lb_:.6f},{ent_:.6f},{mc_:.6f},{acc_:.6f}\n")

        # ====== Confusion matrices per expert ======
        all_true, all_pred, all_expert = [], [], []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, p, _ = model(x)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(logits.argmax(dim=1).cpu().tolist())
                if p is None:
                    all_expert.extend([0] * logits.size(0))
                else:
                    all_expert.extend(p.argmax(dim=1).cpu().tolist())

        num_exp = cfg['model']['num_experts']
        labels = list(range(cfg['model']['num_classes']))
        for e in range(num_exp):
            idxs = [i for i, ex in enumerate(all_expert) if ex == e]
            if not idxs:
                continue
            t_e = [all_true[i] for i in idxs]
            p_e = [all_pred[i] for i in idxs]
            cm = confusion_matrix(t_e, p_e, labels=labels)
            disp = ConfusionMatrixDisplay(cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap='Blues', colorbar=False, include_values=False)
            ax.set_title(f"Seed {seed} Expert {e} Confusion Matrix")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f'expert_{e}_confmat.png'))
            plt.close(fig)

    # ====== Global summary ======
    sum_path = os.path.join(args.out_dir, "summary.txt")
    mean_test  = np.mean([r[1] for r in all_results])
    mean_probe = np.mean([r[2] for r in all_results])
    with open(sum_path, "w") as f:
        f.write("Seed,TestAcc,ProbeAcc\n")
        for sd, ta, pa in all_results:
            f.write(f"{sd},{ta:.4f},{pa:.4f}\n")
        f.write(f"mean,{mean_test:.4f},{mean_probe:.4f}\n")


if __name__ == '__main__':
    main()
