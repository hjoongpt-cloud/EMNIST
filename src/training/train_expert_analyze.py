#!/usr/bin/env python3
"""
Train + Analyze MoE (Stage E) in one place.
- Keeps existing ExpertModel training logic
- Adds rich diagnostics (routing stats, entropy, usage, diversity, grad norms)
- Saves per-seed results (metrics, confusion matrices) AND per-epoch analysis JSON/NPY

Usage
-----
python -m src.training.train_expert_analyze \
  --config configs/e.yaml \
  --out_dir outputs/stage_E/vanilla_analyze

Notes
-----
* Requires src/utils/moe_analysis.py (see previous commit).
* If you don't want analysis overhead, pass --no_analysis.
"""
import os
import math
import json
import argparse
import yaml
import random
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
from src.utils.moe_analysis import MoeAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import torch

# ---------------------- utils ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------- Model ----------------------
class ExpertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        M = cfg['model']
        in_ch = 1
        c1 = M.get('conv1_channels', 55)
        self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=9)
        pf_path = M.get('pretrained_filter_path', None)
        if pf_path:
            filters = np.load(pf_path)  # e.g. shape (C_out, K, K)
            # conv1.weight shape = (C_out, C_in, K, K)
            if filters.ndim == 3:
                filters = filters[:, None, :, :]
            self.conv1.weight.data.copy_(
                torch.from_numpy(filters).to(self.conv1.weight.dtype)
            )
            if self.conv1.bias is not None:
                self.conv1.bias.data.zero_()        
        
        self.conv2 = nn.Conv2d(c1, M['embed_dim'], kernel_size=3)
        self.dropout2 = nn.Dropout(M.get('conv2_dropout', 0.0))

        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)
        self.attn = nn.MultiheadAttention(M['embed_dim'], M['num_heads'], batch_first=True)

        # MoE layer
        self.moe = MoELayer(
            input_dim=M['embed_dim'],
            num_experts=M['num_experts'],
            expert_hidden=M['expert_hidden'],
            expert_out=M['embed_dim'],
            top_k=M.get('top_k', 1)
        )
        self.classifier = nn.Linear(M['embed_dim'], M['num_classes'])

        # penalties
        self.lb_loss = LoadBalanceLoss(M['num_experts'])
        self.ent_loss = EntropyLoss()
        self.conv2corr_lambda = cfg.get('conv2corr_lambda', 0.0)

    # -------- helpers --------
    def conv2corr_penalty(self):
        if self.conv2corr_lambda == 0:
            return torch.tensor(0.0, device=self.conv2.weight.device)
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

    # -------- forward paths --------
    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)
        pe = self._2d_positional_encoding(h, w)
        attn_in = seq + pe.unsqueeze(0)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        return attn_out.mean(dim=1)  # [B, embed]

    def forward(self, x):
        pooled = self.forward_features(x)
        moe_out, p = self.moe(pooled)  # [B, D], [B, E]
        logits = self.classifier(moe_out)
        return logits, p

    def forward_all_expert_logits(self, x):
        pooled = self.forward_features(x)          # [B,D]
        outs_all, p, gate_logits = self.moe.forward_all(pooled)  # [B,E,D], [B,E], [B,E]
        # classifier to logits per expert
        B,E,D = outs_all.shape
        logits_all = self.classifier(outs_all.view(B*E, D)).view(B, E, -1)  # [B,E,C]
        return logits_all, p, gate_logits

# ---------------------- Train / Eval ----------------------
def train_epoch(model, loader, optimizer, device, cfg, analyzer=None, epoch=1):
    """
    One epoch of MoE training + (optional) analysis.
    - Uses full data for training.
    - For analysis (oracle CE, routing stats, grads), only a sampled subset is logged
      if cfg['analyze'] has sample_ratio < 1.0 or every_n_epochs > 1.

    Returns:
        avg_ce, avg_lb, avg_ent, avg_acc
    """
    model.train()
    tot_ce = tot_lb = tot_ent = tot_acc = 0.0
    n_tot = 0

    # analysis options
    an_cfg      = cfg.get('analyze', {})
    samp_ratio  = float(an_cfg.get('sample_ratio', 1.0))
    ep_period   = int(an_cfg.get('every_n_epochs', 1))
    do_analyze  = (analyzer is not None) and (epoch % ep_period == 0)

    g_cfg       = cfg['model']['gating']
    gtype       = g_cfg.get('type', 'vanilla')
    use_ohem    = cfg['model']['ohem'].get('use_ohem', False)
    keep_ratio  = cfg['model']['ohem'].get('keep_ratio', 0.5)

    for x, y in loader:
        bs = y.size(0)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if do_analyze:
            # === 1) 학습 경로 ===
            logits_mix, p_mix = model(x)
            ce_per = F.cross_entropy(logits_mix, y, reduction='none')
            ce     = ce_per.mean()

            # OHEM & penalty
            if use_ohem:
                k = max(1, int(keep_ratio * bs))
                hard_idx = torch.topk(ce_per, k).indices
                p_sel = p_mix[hard_idx]
            else:
                p_sel = p_mix

            lb = ent = torch.tensor(0.0, device=device)
            if gtype in ['load_balance', 'both']:
                lb = g_cfg['lambda_load'] * model.lb_loss(p_sel)
            if gtype in ['entropy', 'both']:
                ent = g_cfg['lambda_ent'] * model.ent_loss(p_sel)

            penalty = model.conv2corr_penalty()
            loss = ce + lb + ent + penalty
            loss.backward()
            optimizer.step()

            # === 2) 분석 경로 ===
            with torch.no_grad():
                logits_all, p_full, gate_logits = model.forward_all_expert_logits(x)
                B, E, C = logits_all.shape
                idx = torch.arange(B, device=device)
                chosen_e = p_mix.argmax(dim=1)

                ce_all = F.cross_entropy(
                    logits_all.view(-1, C),
                    y.repeat_interleave(E),
                    reduction='none'
                ).view(B, E)
                best_e  = ce_all.argmin(dim=1)
                ce_best = ce_all[idx, best_e]

                if samp_ratio < 1.0:
                    mask = torch.rand(B, device=device) < samp_ratio
                else:
                    mask = slice(None)

                analyzer.accumulate_batch(
                    p           = p_mix[mask],
                    y           = y[mask],
                    gate_logits = gate_logits[mask],
                    ce_chosen   = ce_per[mask],
                    ce_best     = ce_best[mask],
                    best_e      = best_e[mask],
                    chosen_e    = chosen_e[mask]
                )
                analyzer.log_grad_norms(model)

            # === 공통 누적 ===
            with torch.no_grad():
                pred = logits_mix.argmax(dim=1)
                acc  = (pred == y).sum().item()
        else:
            # light path
            logits_mix, p_mix = model(x)
            ce_per = F.cross_entropy(logits_mix, y, reduction='none')
            ce     = ce_per.mean()

            if use_ohem:
                k = max(1, int(keep_ratio * bs))
                hard_idx = torch.topk(ce_per, k).indices
                p_sel = p_mix[hard_idx]
            else:
                p_sel = p_mix

            lb = ent = torch.tensor(0.0, device=device)
            if gtype in ['load_balance', 'both']:
                lb = g_cfg['lambda_load'] * model.lb_loss(p_sel)
            if gtype in ['entropy', 'both']:
                ent = g_cfg['lambda_ent'] * model.ent_loss(p_sel)

            penalty = model.conv2corr_penalty()
            #loss = ce + lb + ent + penalty
            loss = ce + lb + ent
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits_mix.argmax(dim=1)
                acc  = (pred == y).sum().item()

        # === 여기서 공통으로 누적 ===
        tot_ce  += ce.item()  * bs
        tot_lb  += lb.item()  * bs
        tot_ent += ent.item() * bs
        tot_acc += acc
        n_tot   += bs

    return tot_ce / n_tot, tot_lb / n_tot, tot_ent / n_tot, tot_acc / n_tot




def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total


def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feat = model.forward_features(x)
            feats.append(feat.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = cfg.get('seeds', [42,43,44])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean_test_all = []
    mean_probe_all = []

    for seed in seeds:
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        analysis_dir = os.path.join(run_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Data
        nm = cfg.get('normalize', {'mean':[0.1307], 'std':[0.3081]})
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(nm['mean'], nm['std'])
        ])
        train_ds = EMNIST(root='data', split='balanced', train=True, download=True, transform=transform)
        test_ds  = EMNIST(root='data', split='balanced', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=cfg['opt']['batch_size'], shuffle=True,  num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=cfg['opt']['batch_size'], shuffle=False, num_workers=4)

        # Model
        model = ExpertModel(cfg).to(device)
        opt = optim.AdamW(model.parameters(), lr=float(cfg['opt']['lr']))

        analyzer = MoeAnalyzer(cfg['model']['num_experts'], cfg['model']['num_classes'])
        history = {'epoch':[], 'ce':[], 'lb':[], 'ent':[], 'acc':[]}

        for ep in range(1, cfg['opt']['epochs']+1):
            analyzer.begin_epoch()
            ce, lb, ent, acc = train_epoch(model, train_loader, opt, device, cfg, analyzer)
            stats = analyzer.end_epoch()
            analyzer.finalize_epoch(analysis_dir, ep, model)

            # save epoch stats json
            with open(os.path.join(analysis_dir, f'stats_ep{ep:03d}.json'),'w') as f:
                json.dump(stats, f, indent=2)

            history['epoch'].append(ep); history['ce'].append(ce); history['lb'].append(lb); history['ent'].append(ent); history['acc'].append(acc)
            print(f"Seed {seed} Ep {ep:02d} | CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} Acc {acc:.4f} | misroute {stats['misroute_rate']:.3f} regret {stats['gate_regret_ce']:.4f}")

        # Eval & probe
        test_acc = evaluate(model, test_loader, device)
        Xtr, ytr = extract_features(model, train_loader, device)
        Xte, yte = extract_features(model, test_loader, device)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        probe_acc = clf.score(Xte, yte)
        mean_test_all.append(test_acc); mean_probe_all.append(probe_acc)
        # save per-seed
        torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
        with open(os.path.join(run_dir, 'results.txt'),'w') as f:
            f.write(f"test_acc: {test_acc:.4f}\nprobe_acc: {probe_acc:.4f}\n")
        with open(os.path.join(run_dir, 'history.csv'),'w') as f:
            f.write('epoch,ce,lb,ent,acc\n')
            for e,ce_,lb_,ent_,acc_ in zip(history['epoch'],history['ce'],history['lb'],history['ent'],history['acc']):
                f.write(f"{e},{ce_:.6f},{lb_:.6f},{ent_:.6f},{acc_:.6f}\n")

    # global summary
    with open(os.path.join(args.out_dir, 'summary.txt'),'w') as f:
        f.write('Seed,TestAcc,ProbeAcc\n')
        for sd,ta,pa in zip(seeds, mean_test_all, mean_probe_all):
            f.write(f"{sd},{ta:.4f},{pa:.4f}\n")
        f.write(f"mean,{np.mean(mean_test_all):.4f},{np.mean(mean_probe_all):.4f}\n")

if __name__ == '__main__':
    import json
    main()

