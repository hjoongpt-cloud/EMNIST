# ==========================
# File: scripts/dump_logits.py
# ==========================
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
# adjust imports to your project
from src.training.train_expert_analyze import ExpertModel  # or your model file
from torchvision.datasets import EMNIST
from torchvision import transforms


def dump_logits(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load cfg if needed to build model (simplified here)
    
    cfg = yaml.safe_load(open(args.config))

    model = ExpertModel(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = EMNIST(root='data', split='balanced', train=False, download=True, transform=transform)
    if args.sample_ratio < 1.0:
        n = len(ds)
        sel = np.random.choice(n, int(n*args.sample_ratio), replace=False)
        ds = torch.utils.data.Subset(ds, sel)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_logits = []
    all_labels = []
    all_preds  = []
    all_gate_p = []
    all_preds_top2 = []

    all_logits_all = []   # [B,E,C]
    all_best_e     = []   # oracle expert idx per sample
    all_chosen_e   = []   # router chosen expert idx
    all_ce_best    = []   # oracle CE per sample
    all_ce_chosen  = []   # chosen CE per sample
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.oracle:
                logits_all, p, gate_logits = model.forward_all_expert_logits(x)  # [B,E,C]
                B,E,C = logits_all.shape
                idx = torch.arange(B, device=device)
                chosen_e = p.argmax(1)
                logits_chosen = logits_all[idx, chosen_e]
                ce = torch.nn.CrossEntropyLoss(reduction='none')
                ce_chosen = ce(logits_chosen, y)
                ce_all = ce(logits_all.view(-1, C), y.repeat_interleave(E)).view(B,E)
                best_e  = ce_all.argmin(1)
                ce_best = ce_all[idx, best_e]
                topv, topi = p.topk(k=min(2, p.size(1)), dim=1)     # [B,2], [B,2]
                p2 = torch.zeros_like(p).scatter(1, topi, topv)
                p2 = p2 / (p2.sum(dim=1, keepdim=True) + 1e-9)
                mix_top2 = (p2.unsqueeze(-1) * logits_all).sum(dim=1)   # [B,C]
                preds_top2 = mix_top2.argmax(1)
                all_logits.append(logits_chosen.cpu().numpy())
                all_preds.append(logits_chosen.argmax(1).cpu().numpy())
                all_gate_p.append(p.cpu().numpy())
                all_logits_all.append(logits_all.cpu().numpy())
                all_best_e.append(best_e.cpu().numpy())
                all_chosen_e.append(chosen_e.cpu().numpy())
                all_ce_best.append(ce_best.cpu().numpy())
                all_ce_chosen.append(ce_chosen.cpu().numpy())
                all_preds_top2.append(preds_top2.cpu().numpy())
            else:
                logits, p = model(x)
                all_logits.append(logits.cpu().numpy())
                all_preds.append(logits.argmax(1).cpu().numpy())
                if p is not None:
                    all_gate_p.append(p.cpu().numpy())

            all_labels.append(y.cpu().numpy())

    # 저장
    data = {
        "logits": np.concatenate(all_logits),
        "labels": np.concatenate(all_labels),
        "preds":  np.concatenate(all_preds),
    }
    if all_gate_p:      data["gate_p"]      = np.concatenate(all_gate_p)
    if args.oracle:
        data.update({
            "logits_all":  np.concatenate(all_logits_all),
            "best_e":      np.concatenate(all_best_e),
            "chosen_e":    np.concatenate(all_chosen_e),
            "ce_best":     np.concatenate(all_ce_best),
            "ce_chosen":   np.concatenate(all_ce_chosen),
            "preds_top2":  np.concatenate(all_preds_top2),
            
        })
    np.savez_compressed(args.out, **data)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_ratio', type=float, default=1.0)
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--oracle', action='store_true')
    args = ap.parse_args()
    dump_logits(args)