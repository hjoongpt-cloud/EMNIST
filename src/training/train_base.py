#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

from src.data.transforms import InvertIfSmart
from src.data.augmentations import build_augment_pipeline
from src.data.augmentations import CenterBoxTransform
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(">>> RUNNING UPDATED train_base.py <<<")
move_center = CenterBoxTransform()
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BasicModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes,
                 conv2_dropout=0.0, conv2corr_lambda=0.0,
                 conv1_channels=55):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=9)
        self.conv2 = nn.Conv2d(conv1_channels, embed_dim, kernel_size=3)
        self.dropout2 = nn.Dropout(conv2_dropout)
        self.conv2corr_lambda = conv2corr_lambda
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def conv2corr_penalty(self):
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
        penalty = self.conv2corr_penalty()
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)
        pe = self._2d_positional_encoding(h, w)
        seq = seq + pe.unsqueeze(0)
        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(dim=1)
        head_out = self.head(pooled)
        logits = self.classifier(head_out)
        return logits, penalty

    def forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)
        pe = self._2d_positional_encoding(h, w)
        seq = seq + pe.unsqueeze(0)
        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(dim=1)
        return self.head(pooled)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", required=True,
                        help="YAML config files in order: base, c, ...")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cfg = {}
    for cfile in args.config:
        with open(cfile) as f:
            cfg.update(yaml.safe_load(f))

    transforms_list = []
    if "augmentations" in cfg:
        transforms_list.append(build_augment_pipeline(config_path=args.config[1]))
    transforms_list += [
        move_center,
        transforms.ToTensor(),
        transforms.Normalize(**cfg.get("normalize", {"mean": [0.1307], "std": [0.3081]}))
    ]
    train_tfm = transforms.Compose(transforms_list)
    test_tfm = transforms.Compose([
        move_center,
        transforms.ToTensor(),
        transforms.Normalize(**cfg.get("normalize", {"mean": [0.1307], "std": [0.3081]}))
    ])

    data_root = cfg.get("data_root", "data")
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds = EMNIST(root=data_root, split='balanced', train=True,
                      download=True, transform=train_tfm)
    test_ds = EMNIST(root=data_root, split='balanced', train=False,
                     download=True, transform=test_tfm)
    seeds = cfg.get("seeds", [42, 43, 44])

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        print(f"Running seed {seed}")
        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size",128), shuffle=True, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size",128), shuffle=False, num_workers=4)

        model = BasicModel(
            cfg.get("embed_dim",32), cfg.get("num_heads",4), len(train_ds.classes),
            conv2_dropout=cfg.get("conv2_dropout",0.0),
            conv2corr_lambda=cfg.get("conv2corr_lambda",0.0)
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.get("lr",1e-3))

        history = {"loss": [], "acc": []}
        for epoch in range(1, cfg.get("epochs",20)+1):
            loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
            history["loss"].append(loss)
            history["acc"].append(acc)
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

        test_acc = evaluate_model(model, test_loader, device)
        feats_tr, labs_tr = extract_features(model, train_loader, device)
        feats_te, labs_te = extract_features(model, test_loader, device)
        scaler = StandardScaler()
        feats_tr = scaler.fit_transform(feats_tr)
        feats_te = scaler.transform(feats_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(solver='saga', max_iter=1000, tol=1e-4)
            clf.fit(feats_tr, labs_tr)
        probe_acc = clf.score(feats_te, labs_te)

        exp_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "results.txt"), 'w') as f:
            f.write(f"test_acc: {test_acc:.4f}\n")
            f.write(f"probe_acc: {probe_acc:.4f}\n")
            f.write("epoch,loss,acc\n")
            for i, (l, a) in enumerate(zip(history['loss'], history['acc']), 1):
                f.write(f"{i},{l:.4f},{a:.4f}\n")
        results.append(test_acc)

    # write summary at top level
    summary_path = os.path.join(args.out_dir, 'summary.txt')
    mean_acc = np.mean(results)
    with open(summary_path, 'w') as f:
        f.write("Seed,TestAcc\n")
        for sd, acc in zip(seeds, results):
            f.write(f"{sd},{acc:.4f}\n")
        f.write(f"mean,{mean_acc:.4f}\n")

if __name__ == '__main__':
    main()
