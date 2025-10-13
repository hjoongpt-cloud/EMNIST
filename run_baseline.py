#!/usr/bin/env python3
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import EMNIST
from torchvision import transforms
from src.data.transforms import InvertIfSmart  # 실험 B: 밝기/명암 대비 인버트 함수
from src.data.augmentations import CenterBoxTransform  # CenterBoxTransform 호출 경로 수정
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# CenterBoxTransform 인스턴스 생성
move_center = CenterBoxTransform()

# --- Seed 설정 함수 ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- BasicModel 구현 (Attention + PE 포함) ---
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
        return self.conv2corr_lambda * (off ** 2).sum()

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

# --- 학습/평가 함수 ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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

# --- 실험 실행 함수 ---
def run_attn_with_pe(seeds, data_root, out_root,
                     embed_dim=32, num_heads=4,
                     conv2_dropout=0.0, conv2corr_lambda=0.0,
                     epochs=20, batch_size=128, lr=1e-3):
    test_accs, probe_accs = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        print(f"[attn_with_pe | seed={seed}] 시작")
        set_seed(seed)
        transform = transforms.Compose([
            # move_center,  # 인스턴스를 바로 전달
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = EMNIST(root=data_root, split='balanced', train=True,
                          download=True, transform=transform)
        test_ds  = EMNIST(root=data_root, split='balanced', train=False,
                          download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, num_workers=4)

        model = BasicModel(embed_dim, num_heads,
                           len(train_ds.classes),
                           conv2_dropout, conv2corr_lambda).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = 3e-5)
        history = {'loss': [], 'acc': []}

        for epoch in tqdm(range(1, epochs+1)):
            loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
            history['loss'].append(loss)
            history['acc'].append(acc)

        test_acc = evaluate_model(model, test_loader, device)
        X_train, y_train = extract_features(model, train_loader, device)
        X_test,  y_test  = extract_features(model, test_loader, device)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        probe_acc = clf.score(X_test, y_test)

        test_accs.append(test_acc)
        probe_accs.append(probe_acc)

        exp_dir = os.path.join(out_root, 'attn_with_pe_centered', f"seed_{seed}")
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pt"))
        with open(os.path.join(exp_dir, "results.txt"), 'w') as f:
            f.write(f"seed: {seed}\n")
            f.write(f"test_acc: {test_acc:.4f}\n")
            f.write(f"probe_acc: {probe_acc:.4f}\n")
            f.write("epoch,loss,acc\n")
            for i,(l,a) in enumerate(zip(history['loss'], history['acc']),1):
                f.write(f"{i},{l:.4f},{a:.4f}\n")

    # 평균 결과 저장
    mean_test = np.mean(test_accs)
    mean_probe = np.mean(probe_accs)
    summary_dir = os.path.join(out_root, 'attn_with_pe_adamWdecay3e5')
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "summary.txt"), 'w') as f:
        f.write("Seed,TestAcc,ProbeAcc\n")
        for sd, t, p in zip(seeds, test_accs, probe_accs):
            f.write(f"{sd},{t:.4f},{p:.4f}\n")
        f.write(f"mean,{mean_test:.4f},{mean_probe:.4f}\n")

# --- 메인 ---
if __name__ == '__main__':
    data_root = 'data'
    out_root = 'baseline_test'
    os.makedirs(out_root, exist_ok=True)
    seeds = [42, 43, 44]
    run_attn_with_pe(seeds, data_root, out_root)
