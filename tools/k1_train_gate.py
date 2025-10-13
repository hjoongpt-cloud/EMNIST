import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_k import ExpertModelJ
from src.training.train_expert_analyze import extract_features

NUM_EXPERTS = 4
DEFAULT_CLASS_TO_EXPERT = {
    0:0, 24:0,
    1:1, 18:1, 21:1,
    9:2, 44:2,
    15:3, 40:3
}

class GatingNet(nn.Module):
    def __init__(self, input_dim, num_experts=NUM_EXPERTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_experts)
        )
    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trunk_config', required=True)
    ap.add_argument('--k0_ckpt', required=True, help='k0_single.pt (단일 expert) 경로')
    ap.add_argument('--out_path', required=True, help='게이트 ckpt 저장 경로(.pt)')
    ap.add_argument('--alpha_sup', type=float, default=0.7)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--class_to_expert_json', default=None, help='선택: 매핑 json 파일 경로')
    ap.add_argument('--metrics_out', default=None, help='선택: 최종 게이트 지표 JSON 저장 경로')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === cfg 로딩 및 Trunk 준비 ===
    cfg = OmegaConf.load(args.trunk_config)
    model = ExpertModelJ(cfg).to(device)
    ck = torch.load(args.k0_ckpt, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(ck, strict=False)

    for p in model.trunk.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(nm['mean'], nm['std'])
    ])
    train_ds = EMNIST('data', split='balanced', train=True,  download=True, transform=transform)
    test_ds  = EMNIST('data', split='balanced', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === 32D 임베딩 추출 ===
    Xtr, ytr = extract_features(model, train_loader, device)  # (N,32), (N,)
    Xte, yte = extract_features(model, test_loader, device)

    # numpy -> torch 텐서로 강제 변환 (dtype 고정)
    if isinstance(Xtr, np.ndarray): Xtr = torch.from_numpy(Xtr)
    if isinstance(Xte, np.ndarray): Xte = torch.from_numpy(Xte)
    if isinstance(ytr, np.ndarray): ytr = torch.from_numpy(ytr)
    if isinstance(yte, np.ndarray): yte = torch.from_numpy(yte)
    Xtr = Xtr.float(); Xte = Xte.float()
    ytr = ytr.long();  yte = yte.long()

    # 차원은 shape로 읽는 게 안전
    D = Xtr.shape[1]

    # === 게이트 ===
    gating = GatingNet(input_dim=D, num_experts=NUM_EXPERTS).to(device)
    opt = optim.AdamW(gating.parameters(), lr=1e-3)
    kl = nn.KLDivLoss(reduction='batchmean')

    # 클래스-전문가 매핑
    if args.class_to_expert_json:
        with open(args.class_to_expert_json, 'r') as f:
            class_to_expert = {int(k): int(v) for k,v in json.load(f).items()}
    else:
        class_to_expert = DEFAULT_CLASS_TO_EXPERT

    cls2exp = torch.full((47,), -1, dtype=torch.long)
    for c,e in class_to_expert.items():
        cls2exp[c] = e

    cls2exp_dev = cls2exp.to(device)
    uniform = torch.full((NUM_EXPERTS,), 1.0/NUM_EXPERTS, device=device)

    def make_targets(y):
        e_idx = cls2exp_dev[y]  # (B,)
        mask = e_idx.ge(0)
        target = uniform.repeat(y.size(0), 1)
        if mask.any():
            oh = torch.zeros(mask.sum(), NUM_EXPERTS, device=device)
            oh.scatter_(1, e_idx[mask].unsqueeze(1), 1.0)
            target[mask] = oh
        return target, mask

    def eval_gate(split_name, X, y):
        gating.eval()
        B = args.batch_size
        with torch.no_grad():
            logits_list = []
            for i in range(0, X.size(0), B):
                xb = X[i:i+B].to(device)
                logits_list.append(gating(xb))
            logits = torch.cat(logits_list, dim=0)
            pred = logits.argmax(1).cpu()
        mapped = cls2exp[y].ge(0)
        mapped_cnt = int(mapped.sum().item())
        sup_acc = (pred[mapped] == cls2exp[y][mapped]).float().mean().item() if mapped_cnt > 0 else 0.0
        print(f"[K1][{split_name}] SupAcc: {sup_acc:.4f} | mapped {mapped_cnt}/{y.size(0)}")
        return sup_acc, mapped_cnt

    # === 학습 ===
    print("[K1] Training gate on pooled features...")
    gating.train()
    N = Xtr.size(0)
    B = args.batch_size

    for ep in range(1, args.epochs+1):
        perm = torch.randperm(N)
        total_loss, sup_corr, sup_cnt = 0.0, 0, 0
        for i in range(0, N, B):
            idx = perm[i:i+B]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)

            logits = gating(xb)
            logp = F.log_softmax(logits, dim=1)

            target, mask = make_targets(yb)
            loss_all = kl(logp, target)
            if mask.any():
                loss_sup = kl(logp[mask], target[mask])
                loss = args.alpha_sup * loss_sup + (1 - args.alpha_sup) * loss_all
            else:
                loss = loss_all

            opt.zero_grad(); loss.backward(); opt.step()

            total_loss += loss.item() * xb.size(0)
            if mask.any():
                pred = logits.argmax(1)
                sup_corr += (pred[mask] == cls2exp_dev[yb[mask]]).sum().item()
                sup_cnt  += mask.sum().item()

        avg = total_loss / N
        sup_acc = (sup_corr / sup_cnt) if sup_cnt > 0 else 0.0
        print(f"[K1] Ep{ep:02d} KL {avg:.4f} | SupAcc {sup_acc:.4f} (count {sup_cnt})")

    # === 최종 평가 ===
    train_sup_acc, train_mapped = eval_gate('train', Xtr, ytr)
    test_sup_acc,  test_mapped  = eval_gate('test',  Xte, yte)

    # === 저장 ===
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save({
        'state_dict': gating.state_dict(),
        'input_dim': int(D),
        'num_experts': NUM_EXPERTS,
        'class_to_expert': class_to_expert,
        'alpha_sup': args.alpha_sup,
        'train_sup_acc': float(train_sup_acc),
        'test_sup_acc': float(test_sup_acc),
    }, args.out_path)
    print(f"[K1] Saved gate → {args.out_path}")

    # === 지표 JSON(선택) ===
    metrics = {
        'train_sup_acc': float(train_sup_acc), 'train_mapped': int(train_mapped), 'train_total': int(Xtr.size(0)),
        'test_sup_acc':  float(test_sup_acc),  'test_mapped':  int(test_mapped),  'test_total':  int(Xte.size(0)),
    }
    metrics_path = args.metrics_out or os.path.splitext(args.out_path)[0] + '_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[K1] Wrote metrics → {metrics_path}")


if __name__ == '__main__':
    main()