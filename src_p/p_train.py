# -*- coding: utf-8 -*-
# p_train.py
#
# O-stage from-scratch training with slots (s-only).
# - conv1: load pretrained 9x9 filters -> 14x14x150
# - 1x1 projection -> 64D
# - 1-layer self-attn on tokens
# - learnable slot queries -> per-slot attention maps
# - (optional) spatial mask (pair or comb4) -> re-normalize
# - per-slot pooled features -> shared classifier -> P-weighted sum logits
#
# Safe-by-default training (warm-up, AMP optional, grad clip, NaN guards)

import os, math, argparse
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools

# ----------------------------- utils -----------------------------

def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def assert_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"{name} has NaN/Inf")

# ----------------------------- spatial pair / comb4 mask -----------------------------

def grid_boxes(H: int, W: int, gh: int, gw: int) -> List[Tuple[int,int,int,int]]:
    """Divide HxW into ghxgw cells and return inclusive boxes (x0,y0,x1,y1).
       Special-case 14x14@3x3 to match legacy: edges [0,5,10,14] (5-5-4)."""
    if H == 14 and W == 14 and gh == 3 and gw == 3:
        edges = [0, 5, 10, 14]
        boxes = []
        for r in range(3):
            for c in range(3):
                x0, x1 = edges[c], edges[c+1]-1
                y0, y1 = edges[r], edges[r+1]-1
                boxes.append((x0,y0,x1,y1))
        return boxes
    boxes = []
    for r in range(gh):
        for c in range(gw):
            x0 = int(round(c * (W/float(gw))))
            y0 = int(round(r * (H/float(gh))))
            x1 = int(round((c+1)*(W/float(gw)))) - 1
            y1 = int(round((r+1)*(H/float(gh)))) - 1
            boxes.append((x0,y0,x1,y1))
    return boxes

def round_comb4_for_M(M, gh=3, gw=3):
    """9C4 조합을 슬롯 순번에 원형 할당."""
    idxs = list(itertools.combinations(range(gh*gw), 4))  # 길이 126
    return [idxs[m % len(idxs)] for m in range(M)], idxs

def round_comb4_mask_token(M, H, W, device, gh=3, gw=3):
    """슬롯별(14x14) comb4 고정 마스크 텐서 생성: (M,H,W)"""
    boxes = grid_boxes(H, W, gh, gw)
    comb_for_m, _ = round_comb4_for_M(M, gh, gw)
    mask = torch.zeros(M, H, W, device=device)
    for m, comb in enumerate(comb_for_m):
        for g in comb:
            x0,y0,x1,y1 = boxes[g]
            mask[m, y0:y1+1, x0:x1+1] = 1.0
    return mask

def auto_pairs_for_batch(A_bmhw: torch.Tensor, boxes: List[Tuple[int,int,int,int]]):
    """Pick, per (b,m), the top-2 grid cells by mass."""
    B,M,H,W = A_bmhw.shape
    G = len(boxes)
    pairs_all = []
    for b in range(B):
        mass = torch.zeros(M, G, device=A_bmhw.device, dtype=A_bmhw.dtype)
        for g,(x0,y0,x1,y1) in enumerate(boxes):
            mass[:, g] = A_bmhw[b, :, y0:y1+1, x0:x1+1].flatten(1).sum(dim=1)
        top2 = torch.topk(mass, k=min(2,G), dim=1).indices  # (M,2)
        pairs = [(int(top2[m,0].item()), int(top2[m,1].item())) for m in range(M)]
        pairs_all.append(pairs)
    return pairs_all

def round_pairs_for_slots(M: int, boxes: List[Tuple[int,int,int,int]]):
    G = len(boxes)
    all_pairs = []
    for i in range(G):
        for j in range(i+1, G):
            all_pairs.append((i,j))
    if len(all_pairs) == 0:
        all_pairs = [(0,0)]
    return [all_pairs[m % len(all_pairs)] for m in range(M)]

def _eps_for_dtype(dtype: torch.dtype) -> float:
    return 1e-4 if dtype == torch.float16 else 1e-8

def apply_spatial_pair_mask(A_bmhw: torch.Tensor, enable: bool, grid: int, assign: str):
    """Mask A to union of 2 grid cells per slot. Re-normalize per-slot to sum=1.
       Returns (A_eff, mass_raw) where:
         - A_eff: masked & renormalized maps (B,M,H,W)
         - mass_raw: masked mass BEFORE renorm (B,M), for P computation
    """
    if not enable:
        mass_raw = A_bmhw.float().flatten(2).sum(-1)  # each slot sums to 1 over HxW
        return A_bmhw, mass_raw
    B,M,H,W = A_bmhw.shape
    boxes = grid_boxes(H,W,grid,grid)
    if assign == "auto":
        pairs_all = auto_pairs_for_batch(A_bmhw, boxes)
    else:
        base_pairs = round_pairs_for_slots(M, boxes)
        pairs_all = [base_pairs for _ in range(B)]
    A_masked = torch.zeros_like(A_bmhw)
    for b in range(B):
        for m in range(M):
            g1,g2 = pairs_all[b][m]
            x0a,y0a,x1a,y1a = boxes[g1]
            x0b,y0b,x1b,y1b = boxes[g2]
            A_masked[b,m,y0a:y1a+1, x0a:x1a+1] = A_bmhw[b,m,y0a:y1a+1, x0a:x1a+1]
            A_masked[b,m,y0b:y1b+1, x0b:x1b+1] = A_bmhw[b,m,y0b:y1b+1, x0b:x1b+1]
    eps = _eps_for_dtype(A_bmhw.dtype)
    A_eff32 = A_masked.float()
    mass_raw = A_eff32.flatten(2).sum(-1).clamp_min(eps)         # (B,M)
    A_eff32 = A_eff32 / mass_raw.view(B, M, 1, 1)
    return A_eff32.to(A_bmhw.dtype), mass_raw

# ----------------------------- model -----------------------------

class SlotClassifier(nn.Module):
    def __init__(self,
                 filters_path: str,
                 num_slots: int = 12,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_classes: int = 47):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 150, kernel_size=9, stride=2, padding=4, bias=False)
        self._load_conv1(filters_path)

        self.proj = nn.Conv2d(150, d_model, kernel_size=1, bias=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.token_ln = nn.LayerNorm(d_model)

        self.num_slots = num_slots
        self.d_model = d_model
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * (1.0 / math.sqrt(d_model)))

        self.classifier = nn.Linear(d_model, num_classes)

    def _load_conv1(self, path: str):
        w = np.load(path)  # (150,1,9,9)
        w = torch.from_numpy(w).float()
        if w.ndim != 4 or w.shape[0] != 150 or w.shape[1] != 1 or w.shape[2] != 9:
            raise RuntimeError(f"filters shape mismatch: got {tuple(w.shape)}; need (150,1,9,9)")
        with torch.no_grad():
            self.conv1.weight.copy_(w)

    @torch.no_grad()
    def _check_stats_once(self, x: torch.Tensor):
        y = self.conv1(x)
        return dict(conv1_mean=y.mean().item(), conv1_std=y.std().item())

    def forward(self, x: torch.Tensor, use_spmask: bool, spmask_grid: int, spmask_assign: str, tau_p: float):
        """
        x: (B,1,28,28)
        returns: logits (B,C), aux dict
        """
        B = x.size(0)

        # 1) conv1 -> 14x14x150
        h = F.gelu(self.conv1(x))                       # (B,150,14,14)

        # 2) 1x1 proj -> D
        h = self.proj(h)                                # (B,D,14,14)
        H, W = h.size(2), h.size(3)

        # 3) tokens for self-attn
        tokens = h.permute(0,2,3,1).reshape(B, H*W, self.d_model)   # (B,N,D)
        tokens = self.encoder(tokens)
        tokens = self.token_ln(tokens)
        assert_finite(tokens, "tokens")

        # 4) slot attention logits over pixels
        q = F.normalize(self.slot_queries, dim=-1)
        t = F.normalize(tokens, dim=-1)
        A_logits = torch.einsum("md,bnd->bmn", q, t) / math.sqrt(self.d_model)   # (B,M,N)

        # 5) per-slot spatial softmax over pixels -> A_slot (sum_n = 1)
        A_slot = torch.softmax(A_logits, dim=2).view(B, self.num_slots, H, W)    # (B,M,H,W)

        # 6) spatial mask & renorm  —— comb4는 '면적 보정 + 안/밖 대비'로 P 계산
        # 6) spatial mask & renorm (compute P from PRE-renorm mass)
        if use_spmask and spmask_grid == 3 and spmask_assign == "round":
            M = self.num_slots
            mask_mhw = getattr(self, "_mask_mhw", None)
            if (mask_mhw is None) or (mask_mhw.shape[-2:] != (H, W)) or (mask_mhw.size(0) != M):
                self._mask_mhw = round_comb4_mask_token(M, H, W, device=tokens.device)
                mask_mhw = self._mask_mhw

            # (a) 풀링용 정규화는 기존 그대로: pre-renorm mass로 분모
            A_masked = A_slot * mask_mhw.unsqueeze(0)                 # (B,M,H,W)
            A_eff32  = A_masked.float()
            mass_raw = A_eff32.flatten(2).sum(-1).clamp_min(1e-8)     # (B,M)
            A_eff32  = A_eff32 / mass_raw.view(B, M, 1, 1)
            A_eff    = A_eff32.to(A_slot.dtype)

            # (b) ★ 게이팅 P: 영역 기대값 보정 (E[mass] = area_in / N_px)
            H_px, W_px = A_eff.size(-2), A_eff.size(-1)
            N_px = float(H_px * W_px)
            area_in = mask_mhw.float().flatten(1).sum(-1).to(mass_raw.device)   # (M,)
            exp_mass = (area_in / N_px).view(1, M)                               # (1,M)

            z_base = mass_raw.float() - exp_mass                                 # 기대값 보정
            z = (z_base - z_base.mean(dim=1, keepdim=True)) / max(1e-6, float(tau_p))
            P = torch.softmax(z, dim=1).to(A_slot.dtype)                         # (B,M)
        else:
            A_eff, mass_raw = apply_spatial_pair_mask(
                A_slot,
                enable=bool(use_spmask),
                grid=int(spmask_grid),
                assign=str(spmask_assign),
            )
            # pair/auto 경로는 기대값 보정 미적용(현 세팅에선 comb4만 사용)
            if use_spmask:
                z = (mass_raw.float() - mass_raw.float().mean(dim=1, keepdim=True)) / max(1e-6, float(tau_p))
                P = torch.softmax(z, dim=1).to(A_slot.dtype)
            else:
                s = torch.logsumexp(A_logits.float(), dim=2)
                z = (s - s.mean(dim=1, keepdim=True)) / max(1e-6, float(tau_p))
                P = torch.softmax(z, dim=1).to(A_slot.dtype)


        assert_finite(P, "P")

        # 7) slot embeddings from MASKED maps
        A_flat = A_eff.view(B, self.num_slots, H*W)               # (B,M,N)
        S_masked = torch.bmm(A_flat, tokens)                      # (B,M,D)
        S_masked = F.normalize(S_masked, dim=-1)
        assert_finite(S_masked, "S_masked")

        # 8) per-slot logits & P-weighted sum
        slot_logits = self.classifier(S_masked)                   # (B,M,C)
        logits = (slot_logits * P.unsqueeze(-1)).sum(dim=1)       # (B,C)

        aux = {
            "A_maps": A_eff,
            "A_maps_raw": A_slot,
            "feat_hw": tokens.view(B, H, W, self.d_model).detach(),
            "S_slots": S_masked.detach(),
            "slot_prob": P.detach()
        }
        return logits, aux

# ----------------------------- data -----------------------------

def get_loaders(batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,)):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.EMNIST(root="./data", split="balanced", train=True,  download=True, transform=tf)
    test_ds  = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), prefetch_factor=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), prefetch_factor=4)
    num_classes = int(train_ds.targets.max().item()) + 1
    return train_loader, test_loader, num_classes

# ----------------------------- train / eval -----------------------------

@torch.no_grad()
def evaluate(model: SlotClassifier, loader: DataLoader, device: torch.device,
             use_spmask: bool, spmask_grid: int, spmask_assign: str, tau_p: float):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits, _ = model(x, use_spmask=use_spmask, spmask_grid=spmask_grid,
                          spmask_assign=spmask_assign, tau_p=tau_p)
        pred = logits.argmax(dim=1)
        correct += int((pred==y).sum().item())
        total   += int(y.numel())
    return 100.0 * correct / max(1,total)

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters_path", required=True, help="path to 9x9 conv1 filters .npy (150,1,9,9)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--num_slots", type=int, default=126)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # warm-up & mask
    ap.add_argument("--warmup_epochs", type=int, default=5, help="epochs with no spatial mask")
    ap.add_argument("--spmask_grid", type=int, default=3)
    ap.add_argument("--spmask_assign", choices=["auto","round"], default="round")
    ap.add_argument("--tau", type=float, default=0.7, help="slot weight softmax temp")

    # optimization
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--freeze_conv1_epochs", type=int, default=30)
    ap.add_argument("--amp", type=int, default=0)

    args = ap.parse_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = get_loaders(args.batch_size, args.num_workers)

    model = SlotClassifier(filters_path=args.filters_path,
                           num_slots=args.num_slots,
                           d_model=args.d_model,
                           nhead=args.nhead,
                           num_classes=num_classes).to(device)

    # freeze conv1
    conv1_params = list(model.conv1.parameters())
    for p in conv1_params: p.requires_grad_(False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type=="cuda")

    best = -1.0
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f: f.write("epoch,train_loss,test_acc,P_ent,maxP\n")

    x0, _ = next(iter(train_loader))
    stats = model._check_stats_once(x0[:8].to(device))
    print(f"[sanity] conv1 stats: {stats}")

    for ep in range(1, args.epochs+1):
        model.train()
        if ep == args.freeze_conv1_epochs + 1:
            for p in conv1_params: p.requires_grad_(True)
            optim.add_param_group({"params": conv1_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay})
            print("[info] conv1 unfrozen (+added to optimizer)")

        use_spmask = (ep > args.warmup_epochs)
        tau_p = args.tau

        run_loss, run_Pent, run_maxP, n_batches = 0.0, 0.0, 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type=="cuda"):
                logits, aux = model(x, use_spmask=use_spmask,
                                    spmask_grid=args.spmask_grid,
                                    spmask_assign=args.spmask_assign,
                                    tau_p=tau_p)

            loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                print("[warn] non-finite loss detected; skipping batch")
                continue

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optim)
            scaler.update()

            P = aux["slot_prob"]
            M = P.size(1)
            Pent = (-(P.clamp_min(1e-8).log() * P).sum(dim=1) / math.log(float(M))).mean()
            maxP = P.max(dim=1).values.mean()

            run_loss += float(loss.item())
            run_Pent += float(Pent.item())
            run_maxP += float(maxP.item())
            n_batches += 1

        train_loss = run_loss / max(1, n_batches)
        P_ent = run_Pent / max(1, n_batches)
        maxP  = run_maxP / max(1, n_batches)

        acc = evaluate(model, test_loader, device,
                       use_spmask=use_spmask,
                       spmask_grid=args.spmask_grid,
                       spmask_assign=args.spmask_assign,
                       tau_p=tau_p)

        print(f"[ep {ep:02d}] loss={train_loss:.4f} | acc={acc:.2f}% | P_ent={P_ent:.3f} | maxP={maxP:.3f} | spmask={int(use_spmask)}")

        with open(csv_path, "a") as f:
            f.write(f"{ep},{train_loss:.6f},{acc:.4f},{P_ent:.6f},{maxP:.6f}\n")

        if acc > best:
            best = acc
            ckpt = {
                "model": model.state_dict(),
                "meta": {
                    "epoch": ep,
                    "num_slots": args.num_slots,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "spmask_grid": args.spmask_grid,
                    "spmask_assign": args.spmask_assign,
                    "tau": args.tau,
                    "warmup_epochs": args.warmup_epochs,
                    "filters_path": args.filters_path,
                }
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"[save] best.pt (acc={best:.2f}%)")

if __name__ == "__main__":
    train()
