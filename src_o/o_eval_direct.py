#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json, math, itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------- trunk ----------
from src_o.o_trunk import OTrunk

# ---------- utils: load from ckpt ----------
@torch.no_grad()
def load_slot_queries(ckpt_path, device, normalize=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    q = None
    for pfx in ("module.", ""):
        k = f"{pfx}slot_queries"
        if k in sd:
            q = sd[k].float()
            break
    if q is None:
        raise RuntimeError("slot_queries not found in ckpt")
    q = q.to(device)
    return (F.normalize(q, dim=-1) if normalize else q)

@torch.no_grad()
def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    W = b = None
    for pfx in ("module.", ""):
        wk, bk = f"{pfx}classifier.weight", f"{pfx}classifier.bias"
        if wk in sd and bk in sd:
            W = sd[wk].float().to(device)  # (C,D)
            b = sd[bk].float().to(device)  # (C,)
            break
    if W is None or b is None:
        raise RuntimeError("classifier.{weight,bias} not found in ckpt")
    return W, b

# ---------- data ----------
def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full = datasets.EMNIST("./data", split="balanced", train=True, download=True, transform=tf)
        n = len(full); nv = int(round(n * val_ratio)); nt = n - nv
        g = torch.Generator().manual_seed(seed)
        tr, va = random_split(full, [nt, nv], generator=g)
        ds = tr if split == "train" else va
    else:
        ds = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

# ---------- 3x3 pair masks (round assign) ----------
@torch.no_grad()
def round_pair_mask(M, H, W, device):
    assert H == 14 and W == 14, "assume 14x14 tokens"
    xs, ys = [0,5,10,14], [0,5,10,14]
    cells=[]
    for gy in range(3):
        for gx in range(3):
            x0,x1 = xs[gx], xs[gx+1]
            y0,y1 = ys[gy], ys[gy+1]
            m = torch.zeros(H,W, device=device); m[y0:y1, x0:x1] = 1.0
            cells.append(m)
    pairs=[]
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append(torch.maximum(cells[i], cells[j]))
    idx = torch.arange(M, device=device) % len(pairs)
    return torch.stack([pairs[i] for i in idx.tolist()], 0)  # (M,H,W)

# ---------- slots pipeline (no dump) ----------
@torch.no_grad()
def compute_maps(tokens_bnd, q_md, amap="pxslot", norm_tokens=False, norm_queries=False, scale=1.0):
    """
    tokens_bnd: (B, N, D), q_md: (M, D)
    amap = 'pxslot'  -> per-pixel softmax over slots
         = 'slotsum' -> per-slot softmax over pixels
    """
    B, N, D = tokens_bnd.shape; M = q_md.size(0)
    H = W = int(math.sqrt(N)); assert H*W == N, f"N={N} not square"

    t = F.normalize(tokens_bnd, dim=-1) if norm_tokens else tokens_bnd
    q = F.normalize(q_md, dim=-1) if norm_queries else q_md

    # logits
    # shape 선택: (B, M, N)로 만들어두면 pixel 축이 dim=2라 헷갈림 적음
    logits_bmn = torch.einsum("bnd,md->bmn", t, q) * float(scale)  # (B,M,N)

    if amap == "pxslot":
        # per-pixel softmax over slots -> A_raw: (B,M,H,W), sum_m A=1 @ each (h,w)
        A_raw = torch.softmax(logits_bmn, dim=1).view(B, M, H, W)
    elif amap == "slotsum":
        # per-slot softmax over pixels -> A_raw: (B,M,H,W), sum_hw A=1 per slot
        A_raw = torch.softmax(logits_bmn, dim=2).view(B, M, H, W)
    else:
        raise ValueError("amap must be 'pxslot' or 'slotsum'")
    return A_raw, logits_bmn.view(B, M, H, W), (H, W)

@torch.no_grad()
def apply_mask(A_raw, mask_mode="round"):
    if mask_mode == "none":
        return A_raw
    B, M, H, W = A_raw.shape
    mask = round_pair_mask(M, H, W, A_raw.device).unsqueeze(0)  # (1,M,H,W)
    return A_raw * mask

@torch.no_grad()
def compute_P(logits_bmn, A_masked, p_mode="lse", tau=0.7):
    """
    logits_bmn: (B,M,N)
    A_masked  : (B,M,H,W)  (정규화하지 않은 그대로)
    """
    if p_mode == "uniform":
        B, M, _ = logits_bmn.shape
        return torch.full((B, M), 1.0 / M, device=logits_bmn.device)
    elif p_mode == "mass":
        mass = A_masked.flatten(2).sum(-1)  # (B,M)
        z = (mass - mass.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
        return torch.softmax(z, dim=1)
    elif p_mode == "lse":
        s = torch.logsumexp(logits_bmn, dim=2)  # (B,M)  # over pixels
        z = (s - s.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
        return torch.softmax(z, dim=1)
    else:
        raise ValueError("p_mode must be 'uniform'|'mass'|'lse'")

@torch.no_grad()
def slot_embeddings(tokens_bnd, A_masked, avg=True):
    B, N, D = tokens_bnd.shape; B2, M, H, W = A_masked.shape
    assert B == B2 and N == H*W
    t = tokens_bnd  # 그대로 (정규화 필요시 여기서)
    t = F.normalize(t, dim=-1)
    t_flat = t.view(B, H*W, D)
    A_flat = A_masked.view(B, M, H*W)
    S = torch.bmm(A_flat, t_flat)  # (B,M,D)
    if avg:
        mass = A_flat.sum(-1).clamp_min(1e-8).unsqueeze(-1)  # (B,M,1)
        S = S / mass
    return F.normalize(S, dim=-1)

@torch.no_grad()
def aggregate_logits(S_bmd, P_bm, W_cd, b_c, agg="wsum", slot_topk=5):
    """
    S_bmd: (B,M,D), P_bm: (B,M), W_cd: (C,D), b_c: (C,)
    return logits: (B,C)
    """
    # slot-level logits
    slot_logits = torch.einsum("bmd,cd->bmc", S_bmd, W_cd) + b_c.view(1,1,-1)  # (B,M,C)

    if agg == "sum":
        # 그냥 Σ_m slot_logits
        return slot_logits.sum(dim=1)
    elif agg == "maxslot":
        return slot_logits.max(dim=1).values
    elif agg == "wsum":
        w = P_bm / P_bm.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bmc,bm->bc", slot_logits, w)
    elif agg == "softk":
        # 상위 k 슬롯만 softmax 가중 평균
        B, M, C = slot_logits.shape
        k = min(int(slot_topk), M)
        topv, topi = torch.topk(P_bm, k=k, dim=1)
        mask = torch.zeros_like(P_bm).scatter_(1, topi, 1.0)
        sel = P_bm * mask
        w = sel / sel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bmc,bm->bc", slot_logits, w)
    else:
        raise ValueError("agg must be one of wsum,sum,maxslot,softk")

# ---------- coverage(topX) ----------
@torch.no_grad()
def coverage_topX(S_bmd, W_cd, b_c, y_true, X_list=(1,3,5)):
    # 슬롯별 로짓 -> 클래스별 증거 max-reduce
    sim = torch.einsum("bmd,cd->bmc", S_bmd, W_cd) + b_c.view(1,1,-1)  # (B,M,C)
    evi = sim.max(dim=1).values  # (B,C)
    cov = {}
    for X in X_list:
        top = evi.topk(k=min(X, evi.size(1)), dim=1).indices
        cov[X] = float((top == y_true.view(-1,1)).any(dim=1).float().mean().item())
    return cov

# ---------- plotting ----------
def save_hist(P_list, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P_all = np.concatenate([p.flatten() for p in P_list], axis=0)
    plt.figure(figsize=(6,4))
    plt.hist(P_all, bins=50, density=True)
    plt.xlabel("slot weight P"); plt.ylabel("density"); plt.title("P histogram")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ---------- main eval ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    # trunk meta(가능하면 ckpt meta에서 읽지만, 없는 경우 대비)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    # slots
    ap.add_argument("--amap", choices=["pxslot","slotsum"], default="pxslot")
    ap.add_argument("--mask", choices=["none","round"], default="round")
    ap.add_argument("--p_mode", choices=["uniform","mass","lse"], default="lse")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--norm_tokens", type=int, default=0)   # 0/1
    ap.add_argument("--norm_queries", type=int, default=0)  # 0/1
    ap.add_argument("--scale", type=float, default=1.0)
    # aggregation
    ap.add_argument("--agg", choices=["wsum","sum","maxslot","softk"], default="wsum")
    ap.add_argument("--slot_topk", type=int, default=5)
    # sweeps
    ap.add_argument("--sweep_k", type=str, default="")    # "1,3,5,7"
    ap.add_argument("--sweep_scale", type=str, default="")# "1,2,4,8,16"
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk from meta
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = args.d_model or int(meta.get("d_model", 64))
    nhead   = args.nhead   or int(meta.get("nhead", 4))
    nlayers = args.num_layers or int(meta.get("num_layers", 1))
    filters = np.load(args.conv1_filters)

    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=nlayers, d_ff=256, conv1_filters=filters).to(device).eval()
    q  = load_slot_queries(args.ckpt, device, normalize=bool(args.norm_queries))
    W, b = load_classifier(args.ckpt, device)

    # data
    loader = get_loader(args.split, bs=args.batch_size, nw=args.num_workers)

    # sweeps
    k_list = [args.slot_topk] if not args.sweep_k else [int(x) for x in args.sweep_k.split(",") if x.strip()]
    s_list = [args.scale] if not args.sweep_scale else [float(x) for x in args.sweep_scale.split(",") if x.strip()]

    results = []
    P_accum = []

    for scale in s_list:
        for k in k_list:
            top1 = n = 0
            cov_agg = {"1":0.0,"3":0.0,"5":0.0}; cov_n = 0
            with torch.no_grad():
                for x, y in tqdm(loader, desc=f"[eval k={k} scale={scale}]"):
                    x = x.to(device); y = y.to(device)
                    tok, _ = trunk(x)  # (B,196,D)
                    # maps
                    A_raw, logits_bmn_hw, (H, W) = compute_maps(tok, q, amap=args.amap,
                                                                norm_tokens=bool(args.norm_tokens),
                                                                norm_queries=bool(args.norm_queries),
                                                                scale=scale)
                    # mask
                    A_masked = apply_mask(A_raw, mask_mode=args.mask)
                    # P
                    logits_bmn = (logits_bmn_hw.view(x.size(0), q.size(0), -1))  # (B,M,N)
                    P = compute_P(logits_bmn, A_masked, p_mode=args.p_mode, tau=args.tau)  # (B,M)
                    # S
                    S = slot_embeddings(tok, A_masked, avg=True)  # (B,M,D)
                    # aggregate
                    logits = aggregate_logits(S, P, W, b, agg=args.agg, slot_topk=k)  # (B,C)
                    pred = logits.argmax(dim=1)
                    top1 += (pred == y).sum().item(); n += y.numel()
                    # coverage
                    cov = coverage_topX(S, W, b, y, X_list=(1,3,5))
                    for X in (1,3,5):
                        cov_agg[str(X)] += cov[X]*y.numel()
                    cov_n += y.numel()
                    # collect P
                    P_accum.append(P.detach().cpu().numpy())

            acc = 100.0 * top1 / max(1, n)
            for X in (1,3,5):
                cov_agg[str(X)] = float(cov_agg[str(X)]/max(1, cov_n))
            results.append({
                "amap": args.amap, "mask": args.mask, "p_mode": args.p_mode,
                "norm_tokens": int(args.norm_tokens), "norm_queries": int(args.norm_queries),
                "scale": scale, "agg": args.agg, "slot_topk": k,
                "acc": acc, "coverage@1": cov_agg["1"], "coverage@3": cov_agg["3"], "coverage@5": cov_agg["5"]
            })
            print(f"[result] acc={acc:.2f}% | cov@3={cov_agg['3']:.3f} (k={k}, scale={scale})")

    # save table
    import csv
    with open(os.path.join(args.out_dir, "acc_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # save summary
    # P histogram
    from collections import defaultdict
    hist_Pmax = []
    for P in P_accum:
        Pmax = P.max(axis=1)
        hist_Pmax.extend(Pmax.tolist())
    summary = {
        "num_runs": len(results),
        "best_acc": max(r["acc"] for r in results) if results else 0.0,
        "best_cfg": max(results, key=lambda r: r["acc"]) if results else {},
        "P_max_mean": float(np.mean(hist_Pmax)) if hist_Pmax else 0.0,
        "P_max_median": float(np.median(hist_Pmax)) if hist_Pmax else 0.0,
        "runs": results
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # P histogram png
    save_hist([np.concatenate(P_accum,0)] if P_accum else [np.zeros((1,1))],
              os.path.join(args.out_dir, "P_hist.png"))

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
