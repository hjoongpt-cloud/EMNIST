# -*- coding: utf-8 -*-
# src_n/tools/n_finetune_with_slots.py
import os, json, argparse, math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from src_common.labels import emnist_char
from src_m.tools.m_train import build_model, seed_all
from src_n.tools.n_iou_utils import (
    greedy_diverse_topk, topq_mask_per_channel, pairwise_iou_mean, binary_iou
)

# ----------------------------- small helpers -----------------------------
def get_loaders(batch_size=256, num_workers=2, split="train", mean=(0.1307,), std=(0.3081,)):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    is_train = (split == "train")
    ds = datasets.EMNIST(root="./data", split="balanced", train=is_train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                                       num_workers=num_workers, pin_memory=True)

def normed(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def infer_xy_from_A(A_bmhw: torch.Tensor):
    # A: (B,M,H,W) -> (B,M,2) in [0,1]
    B,M,H,W = A_bmhw.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=A_bmhw.device),
                            torch.arange(W, device=A_bmhw.device), indexing="ij")
    mass = A_bmhw.flatten(2).sum(-1).clamp_min(1e-8)      # (B,M)
    cx = (A_bmhw*xx).flatten(2).sum(-1) / mass            # (B,M)
    cy = (A_bmhw*yy).flatten(2).sum(-1) / mass            # (B,M)
    xy = torch.stack([cx/(W-1+1e-8), cy/(H-1+1e-8)], dim=-1)  # (B,M,2)
    return xy

def build_feature(S_slots, XY, mode="s+xy", xy_weight=1.0):
    if mode == "xy":
        return normed(XY, dim=-1)
    elif mode == "s":
        if S_slots is None: raise RuntimeError("feature_mode='s' requires S_slots")
        return normed(S_slots, dim=-1)
    else:
        if S_slots is None: raise RuntimeError("feature_mode='s+xy' requires S_slots")
        s = normed(S_slots, dim=-1)
        x = normed(XY, dim=-1) * float(xy_weight)
        return torch.cat([s, x], dim=-1)

def soft_topk_weights(p, k=None, beta=0.5, mask=None):
    if mask is not None: p = p * mask
    if (k is not None) and k>0 and k < p.size(1):
        topv, topi = torch.topk(p, k, dim=1)
        keep = torch.zeros_like(p)
        keep.scatter_(1, topi, 1.0)
        q = p * keep
    else:
        q = p
    z = (q - q.mean(dim=1, keepdim=True)) / max(1e-6, float(beta))
    w = F.softmax(z, dim=1) * (mask if mask is not None else 1.0)
    w = w / (w.sum(dim=1, keepdim=True).clamp_min(1e-8))
    return w

def lse_pool(x, tau=0.5, dim=-1):
    xmax, _ = torch.max(x, dim=dim, keepdim=True)
    return (xmax + tau * torch.log(torch.clamp_min(torch.sum(torch.exp((x - xmax)/tau), dim=dim, keepdim=True), 1e-8))).squeeze(dim)

# ----------------------------- spatial mask (same as dump) -----------------------------
def _grid_boxes_torch(H, W, gh, gw):
    boxes = []
    for r in range(gh):
        for c in range(gw):
            x0 = int(round(c * (W/float(gw))))
            y0 = int(round(r * (H/float(gh))))
            x1 = int(round((c+1)*(W/float(gw)))) - 1
            y1 = int(round((r+1)*(H/float(gh)))) - 1
            boxes.append((x0,y0,x1,y1))
    return boxes

def _auto_pairs_for_batch(A_bmhw, boxes):
    """A_bmhw: (B,M,H,W) -> list of per-sample pair list per slot."""
    B,M,H,W = A_bmhw.shape
    G = len(boxes)
    pairs_all = []
    for b in range(B):
        # mass per grid cell per slot
        mass = torch.zeros(M, G, device=A_bmhw.device, dtype=A_bmhw.dtype)
        for g,(x0,y0,x1,y1) in enumerate(boxes):
            mass[:, g] = A_bmhw[b, :, y0:y1+1, x0:x1+1].flatten(1).sum(dim=1)
        top2 = torch.topk(mass, k=min(2,G), dim=1).indices  # (M,2)
        pairs = [(int(top2[m,0].item()), int(top2[m,1].item())) for m in range(M)]
        pairs_all.append(pairs)
    return pairs_all  # length B, each list length M

def _round_pairs_for_slots(M, boxes):
    G = len(boxes)
    all_pairs = []
    for i in range(G):
        for j in range(i+1, G):
            all_pairs.append((i,j))
    return [all_pairs[m % len(all_pairs)] for m in range(M)]

def apply_spatial_pair_mask(A_bmhw, enable=True, grid=3, assign="auto"):
    """Return A_eff (masked)."""
    if not enable:
        return A_bmhw
    B,M,H,W = A_bmhw.shape
    boxes = _grid_boxes_torch(H, W, grid, grid)
    if assign == "auto":
        pairs_all = _auto_pairs_for_batch(A_bmhw, boxes)
    else:
        base_pairs = _round_pairs_for_slots(M, boxes)
        pairs_all = [base_pairs for _ in range(B)]  # reuse

    A_eff = torch.zeros_like(A_bmhw)
    for b in range(B):
        for m in range(M):
            g1,g2 = pairs_all[b][m]
            x0a,y0a,x1a,y1a = boxes[g1]
            x0b,y0b,x1b,y1b = boxes[g2]
            A_eff[b,m,y0a:y1a+1, x0a:x1a+1] = A_bmhw[b,m,y0a:y1a+1, x0a:x1a+1]
            A_eff[b,m,y0b:y1b+1, x0b:x1b+1] = A_bmhw[b,m,y0b:y1b+1, x0b:x1b+1]
    return A_eff

def slot_prob_from_maps(A_bmhw, slot_mask, tau=0.7):
    B,M,H,W = A_bmhw.shape
    mask = slot_mask.view(1,M).to(A_bmhw.device)
    e = A_bmhw.flatten(2).sum(-1) * mask
    z = (e - e.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    p = F.softmax(z, dim=1) * mask
    p = p / (p.sum(dim=1, keepdim=True).clamp_min(1e-8))
    return p

# ----------------------------- prototype head -----------------------------
class SlotProtoHead(nn.Module):
    def __init__(self, Cn, C_cls, feature_dim, num_classes,
                 class_reduce="lse", proto_tau=0.5, slot_topk=3, beta=0.5,
                 slot_agg="support", pmean_p=2.5,
                 support_beta=0.4, support_tsup=1.6,
                 support_kappa_feat=1.0, support_kappa_xy=0.0):
        super().__init__()
        self.register_buffer("C", Cn)
        self.register_buffer("C_cls", C_cls)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_reduce = class_reduce
        self.proto_tau = proto_tau
        self.slot_topk = slot_topk
        self.beta = beta
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.slot_agg = slot_agg
        self.pmean_p = pmean_p
        self.support_beta = support_beta
        self.support_tsup = support_tsup
        self.support_kappa_feat = support_kappa_feat
        self.support_kappa_xy = support_kappa_xy

    def forward(self, feats_bmd, slot_prob, slot_mask, S_slots=None, XY=None, return_evi=False):
        B, M, d = feats_bmd.shape
        assert d == self.feature_dim

        mask = slot_mask.view(1, M).to(slot_prob.device)
        w = soft_topk_weights(slot_prob, k=self.slot_topk, beta=self.beta, mask=mask)  # (B,M)

        f   = F.normalize(feats_bmd, dim=-1)
        sim = torch.einsum("bmd,kd->bmk", f, self.C)  # (B,M,K)

        K = self.C.shape[0]
        classes = torch.unique(self.C_cls)
        Cmax = int(torch.max(classes).item()) + 1

        sim32  = sim.float()
        neg_inf = torch.tensor(-1e9, device=sim32.device, dtype=sim32.dtype)
        evi_all32 = sim32.new_zeros(B, M, Cmax)
        for c in classes.tolist():
            mask_c = (self.C_cls == c).view(1,1,K)
            s_c = sim32.masked_fill(~mask_c, neg_inf)
            if self.class_reduce == "max":
                evi_c, _ = torch.max(s_c, dim=2)
            else:
                evi_c = lse_pool(s_c, tau=self.proto_tau, dim=2)
            evi_all32[:, :, c] = evi_c
        evi_all = evi_all32.to(sim.dtype)  # (B,M,C)

        # across-slot aggregation
        if self.slot_agg == "support":
            proto_logits = top1_plus_support(
                evi_all, w, S_slots=S_slots, XY=XY,
                beta_support=self.support_beta, t_sup=self.support_tsup,
                kappa_feat=self.support_kappa_feat, kappa_xy=self.support_kappa_xy
            )
        elif self.slot_agg == "pmean":
            p = self.pmean_p
            contrib = (evi_all * w.unsqueeze(-1)).clamp_min(0)
            proto_logits = contrib.pow(p).mean(dim=1).pow(1.0/p)
        else:
            proto_logits = torch.sum(w.unsqueeze(-1) * evi_all, dim=1)

        score = self.alpha * proto_logits
        if return_evi:
            return score, evi_all
        return score

# --- support aggregator used above ---
def top1_plus_support(sim_by_class, P, S_slots=None, XY=None,
                      beta_support=0.4, t_sup=1.6,
                      kappa_feat=1.0, kappa_xy=0.0):
    B, M, C = sim_by_class.shape
    score = sim_by_class * P.unsqueeze(-1)                 # (B,M,C)
    top1_val, top1_idx = score.max(dim=1)                  # (B,C), (B,C)

    P_sup = torch.softmax(torch.log(P + 1e-8) / t_sup, dim=1).unsqueeze(-1).expand_as(sim_by_class)
    mask = torch.ones_like(sim_by_class, dtype=torch.bool)
    mask.scatter_(1, top1_idx.unsqueeze(1), False)         # exclude top1 slot

    w_red = 0.0
    if S_slots is not None and kappa_feat > 0:
        S = F.normalize(S_slots, dim=-1)
        gather_idx = top1_idx.unsqueeze(-1).expand(B, C, S.shape[-1])
        top1_feat = torch.gather(S.transpose(1,2), 2, gather_idx.transpose(1,2)).transpose(1,2)
        cos = torch.einsum('bmd,bcd->bmc', S, top1_feat).clamp_min(0)
        w_red = kappa_feat * cos
    if XY is not None and kappa_xy > 0:
        binsig2 = 0.04
        xy_top = torch.gather(XY, 1, top1_idx.max(dim=1).values.unsqueeze(-1).expand(B, C, 2))
        d2 = ((XY.unsqueeze(2) - xy_top.unsqueeze(1))**2).sum(-1)
        w_red = (w_red if isinstance(w_red, torch.Tensor) else 0) + kappa_xy * torch.exp(-d2 / binsig2)

    support_raw = (sim_by_class.float()
                + torch.log(P_sup.float().clamp_min(1e-8))
                - (w_red.float() if isinstance(w_red, torch.Tensor) else 0.0))
    support_raw = support_raw.masked_fill(~mask, -1e4)
    support_val = torch.logsumexp(support_raw, dim=1).to(sim_by_class.dtype)

    return top1_val + beta_support * support_val

def slot_feature_orthogonality(S_bmd, sel_bm):
    """
    S_bmd: (B,M,Ds) raw slot features
    sel_bm: (B,M) bool (선택된 슬롯)
    """
    if S_bmd is None: 
        return torch.zeros((), device=sel_bm.device)
    B, M, Ds = S_bmd.shape
    loss_sum, cnt = 0.0, 0
    for b in range(B):
        m = sel_bm[b]
        if m.sum() <= 1: 
            continue
        Sb = F.normalize(S_bmd[b, m], dim=-1)   # (K,Ds)
        G  = Sb @ Sb.t()                        # (K,K)
        off = (G.triu(1)**2).mean()
        loss_sum = loss_sum + off
        cnt += 1
    if cnt == 0: 
        return torch.zeros((), device=sel_bm.device)
    return loss_sum / cnt

def slot_class_selectivity_loss(evi_bmc, P_bm, y_b, w_norm=True):
    """
    evi_bmc: (B,M,C) per-slot per-class evidence
    P_bm   : (B,M)   slot weights
    y_b    : (B,)    GT class
    """
    B, M, C = evi_bmc.shape
    device = evi_bmc.device
    onehot = F.one_hot(y_b, num_classes=C).float()            # (B,C)
    s_true_bm = (evi_bmc * onehot.unsqueeze(1)).sum(2)        # (B,M)
    s_true_bm = s_true_bm * P_bm                               # (B,M)

    q_mc = torch.full((M, C), 1e-8, device=device)            # eps → 0분모/로그 방지
    for c in range(C):
        mask = (y_b == c).float().unsqueeze(1)                # (B,1)
        q_mc[:, c] += (s_true_bm * mask).sum(0)               # (M,)

    q_mc = q_mc / q_mc.sum(1, keepdim=True).clamp_min(1e-8)
    q_mc = q_mc.clamp_min(1e-8)
    H = -(q_mc * q_mc.log()).sum(1)                           # (M,)
    if w_norm:
        H = H / math.log(float(C))
    return H.mean()

# ----------------------------- training main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--amp", type=int, default=1, help="1=use autocast+GradScaler, 0=disable AMP")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)

    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--proto_tau", type=float, default=0.5)
    ap.add_argument("--slot_topk", type=int, default=3)
    ap.add_argument("--class_reduce", default="lse", choices=["lse","max"])
    ap.add_argument("--slot_agg", default="support", choices=["sum","support","pmean"])
    ap.add_argument("--pmean_p", type=float, default=2.5)

    ap.add_argument("--support_beta", type=float, default=0.4)
    ap.add_argument("--support_tsup", type=float, default=1.6)
    ap.add_argument("--support_kappa_feat", type=float, default=1.0)
    ap.add_argument("--support_kappa_xy", type=float, default=0.0)

    ap.add_argument("--loss_ortho_w", type=float, default=0.01)
    ap.add_argument("--loss_slot_select_w", type=float, default=0.01)

    ap.add_argument("--learn_alpha", type=int, default=1)
    ap.add_argument("--freeze_trunk", type=int, default=1)
    ap.add_argument("--freeze_head", type=int, default=1)
    ap.add_argument("--eval_only", type=int, default=0)

    # spatial diversification & entropy regs
    ap.add_argument("--nms_enable", type=int, default=1)
    ap.add_argument("--nms_iou_q", type=float, default=0.10)
    ap.add_argument("--nms_iou_thr", type=float, default=0.30)
    ap.add_argument("--loss_iou_w", type=float, default=0.0)
    ap.add_argument("--loss_ent_w", type=float, default=0.0)
    ap.add_argument("--ent_min", type=float, default=0.6)
    ap.add_argument("--iou_penalty", default="mean", choices=["mean","elbow","entropy"])
    ap.add_argument("--iou_elbow_lo", type=float, default=0.20)
    ap.add_argument("--iou_elbow_hi", type=float, default=0.50)
    ap.add_argument("--iou_entropy_tau", type=float, default=8.0)
    ap.add_argument("--iou_ent_min", type=float, default=0.80)

    # NEW: spatial pair mask (same semantics as dump)
    ap.add_argument("--spmask_enable", type=int, default=1)
    ap.add_argument("--spmask_grid", type=int, default=3)
    ap.add_argument("--spmask_assign", choices=["auto","round"], default="auto")

    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load cfg
    if args.config.endswith(".json"):
        import json as _json
        with open(args.config, "r") as f: cfg = _json.load(f)
    else:
        import yaml
        with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    mean = tuple(cfg.get("normalize",{}).get("mean", [0.1307]))
    std  = tuple(cfg.get("normalize",{}).get("std",  [0.3081]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    trunk.to(device).eval(); head.to(device).eval()

    # freeze
    for p in trunk.parameters(): p.requires_grad_(not bool(args.freeze_trunk))
    for p in head.parameters():  p.requires_grad_(not bool(args.freeze_head))

    # slot mask
    M = head.M
    slot_mask = getattr(head, "slot_mask", torch.ones(M, device=device)).detach().float().to(device)

    # prototypes (fixed)
    with open(args.proto_json, "r") as f:
        data = json.load(f)
    per = data.get("per_class", {})
    C_list, C_cls = [], []
    for c_str, block in per.items():
        cid = int(c_str)
        vecs = []
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                if isinstance(mus[0], (list, tuple)): vecs = mus
                else: vecs = [mus]
            else:
                inner = block.get("protos") or block.get("clusters") or block.get("items") or []
                for p in inner:
                    v = p.get("mu") if isinstance(p, dict) else p
                    if v is not None: vecs.append(v)
        elif isinstance(block, list):
            for p in block:
                v = p.get("mu") if isinstance(p, dict) else p
                if v is not None: vecs.append(v)
        else:
            vecs = [block]
        for v in vecs:
            C_list.append(np.asarray(v, np.float32).reshape(-1))
            C_cls.append(cid)
    Cn = torch.from_numpy(np.stack(C_list, 0)).float().to(device)
    # 안전 정규화: 0-벡터 방지
    Cn = Cn / Cn.norm(dim=1, keepdim=True).clamp_min(1e-8)

    C_cls = torch.from_numpy(np.asarray(C_cls, np.int64)).to(device)
    feature_dim = Cn.shape[1]
    num_classes = int(torch.max(C_cls).item()) + 1

    # finetune head
    add_head = SlotProtoHead(
        Cn=Cn, C_cls=C_cls,
        feature_dim=feature_dim,
        num_classes=num_classes,
        class_reduce=args.class_reduce,
        proto_tau=args.proto_tau,
        slot_topk=args.slot_topk,
        beta=args.beta,
        slot_agg=args.slot_agg,
        pmean_p=args.pmean_p,
        support_beta=args.support_beta,
        support_tsup=args.support_tsup,
        support_kappa_feat=args.support_kappa_feat,
        support_kappa_xy=args.support_kappa_xy
    ).to(device)
    add_head.alpha.requires_grad_(bool(args.learn_alpha))

    # optimizer
    params = []
    if any(p.requires_grad for p in head.parameters()): params += list(head.parameters())
    if any(p.requires_grad for p in trunk.parameters()): params += list(trunk.parameters())
    params += [p for p in add_head.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr)

    # data
    train_loader = get_loaders(args.batch_size, args.num_workers, split="train", mean=mean, std=std)
    test_loader  = get_loaders(args.batch_size, args.num_workers, split="test",  mean=mean, std=std)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda" and bool(args.amp)))


    # forward
    def forward_batch(x, return_aux=False):
        Z, aux = trunk(x)
        base_logits = head(Z)                      # (B,C)
        A = aux["A_maps"]                          # (B,M,H,W)

        # *** apply spatial mask like dump ***
        A_eff = apply_spatial_pair_mask(A, enable=bool(args.spmask_enable),
                                        grid=int(args.spmask_grid),
                                        assign=str(args.spmask_assign))

        # slot prob / XY from masked maps
        P = slot_prob_from_maps(A_eff, slot_mask, tau=args.tau)  # (B,M)

        # NMS selection (optional)
        if args.nms_enable:
            sel = greedy_diverse_topk(
                scores_bm=P, A_bmhw=A_eff,
                k=args.slot_topk, q_iou=args.nms_iou_q, iou_thr=args.nms_iou_thr
            )  # (B,M) bool
            P = P * sel.float()
            P = P / (P.sum(dim=1, keepdim=True).clamp_min(1e-8))
        else:
            sel = (P > 0)

        XY = infer_xy_from_A(A_eff)                # (B,M,2)

        # obtain slot semantic features if available
        S = None
        for k in ["S_slots","slots_S","slot_embed","slot_repr"]:
            if k in aux and isinstance(aux[k], torch.Tensor):
                S = aux[k]; break

        feats = build_feature(S, XY, args.feature_mode, args.xy_weight)  # (B,M,d)
        score, evi_all = add_head(feats, P, slot_mask, S_slots=S, XY=XY, return_evi=True)
        logits = base_logits + score
        if return_aux:
            return logits, A_eff, P, sel, S, XY, evi_all
        return logits

    def evaluate(loader):
        trunk.eval(); head.eval(); add_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                logits = forward_batch(x)
                pred = logits.argmax(dim=1)
                correct += int((pred==y).sum().item())
                total   += int(y.numel())
        return 100.0 * correct / max(1,total)

    if args.eval_only:
        acc = evaluate(test_loader)
        print(f"[eval only] test_acc={acc:.2f}%")
        return

    # ----------------------------- train -----------------------------
    best = -1.0
    for ep in range(1, args.epochs+1):
        trunk.train(not bool(args.freeze_trunk))
        head.train(not bool(args.freeze_head))
        add_head.train(True)

        run_loss = 0.0
        pbar = tqdm(train_loader, desc=f"ep{ep}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda" and bool(args.amp))):
                logits, A, P, sel, S, XY, evi = forward_batch(x, return_aux=True)
                masks = topq_mask_per_channel(A, q=args.nms_iou_q).to(torch.bool)
                ce = F.cross_entropy(logits, y)

                reg = 0.0
                # IoU penalty
                if args.loss_iou_w > 0 and args.nms_enable:
                    iou_mean = pairwise_iou_mean(masks, sel)
                    if args.iou_penalty == "mean":
                        pen = iou_mean
                    elif args.iou_penalty == "elbow":
                        lo, hi = float(args.iou_elbow_lo), float(args.iou_elbow_hi)
                        x_ = iou_mean.clamp(0., 1.)
                        mid = ((x_ - lo) / max(1e-6, (hi - lo))).clamp(0., 1.)
                        hi_part = (x_ - hi).clamp(min=0.) / max(1e-6, (1. - hi))
                        pen = mid + 1.5 * hi_part
                    else:  # entropy
                        B_ = sel.size(0)
                        ent_list = []
                        for b in range(B_):
                            idx = sel[b]
                            mb = masks[b, idx]
                            K_ = mb.shape[0]
                            if K_ <= 1:
                                ent_list.append(torch.ones((), device=mb.device))
                                continue
                            iou_mat = torch.zeros((K_, K_), device=mb.device)
                            for i in range(K_):
                                for j in range(i+1, K_):
                                    iou = binary_iou(mb[i:i+1], mb[j:j+1])[0]
                                    iou_mat[i, j] = iou_mat[j, i] = iou
                            rows, eye = [], ~torch.eye(K_, dtype=torch.bool, device=mb.device)
                            for i in range(K_):
                                vec = iou_mat[i][eye[i]]
                                p = torch.softmax(vec * (1.0 / max(1e-6, args.iou_entropy_tau)), dim=0)
                                H = -(p.clamp_min(1e-8).log() * p).sum() / math.log(max(2, K_-1))
                                rows.append(H)
                            Hn = torch.stack(rows).mean()
                            lack = (float(args.iou_ent_min) - Hn).clamp(min=0.)
                            ent_list.append(lack)
                        pen = torch.stack(ent_list)
                    reg = reg + args.loss_iou_w * pen.mean()

                # slot entropy floor
                if args.loss_ent_w > 0:
                    M_ = P.size(1)
                    H_ = -(P.clamp_min(1e-8).log() * P).sum(dim=1) / max(1e-6, math.log(float(M_)))
                    lack = (args.ent_min - H_).clamp(min=0)
                    reg = reg + args.loss_ent_w * lack.mean()

                # optional regularizers (kept)
                if args.loss_ortho_w > 0 and S is not None:
                    # simple orthogonality between selected slot features
                    B_, M_, Ds = S.shape
                    loss_sum = 0.0; cnt=0
                    for b in range(B_):
                        m = sel[b]
                        if m.sum() <= 1: continue
                        Sb = F.normalize(S[b, m], dim=-1)
                        G  = Sb @ Sb.t()
                        off = (G.triu(1)**2).mean()
                        loss_sum = loss_sum + off
                        cnt += 1
                    if cnt>0:
                        reg = reg + args.loss_ortho_w * (loss_sum/cnt)

                if args.loss_slot_select_w > 0 and evi is not None:
                    # encourage slot-class selectivity
                    B_, M_, C_ = evi.shape
                    y_onehot = F.one_hot(y, num_classes=C_).float().unsqueeze(1)  # (B,1,C)
                    s_true = (evi * y_onehot).sum(dim=2) * P  # (B,M)
                    q_mc = torch.zeros(M_, C_, device=evi.device)
                    for c in range(C_):
                        mask = (y == c).float().unsqueeze(1)
                        q_mc[:, c] = (s_true * mask).sum(dim=0) + 1e-8
                    q_mc = q_mc / q_mc.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    H = -(q_mc * q_mc.log()).sum(dim=1) / math.log(C_)
                    reg = reg + args.loss_slot_select_w * H.mean()

                loss = ce + reg

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            run_loss += float(loss.item())
            pbar.set_postfix(loss=f"{run_loss / max(1, pbar.n):.4f}")

        acc = evaluate(test_loader)
        train_loss_ep = run_loss/len(train_loader)
        print(f"[ep {ep}] train_loss={train_loss_ep:.4f} | test_acc={acc:.2f}%")
        csv_path = os.path.join(args.out_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("epoch,train_loss,test_acc\n")
        with open(csv_path, "a") as f:
            f.write(f"{ep},{train_loss_ep:.6f},{acc:.4f}\n")
        if acc > best:
            best = acc
            torch.save({
                "trunk": trunk.state_dict(),
                "head": head.state_dict(),
                "add_head": add_head.state_dict(),
                "cfg": cfg
            }, os.path.join(args.out_dir, "best_with_slots.pt"))
            print(f"[save] best_with_slots.pt (acc={best:.2f}%)")
            with open(os.path.join(args.out_dir, "last.json"), "w") as fh:
                json.dump({"epoch": int(ep), "test_acc": float(acc)}, fh, indent=2)

if __name__ == "__main__":
    main()
