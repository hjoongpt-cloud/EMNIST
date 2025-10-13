#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습 ckpt를 그대로 사용하여, dump 없이 '슬롯 evidence'를 시각화.

추가/개선:
- select_for(true/pred) 기준으로 슬롯 기여도 계산
- k_slots / cumulative / min_contrib 로 슬롯 선택 유연화
- 패널은 '기여도 내림차순' 정렬, 슬롯별 top-3 클래스 바차트 포함
- 메인 오버레이는 overlay_topk 로 간단히(겹침 방지), 자세한 건 패널에서

예시:
python src_o/o_viz_slots.py \
  --ckpt outputs/o/train/best.pt \
  --conv1_filters "$FILT" \
  --out_dir outputs/o/viz \
  --split val \
  --amap pxslot --mask round --p_mode lse --tau 0.7 \
  --norm_tokens 0 --norm_queries 0 --scale 1.0 \
  --agg wsum --overlay_topk 3 \
  --select_for pred --k_slots 16 --cumulative 0.9 \
  --panel 1
"""

import os, json, math, csv, argparse, random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk  # trunk

# ---------------- CKPT 로딩 ----------------
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

# ---------------- Data & Label Mapping ----------------
def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if split in ("train","val"):
        full = datasets.EMNIST("./data", split="balanced", train=True, download=True, transform=tf)
        n = len(full); nv = int(round(n*val_ratio)); nt = n-nv
        g = torch.Generator().manual_seed(seed)
        tr, va = random_split(full, [nt, nv], generator=g)
        ds = tr if split=="train" else va
    else:
        ds = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

def get_label_mapping():
    tmp = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=transforms.ToTensor())
    classes = [str(c) for c in list(tmp.classes)]
    return classes  # index -> char(str)

# ---------------- 3x3 → 36 페어 (round-robin) ----------------
@torch.no_grad()
def grid_cells_pix():
    # 입력 28x28을 3분할: 폭 [0,10,20,28]
    edges = [0, 10, 20, 28]
    cells = []
    for gy in range(3):
        for gx in range(3):
            x0, x1 = edges[gx], edges[gx+1]
            y0, y1 = edges[gy], edges[gy+1]
            cells.append((x0, y0, x1, y1))  # x:[x0,x1), y:[y0,y1)
    return cells  # len=9

@torch.no_grad()
def round_pair_map(M):
    pairs = []
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append((i,j))
    return [pairs[m % len(pairs)] for m in range(M)], pairs  # (len=M), (len=36)

@torch.no_grad()
def round_pair_mask_token(M, H, Wtok, device):
    # 14x14 토큰 그리드에서의 마스크(계산용)
    assert H==14 and Wtok==14, "assume 14x14 tokens"
    xs, ys = [0,5,10,14], [0,5,10,14]
    cells=[]
    for gy in range(3):
        for gx in range(3):
            x0,x1=xs[gx], xs[gx+1]
            y0,y1=ys[gy], ys[gy+1]
            m = torch.zeros(H,Wtok,device=device); m[y0:y1, x0:x1] = 1.0
            cells.append(m)
    pairs=[]
    for i in range(9):
        for j in range(i+1,9):
            pairs.append(torch.maximum(cells[i], cells[j]))
    idx = torch.arange(M, device=device) % len(pairs)
    return torch.stack([pairs[i] for i in idx.tolist()], 0)  # (M,H,Wtok)

# ---------------- 슬롯맵 / 가중치 / 임베딩 ----------------
@torch.no_grad()
def compute_maps(tokens_bnd, q_md, amap="pxslot", norm_tokens=False, norm_queries=False, scale=1.0):
    """
    tokens_bnd: (B,N,D), q_md:(M,D)
    amap: 'pxslot' (픽셀별 softmax-슬롯축) | 'slotsum' (슬롯별 softmax-픽셀축)
    반환: A_raw:(B,M,H,Wtok), logits_bmn:(B,M,N), (H,Wtok)
    """
    B, N, D = tokens_bnd.shape; M = q_md.size(0)
    H = Wtok = int(math.sqrt(N)); assert H*Wtok == N
    t = F.normalize(tokens_bnd, dim=-1) if norm_tokens else tokens_bnd
    q = F.normalize(q_md, dim=-1) if norm_queries else q_md
    logits_bmn = torch.einsum("bnd,md->bmn", t, q) * float(scale)  # (B,M,N)
    if amap == "pxslot":
        A_raw = torch.softmax(logits_bmn, dim=1).view(B, M, H, Wtok)
    elif amap == "slotsum":
        A_raw = torch.softmax(logits_bmn, dim=2).view(B, M, H, Wtok)
    else:
        raise ValueError("amap must be pxslot|slotsum")
    return A_raw, logits_bmn, (H, Wtok)

@torch.no_grad()
def apply_mask(A_raw, mode="round"):
    if mode == "none": return A_raw
    B, M, H, Wtok = A_raw.shape
    mask = round_pair_mask_token(M, H, Wtok, A_raw.device).unsqueeze(0)
    return A_raw * mask  # 정규화는 하지 않음(질량 보존)

@torch.no_grad()
def compute_P(logits_bmn, A_masked, p_mode="lse", tau=0.7):
    if p_mode == "uniform":
        B, M, _ = logits_bmn.shape
        return torch.full((B, M), 1.0/M, device=logits_bmn.device)
    elif p_mode == "mass":
        mass = A_masked.flatten(2).sum(-1)  # (B,M)
        z = (mass - mass.mean(dim=1, keepdim=True))/max(1e-6,float(tau))
        return torch.softmax(z, dim=1)
    elif p_mode == "lse":
        s = torch.logsumexp(logits_bmn, dim=2)  # (B,M) over pixels
        z = (s - s.mean(dim=1, keepdim=True))/max(1e-6,float(tau))
        return torch.softmax(z, dim=1)
    else:
        raise ValueError("p_mode must be uniform|mass|lse")

@torch.no_grad()
def slot_embeddings(tokens_bnd, A_masked, avg=True):
    B, N, D = tokens_bnd.shape; B2, M, H, Wtok = A_masked.shape
    assert B==B2 and N==H*Wtok
    t = F.normalize(tokens_bnd, dim=-1)
    t_flat = t.view(B, H*Wtok, D)
    A_flat = A_masked.view(B, M, H*Wtok)
    S = torch.bmm(A_flat, t_flat)  # (B,M,D)
    if avg:
        mass = A_flat.sum(-1).clamp_min(1e-8).unsqueeze(-1)
        S = S / mass
    return F.normalize(S, dim=-1)

@torch.no_grad()
def aggregate_logits(S_bmd, P_bm, W_cd, b_c, agg="wsum", slot_topk=3):
    slot_logits = torch.einsum("bmd,cd->bmc", S_bmd, W_cd) + b_c.view(1,1,-1)  # (B,M,C)
    if agg == "wsum":
        w = P_bm / P_bm.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bmc,bm->bc", slot_logits, w), slot_logits
    elif agg == "softk":
        B, M, C = slot_logits.shape; k = min(int(slot_topk), M)
        topv, topi = torch.topk(P_bm, k=k, dim=1)
        mask = torch.zeros_like(P_bm).scatter_(1, topi, 1.0)
        sel = P_bm * mask
        w = sel / sel.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bmc,bm->bc", slot_logits, w), slot_logits
    elif agg == "sum":
        return slot_logits.sum(dim=1), slot_logits
    elif agg == "maxslot":
        return slot_logits.max(dim=1).values, slot_logits
    else:
        raise ValueError("agg must be wsum|softk|sum|maxslot")

# ---------------- 색상(슬롯별 고정) & 범례 ----------------
def make_slot_palette(M):
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20")
    cols = []
    for m in range(M):
        c = cmap(m % 20)  # RGBA
        cols.append((c[0], c[1], c[2], 0.35))  # alpha=0.35
    return cols

def save_slot_legend(palette, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    M = len(palette)
    cols = 4
    rows = int(np.ceil(M/cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols*2.2, rows*0.9), dpi=150)
    ax = np.atleast_2d(ax)
    for m in range(M):
        r, c = divmod(m, cols)
        a = ax[r, c]
        a.add_patch(Rectangle((0,0),1,1,facecolor=palette[m], edgecolor=None))
        a.set_title(f"slot {m}", fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(0,1); a.set_ylim(0,1)
    for m in range(M, rows*cols):
        r, c = divmod(m, cols)
        ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_png); plt.close(fig)

# ---------------- 시각화 ----------------
def denorm_img(x):
    return (x*0.3081 + 0.1307).clamp(0,1)

def draw_overlay(img_1x28x28, slot_pairs_for_m, top_slots, out_png, meta_text, palette):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = denorm_img(img_1x28x28).squeeze(0).cpu().numpy()
    cells = grid_cells_pix()

    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    for slotinfo in top_slots:
        m = slotinfo["m"]; (i,j) = slotinfo["pair"]
        assert i != j, "pair must be two different cells"
        col = palette[m]
        for cell_idx in (i,j):
            x0,y0,x1,y1 = cells[cell_idx]
            rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=0, edgecolor=None,
                             facecolor=col, fill=True)
            ax.add_patch(rect)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(meta_text, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)

def draw_slot_panel_grid(img_1x28x28, selected_slots, slot_pairs_for_m,
                         slot_logits_row, P_row, mass_row, idx2char, palette,
                         out_png, title="slots by contribution (desc)"):
    """
    selected_slots: [slot indices] (기여도 내림차순 정렬 상태)
    각 슬롯 패널: 왼쪽 원본+영역, 오른쪽 top-3 클래스 바차트
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells = grid_cells_pix()
    img = denorm_img(img_1x28x28).squeeze(0).cpu().numpy()

    n = len(selected_slots)
    if n == 0:
        return
    cols = 2  # (overlay, bar)
    # 한 행에 몇 슬롯?
    slots_per_row = 4  # 한 줄에 4슬롯(=8 axes)
    rows = int(np.ceil(n / slots_per_row))

    fig = plt.figure(figsize=(cols*slots_per_row*2.2, rows*2.2), dpi=140)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(rows, slots_per_row*cols, figure=fig)
    # 전체 제목
    fig.suptitle(title, fontsize=12, y=1.02)

    for idx, m in enumerate(selected_slots):
        r = idx // slots_per_row
        c = (idx % slots_per_row) * cols

        # ---- (A) overlay ----
        ax1 = fig.add_subplot(gs[r, c])
        ax1.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        i, j = slot_pairs_for_m[m]; assert i != j
        col = palette[m]
        for cell_idx in (i,j):
            x0,y0,x1,y1 = cells[cell_idx]
            rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=0, edgecolor=None,
                             facecolor=col, fill=True)
            ax1.add_patch(rect)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title(f"slot {m}  P={P_row[m]:.3f}  mass={mass_row[m]:.2f}", fontsize=8)

        # ---- (B) bar chart top-3 ----
        ax2 = fig.add_subplot(gs[r, c+1])
        # top-3: slot_logits_row[m] 기준
        vals, idxs = torch.topk(slot_logits_row[m], k=min(3, slot_logits_row.size(1)))
        labels = [idx2char[int(i)] for i in idxs.tolist()]
        ax2.bar(range(len(vals)), [float(v) for v in vals.tolist()])
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=0)
        ax2.set_title("slot-top3 logits", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------- 슬롯 선택(정렬/규칙) ----------------
def select_slots(P_row, slot_logits_row, mass_row, cls_id,
                 mode="mass", k_slots=12, cumulative=0.0, min_thresh=0.0):
    """
    mode:
      - "mass": 슬롯 질량(mass_row)을 점수로 사용 (A_masked의 합; 크면 넓고 강하게 본 슬롯)
      - "contrib": 기여도 w*slot_logit[:, cls_id] 사용
    선택 규칙:
      - cumulative>0: 양(+) 점수를 큰 순으로 누적하여 목표비율 도달할 때까지
      - min_thresh>0: 양(+) 점수가 임계 이상인 슬롯만
      - 그 외: top-k
    반환: selected_idx(list), score(torch.Tensor)
    """
    if mode == "mass":
        score = mass_row.clone().clamp_min(0)            # 음수 방지
        total = float(score.sum().item())
        score_norm = score / max(total, 1e-8)            # 누적 규칙용 정규화
    else:  # contrib
        w = P_row / P_row.sum().clamp_min(1e-8)
        score = (w * slot_logits_row[:, cls_id]).clone()
        # 양(+) 기여만 우선 고려
        score = torch.where(score > 0, score, torch.zeros_like(score))
        ssum = float(score.sum().item())
        score_norm = score / max(ssum, 1e-8) if ssum > 0 else score

    M = score.numel()
    if M == 0:
        return [], score

    # 양(+) 점수 인덱스 정렬(desc)
    idx_all = torch.arange(M)
    pos_mask = score > 0
    pos_idx = idx_all[pos_mask]
    pos_vals = score_norm[pos_mask]
    order = torch.argsort(pos_vals, descending=True)
    pos_idx_sorted = pos_idx[order]
    pos_vals_sorted = pos_vals[order]

    selected = []

    # (1) 누적 비율 선택
    if cumulative and cumulative > 0.0 and pos_idx_sorted.numel() > 0:
        target = min(1.0, float(cumulative))
        acc = 0.0
        for m, v in zip(pos_idx_sorted.tolist(), pos_vals_sorted.tolist()):
            selected.append(m)
            acc += v
            if acc >= target:
                break
        # 전혀 선택 못했으면 top-k로 백업
        if not selected and k_slots > 0:
            k = min(int(k_slots), M)
            selected = torch.topk(score, k=k).indices.tolist()
        return selected, score

    # (2) 임계값 선택
    if min_thresh and min_thresh > 0.0 and pos_idx_sorted.numel() > 0:
        for m, v in zip(pos_idx_sorted.tolist(), pos_vals_sorted.tolist()):
            if v >= float(min_thresh):
                selected.append(m)
        # 부족하면 top-k로 보충
        need = max(0, min(int(k_slots), M) - len(selected))
        if need > 0:
            rest_mask = torch.ones(M, dtype=torch.bool)
            if selected:
                rest_mask[torch.tensor(selected, dtype=torch.long)] = False
            rest_idx = idx_all[rest_mask]
            rest_vals = score[rest_mask]
            add = torch.topk(rest_vals, k=min(need, rest_idx.numel())).indices
            selected.extend(rest_idx[add].tolist())
        return selected, score

    # (3) 기본: top-k
    k = min(int(k_slots), M)
    selected = torch.topk(score, k=k).indices.tolist()
    return selected, score


# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    # (기존 인자들 아래)
    ap.add_argument("--sort_by", choices=["mass","contrib"], default="mass")


    # trunk meta(ckpt meta가 있으면 우선)
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)

    # 파이프라인 옵션
    ap.add_argument("--amap", choices=["pxslot","slotsum"], default="pxslot")
    ap.add_argument("--mask", choices=["none","round"], default="round")
    ap.add_argument("--p_mode", choices=["uniform","mass","lse"], default="lse")
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--norm_tokens", type=int, default=0)
    ap.add_argument("--norm_queries", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--agg", choices=["wsum","softk","sum","maxslot"], default="wsum")

    # 오버레이(입력 위): 겹침 최소화를 위해 소수만
    ap.add_argument("--overlay_topk", type=int, default=3)

    # 슬롯 선택/정렬(패널용)
    ap.add_argument("--select_for", choices=["true","pred"], default="pred")
    ap.add_argument("--k_slots", type=int, default=12)
    ap.add_argument("--cumulative", type=float, default=0.0)  # 0이면 비활성, 예: 0.9
    ap.add_argument("--min_contrib", type=float, default=0.0) # 0이면 비활성

    # 시각화 뽑을 수량
    ap.add_argument("--num_correct", type=int, default=12)
    ap.add_argument("--num_confused", type=int, default=12)
    ap.add_argument("--num_random", type=int, default=12)

    # 패널(subplot) 생성 여부
    ap.add_argument("--panel", type=int, default=1)

    # 최대 처리 배치 수
    ap.add_argument("--max_batches", type=int, default=999999)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk & meta
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = args.d_model or int(meta.get("d_model", 64))
    nhead   = args.nhead   or int(meta.get("nhead", 4))
    nlayers = args.num_layers or int(meta.get("num_layers", 1))
    filters = np.load(args.conv1_filters)

    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=nlayers, d_ff=256, conv1_filters=filters).to(device).eval()
    q  = load_slot_queries(args.ckpt, device, normalize=bool(args.norm_queries))
    W_cls, b_cls = load_classifier(args.ckpt, device)

    # label mapping
    idx2char = get_label_mapping()

    # data
    loader = get_loader(args.split, bs=256, nw=2)

    # palette & legend
    M = q.size(0)
    palette = make_slot_palette(M)
    save_slot_legend(palette, os.path.join(args.out_dir, "slot_legend.png"))

    all_preds = []; all_logits = []; all_y = []; all_imgs = []; all_slots_meta = []

    # -------- 1) 전체 forward --------
    with torch.no_grad():
        batches = 0
        for x, y in tqdm(loader, desc="[viz-forward]"):
            x = x.to(device); y = y.to(device)
            tok, _ = trunk(x)  # (B,196,D)
            A_raw, logits_bmn, (Htok, Wtok) = compute_maps(
                tok, q, amap=args.amap,
                norm_tokens=bool(args.norm_tokens),
                norm_queries=bool(args.norm_queries),
                scale=args.scale
            )
            A_masked = apply_mask(A_raw, mode=args.mask)
            P = compute_P(logits_bmn, A_masked, p_mode=args.p_mode, tau=args.tau)  # (B,M)
            S = slot_embeddings(tok, A_masked, avg=True)                            # (B,M,D)
            logits, slot_logits = aggregate_logits(S, P, W_cls, b_cls, agg=args.agg, slot_topk=args.overlay_topk)
            pred = logits.argmax(dim=1)

            all_preds.append(pred.cpu())
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
            all_imgs.append(x.cpu())
            # 메타: P, 슬롯별 로짓, 마스크 질량
            all_slots_meta.append({
                "P": P.cpu(),
                "slot_logits_full": (torch.einsum("bmd,cd->bmc", S.cpu(), W_cls.cpu()) + b_cls.view(1,1,-1).cpu()),
                "A_mass": A_masked.flatten(2).sum(-1).cpu()
            })

            batches += 1
            if batches >= args.max_batches:
                break

    pred_all   = torch.cat(all_preds, 0)
    logits_all = torch.cat(all_logits, 0)
    y_all      = torch.cat(all_y, 0)
    imgs_all   = torch.cat(all_imgs, 0)
    P_all      = torch.cat([m["P"] for m in all_slots_meta], 0)                    # (N,M)
    slotlog_all= torch.cat([m["slot_logits_full"] for m in all_slots_meta], 0)     # (N,M,C)
    Amass_all  = torch.cat([m["A_mass"] for m in all_slots_meta], 0)               # (N,M)

    # scores dict
    top2_all = torch.topk(logits_all, k=2, dim=1)
    scores = {
        "pred": pred_all.numpy().tolist(),
        "logits": logits_all,
        "margin": (top2_all.values[:,0] - top2_all.values[:,1])
    }

    # -------- 2) 샘플 선별 --------
    def select_samples(scores, y_true, num_correct=12, num_confused=12, num_random=12):
        N = len(scores["pred"])
        idx_all = list(range(N))
        conf = scores["logits"].max(dim=1).values
        correct_idx = [i for i in idx_all if scores["pred"][i] == y_true[i]]
        correct_idx.sort(key=lambda i: float(conf[i]), reverse=True)
        pick_correct = correct_idx[:num_correct]
        margin = scores["margin"]
        wrong_idx = [i for i in idx_all if scores["pred"][i] != y_true[i]]
        wrong_idx.sort(key=lambda i: float(abs(margin[i])))
        pick_confused = wrong_idx[:num_confused]
        rest = [i for i in idx_all if i not in pick_correct and i not in pick_confused]
        random.shuffle(rest)
        pick_random = rest[:num_random]
        return pick_correct, pick_confused, pick_random

    pick_correct, pick_confused, pick_random = select_samples(
        scores, y_all.numpy().tolist(),
        num_correct=args.num_correct, num_confused=args.num_confused, num_random=args.num_random
    )

    # 인덱스 CSV
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","group","pred","pred_char","true","true_char","margin"])
        for gname, picks in [("correct", pick_correct), ("confused", pick_confused), ("random", pick_random)]:
            for i in picks:
                pr, yt = int(pred_all[i]), int(y_all[i])
                w.writerow([i, gname, pr, idx2char[pr], yt, idx2char[yt], float(scores["margin"][i])])

    # -------- 3) per-sample 시각화 + JSON --------
    pair_for_m, all_pairs = round_pair_map(P_all.size(1))
    groups = [("correct", pick_correct), ("confused", pick_confused), ("random", pick_random)]

    for gname, picks in groups:
        out_group = os.path.join(args.out_dir, "samples", gname)
        os.makedirs(out_group, exist_ok=True)

        for i in picks:
            img = imgs_all[i]        # (1,28,28)
            yt  = int(y_all[i]); yt_c = idx2char[yt]
            pr  = int(pred_all[i]); pr_c = idx2char[pr]
            P   = P_all[i]           # (M,)
            sl  = slotlog_all[i]     # (M,C)
            mass= Amass_all[i]       # (M,)

            # 개입 슬롯 수 (질량 기준): mass > 1% * 총질량
            total_mass = float(mass.sum())
            eff_slots = int((mass > max(1e-8, 0.01*total_mass)).sum().item())

            # pred top2 문자
            top2v, top2i = torch.topk(logits_all[i], k=2)
            top2_chars = [(idx2char[int(c)], float(v)) for v, c in zip(top2v.tolist(), top2i.tolist())]

            # 메인 오버레이: overlay_topk (겹침 최소화)
            # 기준 클래스
            cls_for = pr if args.select_for == "pred" else yt
            w_norm = (P / P.sum().clamp_min(1e-8))
            contrib_vec = w_norm * sl[:, cls_for]  # (M,)

            # overlay_topk 만큼만 메인 오버레이에 표시
            k_overlay = max(0, int(args.overlay_topk))
            top_overlay_idx = torch.topk(contrib_vec, k=min(k_overlay, contrib_vec.numel())).indices.tolist() if k_overlay>0 else []
            overlay_slots = [{"m": m, "pair": pair_for_m[m]} for m in top_overlay_idx]
            cap = f"idx={i} | TRUE={yt_c} | PRED={pr_c} top2={[(c,round(v,2)) for c,v in top2_chars]} margin={float(scores['margin'][i]):.2f}"
            png_overlay = os.path.join(out_group, f"{i:06d}_overlay.png")
            draw_overlay(img, pair_for_m, overlay_slots, png_overlay, cap, palette)

            # 패널용 슬롯 선택(정렬)
            selected_slots, score_all = select_slots(
                P, sl, mass, cls_for,
                mode=args.sort_by,
                k_slots=int(args.k_slots),
                cumulative=float(args.cumulative),
                min_thresh=float(args.min_contrib)
            )

            # 선택 기준에 따라 정렬 (기본 mass 내림차순)
            score_np = score_all.detach().cpu().numpy()
            selected_slots.sort(key=lambda m: score_np[m], reverse=True)

            # 패널 PNG
            if args.panel and len(selected_slots)>0:
                png_panel = os.path.join(out_group, f"{i:06d}_slots.png")
                draw_slot_panel_grid(img, selected_slots, pair_for_m,
                                     sl, P, mass, idx2char, palette,
                                     png_panel,
                                     title=f"idx={i} | basis={args.select_for}({idx2char[cls_for]}) | slots={len(selected_slots)} | eff={eff_slots}")

            # JSON 메타 저장(선별 슬롯 상세)
            def slot_top3_chars(slot_logits_row):
                vals, idxs = torch.topk(slot_logits_row, k=min(3, slot_logits_row.numel()))
                return [{"id": int(c), "char": idx2char[int(c)], "logit": float(v)} for v, c in zip(vals.tolist(), idxs.tolist())]

            slots_detail = []
            for m in selected_slots:
                i1, i2 = pair_for_m[m]
                slots_detail.append({
                    "slot": m, "pair": [i1, i2],
                    "P": float(P[m]), "mass": float(mass[m]),
                    "slot_top3": slot_top3_chars(sl[m])
                })

            jmeta = {
                "idx": i,
                "group": gname,
                "true": {"id": yt, "char": yt_c},
                "pred": {"id": pr, "char": pr_c},
                "pred_top2": [{"id": int(top2i[j]), "char": idx2char[int(top2i[j])], "logit": float(top2v[j])} for j in range(min(2, top2i.numel()))],
                "margin": float(scores["margin"][i]),
                "P_max": float(P.max().item()),
                "effective_slots": eff_slots,
                "basis": {"select_for": args.select_for, "class_id": cls_for, "class_char": idx2char[cls_for]},
                "selected_slots": slots_detail,
                "pairs_reference": [{"pair_index": pidx, "cells": list(all_pairs[pidx])} for pidx in range(len(all_pairs))],
                "config": {
                    "amap": args.amap, "mask": args.mask, "p_mode": args.p_mode,
                    "tau": args.tau, "norm_tokens": int(args.norm_tokens), "norm_queries": int(args.norm_queries),
                    "scale": args.scale, "agg": args.agg,
                    "overlay_topk": int(args.overlay_topk),
                    "select_for": args.select_for, "k_slots": int(args.k_slots),
                    "cumulative": float(args.cumulative), "min_contrib": float(args.min_contrib),
                    "panel": int(args.panel)
                }
            }
            with open(os.path.join(out_group, f"{i:06d}.json"), "w") as f:
                json.dump(jmeta, f, indent=2)

    print(f"[done] outputs:")
    print(f" - Slot legend: {os.path.join(args.out_dir,'slot_legend.png')}")
    print(f" - Index CSV  : {os.path.join(args.out_dir,'index.csv')}")
    print(f" - Samples    : {os.path.join(args.out_dir,'samples')} (correct/confused/random)")
    print("메인 오버레이는 소수만, 자세한 슬롯 분석은 패널 PNG/JSON을 확인하세요.")

if __name__ == "__main__":
    main()
