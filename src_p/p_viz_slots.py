#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
o_viz_slots.py  (9C2 -> 9C4 대응판)
- 덤프 없이 ckpt와 conv1 filters로 바로 시각화/점검.
- 학습과 동일 가정: 3x3 중 4칸(=9C4) 유니온 마스크를 'round' 매핑으로 사용.
- 최종 로짓 집계는 기본 wsum(가중합). 슬롯 패널은 기본 '질량(mass)' 기준 정렬.
- 히스토그램 y축은 'probability mass per bin'(bin 폭×밀도), 막대 합=1.
"""

import os, json, math, csv, argparse, random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk

# ----------------------------- 유틸 -----------------------------

def denorm_img(x):  # x: (1,28,28) normalized
    return (x*0.3081 + 0.1307).clamp(0,1)

def get_label_mapping():
    tmp = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=transforms.ToTensor())
    return [str(c) for c in list(tmp.classes)]

def save_hist_mass(values, out_png, bins=50, rng=(0.0, 1.0), xlabel="slot weight P"):
    """
    y축이 '각 bin의 확률질량(probability mass)'가 되도록 그림 (합=1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = np.asarray(values).ravel()
    hist_density, bin_edges = np.histogram(values, bins=bins, range=rng, density=True)
    bin_widths = np.diff(bin_edges)
    mass = hist_density * bin_widths
    centers = bin_edges[:-1] + bin_widths / 2.0
    plt.figure(figsize=(6, 4))
    plt.bar(centers, mass, width=bin_widths, align="center", edgecolor="none")
    plt.xlabel(xlabel); plt.ylabel("probability mass per bin")
    plt.title("Histogram (mass per bin; sums to 1)")
    plt.ylim(0, max(1e-9, mass.max()) * 1.1)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ----------------------------- 3x3 셀/마스크 -----------------------------

def grid_cells_pix():
    """28x28 픽셀에서 3분할(10,10,8). 반환: 9개 셀의 (x0,y0,x1,y1)"""
    edges = [0, 10, 20, 28]
    cells = []
    for gy in range(3):
        for gx in range(3):
            x0, x1 = edges[gx], edges[gx+1]
            y0, y1 = edges[gy], edges[gy+1]
            cells.append((x0, y0, x1, y1))
    return cells  # len=9

@torch.no_grad()
def round_comb4_mask_token(M, H, W, device):
    """
    14x14 토큰 격자에서 9C4 조합(=126)을 미리 만들고 슬롯 M개에 라운드 매핑.
    반환: (M,14,14) {0,1}
    """
    from itertools import combinations
    assert H == 14 and W == 14
    xs, ys = [0,5,10,14], [0,5,10,14]
    cell_masks = []
    for gy in range(3):
        for gx in range(3):
            m = torch.zeros(H, W, device=device)
            m[ys[gy]:ys[gy+1], xs[gx]:xs[gx+1]] = 1.0
            cell_masks.append(m)        # (9,14,14)
    combs = list(combinations(range(9), 3))  # 126
    unions = []
    for comb in combs:
        u = torch.zeros(H, W, device=device)
        for idx in comb:
            u = torch.maximum(u, cell_masks[idx])
        unions.append(u)
    unions = torch.stack(unions, 0)  # (126,14,14)
    if M <= unions.size(0): return unions[:M]
    reps = (M + unions.size(0) - 1)//unions.size(0)
    return unions.repeat(reps, 1, 1)[:M]

@torch.no_grad()
def round_comb4_for_M(M):
    """시각화용: 슬롯 m -> (4칸 셀 인덱스)의 라운드 매핑(고정)."""
    from itertools import combinations
    combs = list(combinations(range(9), 3))  # 126
    return [combs[m % len(combs)] for m in range(M)], combs

# ----------------------------- trunk/헤드 로딩 -----------------------------

@torch.no_grad()
def load_slot_queries(ckpt_path, device, normalize=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    q = None
    for pfx in ("module.", ""):
        k = f"{pfx}slot_queries"
        if k in sd:
            q = sd[k].float(); break
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
            W = sd[wk].float().to(device); b = sd[bk].float().to(device); break
    if W is None or b is None:
        raise RuntimeError("classifier.{weight,bias} not found in ckpt")
    return W, b

# ----------------------------- 맵/풀링/집계 -----------------------------

@torch.no_grad()
def compute_maps(tokens_bnd, q_md, amap="pxslot", norm_tokens=False, norm_queries=False, scale=1.0):
    """
    tokens_bnd: (B,N,D), q_md: (M,D)
    A_raw: (B,M,14,14)  — 기본은 'slotsum'(슬롯 분포) 사용(softmax over tokens N).
    """
    B,N,D = tokens_bnd.shape; M = q_md.size(0)
    H = Wtok = int(math.sqrt(N)); assert H*Wtok==N==196
    t = F.normalize(tokens_bnd, dim=-1) if norm_tokens else tokens_bnd
    q = F.normalize(q_md, dim=-1) if norm_queries else q_md
    logits_bmn = torch.einsum("bnd,md->bmn", t, q) * float(scale)  # (B,M,N)
    if amap == "pxslot":
        # 픽셀별 softmax over slots — 시각적 의미는 있으나 S 풀링엔 부적합
        A_raw = torch.softmax(logits_bmn, dim=1).view(B,M,H,Wtok)
    elif amap == "slotsum":
        # 슬롯별 softmax over tokens — S 풀링에 적합(학습과 동일)
        A_raw = torch.softmax(logits_bmn, dim=2).view(B,M,H,Wtok)
    else:
        raise ValueError("amap must be pxslot|slotsum")
    return A_raw, logits_bmn, (H,Wtok)

@torch.no_grad()
def aggregate_logits(S_bmd, P_bm, W_cd, b_c, agg="wsum", slot_topk=0):
    """
    S_bmd: (B,M,D), P_bm: (B,M), W_cd: (C,D), b_c: (C,)
    반환: (B,C), slot_logits(bmc)
    """
    slot_logits = torch.einsum("bmd,cd->bmc", S_bmd, W_cd) + b_c.view(1,1,-1)
    if agg == "wsum":
        w = P_bm / P_bm.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bmc,bm->bc", slot_logits, w), slot_logits
    elif agg == "softk":
        B,M,C = slot_logits.shape
        k = min(int(slot_topk), M)
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

# ----------------------------- 팔레트/범례 -----------------------------

def make_slot_palette(M):
    import matplotlib
    cmap = matplotlib.cm.get_cmap("tab20")
    cols=[]
    for m in range(M):
        c = cmap(m % 20); cols.append((c[0], c[1], c[2], 0.35))
    return cols

def save_slot_legend(palette, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    M = len(palette); cols = 4; rows = int(np.ceil(M/cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols*2.2, rows*0.9), dpi=150)
    ax = np.atleast_2d(ax)
    for m in range(M):
        r,c = divmod(m, cols)
        a = ax[r,c]; a.add_patch(Rectangle((0,0),1,1,facecolor=palette[m], edgecolor=None))
        a.set_title(f"slot {m}", fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(0,1); a.set_ylim(0,1)
    for m in range(M, rows*cols):
        r,c = divmod(m, cols); ax[r,c].axis("off")
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

# ----------------------------- 시각화(오버레이/패널) -----------------------------

def draw_overlay(img_1x28x28, comb_for_m, top_slots, out_png, meta_text, palette):
    """
    top_slots: [{"m":slot_index}, ...]
    지정 슬롯들의 4칸 셀을 '채움(fill=True)'으로 표시(테두리X).
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    img = denorm_img(img_1x28x28).squeeze(0).cpu().numpy()
    cells = grid_cells_pix()
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for slotinfo in top_slots:
        m = int(slotinfo["m"]); comb = comb_for_m[m]
        col = palette[m]
        for cell_idx in comb:
            x0,y0,x1,y1 = cells[cell_idx]
            rect = Rectangle((x0,y0), x1-x0, y1-y0, linewidth=0, facecolor=col, fill=True)
            ax.add_patch(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(meta_text, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

def draw_slot_panel_grid(img_1x28x28, selected_slots, comb_for_m,
                         slot_logits_row, P_row, mass_row, idx2char, palette,
                         out_png, title="slots by mass (desc)"):
    """
    슬롯별 패널: (좌) 네 칸 채움 오버레이, (우) top-3 로짓 바차트.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells = grid_cells_pix()
    img = denorm_img(img_1x28x28).squeeze(0).cpu().numpy()

    n = len(selected_slots)
    if n == 0: return
    cols = 2; slots_per_row = 4; rows = int(np.ceil(n / slots_per_row))
    fig = plt.figure(figsize=(cols*slots_per_row*2.2, rows*2.2), dpi=140)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(rows, slots_per_row*cols, figure=fig)
    fig.suptitle(title, fontsize=12, y=1.02)

    for idx, m in enumerate(selected_slots):
        r = idx // slots_per_row; c = (idx % slots_per_row) * cols

        # (A) overlay
        ax1 = fig.add_subplot(gs[r, c]); ax1.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        comb = comb_for_m[m]; col = palette[m]
        for cell_idx in comb:
            x0,y0,x1,y1 = cells[cell_idx]
            ax1.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, linewidth=0, facecolor=col, fill=True))
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title(f"slot {m}  P={P_row[m]:.3f}  mass={mass_row[m]:.2f}", fontsize=8)

        # (B) top-3 bar (로짓)
        ax2 = fig.add_subplot(gs[r, c+1])
        vals, idxs = torch.topk(slot_logits_row[m], k=min(3, slot_logits_row.size(1)))
        labels = [idx2char[int(i)] for i in idxs.tolist()]
        ax2.bar(range(len(vals)), [float(v) for v in vals.tolist()])
        ax2.set_xticks(range(len(vals))); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_title("slot-top3 logits", fontsize=8)

    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# ----------------------------- 슬롯 선택(질량/기여 기준) -----------------------------

def select_slots(P_row, slot_logits_row, mass_row, cls_id,
                 mode="mass", k_slots=16, cumulative=0.9, min_thresh=0.0):
    """
    mode:
      - "mass": 슬롯 질량으로 정렬(기본)
      - "contrib": (정규화 P)*slot_logit[:, cls_id]로 정렬
    cumulative>0: 점수 큰 순으로 누적하여 목표비율 도달할 때까지
    """
    if mode == "mass":
        score = mass_row.clone()
    else:
        w = P_row / P_row.sum().clamp_min(1e-8)
        score = w * slot_logits_row[:, cls_id]

    M = score.numel()
    order = torch.argsort(score, descending=True)
    selected = []
    acc = 0.0
    for m in order.tolist():
        v = float(score[m])
        if min_thresh > 0.0 and v < min_thresh: break
        selected.append(m); acc += max(0.0, v)
        if cumulative and acc >= float(cumulative): break
        if not cumulative and len(selected) >= int(k_slots): break
    if not selected and M > 0:
        selected = order[:int(k_slots)].tolist()
    return selected, score

# ----------------------------- 데이터 로더 -----------------------------

def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full = datasets.EMNIST("./data", split="balanced", train=True, download=True, transform=tf)
        n = len(full); nv = int(round(n*val_ratio)); nt = n-nv
        g = torch.Generator().manual_seed(seed)
        tr, va = random_split(full, [nt, nv], generator=g)
        ds = tr if split=="train" else va
    else:
        ds = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

# ----------------------------- 메인 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", type=str, default="val")

    ap.add_argument("--amap", type=str, default="slotsum")      # slotsum(학습과 동일) | pxslot
    ap.add_argument("--mask", type=str, default="round")        # round만 사용(학습 가정)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--norm_tokens", type=int, default=0)
    ap.add_argument("--norm_queries", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)

    ap.add_argument("--agg", type=str, default="wsum")          # wsum|softk|sum|maxslot
    ap.add_argument("--overlay_topk", type=int, default=6)      # 오버레이에 칠할 슬롯 개수
    ap.add_argument("--select_for", type=str, default="pred")   # pred|true
    ap.add_argument("--sort_by", choices=["mass","contrib"], default="mass")
    ap.add_argument("--k_slots", type=int, default=24)          # 패널 최대 슬롯 수
    ap.add_argument("--cumulative", type=float, default=0.95)   # 누적 점수 기준
    ap.add_argument("--min_contrib", type=float, default=0.0)

    ap.add_argument("--num_correct", type=int, default=12)
    ap.add_argument("--num_confused", type=int, default=12)
    ap.add_argument("--num_random", type=int, default=12)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk & head
    import numpy as np
    filters = np.load(args.conv1_filters)
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = int(meta.get("d_model", 64)); nhead = int(meta.get("nhead", 4)); num_layers = int(meta.get("num_layers", 2))
    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=int(meta.get("d_ff", 256)),
                   conv1_filters=filters).to(device).eval()
    slot_q = load_slot_queries(args.ckpt, device, normalize=bool(args.norm_queries))
    W, b = load_classifier(args.ckpt, device)
    M = slot_q.size(0)

    # 데이터 & 레이블
    loader = get_loader(args.split, bs=args.batch_size, nw=args.num_workers)
    idx2char = get_label_mapping()

    # comb4 고정 매핑(학습과 동일 가정)
    comb_for_m, all_combs = round_comb4_for_M(M)
    # 토큰 마스크(14x14) 고정 텐서
    mask_mhw = round_comb4_mask_token(M, 14, 14, device=device)  # (M,14,14)

    # 팔레트/범례
    palette = make_slot_palette(M)
    save_slot_legend(palette, os.path.join(args.out_dir, "slot_legend.png"))

    # 누적
    imgs_all=[]; y_all=[]; pred_all=[]; logits_all=[]; P_all=[]; slotlog_all=[]; Amass_all=[]

    with torch.no_grad():
        for x, y in tqdm(loader, desc="[viz-forward]"):
            x = x.to(device); y = y.to(device)
            tokens, _ = trunk(x)   # tokens:(B,196,d)

            # 맵 계산
            A_raw, logits_bmn, (H,Wtok) = compute_maps(tokens, slot_q, amap=args.amap,
                                                       norm_tokens=bool(args.norm_tokens),
                                                       norm_queries=False,  # 이미 위에서 처리
                                                       scale=args.scale)
            # (중요) mass는 '정규화 전' 마스크 적용 합으로 계산해야 변별력이 생김
            A_masked = A_raw * mask_mhw.view(1,M,H,Wtok)               # (B,M,14,14)
            mass = A_masked.flatten(2).sum(-1)                          # (B,M)

            # 풀링용 정규화(슬롯별 합=1) -> S 계산
            s = A_masked.flatten(2).sum(-1, keepdim=True).clamp_min(1e-8)
            A_eff = (A_masked.flatten(2) / s).view(x.size(0), M, H, Wtok)

            # 슬롯 확률 P
            z = (mass - mass.mean(dim=1, keepdim=True)) / max(1e-6, float(args.tau))
            P = torch.softmax(z, dim=1)                                  # (B,M)

            # 슬롯 임베딩 & 로짓
            S = torch.bmm(A_eff.view(x.size(0), M, -1), tokens)          # (B,M,d)
            S = F.normalize(S, dim=-1)
            logits, slot_logits = aggregate_logits(S, P, W, b, agg=args.agg, slot_topk=args.overlay_topk)

            imgs_all.append(x.cpu()); y_all.append(y.cpu())
            pred_all.append(logits.argmax(dim=1).cpu())
            logits_all.append(logits.cpu())
            P_all.append(P.cpu())
            slotlog_all.append(slot_logits.cpu())
            Amass_all.append(mass.cpu())

    imgs_all = torch.cat(imgs_all, 0); y_all = torch.cat(y_all, 0)
    pred_all = torch.cat(pred_all, 0); logits_all = torch.cat(logits_all, 0)
    P_all = torch.cat(P_all, 0); slotlog_all = torch.cat(slotlog_all, 0)
    Amass_all = torch.cat(Amass_all, 0)

    # 인덱스 선정 (correct/confused/random)
    scores = {}
    scores["logits"] = logits_all
    scores["margin"] = (logits_all.topk(2, dim=1).values[:,0] - logits_all.topk(2, dim=1).values[:,1])

    out_csv = os.path.join(args.out_dir, "index.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","group","pred","pred_char","true","true_char","margin"])
        conf = scores["logits"].max(dim=1).values
        idx_all = list(range(len(pred_all)))
        correct_idx = [i for i in idx_all if int(pred_all[i]) == int(y_all[i])]
        correct_idx.sort(key=lambda i: float(conf[i]), reverse=True)
        wrong_idx = [i for i in idx_all if int(pred_all[i]) != int(y_all[i])]
        wrong_idx.sort(key=lambda i: float(abs(scores["margin"][i])))
        rest = [i for i in idx_all if i not in set(correct_idx[:args.num_correct])|set(wrong_idx[:args.num_confused])]
        random.shuffle(rest)
        pick = [("correct", correct_idx[:args.num_correct]),
                ("confused", wrong_idx[:args.num_confused]),
                ("random", rest[:args.num_random])]
        for gname, picks in pick:
            for i in picks:
                pr, yt = int(pred_all[i]), int(y_all[i])
                w.writerow([i, gname, pr, idx2char[pr], yt, idx2char[yt], float(scores["margin"][i])])

    # 샘플별 시각화 + JSON
    cells_pix = grid_cells_pix()
    for gname, picks in pick:
        out_group = os.path.join(args.out_dir, "samples", gname); os.makedirs(out_group, exist_ok=True)
        for i in picks:
            img = imgs_all[i]; yt=int(y_all[i]); pr=int(pred_all[i])
            yt_c = idx2char[yt]; pr_c = idx2char[pr]
            P_row = P_all[i]; sl = slotlog_all[i]; mass_row = Amass_all[i]

            # 기준 클래스
            cls_for = pr if args.select_for == "pred" else yt
            w_norm = (P_row / P_row.sum().clamp_min(1e-8))
            contrib_vec = w_norm * sl[:, cls_for]  # (M,)

            # 오버레이: 상위 overlay_topk 슬롯(기여 기준)
            k_overlay = max(0, int(args.overlay_topk))
            if k_overlay > 0:
                top_overlay_idx = torch.topk(contrib_vec, k=min(k_overlay, contrib_vec.numel())).indices.tolist()
            else:
                top_overlay_idx = []
            overlay_slots = [{"m": m} for m in top_overlay_idx]
            cap = f"idx={i} | TRUE={yt_c} | PRED={pr_c} margin={float(scores['margin'][i]):.2f}"
            draw_overlay(img, comb_for_m, overlay_slots, os.path.join(out_group, f"{i:06d}_overlay.png"), cap, palette)

            # 패널: 정렬 기준 (mass|contrib)
            selected_slots, score_all = select_slots(P_row, sl, mass_row, cls_for,
                                                     mode=args.sort_by,
                                                     k_slots=int(args.k_slots),
                                                     cumulative=float(args.cumulative),
                                                     min_thresh=float(args.min_contrib))
            score_np = score_all.detach().cpu().numpy()
            selected_slots.sort(key=lambda m: score_np[m], reverse=True)
            if len(selected_slots)>0:
                draw_slot_panel_grid(img, selected_slots, comb_for_m, sl, P_row, mass_row, idx2char, palette,
                                     os.path.join(out_group, f"{i:06d}_slots.png"),
                                     title=f"idx={i} basis={args.select_for}({idx2char[cls_for]}) sort={args.sort_by} slots={len(selected_slots)}")

            # JSON 메타
            def top3_chars(row):
                vals, idxs = torch.topk(row, k=min(3, row.numel()))
                return [{"id": int(c), "char": idx2char[int(c)], "logit": float(v)} for v,c in zip(vals.tolist(), idxs.tolist())]

            slots_detail=[]
            for m in selected_slots:
                slots_detail.append({"slot": m, "comb": list(comb_for_m[m]),
                                     "P": float(P_row[m]), "mass": float(mass_row[m]),
                                     "slot_top3": top3_chars(sl[m])})
            with open(os.path.join(out_group, f"{i:06d}.json"), "w") as f:
                json.dump({
                    "idx": i, "group": gname,
                    "true": {"id": yt, "char": yt_c},
                    "pred": {"id": pr, "char": pr_c},
                    "margin": float(scores["margin"][i]),
                    "selected_slots": slots_detail,
                    "combs_reference": [{"comb_index": k, "cells": list(c)} for k,c in enumerate(round_comb4_for_M(84)[1])]
                }, f, indent=2)

    # P 히스토그램(질량축)
    save_hist_mass(P_all.numpy().flatten(), os.path.join(args.out_dir, "P_hist.png"),
                   bins=50, rng=(0.0, 1.0), xlabel="slot weight P")

    print("[done] outputs ->", args.out_dir)

if __name__ == "__main__":
    main()
