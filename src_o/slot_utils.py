#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, json, itertools
import numpy as np
import torch
import torch.nn.functional as F

# ----------------- ckpt -> slot_queries -----------------
def load_slot_queries_from_ckpt(ckpt_path, device):
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
    return F.normalize(q.to(device), dim=-1)  # (M,D)

# ----------------- 3x3 grid pair mask fallback -----------------
def _grid14_edges():
    # 14 -> 5,5,4 split: edges [0,5,10,14]
    return [0, 5, 10, 14]

def _build_grid3_boxes_14x14():
    xs = _grid14_edges(); ys = _grid14_edges()
    boxes = []
    for gy in range(3):
        for gx in range(3):
            x0, x1 = xs[gx], xs[gx+1]-1
            y0, y1 = ys[gy], ys[gy+1]-1
            boxes.append((x0,y0,x1,y1))
    return boxes  # 9개

def _build_pair_masks_14x14(device):
    boxes = _build_grid3_boxes_14x14()
    from itertools import combinations
    pairs = list(combinations(range(9), 2))  # 36
    M = len(pairs); H=W=14
    masks = torch.zeros((M, H, W), dtype=torch.float32, device=device)
    for m,(i,j) in enumerate(pairs):
        for idx in (i,j):
            x0,y0,x1,y1 = boxes[idx]
            masks[m, y0:y1+1, x0:x1+1] = 1.0
    return masks, pairs, boxes

def _apply_pair_mask_fallback(A_bmhw, assign="round"):
    """
    A_bmhw: (B,M,H,W)  # per-pixel softmax over slots
    assign=="round": 슬롯 m -> 미리 정의된 36개 페어를 m%36로 순환 배정(겹침 방지).
    assign=="auto" : (옵션) 슬롯별로 좋은 페어를 독립 선택.
    반환: 정규화하지 않은 마스크 적용 맵 A_masked (B,M,H,W)
    """
    B, M, H, W = A_bmhw.shape
    assert H == 14 and W == 14, "fallback은 14x14에서만 지원"
    masks_phw, pairs, _ = _build_pair_masks_14x14(A_bmhw.device)  # (36,14,14)

    if assign == "round":
        P = masks_phw.size(0)  # 36
        idx_m = torch.arange(M, device=A_bmhw.device) % P            # (M,)
        mask_mhw = masks_phw[idx_m]                                   # (M,14,14)
        A_masked = A_bmhw * mask_mhw.unsqueeze(0)                     # (B,M,14,14)
    elif assign == "auto":
        score_bmp = torch.einsum("bmhw,phw->bmp", A_bmhw, masks_phw)  # (B,M,36)
        topi_bm = score_bmp.argmax(dim=2)                             # (B,M)
        mask_bmhw = masks_phw[topi_bm.view(-1)].view(B, M, H, W)
        A_masked = A_bmhw * mask_bmhw
    else:
        raise ValueError("assign must be 'round' or 'auto'")
    return A_masked  # <-- 정규화하지 않음


# ----------------- slot-query 기반 슬롯 추출 -----------------
@torch.no_grad()
def extract_slots_with_queries(tokens_bnd, slot_q_md, use_pair_mask=True,
                               tau_p=0.7, grid=3, assign="round",
                               heat_map_bhw=None):
    """
    return: A_eff(=A_masked), P, S, A_raw
    """
    B, N, D = tokens_bnd.shape
    M = slot_q_md.size(0)
    H = W = int(math.sqrt(N)); assert H*W == N

    # tokens 정규화
    t = F.normalize(tokens_bnd, dim=-1)
    t_flat = t.view(B, H*W, D)

    # 1) per-pixel softmax(슬롯축)
    logits_bnm = torch.einsum("bnd,md->bnm", t, slot_q_md) / math.sqrt(D)
    A_raw = torch.softmax(logits_bnm, dim=2).permute(0,2,1).contiguous().view(B, M, H, W)

    # 2) 마스크 적용 (정규화 금지)
    if use_pair_mask:
        try:
            from src_o.o_spmask import apply_spatial_pair_mask
            A_masked = apply_spatial_pair_mask(A_raw, enable=True, grid=grid, assign=assign)
        except Exception:
            A_masked = _apply_pair_mask_fallback(A_raw, assign=assign)
    else:
        A_masked = A_raw

    # 3) P: 슬롯 질량 기반 (정규화 없이 집계)
    mass = A_masked.flatten(2).sum(-1)                               # (B,M)
    z = (mass - mass.mean(dim=1, keepdim=True)) / max(1e-6, float(tau_p))
    P = torch.softmax(z, dim=1)

    # 4) S: 질량으로 평균낸 뒤 L2 정규화
    A_flat = A_masked.view(B, M, -1)                                 # (B,M,HW)
    S = torch.bmm(A_flat, t_flat)                                    # (B,M,D)
    S = S / mass.clamp_min(1e-8).unsqueeze(-1)                       # 평균
    S = F.normalize(S, dim=-1)

    # 저장용: A_eff는 마스크 적용본을 그대로 반환
    return A_masked, P, S, A_raw

# ----------------- 프로토 로딩(0-벡터 방어 포함) -----------------
def load_prototypes_json(path, device, filter_zero_proto=True, zero_eps=1e-6):
    with open(path, "r") as f:
        data = json.load(f)
    per = data.get("per_class", data)
    C_list, C_cls = [], []

    def _as_vecs(block):
        vecs = []
        if isinstance(block, dict):
            if "mu" in block:
                mus = block["mu"]
                if isinstance(mus, list) and mus and isinstance(mus[0], (list, tuple)):
                    vecs.extend(mus)
                else:
                    vecs.append(mus)
            else:
                for key in ("center","centers","protos","clusters","items","vectors"):
                    if key in block:
                        v = block[key]
                        if isinstance(v, list):
                            for p in v:
                                vecs.append(p.get("mu", p) if isinstance(p, dict) else p)
                        else:
                            vecs.append(v)
        elif isinstance(block, list):
            for p in block:
                vecs.append(p.get("mu", p) if isinstance(p, dict) else p)
        else:
            vecs.append(block)
        return vecs

    items = per.items() if isinstance(per, dict) else [("0", per)]
    for c_str, block in items:
        try: cid = int(c_str)
        except: cid = sum(map(ord, str(c_str)))
        for v in _as_vecs(block):
            C_list.append(np.asarray(v, np.float32).reshape(-1))
            C_cls.append(cid)
    if not C_list:
        raise RuntimeError("no prototypes parsed")
    C = torch.from_numpy(np.stack(C_list, 0)).float().to(device)
    C_norm = torch.linalg.norm(C, dim=1)
    if filter_zero_proto:
        keep = (C_norm > zero_eps)
        C = C[keep]
        C_cls = np.asarray(C_cls, np.int64)[keep.cpu().numpy()]
    else:
        C_cls = np.asarray(C_cls, np.int64)
    C = F.normalize(C, dim=1)
    C_cls = torch.from_numpy(C_cls).to(device)
    return C, C_cls, data.get("meta", {})

# ----------------- 라벨 압축 & evidence -----------------
@torch.no_grad()
def build_class_index(C_cls: torch.Tensor):
    labels_sorted = sorted(int(x) for x in torch.unique(C_cls).tolist())
    label_to_col = {lab: j for j, lab in enumerate(labels_sorted)}
    col_to_label = {j: lab for lab, j in label_to_col.items()}
    return labels_sorted, label_to_col, col_to_label

@torch.no_grad()
def per_slot_evidence_compact(S_bmd, C_kd, C_cls, labels_sorted,
                              class_reduce="lse", proto_tau=0.4):
    batched = (S_bmd.dim() == 3)
    if not batched: S_bmd = S_bmd.unsqueeze(0)
    B, M, D = S_bmd.shape; K = C_kd.size(0)
    sim = torch.einsum("bmd,kd->bmk", S_bmd, C_kd).float()  # (B,M,K)
    neg_inf = torch.tensor(-1e9, device=S_bmd.device, dtype=sim.dtype)
    Cp = len(labels_sorted); evi = sim.new_zeros(B, M, Cp)
    C_cls = C_cls.view(1,1,K)
    for j, lab in enumerate(labels_sorted):
        mask_c = (C_cls == lab)
        s_c = sim.masked_fill(~mask_c, neg_inf)
        if class_reduce == "max":
            evi_c, _ = torch.max(s_c, dim=2)
        else:
            xmax, _ = torch.max(s_c, dim=2, keepdim=True)
            lse = torch.sum(torch.exp((s_c - xmax)/proto_tau), dim=2, keepdim=True).clamp_min(1e-8)
            evi_c = (xmax + proto_tau * torch.log(lse)).squeeze(2)
        evi[:, :, j] = evi_c
    return evi if batched else evi.squeeze(0)
