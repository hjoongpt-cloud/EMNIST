#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slot-k / Pct-of-max 스윕 실험 스크립트
- validation dump(npz)와 prototypes JSON을 읽어 고정 k / pct-of-max 규칙을 스윕
- 정확도 곡선(acc_curves.csv), P 히스토그램(p_hist.png), 정답 커버리지(coverage.csv),
  추천 설정(recommended.json), 슬롯 prior(idf/sel: slot_priors.npz)를 저장

사용 예:
python src_o/o_slotk_experiments.py \
  --dump_dir outputs/o/dumps_val \
  --proto_json outputs/o/proto/prototypes_auto.json \
  --out_dir outputs/o/slotk_exp --k_max 36 \
  --pct_list "0.3,0.4,0.5,0.6,0.7,0.8" --topX 3
"""

import os
import json
import glob
import math
import argparse
import numpy as np

from tqdm import tqdm

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- Prototype I/O -----------------------------
def load_prototypes(path, device, strict_dim=None, filter_zero_proto=True, zero_eps=1e-6):
    """
    prototypes_auto.json 을 읽어서 (K,dC) 텐서와 클래스 인덱스 (K,)를 반환.
    다양한 JSON 형태(per_class:{cid:{mu:[...]}}, {cid:[{mu:...}, ...]}, 등)를 허용.
    """
    with open(path, "r") as f:
        data = json.load(f)

    per = data.get("per_class", data)  # per_class 키가 없으면 루트 그대로 사용 시도
    C_list, C_cls = [], []

    def _as_vecs(block):
        """block에서 벡터 리스트를 추출"""
        vecs = []
        if isinstance(block, dict):
            # 흔한 케이스: {"mu": [...]} 또는 {"mu":[[...],[...]]}
            if "mu" in block:
                mus = block["mu"]
                if len(mus) > 0 and isinstance(mus[0], (list, tuple, float, int)):
                    if isinstance(mus[0], (list, tuple)):
                        vecs.extend(mus)
                    else:
                        vecs.append(mus)
            else:
                # 다른 키 후보들
                for key in ("center", "centers", "protos", "clusters", "items", "vectors"):
                    if key in block:
                        v = block[key]
                        if isinstance(v, list):
                            for p in v:
                                if isinstance(p, dict) and "mu" in p:
                                    vecs.append(p["mu"])
                                else:
                                    vecs.append(p)
                        else:
                            vecs.append(v)
        elif isinstance(block, list):
            for p in block:
                if isinstance(p, dict) and "mu" in p:
                    vecs.append(p["mu"])
                else:
                    vecs.append(p)
        else:
            vecs.append(block)
        return vecs

    # per 가 dict(class_id -> block) 라고 가정
    if isinstance(per, dict):
        items = per.items()
    else:
        # 혹시 루트가 리스트이고 내부에 {"class":cid,"mu":[...]} 같은 형식일 수도
        items = []
        for entry in per:
            if isinstance(entry, dict) and "class" in entry:
                items.append((str(entry["class"]), entry))
            else:
                # 마지막 수단: 전부 같은 클래스로 본다(0)
                items.append(("0", entry))

    for c_str, block in items:
        try:
            cid = int(c_str)
        except Exception:
            # cid가 문자열(예: "A") 등일 경우 해시 안정화를 위해 ord 합
            cid = sum(map(ord, str(c_str)))
        vecs = _as_vecs(block)
        for v in vecs:
            v_np = np.asarray(v, np.float32).reshape(-1)
            C_list.append(v_np)
            C_cls.append(cid)

    if len(C_list) == 0:
        raise RuntimeError("no prototypes parsed from json")

    C_proto = torch.from_numpy(np.stack(C_list, 0)).float().to(device)  # (K, dC)
    C_norm = torch.linalg.norm(C_proto, dim=1)
    if filter_zero_proto:
        keep = C_norm > zero_eps
        if not torch.all(keep):
            C_proto = C_proto[keep]
            C_cls = np.asarray(C_cls, np.int64)[keep.cpu().numpy()]
        else:
            C_cls = np.asarray(C_cls, np.int64)
    else:
        C_cls = np.asarray(C_cls, np.int64)

    C_proto = F.normalize(C_proto, dim=1)
    C_cls = torch.from_numpy(C_cls).to(device)

    meta = data.get("meta", {})
    return C_proto, C_cls, meta


def adapt_proto_and_feats(C_proto, S):
    """
    C_proto: (K, dC), S: (M, dF)
    서로 다른 차원이면 앞쪽 dmin만 쓰고 다시 L2정규화 (정보 손실 가능)
    방어적 타입/차원 체크 포함
    """
    if not torch.is_tensor(C_proto):
        C_proto = torch.as_tensor(C_proto)
    if not torch.is_tensor(S):
        S = torch.as_tensor(S)

    if C_proto.dim() != 2 or S.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got C_proto.shape={tuple(C_proto.shape)}, S.shape={tuple(S.shape)}"
        )

    dev = S.device
    C_proto = C_proto.to(dev, dtype=torch.float32)
    S = S.to(dev, dtype=torch.float32)

    dC = C_proto.size(1)
    dF = S.size(1)
    if dC == dF:
        return F.normalize(C_proto, dim=1), F.normalize(S, dim=1)

    dmin = min(dC, dF)
    C_use = F.normalize(C_proto[:, :dmin], dim=1)
    S_use = F.normalize(S[:, :dmin], dim=1)
    return C_use, S_use


# ----------------------------- Evidence & Aggregation -----------------------------
@torch.no_grad()
def per_slot_evidence(S_md, C_kd, C_cls, class_reduce="lse", proto_tau=0.5):
    """
    S_md: (M, d)  — 슬롯 임베딩 (정규화 가정)
    C_kd: (K, d)  — 프로토타입 (정규화 가정)
    C_cls: (K,)   — 각 프로토의 클래스 인덱스
    return: evi_mc: (M, Cmax) — 슬롯별 클래스 evidence
    """
    device = S_md.device
    M, d = S_md.shape
    K = C_kd.size(0)
    classes = torch.unique(C_cls)
    Cmax = int(torch.max(classes).item()) + 1

    sim = torch.matmul(S_md, C_kd.t()).float()  # (M, K)
    neg_inf = torch.tensor(-1e9, device=device, dtype=sim.dtype)

    evi = sim.new_zeros(M, Cmax)
    if class_reduce not in ("lse", "max"):
        raise ValueError(f"class_reduce must be 'lse' or 'max', got {class_reduce}")

    for c in classes.tolist():
        mask_c = (C_cls == c).view(1, K)
        s_c = sim.masked_fill(~mask_c, neg_inf)  # (M, K_c 유효)
        if class_reduce == "max":
            evi_c, _ = torch.max(s_c, dim=1)     # (M,)
        else:  # lse
            xmax, _ = torch.max(s_c, dim=1, keepdim=True)
            lse = torch.sum(torch.exp((s_c - xmax) / proto_tau), dim=1, keepdim=True).clamp_min(1e-8)
            evi_c = (xmax + proto_tau * torch.log(lse)).squeeze(1)
        evi[:, c] = evi_c
    return evi


def soft_topk_weights(p_m, k=None, beta=0.5):
    """
    p_m: (M,) 슬롯 확률
    k: 상위 몇 개를 남길지(없으면 전체 사용)
    beta: softmax 온도 역할(표준화 후 softmax)
    """
    if p_m.dim() != 1:
        p_m = p_m.view(-1)

    if (k is not None) and k > 0 and k < p_m.numel():
        topv, topi = torch.topk(p_m, int(k), dim=0)
        keep = torch.zeros_like(p_m)
        keep.scatter_(0, topi, 1.0)
        q = p_m * keep
    else:
        q = p_m

    z = (q - q.mean()) / max(1e-6, float(beta))
    w = torch.softmax(z, dim=0)
    return w / w.sum().clamp_min(1e-8)


def agg_fixed_k(evi_mc, p_m, k, beta=1.2, mode="softk"):
    """
    evi_mc: (M, C)
    p_m:    (M,)
    mode:   "softk" | "topk_wsum"
    """
    if p_m.dim() != 1:
        p_m = p_m.view(-1)

    if mode == "softk":
        w = soft_topk_weights(p_m, k=k, beta=beta)        # (M,)
        return (evi_mc * w.view(-1, 1)).sum(dim=0)        # (C,)
    elif mode == "topk_wsum":
        k = min(int(k), p_m.numel())
        topv, topi = torch.topk(p_m, k)
        mask = torch.zeros_like(p_m)
        mask.scatter_(0, topi, 1.0)
        w = (p_m * mask)
        w = w / w.sum().clamp_min(1e-8)
        return (evi_mc * w.view(-1, 1)).sum(dim=0)
    else:
        raise ValueError(f"unknown agg mode: {mode}")


def agg_pct_of_max(evi_mc, p_m, alpha):
    """
    p_m >= alpha * max(p_m) 인 슬롯만 남겨 가중합
    """
    if p_m.dim() != 1:
        p_m = p_m.view(-1)

    thr = float(p_m.max().item()) * float(alpha)
    mask = (p_m >= thr).float()
    w = p_m * mask
    if w.sum() <= 0:
        # 모든게 0이면 그냥 max 슬롯 하나만 사용
        idx = int(torch.argmax(p_m).item())
        w = torch.zeros_like(p_m)
        w[idx] = 1.0
    else:
        w = w / w.sum().clamp_min(1e-8)
    return (evi_mc * w.view(-1, 1)).sum(dim=0)


# ----------------------------- Slot Selectivity (진단용) -----------------------------
def compute_slot_selectivity(files, C_proto, C_cls, proto_tau=0.4):
    """
    슬롯 m이 어떤 클래스를 top-1로 더 자주 내는지 전역 통계를 계산.
    반환:
      idf (M,): log(C / (1 + #classes with freq>0))   -> 비특이 슬롯 down-weight에 사용 가능
      sel (M,C): 슬롯별 클래스 선호도(정규화) - 균등분포(1/C) -> >0 선호 / <0 비선호
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    it0 = np.load(files[0], allow_pickle=True)
    M = int(np.asarray(it0["slot_prob"]).shape[0])

    Cmax = int(torch.max(torch.unique(C_cls)).item()) + 1
    freq = np.zeros((M, Cmax), np.int64)

    for fp in tqdm(files, desc="[selectivity]"):
        it = np.load(fp, allow_pickle=True)
        P = torch.from_numpy(np.asarray(it["slot_prob"], np.float32)).to(device)  # (M,)
        S = torch.from_numpy(np.asarray(it["S_slots"],  np.float32)).to(device)  # (M,D)
        S = F.normalize(S, dim=-1)

        C_use, S_use = adapt_proto_and_feats(C_proto, S)
        evi = per_slot_evidence(S_use, C_use, C_cls, class_reduce="lse", proto_tau=proto_tau)  # (M,C)
        topc = torch.argmax(evi, dim=1).cpu().numpy()
        for m in range(M):
            freq[m, int(topc[m])] += 1

    cls_active = (freq > 0).sum(axis=1)  # 슬롯별 활성 클래스 수
    idf = np.log((Cmax) / (1.0 + cls_active))

    prob = freq / np.clip(freq.sum(axis=1, keepdims=True), 1, None)  # 슬롯별 클래스 확률
    sel = prob - (1.0 / Cmax)  # 균등 대비 편차
    return idf.astype(np.float32), sel.astype(np.float32)


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="validation용 dump 디렉토리 (npz)")
    ap.add_argument("--proto_json", required=True, help="prototypes json 경로")
    ap.add_argument("--out_dir", required=True, help="결과 저장 디렉토리")
    ap.add_argument("--k_max", type=int, default=36)
    ap.add_argument("--pct_list", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--proto_tau", type=float, default=0.4)
    ap.add_argument("--topX", type=int, default=3, help="정답 evidence가 슬롯별 topX 안에 드는 비율 측정")
    ap.add_argument("--class_reduce", choices=["lse", "max"], default="lse")
    ap.add_argument("--filter_zero_proto", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 프로토타입 로딩
    C_proto, C_cls, _ = load_prototypes(
        args.proto_json,
        device=device,
        filter_zero_proto=bool(args.filter_zero_proto)
    )

    # 2) 파일 목록
    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    assert files, f"no npz in {args.dump_dir}"

    # 3) 히스토그램 준비
    hist_bins = np.linspace(0, 1, 51)
    hist_counts = np.zeros_like(hist_bins[:-1], dtype=np.int64)

    # 4) 스윕 설정/결과 컨테이너
    k_list = list(range(1, args.k_max + 1))
    pct_list = [float(s) for s in args.pct_list.split(",") if s.strip()]
    acc_rows = [("rule", "param", "acc", "correct", "total")]
    cov_rows = [("X", "mean_cov", "median_cov")]

    # 5) 슬롯 전역 통계 (선택적 priors)
    idf, sel = compute_slot_selectivity(files, C_proto, C_cls, proto_tau=args.proto_tau)
    np.savez(os.path.join(args.out_dir, "slot_priors.npz"), idf=idf, sel=sel)

    coverages = []
    correct_total = {("fixedk", k): [0, 0] for k in k_list}
    correct_total.update({("pct", a): [0, 0] for a in pct_list})

    # (선택) 정답 순위 히스토그램 수집하려면 주석 해제
    # y_rank_all = []

    # 6) 검증 루프
    for fp in tqdm(files, desc="[val]"):
        it = np.load(fp, allow_pickle=True)

        # 필수 키 파싱
        if "clazz" not in it and "class" in it:
            y = int(it["class"])
        else:
            y = int(it["clazz"])

        P_np = np.asarray(it["slot_prob"], np.float32)
        S_np = np.asarray(it["S_slots"],  np.float32)

        P = torch.from_numpy(P_np).to(device)  # (M,)
        S = torch.from_numpy(S_np).to(device)  # (M,D)
        S = F.normalize(S, dim=-1)

        # P 히스토그램 집계
        cts, _ = np.histogram(P_np, bins=hist_bins)
        hist_counts += cts

        # 차원 적응 & evidence
        try:
            C_use, S_use = adapt_proto_and_feats(C_proto, S)
        except Exception as e:
            print(f"[shape-debug] file={fp}")
            print(f"  P.shape={tuple(P.shape)}")
            print(f"  S.shape={tuple(S.shape)}")
            print(f"  C_proto.shape={tuple(C_proto.shape)}")
            raise

        evi = per_slot_evidence(S_use, C_use, C_cls, class_reduce=args.class_reduce, proto_tau=args.proto_tau)  # (M,C)

        # 커버리지: 정답 클래스가 슬롯 evidence topX 안 비율
        ranks = torch.argsort(evi, dim=1, descending=True)  # (M,C)
        rows, cols = (ranks == y).nonzero(as_tuple=True)    # rows: 슬롯 index, cols: 순위(0=top1)

        M_cur, Cmax = evi.size(0), evi.size(1)
        topX_eff = min(int(args.topX), Cmax)
        y_rank = torch.full((M_cur,), fill_value=Cmax, device=evi.device, dtype=torch.long)
        if rows.numel() > 0:
            y_rank[rows] = cols
        hit_mask = (y_rank < topX_eff)
        coverages.append(float(hit_mask.float().mean().item()))

        # (선택) 순위 분포 저장
        # y_rank_all.append(y_rank.detach().cpu().numpy())

        # A) fixed-k 스윕
        for k in k_list:
            logits = agg_fixed_k(evi, P, k=k, beta=args.beta, mode="softk")
            pred = int(torch.argmax(logits).item())
            c, t = correct_total[("fixedk", k)]
            correct_total[("fixedk", k)] = [c + int(pred == y), t + 1]

        # B) pct-of-max 스윕
        for a in pct_list:
            logits = agg_pct_of_max(evi, P, alpha=a)
            pred = int(torch.argmax(logits).item())
            c, t = correct_total[("pct", a)]
            correct_total[("pct", a)] = [c + int(pred == y), t + 1]

    # 7) 저장물 작성
    mean_cov = float(np.mean(coverages))
    med_cov = float(np.median(coverages))
    cov_rows.append((args.topX, mean_cov, med_cov))
    with open(os.path.join(args.out_dir, "coverage.csv"), "w") as f:
        f.write("\n".join([",".join(map(str, r)) for r in cov_rows]))

    for k in k_list:
        c, t = correct_total[("fixedk", k)]
        acc_rows.append(("fixedk", k, 100.0 * c / max(1, t), c, t))
    for a in pct_list:
        c, t = correct_total[("pct", a)]
        acc_rows.append(("pct", a, 100.0 * c / max(1, t), c, t))
    with open(os.path.join(args.out_dir, "acc_curves.csv"), "w") as f:
        f.write("\n".join([",".join(map(str, r)) for r in acc_rows]))

    centers = 0.5 * (hist_bins[1:] + hist_bins[:-1])
    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist_counts, width=centers[1] - centers[0])
    plt.xlabel("slot_prob")
    plt.ylabel("count")
    plt.title("P histogram (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "p_hist.png"))

    # (선택) 정답 순위 히스토그램 저장
    # if len(y_rank_all) > 0:
    #     y_rank_all = np.concatenate(y_rank_all, axis=0)
    #     Cmax_est = int(np.max(y_rank_all)) + 1
    #     bins = np.arange(0, Cmax_est + 2)  # 0..C (C는 '못찾음')
    #     hist, _ = np.histogram(y_rank_all, bins=bins)
    #     with open(os.path.join(args.out_dir, "y_rank_hist.csv"), "w") as f:
    #         f.write("rank,count\n")
    #         for r, c in enumerate(hist):
    #             f.write(f"{r},{int(c)}\n")

    # 8) 추천/요약
    best_fixedk = max([(r[2], r[1]) for r in acc_rows if r[0] == "fixedk"], default=(0, 1))
    best_pct = max([(r[2], r[1]) for r in acc_rows if r[0] == "pct"], default=(0, 0.5))
    rec = {
        "best_fixedk": {"k": best_fixedk[1], "acc": best_fixedk[0]},
        "best_pct": {"alpha": best_pct[1], "acc": best_pct[0]},
        "coverage_topX": {"X": args.topX, "mean": mean_cov, "median": med_cov},
        "notes": "slot_priors.npz(idf, sel)는 선택적 보정용",
    }
    with open(os.path.join(args.out_dir, "recommended.json"), "w") as f:
        json.dump(rec, f, indent=2)

    print("[done] results saved under", args.out_dir)


if __name__ == "__main__":
    main()
