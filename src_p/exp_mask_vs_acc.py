# src_p/exp_mask_vs_acc.py
# -*- coding: utf-8 -*-
import os, json, math, argparse, warnings, re
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 불필요한 트랜스포머 경고 숨김
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True"
)

# 내부 모듈
try:
    from .p_train import SlotClassifier, round_comb4_mask_token
except Exception:
    from p_train import SlotClassifier, round_comb4_mask_token


def get_loader(batch_size=512, num_workers=8, split="test"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.EMNIST("./data", split="balanced", train=(split == "train"), download=True, transform=tf)

    # 단발 실행 스크립트 → persistent_workers=False 권장
    dl_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4  # workers 있을 때만 지정

    loader = DataLoader(**dl_kwargs)
    n_classes = int(ds.targets.max().item()) + 1
    return loader, n_classes


import re

def _parse_alt_name(name: str):
    """
    switch_alt 문자열을 파싱해서 (kind, param, k)로 돌려줌.
    허용 예:
      - "none", "base", "psum" (스위칭 안 함)
      - "lse_t3", "lse_t2.5"
      - "psharp_tau0.2", "psharp0.2"
      - "topk4"
      - "drop1"
      - "consensus2.0"
      - "lse_topk4_t3"  # LSE(t=3)와 TopK=4 평균 혼합
    """
    if not name:
        return ("none", 0.0, 0)
    s = name.strip().lower()
    if s in ("none", "base", "psum"):
        return ("none", 0.0, 0)

    # lse_topkK_tT
    m = re.match(r"^lse_topk(\d+)_t(\d+(?:\.\d+)?)$", s)
    if m:
        k = int(m.group(1)); t = float(m.group(2))
        return ("lse_topk", t, k)

    # lse_tT
    m = re.match(r"^lse_t(\d+(?:\.\d+)?)$", s)
    if m:
        return ("lse", float(m.group(1)), 0)

    # psharp[_tau]T
    m = re.match(r"^psharp(?:_tau)?(\d+(?:\.\d+)?)$", s)
    if m:
        return ("psharp", float(m.group(1)), 0)

    # topkK
    m = re.match(r"^topk(\d+)$", s)
    if m:
        return ("topk", 0.0, int(m.group(1)))

    # drop1
    if s == "drop1":
        return ("drop1", 0.0, 0)

    # consensusW
    m = re.match(r"^consensus(\d+(?:\.\d+)?)$", s)
    if m:
        return ("consensus", float(m.group(1)), 0)

    # 매칭 실패 시 친절한 에러
    raise ValueError(
        f"Unrecognized --switch_alt '{name}'. "
        "Examples: lse_t3, psharp_tau0.2, topk4, drop1, consensus2.0, lse_topk4_t3, none"
    )



def _apply_alt_logits(kind: str, param: float, k: int,
                      slot_logits: torch.Tensor,  # (B,M,C)
                      P: torch.Tensor,            # (B,M)
                      z_base: torch.Tensor        # (B,M) : mass - E[mass] (center 전)
                      ) -> torch.Tensor:
    """
    저마진(low-margin) 구간에서 사용할 대체 집계 로짓 생성.
    반환: logits_alt (B,C)
    """
    B, M, C = slot_logits.shape

    if kind == "lse":
        t = max(1e-6, float(param))
        # temp-scaled logsumexp over slots
        return torch.logsumexp(slot_logits / t, dim=1) * t

    if kind == "psharp":
        # 기존 z_base를 더 작은 tau로 샤프닝
        t = max(1e-6, float(param))
        z = (z_base - z_base.mean(dim=1, keepdim=True)) / t
        P_sharp = torch.softmax(z, dim=1)
        return (slot_logits * P_sharp.unsqueeze(-1)).sum(dim=1)

    if kind == "topk":
        kk = max(1, min(int(k), M))
        idx = torch.topk(P, kk, dim=1).indices                      # (B,kk)
        gathered = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        return gathered.mean(dim=1)

    if kind == "drop1":
        # P top-1 슬롯 제외하고 P 재정규화 후 합산
        top1 = P.argmax(dim=1, keepdim=True)                        # (B,1)
        mask = torch.ones_like(P)
        mask.scatter_(1, top1, 0.0)
        P2 = P * mask
        denom = P2.sum(dim=1, keepdim=True).clamp_min(1e-8)
        P2 = P2 / denom
        return (slot_logits * P2.unsqueeze(-1)).sum(dim=1)

    if kind == "consensus":
        # P top2 슬롯 argmax 클래스가 합치면 평균, 아니면 Psum
        top2_idx = torch.topk(P, k=2, dim=1).indices                # (B,2)
        s1 = slot_logits[torch.arange(B, device=slot_logits.device), top2_idx[:, 0]]  # (B,C)
        s2 = slot_logits[torch.arange(B, device=slot_logits.device), top2_idx[:, 1]]  # (B,C)
        c1 = s1.argmax(dim=1)
        c2 = s2.argmax(dim=1)
        agree = (c1 == c2).float().unsqueeze(-1)
        w = float(param) if param > 0 else 2.0
        mean2 = (s1 + s2) / 2.0
        psum = (slot_logits * P.unsqueeze(-1)).sum(dim=1)
        return agree * mean2 + (1.0 - agree) * psum

    if kind == "lse_topk":
        # 먼저 LSE(t), 그 다음 top-k(P 기준) 평균과 평균(50:50) – 간단 합성
        t = max(1e-6, float(param))
        logits_lse = torch.logsumexp(slot_logits / t, dim=1) * t
        kk = max(1, min(int(k), M))
        idx = torch.topk(P, kk, dim=1).indices
        gathered = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        logits_topk = gathered.mean(dim=1)
        return 0.5 * logits_lse + 0.5 * logits_topk

    # 기본: 아무 것도 하지 않음(안전)
    return (slot_logits * P.unsqueeze(-1)).sum(dim=1)


@torch.no_grad()
def evaluate_mass_and_aggregators_ckpt(
    ckpt_path, out_json, batch_size=512, num_workers=8,
    tau=0.7, topk_list=(2, 4, 8), mask_repeat=1, device_str=None,
    switch_margin: float = None, switch_alt: str = None
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, num_classes = get_loader(batch_size, num_workers)

    # --- ckpt & model ---
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    num_slots    = int(meta.get("num_slots", 126))
    d_model      = int(meta.get("d_model", 64))
    nhead        = int(meta.get("nhead", 4))
    filters_path = meta.get("filters_path", None)
    if filters_path is None:
        raise RuntimeError("filters_path(meta)가 없습니다.")

    model = SlotClassifier(
        filters_path=filters_path, num_slots=num_slots,
        d_model=d_model, nhead=nhead, num_classes=num_classes
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- comb4 mask (M,H,W) ---
    masks_126 = round_comb4_mask_token(126, 14, 14, device=device)
    masks_mhw = masks_126[:num_slots] if mask_repeat == 1 else masks_126.repeat(mask_repeat, 1, 1)[:num_slots]
    if masks_mhw.shape[0] < num_slots:
        times = math.ceil(num_slots / masks_mhw.shape[0])
        masks_mhw = masks_mhw.repeat(times, 1, 1)[:num_slots]
    M, H_px, W_px = masks_mhw.shape
    N_px = float(H_px * W_px)
    area_in = masks_mhw.float().flatten(1).sum(-1)  # (M,)

    # --- 누적기 ---
    total = 0
    correct = {"Psum": 0, "Final": 0, "Top1": 0, **{f"Top{k}": 0 for k in topk_list}, "LSE": 0}

    nbins = 10
    bin_total = torch.zeros(nbins, dtype=torch.long)
    bin_hits  = torch.zeros(nbins, dtype=torch.long)

    # (b,m) 전역: mass vs acc & 지배 슬롯 통계
    mass_bins_edges = torch.linspace(0.0, 1.0, steps=11)
    bin_total_slots = torch.zeros(10, dtype=torch.long)
    bin_hits_slots  = torch.zeros(10, dtype=torch.long)
    top_slot_hist   = torch.zeros(M, dtype=torch.long)
    top_slot_hits   = torch.zeros(M, dtype=torch.long)
    slot_mass_sum   = torch.zeros(M, dtype=torch.double)

    # 스위칭 통계
    do_switch = (switch_margin is not None) and (switch_alt is not None) and (switch_alt.lower() != "none")
    switch_kind, switch_param, switch_k = _parse_alt_name(switch_alt or "none")
    switch_hi_total = 0
    switch_hi_hits  = 0
    switch_lo_total = 0
    switch_lo_hits_base = 0
    switch_lo_hits_alt  = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.size(0)

        # 1) 원시 맵/토큰 (spmask 끔 → A_raw per-slot 합≈1)
        _, aux = model(x, use_spmask=False, spmask_grid=3, spmask_assign="auto", tau_p=tau)
        A_raw = aux["A_maps_raw"]                   # (B,M,H,W)
        feat  = aux["feat_hw"]                      # (B,H,W,D)
        tokens = feat.view(B, H_px * W_px, d_model) # (B,N,D)

        # 2) comb4 마스크 적용 + (pooling용) 정규화
        mask = masks_mhw.unsqueeze(0).to(A_raw.dtype)    # (1,M,H,W)
        A_masked = A_raw * mask                          # (B,M,H,W)
        mass = A_masked.float().flatten(2).sum(-1).clamp_min(1e-8)  # (B,M)  pre-renorm mass
        A_eff = (A_masked.float() / mass.view(B, M, 1, 1)).to(A_raw.dtype)

        # 3) 슬롯 임베딩/로짓
        A_flat  = A_eff.view(B, M, H_px * W_px)          # (B,M,N)
        S       = torch.bmm(A_flat, tokens)              # (B,M,D)
        S       = F.normalize(S, dim=-1)
        slot_logits = model.classifier(S)                # (B,M,C)

        # 4) P = softmax( (mass - E[mass]) centered / tau )  — 기대값(area/N) 보정
        exp_mass = (area_in.to(mass.device) / N_px).view(1, M)  # (1,M)
        z_base   = mass.float() - exp_mass                      # (B,M)
        z        = (z_base - z_base.mean(dim=1, keepdim=True)) / float(tau)
        P        = torch.softmax(z, dim=1)                      # (B,M)

        # 5) 기본 집계 로짓들
        logits_psum  = (slot_logits * P.unsqueeze(-1)).sum(dim=1)           # (B,C)
        top1_idx     = P.argmax(dim=1)
        logits_top1  = slot_logits[torch.arange(B, device=device), top1_idx]
        logits_topk  = {}
        for k in topk_list:
            kk = min(int(k), M)
            idx = torch.topk(P, kk, dim=1).indices
            gather = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, slot_logits.size(-1)))
            logits_topk[f"Top{kk}"] = gather.mean(dim=1)
        logits_lse = torch.logsumexp(slot_logits, dim=1)

        # 6) 스위칭(최종 로짓)
        logits_final = logits_psum
        if do_switch:
            # P 마진
            P_sorted, _ = torch.sort(P, dim=1, descending=True)
            margin = (P_sorted[:, 0] - P_sorted[:, 1])  # (B,)

            hi_mask = (margin >= float(switch_margin))
            lo_mask = ~hi_mask

            # hi: 그대로 psum
            logits_hi = logits_psum

            # lo: 대체 집계
            if lo_mask.any():
                logits_alt = _apply_alt_logits(switch_kind, switch_param, switch_k, slot_logits, P, z_base)
                # blend by mask
                logits_final = torch.where(lo_mask.view(B, 1), logits_alt, logits_hi)
            else:
                logits_final = logits_hi

            # 통계(정확도)
            switch_hi_total += int(hi_mask.sum().item())
            switch_lo_total += int(lo_mask.sum().item())
            # hi 구간 Psum 정답 수
            if switch_hi_total > 0:
                switch_hi_hits += int((logits_psum.argmax(1)[hi_mask] == y[hi_mask]).sum().item())
            # lo 구간 base / alt 정답 수
            if lo_mask.any():
                switch_lo_hits_base += int((logits_psum.argmax(1)[lo_mask] == y[lo_mask]).sum().item())
                switch_lo_hits_alt  += int((logits_final.argmax(1)[lo_mask] == y[lo_mask]).sum().item())

        # --- 예측/정답 누적 ---
        pred_map = {
            "Psum": logits_psum.argmax(1),
            "Top1": logits_top1.argmax(1),
            **{k: v.argmax(1) for k, v in logits_topk.items()},
            "LSE":  logits_lse.argmax(1),
        }
        for k in ["Psum", "Top1", *logits_topk.keys(), "LSE"]:
            correct[k] += (pred_map[k] == y).sum().item()
        correct["Final"] += (logits_final.argmax(1) == y).sum().item()
        total += B

        # 7) confidence decile (P.max 균등 보정)
        conf = (P.max(dim=1).values - 1.0 / M) / (1 - 1.0 / M)
        bins = torch.clamp((conf * nbins).long(), 0, nbins - 1)
        for b in range(nbins):
            sel = (bins == b)
            n = sel.sum().item()
            if n > 0:
                bin_total[b] += n
                bin_hits[b]  += (pred_map["Psum"][sel] == y[sel]).sum().item()

        # 8) (b,m) 전역 mass vs acc
        correct_slot = (slot_logits.argmax(dim=-1) == y.unsqueeze(1))   # (B,M)
        mass_clamped = mass.clamp(0, 1)
        bins_idx = torch.bucketize(mass_clamped.reshape(-1).cpu(), mass_bins_edges[1:-1].cpu())  # 0..9
        bin_total_slots += torch.bincount(bins_idx, minlength=10)
        bin_hits_slots  += torch.bincount(
            bins_idx, weights=correct_slot.reshape(-1).float().cpu(), minlength=10
        ).long()

        # 9) 지배 슬롯(=mass top1) 통계
        top_idx = mass.argmax(dim=1).cpu()
        top_slot_hist += torch.bincount(top_idx, minlength=M)
        top_logits   = slot_logits[torch.arange(B, device=slot_logits.device), mass.argmax(dim=1)]
        top_correct  = (top_logits.argmax(dim=-1) == y).cpu()
        if top_correct.any():
            top_slot_hits += torch.bincount(top_idx[top_correct], minlength=M)

        # 슬롯별 평균 질량 누적
        slot_mass_sum += mass.sum(dim=0).cpu().double()

    # --- 요약/저장 ---
    accs = {k: correct[k] / max(1, total) for k in correct}

    decile = []
    for i in range(nbins):
        n = int(bin_total[i].item())
        acc_b = (bin_hits[i].item() / n) if n > 0 else 0.0
        decile.append({"decile": int(i), "n": n, "acc": acc_b})

    mass_acc_curve = []
    for b in range(10):
        n = int(bin_total_slots[b].item())
        acc_b = (bin_hits_slots[b].item() / n) if n > 0 else 0.0
        mass_acc_curve.append({
            "bin": b,
            "lo": float(mass_bins_edges[b].item()),
            "hi": float(mass_bins_edges[b+1].item()),
            "n": n,
            "acc": acc_b
        })

    slot_mean_mass = (slot_mass_sum / max(1, total)).tolist()
    top_slot_total = int(top_slot_hist.sum().item())
    top_slot_freq  = (top_slot_hist.double() / max(1, top_slot_total)).tolist()

    # 엔트로피/지배비율
    p = torch.tensor(top_slot_freq, dtype=torch.double)
    eps = 1e-12
    H = float((-(p.clamp_min(eps) * (p.clamp_min(eps)).log()).sum()).item()) if top_slot_total > 0 else 0.0
    H_norm = (H / math.log(M)) if M > 1 and H > 0 else (1.0 if top_slot_total == 0 else 0.0)
    dom_ratio = float((top_slot_hist.max().double() / top_slot_hist.double().mean().clamp_min(1.0)).item()) if top_slot_total > 0 else 0.0

    out = {
        "acc": accs,                               # Final 포함
        "decile": decile,
        "mass_acc_all_slots": mass_acc_curve,
        "slot_dominance": {
            "entropy_norm": H_norm,
            "dominance_ratio": dom_ratio,
            "top_slot_freq": top_slot_freq,
            "slot_mean_mass": slot_mean_mass
        },
        "meta": {
            "num_slots": num_slots, "d_model": d_model, "nhead": nhead,
            "tau": tau, "topk": list(topk_list), "mask_repeat": mask_repeat,
            "M": M, "H_px": H_px, "W_px": W_px,
            "switch_margin": switch_margin, "switch_alt": switch_alt
        },
        "switch": (None if not do_switch else {
            "hi_total": switch_hi_total,
            "hi_hits":  switch_hi_hits,
            "lo_total": switch_lo_total,
            "lo_hits_base": switch_lo_hits_base,
            "lo_hits_alt":  switch_lo_hits_alt,
        })
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # --- 콘솔 요약 ---
    print("[exp_mask_vs_acc] saved ->", out_json)
    print("[acc] " + " | ".join([f"{k}={v*100:.2f}%" for k, v in accs.items()]))

    if do_switch:
        hi_cov = (switch_hi_total / max(1, switch_hi_total + switch_lo_total)) * 100.0
        hi_acc = (switch_hi_hits  / max(1, switch_hi_total)) * 100.0 if switch_hi_total > 0 else 0.0
        lo_acc_base = (switch_lo_hits_base / max(1, switch_lo_total)) * 100.0 if switch_lo_total > 0 else 0.0
        lo_acc_alt  = (switch_lo_hits_alt  / max(1, switch_lo_total)) * 100.0 if switch_lo_total > 0 else 0.0
        overall_final = accs["Final"] * 100.0
        overall_psum  = accs["Psum"]  * 100.0
        print(f"[switch] thr={switch_margin}, alt={switch_alt} | "
              f"hi={hi_cov:.2f}% acc={hi_acc:.2f}% | "
              f"lo acc base={lo_acc_base:.2f}% → alt={lo_acc_alt:.2f}% | "
              f"overall Final={overall_final:.2f}% (Δ={overall_final - overall_psum:+.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--topk", type=str, default="2,4,8")
    ap.add_argument("--mask_repeat", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)

    # 스위칭 옵션
    ap.add_argument("--switch_margin", type=float, default=None, help="P margin threshold (e.g., 0.8). If None, no switching.")
    ap.add_argument("--switch_alt", type=str, default=None,
                    help="Alternative aggregator for low-margin: "
                         "[lse_t{tau} | psharp_tau{tau} | topk{k} | drop1 | consensus{w} | lse_topk{k}_t{tau}]")

    args = ap.parse_args()

    topk_list = tuple(int(x) for x in args.topk.split(",") if x.strip())
    evaluate_mass_and_aggregators_ckpt(
        ckpt_path=args.ckpt, out_json=args.out_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        tau=args.tau, topk_list=topk_list, mask_repeat=args.mask_repeat,
        device_str=args.device,
        switch_margin=args.switch_margin, switch_alt=args.switch_alt
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
