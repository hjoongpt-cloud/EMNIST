# -*- coding: utf-8 -*-
# src_q/q_group_diagnostics.py
#
# 목적:
# - 그룹 내 다양성:
#   * 같은 위치 3슬롯의 A_map 코사인 유사도(마스크 내부, 정규화 후)
#   * 같은 위치 3슬롯의 S 임베딩 코사인 유사도
#   * 슬롯별 argmax 클래스의 지배 클래스(슬롯 단독 argmax) 분포 → 그룹 내 서로 다른 지배 클래스 개수(1/2/3)
# - 그룹 합의율/정확도:
#   * (b, g) 단위로 3슬롯 단독 argmax의 2/3 합의 여부, 합의 케이스 정확도 vs 불합의 케이스 정확도
#     - 불합의 정확도는 “그룹 내 P가 가장 큰 슬롯의 예측” 기준과 “3슬롯 중 하나라도 정답이면” 두 기준을 함께 보고
# - 아블레이션(1슬롯 vs 3슬롯 집계 비교):
#   * baseline: 전체 M슬롯 Psum
#   * group_top1: 각 그룹에서 P가 가장 큰 슬롯 하나만 뽑아 그룹 단위로 묶어 집계 (그룹별 가중치는 그 슬롯의 P, 그룹 간 정규화)

import os, json, math, argparse, warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True"
)

try:
    from .q_trunk import SlotClassifier
    from .q_spmask import build_comb4_base_masks, replicate_masks
except Exception:
    from q_trunk import SlotClassifier
    from q_spmask import build_comb4_base_masks, replicate_masks


def get_loader(batch_size=512, num_workers=8, split="test"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.EMNIST("./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    dl_kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True, persistent_workers=False)
    if num_workers>0: dl_kwargs["prefetch_factor"]=4
    loader = DataLoader(**dl_kwargs)
    n_classes = int(ds.targets.max().item()) + 1
    return loader, n_classes


@torch.no_grad()
def run_diagnostics(
    ckpt_path, out_json, batch_size=512, num_workers=8, tau=0.7, device_str=None
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, num_classes = get_loader(batch_size, num_workers)

    # --- ckpt & model ---
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    num_slots_base   = int(meta.get("num_slots_base", meta.get("num_slots", 126)))
    repeats_per_pos  = int(meta.get("repeats_per_pos", 3))
    d_model          = int(meta.get("d_model", 64))
    nhead            = int(meta.get("nhead", 4))
    filters_path     = meta.get("filters_path", None)
    if filters_path is None:
        raise RuntimeError("filters_path(meta)가 없습니다.")

    model = SlotClassifier(
        filters_path=filters_path,
        num_slots_base=num_slots_base,
        repeats_per_pos=repeats_per_pos,
        d_model=d_model, nhead=nhead, num_classes=num_classes
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- comb4 masks (그룹 매핑: m = g + r*G 형태, G=num_slots_base, r in [0..repeats-1]) ---
    base = build_comb4_base_masks(14,14,device=device)                       # (G,14,14)
    masks_mhw = replicate_masks(base, repeats_per_pos)[:num_slots_base*repeats_per_pos]
    M, H_px, W_px = masks_mhw.shape
    G = num_slots_base
    R = repeats_per_pos
    assert M == G*R, f"M={M}, G*R={G*R} mismatch"

    N_px = float(H_px * W_px)
    area_in = masks_mhw.float().flatten(1).sum(-1)  # (M,)

    # --- 누적기 ---
    # 다양성: 코사인 유사도 평균
    a_cos_sum = 0.0; s_cos_sum = 0.0; pair_count = 0

    # 슬롯별 지배 클래스(슬롯 단독 argmax) 카운트 → 최종 그룹의 distinct 개수(1/2/3)
    slot_dom_counts = {}  # slot m -> {class -> count}
    for m in range(M):
        slot_dom_counts[m] = {}

    # 그룹 합의율/정확도
    total_groups = 0
    consensus_groups = 0
    consensus_hits = 0
    nocon_groups = 0
    nocon_hits_bestP = 0
    nocon_hits_any = 0

    # 아블레이션(이미지 단위 집계 정확도)
    total_imgs = 0
    hits_baseline_psum = 0
    hits_group_top1 = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.size(0)
        total_imgs += B

        # 1) 원시 맵/토큰 (spmask 끔 → per-slot softmax over pixels)
        _, aux = model(x, use_spmask=False, spmask_grid=3, spmask_assign="auto", tau_p=tau)
        A_raw = aux["A_maps_raw"]                   # (B,M,H,W)
        feat  = aux["feat_hw"]                      # (B,H,W,D)
        tokens = feat.view(B, H_px * W_px, d_model) # (B,N,D)

        # 2) 마스크 적용 후 풀링 정규화
        mask = masks_mhw.unsqueeze(0).to(A_raw.dtype)      # (1,M,H,W)
        A_masked = A_raw * mask
        mass = A_masked.float().flatten(2).sum(-1).clamp_min(1e-8)    # (B,M)
        A_eff = (A_masked.float() / mass.view(B, M, 1, 1)).to(A_raw.dtype)

        # 3) 슬롯 임베딩/로짓
        A_flat = A_eff.view(B, M, H_px * W_px)
        S = torch.bmm(A_flat, tokens)                       # (B,M,D)
        S = F.normalize(S, dim=-1)
        slot_logits = model.classifier(S)                   # (B,M,C)

        # 4) P (mass 기대값 보정 → residual 중심화 후 softmax)
        exp_mass = (area_in.to(mass.device) / N_px).view(1, M)
        z_base = mass.float() - exp_mass
        z = (z_base - z_base.mean(dim=1, keepdim=True)) / float(tau)
        P = torch.softmax(z, dim=1)                         # (B,M)

        # 5) baseline(전체 M슬롯) / group_top1(그룹별 best-P 하나만) 집계 정확도
        logits_psum = (slot_logits * P.unsqueeze(-1)).sum(dim=1)              # (B,C)
        pred_psum = logits_psum.argmax(1)
        hits_baseline_psum += int((pred_psum == y).sum().item())

        # group_top1: 각 그룹 g에서 r∈{0..R-1} 중 P가 가장 큰 슬롯 선택 → 그 슬롯의 P를 그룹 weight로 사용
        # 그룹 weight를 그룹 간 정규화한 뒤, 선택된 슬롯의 로짓을 가중합
        P_group = []
        L_group = []
        for g in range(G):
            idxs = [g + r*G for r in range(R)]                         # 길이 R
            P_g, arg = P[:, idxs].max(dim=1)                           # (B,)
            chosen = torch.tensor(idxs, device=device)[arg]            # (B,)
            # gather chosen logits
            # slot_logits shape (B,M,C)에서 batch별로 서로 다른 m를 고르므로 advanced indexing 필요
            L_g = slot_logits[torch.arange(B, device=device), chosen]  # (B,C)
            P_group.append(P_g)
            L_group.append(L_g)
        P_group = torch.stack(P_group, dim=1)                           # (B,G)
        L_group = torch.stack(L_group, dim=1)                           # (B,G,C)
        # 그룹 간 정규화
        P_norm = P_group / P_group.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logits_group_top1 = (L_group * P_norm.unsqueeze(-1)).sum(dim=1) # (B,C)
        pred_group_top1 = logits_group_top1.argmax(1)
        hits_group_top1 += int((pred_group_top1 == y).sum().item())

        # 6) 그룹 내 다양성 & 합의율/정확도 (b,g) 단위
        total_groups += B * G
        # (a) A_map 코사인: 마스크 내부 벡터를 정규화한 뒤 R=3 슬롯 쌍(3개) 평균
        # (b) S 임베딩 코사인: 이미 정규화되어 있으므로 내적이 코사인
        # (c) 슬롯 단독 argmax의 지배 클래스 카운트 누적
        # (d) 합의율/정확도: 3슬롯 단독 argmax가 2/3 이상 일치하는지

        # 각 그룹 g에 대해 반복
        # 벡터화를 일부 섞되, 이해 쉬운 루프로 작성
        A_eff_bm = A_eff  # (B,M,H,W)
        for g in range(G):
            idxs = [g + r*G for r in range(R)]  # 세 슬롯 인덱스

            # (a) A_map cosine (마스크 내부 픽셀만)
            # 마스크 내부를 펼쳐서 정규화한 뒤, 코사인 유사도 계산
            # shape: (B, R, K)  (K = area_in[g] 픽셀 수)
            mask_g = masks_mhw[idxs[0]] > 0
            K = int(mask_g.sum().item())
            if K > 0:
                V = []
                for m in idxs:
                    v = A_eff_bm[:, m][:, mask_g].float()              # (B,K)
                    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    V.append(v)
                V = torch.stack(V, dim=1)                               # (B,R,K)
                # 3쌍 코사인 평균
                cos12 = (V[:,0]*V[:,1]).sum(dim=1).mean().item()
                cos13 = (V[:,0]*V[:,2]).sum(dim=1).mean().item()
                cos23 = (V[:,1]*V[:,2]).sum(dim=1).mean().item()
                a_cos_sum += (cos12 + cos13 + cos23) / 3.0
                pair_count += 1

            # (b) S 임베딩 코사인
            Sg = S[:, idxs, :]                                        # (B,R,D) (정규화됨)
            # 평균 코사인
            s12 = (Sg[:,0]*Sg[:,1]).sum(dim=1).mean().item()
            s13 = (Sg[:,0]*Sg[:,2]).sum(dim=1).mean().item()
            s23 = (Sg[:,1]*Sg[:,2]).sum(dim=1).mean().item()
            s_cos_sum += (s12 + s13 + s23) / 3.0

            # (c) 슬롯 단독 argmax의 지배 클래스 카운트
            #    slot_logits.argmax(-1): (B,M) → 각 슬롯별로 전체 배치에서 카운트 누적
            slot_pred = slot_logits.argmax(dim=-1)                     # (B,M)
            for m in idxs:
                cls = slot_pred[:, m].tolist()
                d = slot_dom_counts[m]
                for c in cls:
                    d[c] = d.get(c, 0) + 1

            # (d) 합의율/정확도 (b,g) 단위)
            # 3슬롯 단독 argmax → 2/3 이상 같은지
            preds_3 = slot_pred[:, idxs]                               # (B,3)
            # 합의 label과 합의 클래스
            # 다수결이지만 1-1-1이면 불합의
            # 간단히: 세 원소 중 가장 많이 등장한 값의 카운트 확인
            for b in range(B):
                vals = preds_3[b].tolist()
                # 빈도
                a = vals[0]; b1 = vals[1]; c = vals[2]
                # 최빈값 카운트 계산 (작게 하드코딩)
                if a==b1 or a==c:
                    mode = a
                    cnt = 2 if (a==b1) ^ (a==c) else 3
                elif b1==c:
                    mode = b1; cnt = 2
                else:
                    mode = None; cnt = 1

                if cnt >= 2:
                    consensus_groups += 1
                    if mode == int(y[b].item()):
                        consensus_hits += 1
                else:
                    nocon_groups += 1
                    # 불합의 정확도 2가지: (i) 그룹 내 P가 가장 큰 슬롯의 예측
                    #                   (ii) 3슬롯 중 하나라도 정답이면 1
                    # (i)
                    pvals = P[b, idxs]
                    m_best = idxs[int(pvals.argmax().item())]
                    pred_best = int(slot_pred[b, m_best].item())
                    if pred_best == int(y[b].item()):
                        nocon_hits_bestP += 1
                    # (ii)
                    if any(int(v)==int(y[b].item()) for v in vals):
                        nocon_hits_any += 1

    # --- 요약 ---
    # 다양성
    a_cos_mean = (a_cos_sum / max(1, pair_count)) if pair_count>0 else 0.0
    s_cos_mean = (s_cos_sum / max(1, total_groups/G)) if total_groups>0 else 0.0  # rough norm

    # 슬롯 지배 클래스 → 그룹별 distinct 개수
    # 슬롯 m의 지배 클래스 = counts 중 최댓값의 클래스
    slot_dom_class = []
    for m in range(M):
        d = slot_dom_counts[m]
        if len(d)==0:
            slot_dom_class.append(None)
        else:
            slot_dom_class.append(max(d.items(), key=lambda kv: kv[1])[0])
    # 그룹별 distinct 개수 카운트
    distinct_hist = {1:0, 2:0, 3:0}
    for g in range(G):
        group_classes = [slot_dom_class[g + r*G] for r in range(R)]
        uniq = len(set(group_classes))
        if uniq in distinct_hist:
            distinct_hist[uniq] += 1
    for k in [1,2,3]:
        distinct_hist[k] = int(distinct_hist[k])

    # 합의율/정확도
    consensus_rate = (consensus_groups / max(1, total_groups))
    acc_when_consensus = (consensus_hits / max(1, consensus_groups)) if consensus_groups>0 else 0.0
    acc_when_nocon_bestP = (nocon_hits_bestP / max(1, nocon_groups)) if nocon_groups>0 else 0.0
    acc_when_nocon_any = (nocon_hits_any / max(1, nocon_groups)) if nocon_groups>0 else 0.0

    # 아블레이션 (이미지 단위)
    acc_baseline_psum = hits_baseline_psum / max(1, total_imgs)
    acc_group_top1    = hits_group_top1    / max(1, total_imgs)

    out = {
        "meta": {
            "num_slots_base": num_slots_base,
            "repeats_per_pos": repeats_per_pos,
            "num_slots": int(num_slots_base*repeats_per_pos),
            "d_model": d_model, "nhead": nhead,
            "tau": tau, "H_px": H_px, "W_px": W_px,
        },
        "diversity": {
            "A_map_cosine_mean": a_cos_mean,       # 마스크 내부 정규화 후 코사인 평균
            "S_embed_cosine_mean": s_cos_mean,     # S 임베딩 코사인 평균
            "group_slot_dominant_class_distinct_hist": distinct_hist,  # {1,2,3}
        },
        "group_agreement": {
            "total_pairs": int(total_groups),
            "consensus_pairs": int(consensus_groups),
            "consensus_rate": consensus_rate,
            "acc_when_consensus": acc_when_consensus,
            "acc_when_no_consensus_bestP": acc_when_nocon_bestP,
            "acc_when_no_consensus_any_slot": acc_when_nocon_any
        },
        "ablation": {
            "acc_baseline_psum": acc_baseline_psum,
            "acc_group_top1": acc_group_top1
        }
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # 콘솔 요약
    print("[q_group_diagnostics] saved ->", out_json)
    print(f"[diversity] A_map_cos={a_cos_mean:.3f} | S_cos={s_cos_mean:.3f} | distinct (1/2/3)={distinct_hist}")
    print(f"[agreement] rate={consensus_rate*100:.2f}% | acc(cons)={acc_when_consensus*100:.2f}% | "
          f"acc(nocon bestP)={acc_when_nocon_bestP*100:.2f}% | acc(nocon any)={acc_when_nocon_any*100:.2f}%")
    print(f"[ablation] baseline Psum={acc_baseline_psum*100:.2f}% | group_top1={acc_group_top1*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    run_diagnostics(
        ckpt_path=args.ckpt,
        out_json=args.out_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tau=args.tau,
        device_str=args.device
    )


if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
