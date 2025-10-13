# -*- coding: utf-8 -*-
# src_o/o_spatial_metrics.py
"""
공간/다양성 리포트 (s-only)
- 입력: o_dump.py 의 *.npz (A_upsampled, slot_prob, image, clazz)
- 출력: metrics.csv (이미지별 IoU/엔트로피 등), summary.txt
"""

import os, argparse, glob, math
import numpy as np
from tqdm import tqdm

def topq_mask(a_hw, q=0.10):
    flat = a_hw.reshape(-1)
    k = max(1, int(round(len(flat) * q)))
    thr = np.partition(flat, -k)[-k]
    return (a_hw >= thr)

def binary_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return (inter / max(1, union))

def pairwise_iou_mean(masks):  # masks: list of HxW bool arrays
    K = len(masks)
    if K <= 1: return 0.0
    s = 0.0; n = 0
    for i in range(K):
        for j in range(i+1, K):
            s += binary_iou(masks[i], masks[j]); n += 1
    return s / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--slots_topk", type=int, default=5)
    ap.add_argument("--q_iou", type=float, default=0.10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    if not files: raise FileNotFoundError(args.dump_dir)

    lines = ["id,y,topk,mean_iou,slot_entropy"]
    mean_ious, entropies = [], []

    for fp in tqdm(files, desc="[spatial]"):
        it = np.load(fp, allow_pickle=True)
        y = int(it["clazz"])
        A = it["A_upsampled"]                     # (M,28,28) float32
        P = it["slot_prob"].astype(np.float32)    # (M,)

        M = P.shape[0]
        k = min(args.slots_topk, M)
        top_idx = np.argpartition(-P, kth=k-1)[:k]
        top_idx = top_idx[np.argsort(-P[top_idx])]

        masks = [topq_mask(A[m], q=args.q_iou) for m in top_idx]
        miou = pairwise_iou_mean(masks)

        # slot prob entropy (정규화)
        p_sel = P[top_idx].clip(1e-8, None); p_sel = p_sel / p_sel.sum()
        H = -(p_sel * np.log(p_sel)).sum() / math.log(len(p_sel))

        mean_ious.append(miou); entropies.append(H)
        lines.append(f"{int(it['id'])},{y},{k},{miou:.4f},{H:.4f}")

    with open(os.path.join(args.out_dir, "metrics.csv"), "w") as f:
        f.write("\n".join(lines))

    # summary
    m_iou = float(np.mean(mean_ious)) if mean_ious else float("nan")
    m_ent = float(np.mean(entropies)) if entropies else float("nan")
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"num_items={len(files)}\n")
        f.write(f"slots_topk={args.slots_topk}\n")
        f.write(f"q_iou={args.q_iou}\n")
        f.write(f"mean(pairwise IoU)={m_iou:.4f}\n")
        f.write(f"mean(slot entropy)={m_ent:.4f}\n")

    print("[spatial] saved:", os.path.join(args.out_dir, "metrics.csv"))
    print("[spatial] summary:", os.path.join(args.out_dir, "summary.txt"))


if __name__ == "__main__":
    main()
