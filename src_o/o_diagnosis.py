# -*- coding: utf-8 -*-
# src_o/o_diagnosis.py
"""
집계법(aggregation) 성능 비교 & 슬롯별 커버리지/로짓 시각화 (s-only)
- 입력: o_dump.py 의 *.npz
- 입력: o_protomine.py 의 proto json
- 출력: agg_metrics.csv, (선택 샘플) slots_map_*.png, slots_logits_*.png
"""

import os, json, math, argparse, glob, random
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------------------- prototypes -----------------------------
def load_prototypes(path, device):
    with open(path, "r") as f:
        data = json.load(f)
    per = data.get("per_class", {})
    C_list, C_cls = [], []
    for c_str, block in per.items():
        cid = int(c_str)
        mus = []
        if isinstance(block, dict):
            arr = block.get("mu") or block.get("center") or block.get("centers")
            if arr is not None:
                mus = arr if isinstance(arr[0], (list, tuple)) else [arr]
        elif isinstance(block, list):
            for p in block:
                mus.append(p.get("mu") if isinstance(p, dict) else p)
        else:
            mus = [block]
        for v in mus:
            C_list.append(np.asarray(v, np.float32).reshape(-1))
            C_cls.append(cid)
    C = torch.from_numpy(np.stack(C_list,0)).float().to(device)  # (K,d)
    C = F.normalize(C, dim=1)
    C_cls = torch.from_numpy(np.asarray(C_cls, np.int64)).to(device)
    return C, C_cls


def per_slot_evidence(S_md, C, C_cls, class_reduce="lse", proto_tau=0.5):
    """
    S_md: (M,d) normalized
    C   : (K,d) normalized prototypes
    return evi_mc: (M,Cmax)
    """
    device = S_md.device
    M, d = S_md.shape
    K = C.shape[0]
    classes = torch.unique(C_cls)
    Cmax = int(torch.max(classes).item()) + 1

    sim = torch.matmul(S_md, C.t()).float()  # (M,K)
    neg_inf = torch.tensor(-1e9, device=device, dtype=sim.dtype)

    evi = sim.new_zeros(M, Cmax)
    for c in classes.tolist():
        mask_c = (C_cls == c).view(1, K)
        s_c = sim.masked_fill(~mask_c, neg_inf)
        if class_reduce == "max":
            evi_c, _ = torch.max(s_c, dim=1)
        else:  # lse
            xmax, _ = torch.max(s_c, dim=1, keepdim=True)
            evi_c = (xmax + proto_tau * torch.log(torch.clamp_min(torch.sum(torch.exp((s_c - xmax)/proto_tau), dim=1, keepdim=True), 1e-8))).squeeze(1)
        evi[:, c] = evi_c
    return evi


# ----------------------------- aggregators -----------------------------
def soft_topk_weights(p_m, k=None, beta=0.5):
    if (k is not None) and k > 0 and k < p_m.numel():
        topv, topi = torch.topk(p_m, k, dim=0)
        keep = torch.zeros_like(p_m); keep.scatter_(0, topi, 1.0)
        q = p_m * keep
    else:
        q = p_m
    z = (q - q.mean()) / max(1e-6, float(beta))
    w = torch.softmax(z, dim=0)
    w = w / w.sum().clamp_min(1e-8)
    return w

def aggregate_logits(evi_mc, p_m, method="wsum", slot_topk=5, beta=0.5, pmean_p=2.5):
    M, C = evi_mc.shape
    if method == "sum":
        return evi_mc.sum(dim=0)
    elif method == "wsum":
        return (evi_mc * p_m.view(M,1)).sum(dim=0)
    elif method == "softk":
        w = soft_topk_weights(p_m, k=slot_topk, beta=beta)
        return (evi_mc * w.view(M,1)).sum(dim=0)
    elif method == "topk_wsum":
        topv, topi = torch.topk(p_m, min(slot_topk, M))
        mask = torch.zeros_like(p_m); mask.scatter_(0, topi, 1.0)
        w = (p_m * mask); w = w / w.sum().clamp_min(1e-8)
        return (evi_mc * w.view(M,1)).sum(dim=0)
    elif method == "pmean":
        w = soft_topk_weights(p_m, k=slot_topk, beta=beta)
        contrib = (evi_mc * w.view(M,1)).clamp_min(0)
        return contrib.pow(pmean_p).mean(dim=0).pow(1.0/pmean_p)
    elif method == "maxslot":
        vals, _ = (evi_mc).max(dim=0)
        return vals
    else:
        raise ValueError(f"unknown aggregate method {method}")


# ----------------------------- viz helpers -----------------------------
def _topq_bbox(a_28, q=0.10):
    flat = a_28.reshape(-1)
    k = max(1, int(round(len(flat) * q)))
    thr = np.partition(flat, -k)[-k]
    m = (a_28 >= thr)
    ys, xs = np.where(m)
    if len(xs)==0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def save_slot_maps(image_28, A_m28, P_m, out_png, title=None, cols=6, q=0.10, topn_text=3, evi_mc=None):
    M = A_m28.shape[0]
    rows = int(math.ceil(M/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.4, rows*2.4))
    axes = axes.ravel()
    for m in range(rows*cols):
        ax = axes[m]
        if m < M:
            ax.imshow(image_28, cmap="gray", vmin=0, vmax=255)
            # 빨간 박스: mask 영역의 bbox (top-q)
            bb = _topq_bbox(A_m28[m], q=q)
            if bb is not None:
                x0,y0,x1,y1 = bb
                ax.add_patch(Rectangle((x0,y0), x1-x0+1, y1-y0+1, fill=False, lw=1.2, ec="red"))
            # 로짓 top-N 텍스트
            if (evi_mc is not None) and (topn_text>0):
                tv, ti = torch.topk(evi_mc[m], k=min(topn_text, evi_mc.shape[1]))
                lines = [f"{int(ti[i].item())}:{float(tv[i].item()):.2f}" for i in range(tv.numel())]
                ax.text(0.02, 0.02, "\n".join(lines), color="yellow", fontsize=7,
                        transform=ax.transAxes, bbox=dict(facecolor="black", alpha=0.4, pad=1))
            ax.set_title(f"slot {m}  p={P_m[m]:.2f}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")
    if title: fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140); plt.close(fig)


# ----------------------------- main eval -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--class_reduce", default="lse", choices=["lse","max"])
    ap.add_argument("--proto_tau", type=float, default=0.4)
    ap.add_argument("--agg", type=str, default="sum,wsum,softk,pmean,topk_wsum,maxslot")
    ap.add_argument("--slot_topk", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--pmean_p", type=float, default=2.0)

    ap.add_argument("--num_correct", type=int, default=6)
    ap.add_argument("--num_wrong", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, C_cls = load_prototypes(args.proto_json, device=device)

    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    if not files: raise FileNotFoundError(args.dump_dir)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    agg_list = [s.strip() for s in args.agg.split(",") if s.strip()]
    all_res = {name: {"correct": 0, "total": 0} for name in ["base"] + agg_list}
    correct_fp, wrong_fp = [], []

    for fp in files:
        it = np.load(fp, allow_pickle=True)
        y = int(it["clazz"])
        base_pred = int(it["pred"]) if "pred" in it.files else -1  # base 없음
        # baseline 스텁
        all_res["base"]["total"] += 1
        all_res["base"]["correct"] += int(base_pred == y and base_pred >= 0)

        S = torch.from_numpy(np.asarray(it["S_slots"], np.float32)).to(device)   # (M,D)
        P = torch.from_numpy(np.asarray(it["slot_prob"], np.float32)).to(device) # (M,)

        S = F.normalize(S, dim=-1)
        evi_mc = per_slot_evidence(S, C, C_cls, class_reduce=args.class_reduce, proto_tau=args.proto_tau)  # (M,C)

        # 평가 & 정답/오답 샘플 수집 (wsum 기준)
        logits_wsum = aggregate_logits(evi_mc, P, method="wsum", slot_topk=args.slot_topk, beta=args.beta, pmean_p=args.pmean_p)
        pred_wsum = int(torch.argmax(logits_wsum).item())

        if pred_wsum == y and len(correct_fp) < args.num_correct:
            correct_fp.append(fp)
        if pred_wsum != y and len(wrong_fp) < args.num_wrong:
            wrong_fp.append(fp)

        # 모든 집계법 기록
        for name in agg_list:
            logits_c = aggregate_logits(evi_mc, P, method=name, slot_topk=args.slot_topk, beta=args.beta, pmean_p=args.pmean_p)
            pred_c = int(torch.argmax(logits_c).item())
            all_res[name]["total"] += 1
            all_res[name]["correct"] += int(pred_c == y)

    # 저장
    lines = ["method,acc,correct,total"]
    for name in ["base"] + agg_list:
        corr, tot = all_res[name]["correct"], all_res[name]["total"]
        acc = (100.0 * corr / max(1, tot))
        lines.append(f"{name},{acc:.4f},{corr},{tot}")
    with open(os.path.join(args.out_dir, "agg_metrics.csv"), "w") as f:
        f.write("\n".join(lines))
    print("[diag] saved:", os.path.join(args.out_dir, "agg_metrics.csv"))

    # 시각화 몇 개
    for tag, fps in [("correct", correct_fp), ("wrong", wrong_fp)]:
        for fp in fps:
            it = np.load(fp, allow_pickle=True)
            img = it["image"]; Aup = it["A_upsampled"]; P = it["slot_prob"].astype(np.float32)
            y = int(it["clazz"])
            S = torch.from_numpy(np.asarray(it["S_slots"], np.float32)).to(device)
            S = F.normalize(S, dim=-1)
            evi_mc = per_slot_evidence(S, C, C_cls, class_reduce=args.class_reduce, proto_tau=args.proto_tau)
            out_png = os.path.join(args.out_dir, f"slots_map_{tag}_{int(it['id']):07d}_y{y}.png")
            save_slot_maps(img, Aup, P, out_png, title=f"{tag.upper()} id={int(it['id'])}", cols=6, q=0.10, topn_text=3, evi_mc=evi_mc.cpu())

    print("[diag] done.")


if __name__ == "__main__":
    main()
