# -*- coding: utf-8 -*-
# src_n/tools/n_diagnosis.py
import os, json, math, argparse, glob, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

from src_common.labels import emnist_char

# ----------------------------- prototype & feature -----------------------------
def load_prototypes(path, device):
    with open(path, "r") as f:
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
    if len(C_list)==0:
        raise RuntimeError("no prototypes found in json")
    C = torch.from_numpy(np.stack(C_list,0)).float().to(device)   # (K,d)
    C = F.normalize(C, dim=1)
    C_cls = torch.from_numpy(np.asarray(C_cls, np.int64)).to(device)
    return C, C_cls

def normed_torch(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def build_feature_torch(S_slots, XY, mode="s+xy", xy_weight=0.1):
    if mode == "xy":
        if XY is None: raise RuntimeError("feature_mode='xy' requires XY")
        return normed_torch(XY, dim=-1)
    elif mode == "s":
        if S_slots is None: raise RuntimeError("feature_mode='s' requires S_slots")
        return normed_torch(S_slots, dim=-1)
    else:
        if S_slots is None or XY is None:
            raise RuntimeError("feature_mode='s+xy' requires S_slots and XY")
        s = normed_torch(S_slots, dim=-1)
        x = normed_torch(XY, dim=-1) * float(xy_weight)
        return torch.cat([s, x], dim=-1)

def lse_pool(x, tau=0.5, dim=-1):
    xmax, _ = torch.max(x, dim=dim, keepdim=True)
    return (xmax + tau * torch.log(torch.clamp_min(torch.sum(torch.exp((x - xmax)/tau), dim=dim, keepdim=True), 1e-8))).squeeze(dim)

def per_slot_evidence(feats_md, C, C_cls, class_reduce="lse", proto_tau=0.5):
    device = feats_md.device
    M, d = feats_md.shape
    K = C.shape[0]
    classes = torch.unique(C_cls)
    Cmax = int(torch.max(classes).item()) + 1
    sim = torch.matmul(feats_md, C.t()).float()  # (M,K)
    neg_inf = torch.tensor(-1e9, device=device, dtype=sim.dtype)
    evi = sim.new_zeros(M, Cmax)
    for c in classes.tolist():
        mask_c = (C_cls == c).view(1, K)
        s_c = sim.masked_fill(~mask_c, neg_inf)
        if class_reduce == "max":
            evi_c, _ = torch.max(s_c, dim=1)
        else:
            evi_c = lse_pool(s_c, tau=proto_tau, dim=1)
        evi[:, c] = evi_c
    return evi  # (M,C)

# ----------------------------- aggregators -----------------------------
def soft_topk_weights(p_m, k=None, beta=0.5):
    p = p_m
    if (k is not None) and k > 0 and k < p.numel():
        topv, topi = torch.topk(p, k, dim=0)
        keep = torch.zeros_like(p); keep.scatter_(0, topi, 1.0)
        q = p * keep
    else:
        q = p
    z = (q - q.mean()) / max(1e-6, float(beta))
    w = torch.softmax(z, dim=0)
    w = w / w.sum().clamp_min(1e-8)
    return w

def top1_plus_support(evi_mc, p_m, beta_support=0.4, t_sup=1.6, S_slots=None, XY=None, kappa_feat=1.0, kappa_xy=0.0):
    M, C = evi_mc.shape
    score_mc = evi_mc * p_m.view(M,1)
    top1_val, top1_idx = score_mc.max(dim=0)
    P_sup = torch.softmax(torch.log(p_m.clamp_min(1e-8)) / t_sup, dim=0).view(M,1).expand_as(evi_mc)
    mask = torch.ones_like(evi_mc, dtype=torch.bool)
    mask.scatter_(0, top1_idx.view(1, C), False)
    w_red = 0.0
    if S_slots is not None and kappa_feat > 0:
        S = F.normalize(S_slots, dim=-1)
        top1_feat = S.index_select(0, top1_idx)
        cos = torch.matmul(S, top1_feat.t()).clamp_min(0)
        w_red = kappa_feat * cos
    if XY is not None and kappa_xy > 0:
        binsig2 = 0.04
        xy_top = XY.index_select(0, top1_idx)
        d2 = ((XY.unsqueeze(1) - xy_top.unsqueeze(0))**2).sum(-1)
        w_red = (w_red if isinstance(w_red, torch.Tensor) else 0) + kappa_xy * torch.exp(-d2 / binsig2)
    support_raw = evi_mc + torch.log(P_sup.clamp_min(1e-8)) - (w_red if isinstance(w_red, torch.Tensor) else 0)
    support_raw = support_raw.masked_fill(~mask, -1e4)
    support_val = torch.logsumexp(support_raw, dim=0)
    return top1_val + beta_support * support_val

def aggregate_logits(evi_mc, p_m, method="wsum", slot_topk=3, beta=0.5, pmean_p=2.5,
                     support_beta=0.4, support_tsup=1.6, S_slots=None, XY=None,
                     support_kappa_feat=1.0, support_kappa_xy=0.0):
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
    elif method == "support":
        return top1_plus_support(
            evi_mc, p_m, beta_support=support_beta, t_sup=support_tsup,
            S_slots=S_slots, XY=XY, kappa_feat=support_kappa_feat, kappa_xy=support_kappa_xy
        )
    elif method == "maxslot":
        vals, _ = (evi_mc).max(dim=0)
        return vals
    else:
        raise ValueError(f"unknown aggregate method {method}")

# ----------------------------- viz -----------------------------
def _topq_mask(a_hw: np.ndarray, q=0.10):
    flat = a_hw.reshape(-1)
    k = max(1, int(round(len(flat)*q)))
    thr = np.partition(flat, -k)[-k]
    return (a_hw >= thr)

def _cov_ellipse(a_hw: np.ndarray, q=0.10):
    m = _topq_mask(a_hw, q=q).astype(np.float32)
    ys, xs = np.where(m>0)
    if len(xs) < 5:
        return None
    cx = xs.mean(); cy = ys.mean()
    X = np.stack([xs-cx, ys-cy], 0)
    S = (X @ X.T) / max(1, X.shape[1]-1)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:,order]
    width, height = 2*np.sqrt(np.maximum(vals, 1e-8))
    angle = math.degrees(math.atan2(vecs[1,0], vecs[0,0]))
    return dict(cx=cx, cy=cy, width=width, height=height, angle=angle)

def _bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(xs)==0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _slot_topk_texts(evi_mc, k=3):
    """Return per-slot list of text lines for top-k predictions."""
    M, C = evi_mc.shape
    texts = []
    for m in range(M):
        topv, topi = torch.topk(evi_mc[m], k=min(k, C))
        lines = []
        for t in range(topv.numel()):
            c = int(topi[t].item())
            ch = emnist_char(c)
            val = float(topv[t].item())
            lines.append(f"{ch}:{val:.2f}")
        texts.append(lines)
    return texts

def save_slot_maps_boxes(image_28, A_m28, P_m, region_bbox_m2x4, topk_texts=None,
                         out_png=None, title=None, cols=4, q=0.10, frag_thr=0.4):
    """
    image_28     : (28,28) uint8
    A_m28        : (M,28,28) float32  (masked maps from dump)
    P_m          : (M,) float
    region_bbox_m2x4: (M,2,4) int32 (two grid cells per slot)
    topk_texts   : list of list[str] per slot (optional)
    """
    M = A_m28.shape[0]
    rows = int(math.ceil(M/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = axes.ravel()

    for m in range(rows*cols):
        ax = axes[m]; ax.axis("off")
        if m >= M: continue
        ax.imshow(image_28, cmap="gray", vmin=0, vmax=255)

        # (1) white boxes: the two grid cells this slot was allowed to see
        rb = region_bbox_m2x4[m]
        for kbox in range(2):
            x0,y0,x1,y1 = [int(v) for v in rb[kbox]]
            ax.add_patch(Rectangle((x0,y0), x1-x0+1, y1-y0+1, fill=False, lw=1.0, ec="white"))

        # (2) red box: exact bbox of top-q mask of the **actual activation within allowed cells**
        mask = _topq_mask(A_m28[m], q=q)
        bb = _bbox_of_mask(mask)
        if bb is not None:
            x0,y0,x1,y1 = bb
            ax.add_patch(Rectangle((x0,y0), x1-x0+1, y1-y0+1, fill=False, lw=1.3, ec="red"))

        # (3) yellow ellipse & center (only if mass not too fragmented)
        ell = _cov_ellipse(A_m28[m], q=q)
        if ell is not None:
            ax.add_patch(Ellipse((ell["cx"], ell["cy"]), ell["width"], ell["height"],
                                 angle=ell["angle"], fill=False, lw=1.2, ec="yellow"))
            ax.plot(ell["cx"], ell["cy"], "o", ms=3, mec="yellow", mfc="yellow")

        # (4) labels
        info = f"p={P_m[m]:.2f}"
        ax.text(0.01, 0.02, info, color="yellow", fontsize=8, transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
        if topk_texts is not None:
            tx = "\n".join(topk_texts[m])
            ax.text(0.01, 0.80, tx, color="yellow", fontsize=8, transform=ax.transAxes,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1))
        ax.set_title(f"slot {m}", fontsize=9)

    if title: fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=140)
    plt.close(fig)

# ----------------------------- dim adapter -----------------------------
def _adapt_proto_and_feats(C, feats):
    dC = C.size(1); dF = feats.size(1)
    if dC == dF:
        return C, feats
    dmin = min(dC, dF)
    C_use = F.normalize(C[:, :dmin], dim=1)
    feats_use = feats[:, :dmin]
    return C_use, feats_use

# ----------------------------- evaluation -----------------------------
def eval_dump(dump_dir, proto_json, out_dir,
              feature_mode="s+xy", xy_weight=0.1,
              class_reduce="lse", proto_tau=0.4,
              agg_list=("sum","wsum","softk","support","pmean","topk_wsum","maxslot"),
              slot_topk=5, beta=1.2, pmean_p=2.0,
              support_beta=0.4, support_tsup=1.6, support_kappa_feat=1.0, support_kappa_xy=0.0,
              num_correct=4, num_wrong=4, seed=0, compare_xy=False, topk_text_k=3, q_vis=0.10):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, C_cls = load_prototypes(proto_json, device=device)

    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if not files: raise FileNotFoundError(dump_dir)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    agg_names = list(agg_list)
    all_res = {name: {"correct": 0, "total": 0} for name in ["base"] + agg_names}
    correct_idx, wrong_idx = [], []

    for fp in files:
        it = np.load(fp, allow_pickle=True)
        y = int(it["clazz"]); pred = int(it["pred"])
        base_logits = torch.from_numpy(it["logits"]).float()
        base_pred = int(torch.argmax(base_logits).item())
        assert base_pred == pred

        all_res["base"]["total"] += 1
        all_res["base"]["correct"] += int(base_pred == y)
        if base_pred == y and len(correct_idx) < num_correct: correct_idx.append(fp)
        if base_pred != y and len(wrong_idx) < num_wrong:    wrong_idx.append(fp)

        # features
        S = it["S_slots"] if "S_slots" in it.files else None
        XY = it["XY"] if "XY" in it.files else None
        if S is not None: S = torch.from_numpy(S).float().to(device)
        if XY is not None: XY = torch.from_numpy(XY).float().to(device)
        feats = build_feature_torch(S, XY, mode=feature_mode, xy_weight=xy_weight).to(device)
        feats = F.normalize(feats, dim=-1)
        C_use, feats_use = _adapt_proto_and_feats(C, feats)
        evi_mc = per_slot_evidence(feats_use, C_use, C_cls, class_reduce=class_reduce, proto_tau=proto_tau)  # (M,C)
        P = torch.from_numpy(it["slot_prob"]).float().to(device)  # (M,)

        for name in agg_names:
            logits_c = aggregate_logits(
                evi_mc, P, method=name, slot_topk=slot_topk, beta=beta, pmean_p=pmean_p,
                support_beta=support_beta, support_tsup=support_tsup, S_slots=S, XY=XY,
                support_kappa_feat=support_kappa_feat, support_kappa_xy=support_kappa_xy
            )
            pred_c = int(torch.argmax(logits_c).item())
            all_res[name]["total"] += 1
            all_res[name]["correct"] += int(pred_c == y)

    # summarize
    def _acc(d): return 100.0 * d["correct"] / max(1, d["total"])
    lines = ["method,acc,correct,total"]
    for name in ["base"] + agg_names:
        acc = _acc(all_res[name])
        lines.append(f"{name},{acc:.4f},{all_res[name]['correct']},{all_res[name]['total']}")
    with open(os.path.join(out_dir, "agg_metrics.csv"), "w") as f:
        f.write("\n".join(lines))
    print("[agg] saved:", os.path.join(out_dir, "agg_metrics.csv"))

    # XY ablation
    if compare_xy and feature_mode != "s":
        print("[compare] running s-only for comparison...")
        res_s = eval_dump_simple(files, C, C_cls, device, out_dir,
                                 mode="s", xy_weight=xy_weight, class_reduce=class_reduce,
                                 proto_tau=proto_tau, agg_names=agg_names,
                                 slot_topk=slot_topk, beta=beta, pmean_p=pmean_p,
                                 support_beta=support_beta, support_tsup=support_tsup,
                                 support_kappa_feat=support_kappa_feat, support_kappa_xy=support_kappa_xy)
        with open(os.path.join(out_dir, "agg_metrics.csv"), "r") as f:
            base = f.read().strip().splitlines()
        with open(os.path.join(out_dir, "agg_compare.csv"), "w") as f:
            f.write("method,acc_s+xy,acc_s_only\n")
            acc_map_xy = {row.split(",")[0]: float(row.split(",")[1]) for row in base[1:]}
            for name in ["base"] + agg_names:
                ax = acc_map_xy.get(name, float("nan"))
                as_only = res_s.get(name, float("nan"))
                f.write(f"{name},{ax:.4f},{as_only:.4f}\n")
        print("[compare] saved:", os.path.join(out_dir, "agg_compare.csv"))

    # visualizations
    for tag, fps in [("correct", correct_idx), ("wrong", wrong_idx)]:
        for fp in fps:
            it = np.load(fp, allow_pickle=True)
            img = it["image"]; y = int(it["clazz"]); pred = int(it["pred"])
            Aup = it["A_upsampled"] if "A_upsampled" in it.files else it["A_maps"]  # (M,28,28)
            P = it["slot_prob"].astype(np.float32)
            region_list = it["region_bbox"] if "region_bbox" in it.files else None

            # slot logits text (top-k)
            S = it["S_slots"] if "S_slots" in it.files else None
            XY = it["XY"] if "XY" in it.files else None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if S is not None: S = torch.from_numpy(S).float().to(device)
            if XY is not None: XY = torch.from_numpy(XY).float().to(device)
            feats = build_feature_torch(S, XY, mode=feature_mode, xy_weight=xy_weight).to(device)
            feats = F.normalize(feats, dim=-1)
            C_use, feats_use = _adapt_proto_and_feats(C, feats)
            evi_mc = per_slot_evidence(feats_use, C_use, C_cls, class_reduce=class_reduce, proto_tau=proto_tau)
            topk_texts = _slot_topk_texts(evi_mc.cpu(), k=topk_text_k)

            out_png = os.path.join(out_dir, f"slots_map_{tag}_{int(it['id']):07d}_y{y}_p{pred}.png")
            save_slot_maps_boxes(img, Aup, P, region_list if region_list is not None else np.zeros((Aup.shape[0],2,4), np.int32),
                                 topk_texts=topk_texts, out_png=out_png, title=f"{tag.upper()} id={int(it['id'])}", q=q_vis)

    return os.path.join(out_dir, "agg_metrics.csv")

def eval_dump_simple(files, C, C_cls, device, out_dir, mode="s", xy_weight=0.1,
                     class_reduce="lse", proto_tau=0.4, agg_names=("sum","wsum","softk","support","pmean","topk_wsum","maxslot"),
                     slot_topk=5, beta=1.2, pmean_p=2.0,
                     support_beta=0.4, support_tsup=1.6, support_kappa_feat=1.0, support_kappa_xy=0.0):
    res = {name: {"correct": 0, "total": 0} for name in ["base"] + list(agg_names)}
    for fp in files:
        it = np.load(fp, allow_pickle=True)
        y = int(it["clazz"])
        base_logits = torch.from_numpy(it["logits"]).float()
        base_pred = int(torch.argmax(base_logits).item())
        res["base"]["total"] += 1
        res["base"]["correct"] += int(base_pred == y)

        S = it["S_slots"] if "S_slots" in it.files else None
        XY = it["XY"] if "XY" in it.files else None
        if S is not None: S = torch.from_numpy(S).float().to(device)
        if XY is not None: XY = torch.from_numpy(XY).float().to(device)
        feats = build_feature_torch(S, XY, mode=mode, xy_weight=xy_weight).to(device)
        feats = F.normalize(feats, dim=-1)
        C_use, feats_use = _adapt_proto_and_feats(C, feats)

        evi_mc = per_slot_evidence(feats_use, C_use, C_cls, class_reduce=class_reduce, proto_tau=proto_tau)
        P = torch.from_numpy(it["slot_prob"]).float().to(device)
        for name in agg_names:
            logits_c = aggregate_logits(
                evi_mc, P, method=name, slot_topk=slot_topk, beta=beta, pmean_p=pmean_p,
                support_beta=support_beta, support_tsup=support_tsup,
                S_slots=S, XY=XY, support_kappa_feat=support_kappa_feat, support_kappa_xy=support_kappa_xy
            )
            pred_c = int(torch.argmax(logits_c).item())
            res[name]["total"] += 1
            res[name]["correct"] += int(pred_c == y)
    return {k: (100.0*res[k]["correct"]/max(1,res[k]["total"])) for k in res}

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=0.1)
    ap.add_argument("--class_reduce", default="lse", choices=["lse","max"])
    ap.add_argument("--proto_tau", type=float, default=0.4)

    ap.add_argument("--agg", type=str, default="sum,wsum,softk,support,pmean,topk_wsum,maxslot")
    ap.add_argument("--slot_topk", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--pmean_p", type=float, default=2.0)
    ap.add_argument("--support_beta", type=float, default=0.4)
    ap.add_argument("--support_tsup", type=float, default=1.6)
    ap.add_argument("--support_kappa_feat", type=float, default=1.0)
    ap.add_argument("--support_kappa_xy", type=float, default=0.0)

    ap.add_argument("--num_correct", type=int, default=4)
    ap.add_argument("--num_wrong", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare_xy", type=int, default=1)
    ap.add_argument("--topk_text_k", type=int, default=3, help="slots_map 그림에서 표시할 per-slot top-k")
    ap.add_argument("--q_vis", type=float, default=0.10, help="red bbox를 위한 top-q 비율")
    args = ap.parse_args()

    agg_list = [s.strip() for s in args.agg.split(",") if s.strip()]
    eval_dump(
        dump_dir=args.dump_dir,
        proto_json=args.proto_json,
        out_dir=args.out_dir,
        feature_mode=args.feature_mode,
        xy_weight=args.xy_weight,
        class_reduce=args.class_reduce,
        proto_tau=args.proto_tau,
        agg_list=agg_list,
        slot_topk=args.slot_topk,
        beta=args.beta,
        pmean_p=args.pmean_p,
        support_beta=args.support_beta,
        support_tsup=args.support_tsup,
        support_kappa_feat=args.support_kappa_feat,
        support_kappa_xy=args.support_kappa_xy,
        num_correct=args.num_correct,
        num_wrong=args.num_wrong,
        seed=args.seed,
        compare_xy=bool(args.compare_xy),
        topk_text_k=args.topk_text_k,
        q_vis=args.q_vis,
    )

if __name__ == "__main__":
    main()
