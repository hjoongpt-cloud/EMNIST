# -*- coding: utf-8 -*-
# src_n/tools/n_explain_sample.py
import os, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- utils: prototypes ----------
def load_prototypes(path):
    with open(path, "r") as f:
        data = json.load(f)
    per = data.get("per_class", {})
    C_list, C_cls, map_local = [], [], {}
    gid = 0
    for c_str, block in per.items():
        cid = int(c_str); vecs = []
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                if isinstance(mus[0], (list, tuple)): vecs = mus
                else: vecs = [mus]
            else:
                inner = block.get("protos") or block.get("clusters") or block.get("items") or []
                for p in inner:
                    v = p.get("mu") if isinstance(p, dict) else p
                    if v is None: continue
                    vecs.append(v)
        elif isinstance(block, list):
            for p in block:
                v = p.get("mu") if isinstance(p, dict) else p
                vecs.append(v)
        else:
            vecs = [block]

        map_local[cid] = []
        for v in vecs:
            C_list.append(np.asarray(v, np.float32).reshape(-1))
            C_cls.append(cid); map_local[cid].append(gid); gid += 1

    C = np.stack(C_list, 0) if len(C_list) else np.zeros((0,2), np.float32)
    C_cls = np.asarray(C_cls, np.int64)
    return C, C_cls, map_local

# ---------- utils: dump ----------
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    item = dict(
        path=path,
        image=z["image"].astype(np.uint8),      # (28,28)
        XY=z["XY"].astype(np.float32),          # (M,2)  [0..1]
        slot_mask=z["slot_mask"].astype(np.float32), # (M,)
        clazz=int(z["clazz"]),
        pred=int(z.get("pred", z["clazz"])),
        Aup=z.get("A_upsampled", None),
    )
    # features
    item["S_slots"] = z.get("S_slots", None)
    if item["S_slots"] is not None:
        item["S_slots"] = item["S_slots"].astype(np.float32)
    # slot weights
    if "slot_prob" in z:
        item["slot_prob"] = z["slot_prob"].astype(np.float32)
    elif "energy_norm" in z:
        item["slot_prob"] = z["energy_norm"].astype(np.float32)
    else:
        raise KeyError("dump lacks slot_prob/energy_norm")
    return item

def build_feature_match_dim(S, XY, target_dim, xy_weight=1.0):
    D = S.shape[-1] if S is not None else 0
    if target_dim == 2:  # XY only
        X = XY / (np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8)
        return X
    if target_dim == D:  # S only
        S = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8)
        return S
    if target_dim == D + 2:  # S+XY
        S = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8)
        X = XY / (np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8)
        return np.concatenate([S, X * float(xy_weight)], axis=-1)
    raise ValueError(f"target_dim={target_dim} not compatible (D={D})")

def soft_topk_weights(p, k=3, beta=0.5, mask=None):
    """p: (M,) slot_prob, mask: (M,) 0/1"""
    q = p.copy()
    if mask is not None: q *= mask
    if k is not None and k < len(q):
        idx = np.argsort(-q)[:k]
        keep = np.zeros_like(q); keep[idx] = 1.0
        q *= keep
    # temperature softmax on kept slots
    z = (q - q.mean()) / max(1e-6, float(beta))
    e = np.exp(z); e *= (mask if mask is not None else 1.0)
    s = e.sum(); 
    return e / (s+1e-8)

def lse(x, tau=0.5, axis=-1):
    xmax = np.max(x, axis=axis, keepdims=True)
    return (xmax + tau * np.log(np.sum(np.exp((x - xmax)/tau), axis=axis, keepdims=True))).squeeze(axis)

def class_evidence_per_slot(feat_m, C, C_cls, reduce="lse", proto_tau=0.5):
    """
    feat_m: (d,), C:(K,d), C_cls:(K,)
    return: evidences (num_classes,), also top-3 matches [(cls, pid, sim),...]
    """
    if C.shape[0] == 0:
        return np.zeros((0,), np.float32), []
    # cosine similarity
    f = feat_m / (np.linalg.norm(feat_m)+1e-8)
    Cm = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-8)
    sim = Cm @ f   # (K,)

    classes = np.unique(C_cls)
    evi = np.zeros(int(classes.max())+1, np.float32)  # assumes labels 0..C-1
    for c in classes:
        mask = (C_cls == c)
        if not np.any(mask): 
            continue
        sims = sim[mask]
        if reduce == "max":
            evi[c] = sims.max()
        else:  # LSE
            evi[c] = lse(sims, tau=proto_tau)
    # top-3 matches overall
    top3 = np.argsort(-sim)[:3]
    matches = [(int(C_cls[i]), int(i), float(sim[i])) for i in top3]
    return evi, matches

def compare_topk(item, C, C_cls, feat_mode_dim, xy_weight, proto_tau, beta, k_list):
    # features by slot
    S = item["S_slots"]; XY = item["XY"]; P = item["slot_prob"]; M = P.shape[0]
    if S is None and feat_mode_dim not in (2,):
        raise RuntimeError("Dump lacks S_slots but feature_mode needs it.")
    feats = build_feature_match_dim(S, XY, feat_mode_dim, xy_weight=xy_weight)  # (M,d)

    results = {}
    evi_cache = {}  # per-slot per-class
    for k in k_list:
        w = soft_topk_weights(P, k=k, beta=beta, mask=item["slot_mask"])
        # per-class score
        classes = np.unique(C_cls)
        Cmax = int(classes.max())+1
        z = np.zeros(Cmax, np.float32)
        per_slot = []
        for m in range(M):
            if w[m] <= 0: 
                per_slot.append({"w":0.0})
                continue
            key = m
            if key not in evi_cache:
                evi_m, matches = class_evidence_per_slot(feats[m], C, C_cls, reduce="lse", proto_tau=proto_tau)
                evi_cache[key] = (evi_m, matches)
            evi_m, matches = evi_cache[key]
            z[:len(evi_m)] += w[m] * evi_m
            per_slot.append({"w": float(w[m]), "evi": evi_m, "matches": matches})
        results[k] = {"z": z, "per_slot": per_slot}
    return results

def overlay(img01, heat01):
    H,W = img01.shape
    base = np.stack([img01, img01, img01], axis=-1)
    base[...,0] = np.clip(base[...,0] + 0.7*heat01, 0, 1)
    return (base*255).astype(np.uint8)
# --- add: slot overlay helper ---
def draw_slot_overlays(ax_img, img_hw, A_maps, XY, slot_prob, topk=5):
    import numpy as np, matplotlib.pyplot as plt
    H, W = img_hw
    # (1) 히트맵 얹기(상위 topk 슬롯만)
    idx = np.argsort(slot_prob)[::-1][:topk]
    for rank, m in enumerate(idx, 1):
        A = A_maps[m]
        A = (A - A.min()) / (A.max() - A.min() + 1e-8)
        ax_img.imshow(A, alpha=0.25)  # 반투명
    # (2) 슬롯 중심 찍기(전체 12개)
    for m, (x, y) in enumerate(XY):
        ax_img.scatter([x*W], [y*H], s=50, marker='o')
        ax_img.text(x*W+1, y*H-1, f"{m}", fontsize=8)
    ax_img.set_title("slots: centers (all) + heat (top-k)")

def explain_and_plot(item, results, out_png, topk_show=3):
    import numpy as np
    y, pred = item["clazz"], item["pred"]
    img = item["image"].astype(np.float32)/255.0
    XY = item["XY"]; P = item["slot_prob"]; M = len(P)
    Aup = item["Aup"] if item["Aup"] is not None else None

    # choose k_best (largest in results keys except 1)
    ks = sorted(results.keys())
    k_many = max([k for k in ks if k>1], default=ks[-1])
    rmany = results[k_many]
    r1    = results[1] if 1 in results else None

    # figure layout
    fig = plt.figure(figsize=(14,8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2,1,1.2], height_ratios=[1,1], wspace=0.35, hspace=0.35)

    # A) image + slot overlays via helper
    ax0 = fig.add_subplot(gs[0,0]); ax0.axis("off")
    ax0.set_title(f"image + top{topk_show} slot heats", fontsize=11)
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1)

    # Aup(업샘플 히트맵)가 없을 때도 헬퍼가 동작하도록 안전하게 처리
    import numpy as np
    H, W = img.shape
    if Aup is None:
        A_maps = np.zeros((len(P), H, W), dtype=np.float32)  # 히트맵 없이 점만 찍힘
    else:
        A_maps = Aup

    # 모든 슬롯 중심은 찍고, 히트맵은 topk만 반투명으로 얹음
    draw_slot_overlays(ax0, (H, W), A_maps, XY, P, topk=topk_show)


    # B) slot contribution bars to y & pred for top slots
    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title("slot contributions (to y & pred)", fontsize=11)
    idx = np.argsort(-P)[:topk_show]
    labels = [f"s{m}" for m in idx]
    vy = []; vp = []; weights=[]
    for m in idx:
        info = rmany["per_slot"][m]
        w = info.get("w",0.0); weights.append(w)
        evi = info.get("evi", None)
        vy.append( (w * (0 if evi is None or y>=len(evi) else evi[y])) )
        vp.append( (w * (0 if evi is None or pred>=len(evi) else evi[pred])) )
    x = np.arange(len(labels))
    ax1.bar(x-0.15, vy, width=0.3, label=f"class {y} (true)")
    ax1.bar(x+0.15, vp, width=0.3, label=f"class {pred} (pred)")
    for i,w in enumerate(weights): ax1.text(x[i], max(vy[i],vp[i])+1e-3, f"w={w:.2f}", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.legend(fontsize=9)

    # C) top-3 prototype matches per slot (table-like text)
    ax2 = fig.add_subplot(gs[0,2]); ax2.axis("off")
    ax2.set_title("slot → prototype top-3 (class, pid, sim)", fontsize=11)
    lines=[]
    for m in idx:
        info = rmany["per_slot"][m]; mats = info.get("matches", [])
        line = f"s{m}: " + " | ".join([f"c{c}/p{pid}:{sim:.2f}" for (c,pid,sim) in mats])
        lines.append(line)
    txt = "\n".join(lines) if lines else "(no matches)"
    ax2.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10)

    # D) Top-K comparison of logits (top-5 classes)
    ax3 = fig.add_subplot(gs[1, :]); 
    ax3.set_title("Top-K comparison: slot_topk=1 vs slot_topk=K (top-5 classes)", fontsize=11)
    z_many = rmany["z"]
    top5 = np.argsort(-z_many)[:5]
    ax3.bar(np.arange(5)-0.15, z_many[top5], width=0.3, label=f"topK={k_many}")
    if r1 is not None:
        z1 = r1["z"]
        ax3.bar(np.arange(5)+0.15, z1[top5], width=0.3, label="topK=1")
    ax3.set_xticks(np.arange(5)); ax3.set_xticklabels([str(c) for c in top5])
    ax3.set_ylabel("logit (aggregated)"); ax3.legend()

    # E) (optional) relation readout for top-2 slots
    # simple geometry only; not used in score
    if len(idx) >= 2:
        (m1,m2) = idx[:2]
        dx = XY[m2,0]-XY[m1,0]; dy = XY[m2,1]-XY[m1,1]
        dist = np.sqrt(dx*dx+dy*dy)
        ax0.text(0.02, 0.02, f"Δ=({dx:+.2f},{dy:+.2f}) | ||Δ||={dist:.2f}", color="w",
                 ha="left", va="bottom", transform=ax0.transAxes, fontsize=9,
                 bbox=dict(facecolor="k", alpha=0.35, pad=2))

    fig.suptitle(f"y={y} pred={pred} | file={os.path.basename(item['path'])}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--idx", type=int, default=None, help="절대 인덱스(파일순)")
    ap.add_argument("--find", type=str, default=None, help="예: 'y=16,pred=0' 또는 'y=16' 등")
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--proto_tau", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # load prototypes
    C, C_cls, _ = load_prototypes(args.proto_json)
    if C.shape[0]==0:
        raise RuntimeError("no prototypes")

    # pick sample
    files = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    if not files: raise FileNotFoundError(args.dump_dir)

    target_path = None
    if args.idx is not None:
        target_path = files[args.idx]
    elif args.find is not None:
        cond = {}
        for part in args.find.split(","):
            k,v = part.split("="); cond[k.strip()] = int(v.strip())
        for p in files:
            it = load_npz(p)
            ok = True
            for k,v in cond.items():
                if k not in it or int(it[k])!=v: ok=False; break
            if ok: target_path = p; break
    else:
        target_path = files[0]

    if target_path is None:
        raise RuntimeError("no sample matched condition")

    item = load_npz(target_path)

    # feature-mode → target dim
    Dslot = 0 if item["S_slots"] is None else item["S_slots"].shape[-1]
    if args.feature_mode=="s": d = Dslot
    elif args.feature_mode=="xy": d = 2
    else: d = Dslot + 2

    # compute results for k=1 and k=topk
    res = compare_topk(item, C, C_cls, d, args.xy_weight, args.proto_tau, args.beta, [1, args.topk])
    explain_and_plot(item, res, args.out_png, topk_show=min(args.topk,3))
    print("[save]", args.out_png)

if __name__ == "__main__":
    main()
