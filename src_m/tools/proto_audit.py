# src_m/tools/proto_audit.py
import os, json, argparse, glob
import numpy as np
from tqdm import tqdm

# ===================== 공통 유틸 =====================

def build_feature_match_dim(S_slots, XY, target_dim, xy_weight=1.0):
    if target_dim == 2:
        xy = XY / (np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8)
        return xy
    if S_slots is None:
        raise ValueError("Prototypes expect S or S+XY, but S_slots missing.")
    Dslot = S_slots.shape[-1]
    if target_dim == Dslot:
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        return S
    if target_dim == Dslot + 2:
        S = S_slots / (np.linalg.norm(Slots:=S_slots, axis=-1, keepdims=True) + 1e-8)
        X = XY       / (np.linalg.norm(XY,       axis=-1, keepdims=True) + 1e-8)
        return np.concatenate([S, X * float(xy_weight)], axis=-1)
    raise ValueError("Unsupported target_dim.")

def load_prototypes(path):
    with open(path,"r") as f: data = json.load(f)
    per = data.get("per_class", {})
    C_list, C_cls = [], []
    for c_str, block in per.items():
        c = int(c_str)
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is None: continue
            arr = np.asarray(mus, np.float32)
            if arr.ndim==1: C_list.append(arr); C_cls.append(c)
            else:
                for v in arr: C_list.append(np.asarray(v,np.float32)); C_cls.append(c)
        elif isinstance(block, list):
            for p in block:
                mu = p.get("mu") if isinstance(p, dict) else p
                if mu is None: continue
                C_list.append(np.asarray(mu, np.float32).reshape(-1)); C_cls.append(c)
        else:
            C_list.append(np.asarray(block, np.float32).reshape(-1)); C_cls.append(c)
    C = np.stack(C_list, 0) if len(C_list)>0 else np.zeros((0,2), np.float32)
    return C, np.asarray(C_cls, np.int64)

def iter_dump_items(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if not paths: raise FileNotFoundError(dump_dir)
    for p in paths:
        z = np.load(p, allow_pickle=True)
        y   = int(z["clazz"])
        pred= int(z["pred"]) if "pred" in z else y
        XY  = z["XY"].astype(np.float32)
        P   = z["slot_prob"].astype(np.float32) if "slot_prob" in z else z["energy_norm"].astype(np.float32)
        m   = z["slot_mask"].astype(np.float32)
        S   = z["S_slots"].astype(np.float32) if "S_slots" in z else None
        yield {"path":p, "clazz":y, "pred":pred, "XY":XY, "P":P, "slot_mask":m, "S_slots":S}

def gini(x):
    x = np.asarray(x, np.float64)
    if np.all(x==0): return 0.0
    x = np.sort(x); n=len(x); cum=np.cumsum(x)
    return float(1 + (1.0/n) - 2.0*np.sum(cum)/(n*cum[-1]))

# ===================== 메인 =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)
    ap.add_argument("--assign_top", type=int, default=1)
    ap.add_argument("--only_correct", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    C, C_cls = load_prototypes(args.proto_json)
    if C.shape[0] == 0:
        print("[audit] no prototypes.")
        return
    K, d = C.shape

    support = np.zeros(K, np.float64)
    resid_sum = np.zeros(K, np.float64)
    resid_all = [[] for _ in range(K)]
    per_class_assigns, per_class_total = {}, {}
    used_count = 0

    it = iter_dump_items(args.dump_dir)
    it = tqdm(it, desc="auditing", unit="img", disable=not args.progress)
    for item in it:
        y, pred = item["clazz"], item["pred"]
        if args.only_correct and (y!=pred): continue
        S, XY, P, m = item["S_slots"], item["XY"], item["P"], item["slot_mask"]

        # 특징 구성(프로토 차원에 맞춤)
        try:
            feats = build_feature_match_dim(S, XY, target_dim=d, xy_weight=args.xy_weight)  # (M,d)
        except ValueError:
            continue

        # 게이팅
        e = P * m
        idx_sorted = np.argsort(-e)
        top = idx_sorted[:max(1, args.assign_top)]

        idx_c = np.where(C_cls==y)[0]
        if idx_c.size == 0: continue

        per_class_total[y] = per_class_total.get(y, 0) + 1
        used_count += 1

        best_k = None; best_dist=1e9
        for midx in top:
            f = feats[midx]
            dists = np.linalg.norm(C[idx_c] - f[None,:], axis=1)
            j = int(np.argmin(dists))
            if dists[j] < best_dist:
                best_dist = float(dists[j]); best_k = int(idx_c[j])

        if best_k is not None:
            support[best_k] += 1.0
            resid_sum[best_k] += best_dist
            resid_all[best_k].append(best_dist)
            per_class_assigns[y] = per_class_assigns.get(y, 0) + 1

    mean_resid = np.zeros(K, np.float64)
    p90_resid  = np.zeros(K, np.float64)
    for k in range(K):
        if support[k] > 0:
            mean_resid[k] = resid_sum[k] / support[k]
            p90_resid[k]  = float(np.percentile(resid_all[k], 90))
        else:
            mean_resid[k] = np.nan; p90_resid[k] = np.nan

    # cross-class 최근접
    cross_rows = []
    for i in tqdm(range(K), desc="cross-class", disable=not args.progress):
        dists = np.linalg.norm(C - C[i], axis=1)
        same  = (C_cls == C_cls[i]); dists[i] = 1e9
        d_cross = dists.copy(); d_cross[same] = 1e9
        j = int(np.argmin(d_cross))
        cross_rows.append((i, j, float(d_cross[j])))
    cross_rows.sort(key=lambda t: t[2])
    cross_hits = []
    for (i,j,dc) in cross_rows[:min(500, len(cross_rows))]:
        cross_hits.append({
            "proto_i": i, "class_i": int(C_cls[i]),
            "proto_j": j, "class_j": int(C_cls[j]),
            "dist": dc,
            "mean_resid_i": float(mean_resid[i]), "p90_resid_i": float(p90_resid[i]),
            "mean_resid_j": float(mean_resid[j]), "p90_resid_j": float(p90_resid[j]),
        })

    # 저장(csv)
    try:
        import pandas as pd
        pd.DataFrame({
            "proto": np.arange(K),
            "class": C_cls,
            "support": support,
            "mean_resid": mean_resid,
            "p90_resid": p90_resid
        }).to_csv(os.path.join(args.out_dir,"proto_summary.csv"), index=False)

        class_rows = []
        for c in np.unique(C_cls):
            idx = np.where(C_cls==c)[0]
            sup = support[idx]
            g   = gini(sup)
            tot = per_class_total.get(int(c), 0)
            ass = per_class_assigns.get(int(c), 0)
            coverage = (ass / tot) if tot>0 else np.nan
            class_rows.append({
                "class": int(c), "#protos": int(idx.size),
                "support_sum": float(np.sum(sup)),
                "support_gini": float(g),
                "used_samples": int(tot),
                "assigned_samples": int(ass),
                "coverage": float(coverage),
            })
        pd.DataFrame(class_rows).to_csv(os.path.join(args.out_dir,"class_summary.csv"), index=False)
        pd.DataFrame(cross_hits).to_csv(os.path.join(args.out_dir,"crossclass_hits.csv"), index=False)
    except Exception:
        pass

    print(f"[audit] used_samples={used_count} K={K} d={d}")
    print(f"[audit] saved tables to: {args.out_dir}")


if __name__ == "__main__":
    main()
