# src_n/tools/n_report.py
import os, json, glob, argparse, numpy as np
import matplotlib.pyplot as plt

def l2norm(x, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def build_feature(S, XY, mode="s+xy", xy_weight=0.1):
    if mode=="s":   return l2norm(S)
    if mode=="xy":  return l2norm(XY)
    return np.concatenate([l2norm(S), l2norm(XY)*xy_weight], axis=-1)

def load_protos(path):
    with open(path, "r") as f: data=json.load(f)
    per = data.get("per_class",{})
    C_list, C_cls = [], []
    for cstr, block in per.items():
        cid = int(cstr); vecs=[]
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                vecs = mus if isinstance(mus[0], (list,tuple)) else [mus]
            else:
                inner = block.get("protos") or block.get("clusters") or []
                for p in inner:
                    v = p.get("mu") if isinstance(p, dict) else p
                    if v is not None: vecs.append(v)
        elif isinstance(block, list):
            for p in block:
                v = p.get("mu") if isinstance(p, dict) else p
                if v is not None: vecs.append(v)
        for v in vecs:
            C_list.append(np.asarray(v, np.float32).reshape(-1))
            C_cls.append(cid)
    if not C_list: raise RuntimeError("no prototypes in json")
    C = np.stack(C_list,0).astype(np.float32)
    C = l2norm(C, axis=1)
    return C, np.asarray(C_cls, np.int64)

def topq_mask(a_hw, q=0.1):
    flat = a_hw.reshape(-1)
    k = max(1, int(round(q*flat.size)))
    idx = np.argpartition(-flat, k-1)[:k]
    m = np.zeros_like(flat, dtype=bool)
    m[idx] = True
    return m.reshape(a_hw.shape)

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    H = -(p*np.log(p)).sum()
    return H

def assign_slots_to_protos(item, C, feature_mode="s+xy", xy_weight=0.1):
    XY = item["XY"]                  # (M,2)
    P  = item["slot_prob"]           # (M,)
    M  = XY.shape[0]
    S  = item.get("S_slots", None)
    alive = item["slot_mask"].astype(bool)
    use = alive & (P > 0)
    if S is None and feature_mode in ("s","s+xy"):
        raise RuntimeError("feature_mode uses S but S_slots missing in dump")
    if S is None:
        F = build_feature(None, XY, mode="xy", xy_weight=xy_weight)
    else:
        F = build_feature(S, XY, mode=feature_mode, xy_weight=xy_weight)  # (M,d)
    F = l2norm(F)
    # cosine → nearest proto
    sim = F @ C.T  # (M,K)
    pid = sim.argmax(1)  # (M,)
    return pid, sim, use

def report_proto_space(args):
    os.makedirs(args.out_dir, exist_ok=True)
    C, C_cls = load_protos(args.proto_json)
    K = C.shape[0]
    # 1) 프로토 공간 엔트로피 & 2) 프로토 커버리지(4x4)
    grid = int(args.grid)
    counts = [np.zeros((grid,grid), np.float32) for _ in range(K)]
    assign_cnt = np.zeros(K, np.int64)
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        it = np.load(path, allow_pickle=True)
        item = {k: it[k] if k in it else None for k in it.files}
        pid, sim, use = assign_slots_to_protos(item, C, args.feature_mode, args.xy_weight)
        XY = item["XY"].astype(np.float32)  # (M,2), coords in (x=W, y=H)
        H = item["A_upsampled"].shape[-2]; W = item["A_upsampled"].shape[-1]
        for m in np.where(use)[0]:
            k = pid[m]
            x, y = XY[m,0]/max(1,W-1), XY[m,1]/max(1,H-1)  # [0,1]
            gx, gy = min(grid-1, int(x*grid)), min(grid-1, int(y*grid))
            counts[k][gy, gx] += 1.0
            assign_cnt[k] += 1
    rows = []
    for k in range(K):
        mat = counts[k]
        p = mat / (mat.sum()+1e-8)
        Hn = entropy(p.ravel()) / np.log(grid*grid)  # 정규화 엔트로피 in [0,1]
        cov = float((mat>0).sum()) / float(grid*grid)  # 커버리지
        rows.append((k, int(C_cls[k]), int(assign_cnt[k]), float(Hn), float(cov)))
        if args.per_proto_heatmap and assign_cnt[k] >= args.min_count_heat:
            plt.figure(figsize=(3,3))
            plt.imshow(mat, origin="lower")
            plt.title(f"proto {k} (cls {int(C_cls[k])})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"proto_{k:04d}_heat.png"), dpi=140)
            plt.close()
    import csv
    with open(os.path.join(args.out_dir, "proto_space.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["proto_id","cls","assigned","entropy","coverage"])
        w.writerows(rows)

def report_slot_diversity(args):
    # 3) 슬롯 다양성: per-sample IoU 평균(↓), XY 분산(↑)
    import itertools
    iou_list, xyvar_list = [], []
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        it = np.load(path, allow_pickle=True)
        item = {k: it[k] if k in it else None for k in it.files}
        A = item["A_upsampled"].astype(np.float32) # (M,H,W)
        XY = item["XY"].astype(np.float32)         # (M,2)
        P  = item["slot_prob"].astype(np.float32)  # (M,)
        M  = A.shape[0]
        k = min(args.slot_topk, M)
        idx = np.argsort(-P)[:k]                   # top-k
        masks = np.stack([topq_mask(A[i], q=args.q_iou) for i in idx],0)  # (k,H,W) bool
        # pairwise IoU mean
        ious=[]
        for i,j in itertools.combinations(range(k),2):
            mi, mj = masks[i].astype(bool), masks[j].astype(bool)
            inter = np.logical_and(mi,mj).sum()
            uni   = np.logical_or(mi,mj).sum()
            ious.append(0.0 if uni==0 else inter/uni)
        iou_list.append(0.0 if not ious else float(np.mean(ious)))
        # XY variance
        xy = XY[idx]
        xyvar_list.append(float(np.mean(np.var(xy, axis=0))))
    iou_arr = np.asarray(iou_list, np.float32); xyv_arr = np.asarray(xyvar_list, np.float32)
    np.savez_compressed(os.path.join(args.out_dir, "slot_diversity.npz"),
                        iou_mean=iou_arr, xy_var=xyv_arr)
    for name, arr in [("iou_mean", iou_arr), ("xy_var", xyv_arr)]:
        plt.figure(figsize=(4,3))
        plt.hist(arr, bins=50)
        plt.title(name); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"hist_{name}.png"), dpi=140)
        plt.close()

def report_duplicates(args):
    # 4) 중복 슬롯 탐지: (m,n) 쌍이 "항상" 비슷한 위치에 있는 경우
    # 기준: alive & P>pmin & 거리<radius 조건이 충족된 비율 >= dup_min_frac
    import itertools, csv
    # 먼저 한 개 파일에서 M 크기 파악
    sample = np.load(sorted(glob.glob(os.path.join(args.dump_dir,"*.npz")))[0], allow_pickle=True)
    M = sample["XY"].shape[0]
    hits = np.zeros((M,M), np.int64)
    co   = np.zeros((M,M), np.int64)
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        it = np.load(path, allow_pickle=True)
        XY = it["XY"].astype(np.float32)    # (M,2)
        P  = it["slot_prob"].astype(np.float32)
        alive = it["slot_mask"].astype(bool)
        use = alive & (P > args.dup_pmin)
        for i,j in itertools.combinations(range(M),2):
            if use[i] and use[j]:
                dx, dy = XY[i,0]-XY[j,0], XY[i,1]-XY[j,1]
                d = np.hypot(dx,dy)
                co[i,j] += 1; co[j,i] += 1
                if d < args.dup_radius:
                    hits[i,j] += 1; hits[j,i] += 1
    rows=[]
    for i in range(M):
        for j in range(i+1,M):
            denom = max(1, co[i,j])
            frac = hits[i,j] / denom
            if frac >= args.dup_min_frac:
                rows.append((i,j, frac, co[i,j]))
    rows.sort(key=lambda x: -x[2])
    with open(os.path.join(args.out_dir, "duplicate_slots.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["slot_i","slot_j","close_frac","co_occurrences"])
        w.writerows(rows)

def report_misclf(args):
    # 5) 오분류 케이스: top-3 슬롯의 클래스 기여 막대 + 프로토 매칭 표
    C, C_cls = load_protos(args.proto_json)
    def class_scores_for_slot(feat):  # RBF-LSE
        # 거리^2 ~ 2 - 2*cos ; cos = feat·C
        cos = (feat.reshape(1,-1) @ C.T).ravel()
        # class reduce: LSE over class prototypes with tau
        tau = max(1e-6, args.proto_tau)
        scores = {}
        for c in np.unique(C_cls):
            mask = (C_cls==c)
            val  = tau*np.log(np.exp(cos[mask]/tau).sum() + 1e-8)
            scores[int(c)] = float(val)
        return scores
    os.makedirs(os.path.join(args.out_dir,"misclf"), exist_ok=True)
    picked = 0
    for path in sorted(glob.glob(os.path.join(args.dump_dir, "*.npz"))):
        it = np.load(path, allow_pickle=True)
        logits = it["logits"].astype(np.float32)  # (C,)
        y = int(it["y"]) if "y" in it.files else int(it["clazz"])
        pred = int(np.argmax(logits))
        if pred == y: continue
        item = {k: it[k] if k in it else None for k in it.files}
        XY = item["XY"].astype(np.float32)
        P  = item["slot_prob"].astype(np.float32)
        S  = item.get("S_slots", None)
        if S is None and args.feature_mode in ("s","s+xy"): 
            # S가 없으면 XY만으로 진행
            feat_mode = "xy"
        else:
            feat_mode = args.feature_mode
        k = min(3, len(P))
        idx = np.argsort(-P)[:k]
        feats=[]
        if feat_mode=="xy":
            feats = l2norm(XY[idx])
        elif feat_mode=="s":
            feats = l2norm(S[idx])
        else:
            feats = build_feature(S[idx], XY[idx], mode="s+xy", xy_weight=args.xy_weight)
        # per-slot class score
        per_slot = [class_scores_for_slot(f) for f in feats]
        # 막대: 각 슬롯의 (pred, y) 점수
        labs = [f"slot#{int(i)}" for i in idx]
        pred_scores = [s.get(pred, 0.0) for s in per_slot]
        true_scores = [s.get(y, 0.0) for s in per_slot]
        # 표: 슬롯별 Top-3 프로토 (class, score)
        top_tables = []
        cos_full = (feats @ C.T)  # (k,K)
        for r in range(k):
            order = np.argsort(-cos_full[r])[:3]
            top_tables.append([(int(C_cls[j]), float(cos_full[r,j]), int(j)) for j in order])
        # 그리기
        import pandas as pd
        plt.figure(figsize=(5,3))
        X = np.arange(k)
        width = 0.35
        plt.bar(X - width/2, pred_scores, width, label=f"pred={pred}")
        plt.bar(X + width/2, true_scores, width, label=f"true={y}")
        plt.xticks(X, labs); plt.legend(); plt.tight_layout()
        base = os.path.join(args.out_dir,"misclf", os.path.basename(path).replace(".npz",""))
        plt.savefig(base+"_bars.png", dpi=140); plt.close()
        # 테이블 CSV
        rows=[]
        for r, tab in enumerate(top_tables):
            for rank,(c,score,pid) in enumerate(tab, 1):
                rows.append([int(idx[r]), rank, int(c), float(score), int(pid)])
        pd.DataFrame(rows, columns=["slot","rank","proto_cls","cos","proto_id"])\
          .to_csv(base+"_top3.csv", index=False)
        picked += 1
        if picked >= args.misclf_max: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=0.1)
    # proto space
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--per_proto_heatmap", type=int, default=1)
    ap.add_argument("--min_count_heat", type=int, default=30)
    # slot diversity
    ap.add_argument("--slot_topk", type=int, default=3)
    ap.add_argument("--q_iou", type=float, default=0.10)
    # duplicates
    ap.add_argument("--dup_radius", type=float, default=2.0)
    ap.add_argument("--dup_min_frac", type=float, default=0.60)
    ap.add_argument("--dup_pmin", type=float, default=0.05)
    # misclf
    ap.add_argument("--proto_tau", type=float, default=0.5)
    ap.add_argument("--misclf_max", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_proto_space(args)
    report_slot_diversity(args)
    report_duplicates(args)
    report_misclf(args)
    print(f"[report] saved → {args.out_dir}")

if __name__ == "__main__":
    main()
