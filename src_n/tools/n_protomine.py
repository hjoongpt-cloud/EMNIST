# src_n/tools/n_protomine.py
import os, json, glob, argparse
import numpy as np
from tqdm import tqdm

def build_feature(S, XY, mode="s+xy", xy_weight=1.0):
    if S is None and mode != "xy":
        raise ValueError("S_slots is required for mode != 'xy'")
    if mode == "xy":
        f = XY
    elif mode == "s":
        f = S
    else:
        # L2-normalize then concat
        S_ = S / (np.linalg.norm(S, axis=-1, keepdims=True)+1e-8)
        X_ = XY/ (np.linalg.norm(XY,axis=-1,keepdims=True)+1e-8)
        f  = np.concatenate([S_, X_ * float(xy_weight)], axis=-1)
    return f.astype(np.float32)

def iter_dump_items(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    for p in paths:
        z = np.load(p, allow_pickle=True)
        need = ["clazz","pred","XY","slot_prob","slot_mask","image"]
        for k in need:
            if k not in z: raise KeyError(f"{p} lacks '{k}'")
        item = {
            "path": p,
            "clazz": int(z["clazz"]),
            "pred": int(z.get("pred", z["clazz"])),
            "XY": z["XY"].astype(np.float32),
            "P": (z["slot_prob"].astype(np.float32) * z["slot_mask"].astype(np.float32)),
            "S_slots": z["S_slots"].astype(np.float32) if "S_slots" in z else None,
            "image": z["image"].astype(np.uint8),
        }
        yield item

def top_slot_feature(item, mode, xy_weight):
    S = item["S_slots"]; XY = item["XY"]; P = item["P"]
    m = int(np.argmax(P))  # 샘플당 대표 슬롯
    if mode == "xy":
        return XY[m]
    elif mode == "s":
        return S[m]
    else:
        S_ = S[m] / (np.linalg.norm(S[m])+1e-8)
        X_ = XY[m]/ (np.linalg.norm(XY[m])+1e-8)
        return np.concatenate([S_, X_ * float(xy_weight)], axis=0)

def kmeans_lloyd(X, K, iters=30, seed=0):
    # 간단한 kmeans (numpy만 사용) – kmeans++ 초기화
    rng = np.random.default_rng(seed)
    N, D = X.shape
    centers = np.empty((K, D), np.float32)
    idx0 = rng.integers(0, N)
    centers[0] = X[idx0]
    dist2 = np.sum((X - centers[0])**2, axis=1)
    for k in range(1, K):
        probs = dist2 / (dist2.sum()+1e-8)
        centers[k] = X[rng.choice(N, p=probs)]
        dist2 = np.minimum(dist2, np.sum((X - centers[k])**2, axis=1))
    assign = np.zeros(N, np.int32)
    for _ in range(iters):
        d2 = ((X[:,None,:]-centers[None,:,:])**2).sum(-1) # (N,K)
        assign_new = np.argmin(d2, axis=1)
        if np.all(assign_new==assign): break
        assign = assign_new
        for k in range(K):
            idx = np.where(assign==k)[0]
            if idx.size>0: centers[k] = X[idx].mean(0)
    return centers, assign

def mine_per_class(Xc, kmax, auto_tune):
    """Xc: [Nc,d] 해당 클래스의 대표 슬롯 피처"""
    Nc = Xc.shape[0]
    if Nc == 0:
        return []
    K = min(kmax, max(1, int(np.sqrt(Nc/8))))  # 대략 초깃값
    centers, assign = kmeans_lloyd(Xc, K=K, iters=40, seed=0)
    # 클러스터 크기 기반 가지치기 (통계적)
    sizes = np.bincount(assign, minlength=K)
    mu, sd = sizes.mean(), sizes.std()
    thr = max(2, int(mu - sd)) if auto_tune else 2
    keep = [k for k in range(K) if sizes[k] >= thr]
    out = [centers[k].tolist() for k in keep]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)
    ap.add_argument("--only_correct", type=int, default=1)
    ap.add_argument("--auto_tune", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    # 수집
    feats_by_c = {}
    it = iter_dump_items(args.dump_dir)
    it = tqdm(it, desc="collect", disable=not args.progress)
    for item in it:
        if args.only_correct and (item["clazz"] != item["pred"]):
            continue
        try:
            f = top_slot_feature(item, args.feature_mode, args.xy_weight)
        except Exception:
            continue
        c = int(item["clazz"])
        feats_by_c.setdefault(c, []).append(f.astype(np.float32))

    # 채굴
    per_class = {}
    for c, lst in tqdm(feats_by_c.items(), desc="mine", disable=not args.progress):
        Xc = np.stack(lst, axis=0)
        mus = mine_per_class(Xc, args.kmax, auto_tune=bool(args.auto_tune))
        per_class[str(c)] = {"mu": mus}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"per_class": per_class}, f, indent=2)
    print(f"[n_protomine] saved → {args.out_json}")

if __name__ == "__main__":
    main()
