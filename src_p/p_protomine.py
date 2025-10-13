# -*- coding: utf-8 -*-
# src_o/o_protomine.py
"""
EMNIST (o-stage, s-only) 프로토타입 마이닝
- 입력: o_dump.py 가 생성한 *.npz (필수: S_slots, slot_prob, clazz)
- 출력: per_class 프로토타입 JSON (load_prototypes와 호환)
"""

import os, json, glob, argparse, math
import numpy as np
from tqdm import tqdm


def iter_dump_items(dump_dir):
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    for p in files:
        z = np.load(p, allow_pickle=True)
        for k in ("S_slots", "slot_prob", "clazz"):
            if k not in z.files:
                raise KeyError(f"{p} lacks '{k}'")
        yield p, z


def collect_class_features(dump_dir, slots_topk=5, only_correct=False, min_prob=0.0):
    per = {}
    warned_pred = False
    for p, z in tqdm(iter_dump_items(dump_dir), desc="[collect]"):
        y = int(z["clazz"])
        S = np.asarray(z["S_slots"], np.float32)    # (M,D)
        P = np.asarray(z["slot_prob"], np.float32)  # (M,)

        use_this = True
        if only_correct:
            pred_ok = False
            if "pred" in z.files:
                pr = int(z["pred"])
                if "logits" in z.files and np.abs(np.asarray(z["logits"])).sum() > 1e-6:
                    pred_ok = True
                    use_this = (pr == y)
            if not pred_ok and not warned_pred:
                print("[o_protomine] only_correct=1 이지만 dump에 유효한 예측 정보가 없어 전체 샘플 사용")
                warned_pred = True
        if not use_this:
            continue

        M = P.shape[0]
        k = min(slots_topk if slots_topk > 0 else M, M)
        top_idx = np.argpartition(-P, kth=k-1)[:k]
        top_idx = top_idx[np.argsort(-P[top_idx])]
        sel = top_idx[P[top_idx] >= float(min_prob)]

        X = S[sel]   # (K',D)
        W = P[sel]   # (K',)

        if y not in per:
            per[y] = {"X": [], "W": []}
        per[y]["X"].append(X)
        per[y]["W"].append(W)

    for c in list(per.keys()):
        X = np.concatenate(per[c]["X"], axis=0) if per[c]["X"] else np.zeros((0, 1), np.float32)
        W = np.concatenate(per[c]["W"], axis=0) if per[c]["W"] else np.zeros((0,), np.float32)
        if X.shape[0] == 0:
            del per[c]
            continue
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / n
        W = np.clip(W, 0.0, None).astype(np.float32)
        W = (W / W.sum()) if W.sum() > 0 else np.ones_like(W) / max(1, W.size)
        per[c]["X"] = X.astype(np.float32)
        per[c]["W"] = W.astype(np.float32)
    return per


def _kmeans_pp_init(X, K, rng):
    N, D = X.shape
    centers = np.empty((K, D), dtype=np.float32)
    i0 = rng.integers(0, N)
    centers[0] = X[i0]
    dist2 = np.sum((X - centers[0])**2, axis=1)
    for k in range(1, K):
        probs = dist2 / (dist2.sum() + 1e-8)
        i = rng.choice(N, p=probs)
        centers[k] = X[i]
        d2_new = np.sum((X - centers[k])**2, axis=1)
        dist2 = np.minimum(dist2, d2_new)
    return centers


def weighted_kmeans(X, W, K, iters=40, seed=123):
    N, D = X.shape
    K = int(min(max(1, K), N))
    if N == 0:
        return np.zeros((0, D), np.float32)
    rng = np.random.default_rng(seed)
    W = np.clip(W, 0.0, None)
    W = (W / W.sum()) if W.sum() > 0 else np.ones_like(W) / float(max(1, N))
    C = _kmeans_pp_init(X, K, rng)
    assign = np.zeros(N, dtype=np.int64)
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - C[None, :, :])**2, axis=2)
        assign_new = np.argmin(d2, axis=1)
        if np.array_equal(assign_new, assign):
            break
        assign = assign_new
        for k in range(K):
            mask = (assign == k)
            if not np.any(mask):
                j = rng.integers(0, N)
                C[k] = X[j]
                continue
            w = W[mask][:, None]
            xc = (w * X[mask]).sum(axis=0) / (w.sum() + 1e-8)
            n = np.linalg.norm(xc) + 1e-8
            C[k] = (xc / n).astype(np.float32)
    return C.astype(np.float32)


def choose_k_auto(N, kmax, floor=1):
    if N <= 0: return 0
    k = int(round(math.sqrt(max(1.0, N / 50.0))))
    return int(min(kmax, max(floor, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--slots_topk", type=int, default=5)
    ap.add_argument("--min_prob", type=float, default=0.0)
    ap.add_argument("--only_correct", type=int, default=0)
    ap.add_argument("--auto_tune", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    per = collect_class_features(
        dump_dir=args.dump_dir,
        slots_topk=args.slots_topk,
        only_correct=bool(args.only_correct),
        min_prob=args.min_prob,
    )

    out = {"per_class": {}, "meta": {
        "feature": "S", "normalize": "l2",
        "kmax": int(args.kmax), "slots_topk": int(args.slots_topk),
        "only_correct": int(args.only_correct), "auto_tune": int(args.auto_tune),
    }}

    for c in sorted(per.keys()):
        X = per[c]["X"]; W = per[c]["W"]
        K = choose_k_auto(X.shape[0], args.kmax, floor=1) if args.auto_tune else min(args.kmax, max(1, X.shape[0]))
        C = weighted_kmeans(X, W, K=K, iters=40, seed=args.seed)
        out["per_class"][str(int(c))] = {"mu": [v.tolist() for v in C]}

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    tot_k = sum(len(out["per_class"][k]["mu"]) for k in out["per_class"])
    print(f"[o_protomine] saved: {args.out_json}  |  classes={len(out['per_class'])}, total_protos={tot_k}")


if __name__ == "__main__":
    main()
