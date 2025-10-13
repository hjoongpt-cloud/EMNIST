# src_m/tools/proto_mine.py
import os, json, argparse, glob
import numpy as np
from tqdm import tqdm

# ===================== 공통 유틸 (dump reader / feature builder) =====================

def iter_dump_items(dump_dir):
    """
    새 스키마 호환:
      - 필수: clazz, image, XY, slot_mask, (slot_prob 또는 energy_norm)
      - 선택: pred, S_slots, A_upsampled, energy_raw, slot_mass, logits
    산출 item 키:
      clazz(int), pred(int), path(str),
      XY:(M,2), P:(M,), slot_mask:(M,), S_slots:(M,D) or None,
      image:(28,28), A_upsampled:(M,28,28) or None
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz in {dump_dir}")
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for k in ["clazz", "XY", "slot_mask", "image"]:
            if k not in z: raise KeyError(f"{p} lacks '{k}'")

        y    = int(z["clazz"])
        pred = int(z["pred"]) if "pred" in z else y

        if "slot_prob" in z:
            P = z["slot_prob"].astype(np.float32)
        elif "energy_norm" in z:
            P = z["energy_norm"].astype(np.float32)
        else:
            raise KeyError(f"{p} lacks 'slot_prob'/'energy_norm'")

        yield {
            "path": p,
            "clazz": y,
            "pred": pred,
            "XY": z["XY"].astype(np.float32),
            "P":  P.astype(np.float32),
            "slot_mask": z["slot_mask"].astype(np.float32),
            "image": z["image"].astype(np.uint8),
            "A_upsampled": z["A_upsampled"].astype(np.float32) if "A_upsampled" in z else None,
            "S_slots": z["S_slots"].astype(np.float32) if "S_slots" in z else None,
        }

def build_feature_match_dim(S_slots, XY, target_dim, xy_weight=1.0):
    """
    프로토타입 벡터 차원(target_dim)에 맞춰 슬롯 특징을 생성.
    - S_slots가 None일 수 있음 → XY-only 모드만 가능
    - target_dim ∈ {2, Dslot, Dslot+2}
    """
    if target_dim == 2:
        xy = XY / (np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8)
        return xy

    if S_slots is None:
        raise ValueError("Prototypes expect S or S+XY, but S_slots is missing. Re-dump with --save_s_slots.")

    Dslot = S_slots.shape[-1]
    if target_dim == Dslot:
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        return S

    if target_dim == (Dslot + 2):
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        X = XY       / (np.linalg.norm(XY,       axis=-1, keepdims=True) + 1e-8)
        return np.concatenate([S, X * float(xy_weight)], axis=-1)

    raise ValueError(f"Unsupported target_dim={target_dim} (Dslot={None if S_slots is None else S_slots.shape[-1]})")


# ===================== 핵심 로직 =====================

def build_candidates_per_class(dump_dir, top_p=0.9, only_correct=True, progress=False):
    """
    반환: dict[cid] = list of records {XY:(2,), P:float, S:(D,) or None}
    - 게이팅: slot 확률 P 내림차순 누적합이 top_p 이하인 슬롯만 사용
    """
    per = {}
    it = iter_dump_items(dump_dir)
    it = tqdm(it, desc="collect", disable=not progress)
    for item in it:
        if only_correct and (item["clazz"] != item["pred"]):
            continue
        y, XY, P, S = int(item["clazz"]), item["XY"], item["P"], item["S_slots"]

        idx = np.argsort(-P)  # desc
        if top_p >= 1.0:
            keep = idx
        else:
            csum = np.cumsum(P[idx])
            k = max(1, int((csum <= (top_p * (P.sum() + 1e-8))).sum()))
            keep = idx[:k]

        lst = per.setdefault(y, [])
        for m in keep:
            lst.append({"XY": XY[m], "P": float(P[m]), "S": (None if S is None else S[m])})
    return per


def stack_features_for_class(c_list, target_dim, xy_weight):
    """
    c_list: [{XY, P, S}, ...]
    return: F: [N, target_dim], W: [N], meta dict
    """
    N = len(c_list)
    if N == 0:
        return np.zeros((0, target_dim), np.float32), np.zeros((0,), np.float32), {}

    has_S = (c_list[0]["S"] is not None)
    XY = np.stack([e["XY"] for e in c_list], 0)  # [N,2]
    W  = np.asarray([e["P"] for e in c_list], np.float32)

    if target_dim == 2:
        F = build_feature_match_dim(None, XY, 2)
    else:
        if not has_S:
            raise ValueError("target_dim requires S_slots; re-dump with --save_s_slots.")
        S = np.stack([e["S"] for e in c_list], 0)  # [N,D]
        F = build_feature_match_dim(S, XY, target_dim, xy_weight)

    return F.astype(np.float32), W, {"N": N, "P_sum": float(W.sum())}


def simple_kmeans(X, K, iters=50, seed=0, W=None):
    """
    가중치 W(선택)를 지원하는 간단 k-means. sklearn 없이 동작.
    X: [N,d], W: [N] or None
    """
    rng = np.random.RandomState(seed)
    N, d = X.shape
    K = min(K, max(1, N))
    # kmeans++ 초기화(간소화)
    centers = np.empty((K, d), np.float32)
    centers[0] = X[rng.randint(0, N)]
    dist = ((X - centers[0])**2).sum(-1)
    for k in range(1, K):
        probs = dist / (dist.sum() + 1e-8)
        idx = rng.choice(N, p=probs)
        centers[k] = X[idx]
        dist = np.minimum(dist, ((X - centers[k])**2).sum(-1))

    # 반복
    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(-1)  # [N,K]
        lab = np.argmin(d2, axis=1)
        # update
        new_cent = np.zeros_like(centers)
        for k in range(K):
            sel = (lab == k)
            if not np.any(sel):
                new_cent[k] = centers[k]
            else:
                if W is None:
                    new_cent[k] = X[sel].mean(0)
                else:
                    ww = W[sel][:, None]
                    new_cent[k] = (X[sel] * ww).sum(0) / (ww.sum(0) + 1e-8)
        if np.allclose(new_cent, centers):
            centers = new_cent
            break
        centers = new_cent
    sizes = np.bincount(lab, minlength=K)
    return centers, lab, sizes


def auto_select_clusters(centers, labels, sizes, min_count=None):
    """
    너무 작은 클러스터 제거:
      - 기준1: size >= max(min_count, floor(mean-1*std))
      - 기준2(보너스): 0개가 되면 상위 size 순으로 1개는 남김
    """
    K = centers.shape[0]
    if K == 0:
        return centers

    m, s = float(sizes.mean()), float(sizes.std())
    thr = int(np.floor(m - s))
    thr = max(thr, 1)
    if min_count is not None:
        thr = max(thr, int(min_count))

    keep = [i for i in range(K) if sizes[i] >= thr]
    if len(keep) == 0:
        keep = [int(np.argmax(sizes))]

    return centers[keep]


def save_proto_json(path, per_class_centers):
    out = {"per_class": {}}
    for c, Cc in per_class_centers.items():
        out["per_class"][str(int(c))] = {"mu": [[float(x) for x in row] for row in Cc]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[proto_mine] saved → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)

    ap.add_argument("--only_correct", type=int, default=1)
    ap.add_argument("--auto_tune", type=int, default=1)
    ap.add_argument("--report_json", type=str, default=None)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    # 1) 후보 수집
    per_raw = build_candidates_per_class(
        args.dump_dir,
        top_p=args.top_p,
        only_correct=bool(args.only_correct),
        progress=args.progress
    )

    # 2) target_dim 결정
    #   - xy → 2
    #   - s  → D
    #   - s+xy → D+2
    # D 추출: 첫 클래스 첫 항목의 S 길이 사용
    has_S = None
    Dslot = None
    for lst in per_raw.values():
        if len(lst) == 0: continue
        has_S = (lst[0]["S"] is not None)
        if has_S: Dslot = int(len(lst[0]["S"]))
        break
    if args.feature_mode == "xy":
        target_dim = 2
    elif args.feature_mode == "s":
        if not has_S: raise ValueError("feature_mode=s requires S_slots. Re-dump with --save_s_slots.")
        target_dim = int(Dslot)
    else:  # s+xy
        if not has_S: raise ValueError("feature_mode=s+xy requires S_slots. Re-dump with --save_s_slots.")
        target_dim = int(Dslot) + 2

    # 3) 클래스별 클러스터링
    per_class_centers = {}
    report = {"classes": []}

    for c in sorted(per_raw.keys()):
        c_list = per_raw[c]
        if len(c_list) == 0:
            per_class_centers[c] = np.zeros((0, target_dim), np.float32)
            continue

        F, W, meta = stack_features_for_class(c_list, target_dim, args.xy_weight)
        K0 = min(args.kmax, max(1, F.shape[0]))
        centers, labels, sizes = simple_kmeans(F, K0, iters=60, seed=0, W=W)

        if args.auto_tune:
            # min_count: 전체 샘플의 ~0.2% 또는 10 중 큰 값
            min_count = max(10, int(0.002 * F.shape[0]))
            centers = auto_select_clusters(centers, labels, sizes, min_count=min_count)

        per_class_centers[c] = centers.astype(np.float32)

        report["classes"].append({
            "class": int(c),
            "N": int(F.shape[0]),
            "k_init": int(K0),
            "k_final": int(centers.shape[0]),
            "size_mean": float(sizes.mean()),
            "size_std": float(sizes.std()),
        })

    # 4) 저장
    save_proto_json(args.out_json, per_class_centers)

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[proto_mine] report → {args.report_json}")


if __name__ == "__main__":
    main()
