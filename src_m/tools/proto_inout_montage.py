# src_n/tools/n_inout_montage.py
import os, json, glob, argparse, heapq
import numpy as np
from tqdm import tqdm

try:
    from imageio import imwrite
    HAS_IMGIO = True
except Exception:
    HAS_IMGIO = False


def load_protos(path):
    with open(path, "r") as f:
        data = json.load(f)
    per = data.get("per_class", {})
    C_list, C_cls = [], []
    for c_str, block in per.items():
        cid = int(c_str)
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                arr = np.asarray(mus, np.float32)
                if arr.ndim == 1:
                    C_list.append(arr); C_cls.append(cid)
                else:
                    for v in arr:
                        C_list.append(np.asarray(v, np.float32)); C_cls.append(cid)
            else:
                inner = block.get("protos") or block.get("clusters") or block.get("items") or []
                if isinstance(inner, list):
                    for p in inner:
                        v = p.get("mu") if isinstance(p, dict) else p
                        if v is None: continue
                        C_list.append(np.asarray(v, np.float32).reshape(-1)); C_cls.append(cid)
        elif isinstance(block, list):
            for p in block:
                v = p.get("mu") if isinstance(p, dict) else p
                if v is None: continue
                C_list.append(np.asarray(v, np.float32).reshape(-1)); C_cls.append(cid)
        else:
            C_list.append(np.asarray(block, np.float32).reshape(-1)); C_cls.append(cid)
    C = np.stack(C_list, 0) if len(C_list) > 0 else np.zeros((0, 2), np.float32)
    return C, np.asarray(C_cls, np.int64)


def build_feature_match_dim(S_slots, XY, target_dim, xy_weight=1.0):
    """
    target_dim ∈ {2, Dslot, Dslot+2}에 맞춰 슬롯 피처를 생성.
    """
    Dslot = None if S_slots is None else S_slots.shape[-1]
    if target_dim == 2:
        # XY만 정규화
        denom = np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8
        return XY / denom
    if (Dslot is not None) and target_dim == Dslot:
        denom = np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8
        return S_slots / denom
    if (Dslot is not None) and target_dim == Dslot + 2:
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        X = XY       / (np.linalg.norm(XY,       axis=-1, keepdims=True) + 1e-8)
        return np.concatenate([S, X * float(xy_weight)], axis=-1)
    raise ValueError(f"Cannot build feature of dim={target_dim} with Dslot={Dslot}.")


def overlay_sum(gray_01, heats_01_list):
    """
    여러 슬롯 히트맵(0..1)을 합쳐서 오버레이 (빨간 채널 강화).
    """
    H, W = gray_01.shape
    base = np.stack([gray_01, gray_01, gray_01], axis=-1)  # (H,W,3)
    if len(heats_01_list) > 0:
        heat = np.clip(np.sum(heats_01_list, axis=0), 0, 1)
        base[..., 0] = np.clip(base[..., 0] + 0.7 * heat, 0, 1)
    return (base * 255).astype(np.uint8)


def draw_points(rgb_img, xy_pixels, color=(0, 255, 255)):
    """
    xy_pixels: [(x,y), ...] in pixel coords, draw small 2x2 square.
    """
    H, W, _ = rgb_img.shape
    for (px, py) in xy_pixels:
        for dy in range(-1, 1):
            for dx in range(-1, 1):
                yy = np.clip(py + dy, 0, H - 1); xx = np.clip(px + dx, 0, W - 1)
                rgb_img[yy, xx, 0] = color[0]
                rgb_img[yy, xx, 1] = color[1]
                rgb_img[yy, xx, 2] = color[2]
    return rgb_img


def upsample_2x(a):
    # a: (H,W) -> (2H,2W) nearest repeat
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--feature_mode", default="s+xy", choices=["s", "xy", "s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)

    ap.add_argument("--slots_topk", type=int, default=3,
                    help="한 타일에 겹쳐서 보여줄 슬롯 개수(프로토타입별로 선택된 슬롯 중 Top-K)")
    ap.add_argument("--energy_source", default="softmax", choices=["softmax", "topq", "mass"],
                    help="슬롯 선택 기준: softmax(slot_prob) / topq(각 슬롯 top-q 픽셀 합) / mass(슬롯 총 질량)")
    ap.add_argument("--q", type=float, default=0.02, help="energy_source=topq 일 때 픽셀 상위 비율")
    ap.add_argument("--keep_frac", type=float, default=0.8, help="샘플 내 상위 에너지 슬롯 비율(0~1)")

    ap.add_argument("--topN", type=int, default=32, help="프로토타입당 IN/OUT 타일 개수")
    ap.add_argument("--only_correct", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 프로토타입 로드
    C, C_cls = load_protos(args.proto_json)  # (K,d), (K,)
    K, d = C.shape
    if K == 0:
        print("[n_inout_montage] no prototypes.")
        return

    # 2) 덤프 로드 + per-sample 처리
    paths = sorted(glob.glob(os.path.join(args.dump_dir, "*.npz")))
    inbox = []
    it = tqdm(paths, desc="collect", disable=not args.progress)
    for p in it:
        z = np.load(p, allow_pickle=True)

        y = int(z["clazz"])
        pred = int(z.get("pred", y))
        if args.only_correct and (pred != y):
            continue

        XY = z["XY"].astype(np.float32)              # (M,2)
        S = z["S_slots"].astype(np.float32) if ("S_slots" in z) else None
        msk = z["slot_mask"].astype(np.float32)      # (M,)
        A = z["A_maps"].astype(np.float32)           # (M,H,W)
        img = z["image"].astype(np.uint8)            # (28,28)

        # 에너지 소스 선택
        if args.energy_source == "softmax":
            if "slot_prob" in z:
                e = z["slot_prob"].astype(np.float32)  # (M,)
            elif "energy_norm" in z:
                e = z["energy_norm"].astype(np.float32)
            else:
                e = A.reshape(A.shape[0], -1).sum(-1).astype(np.float32)
        elif args.energy_source == "mass":
            e = A.reshape(A.shape[0], -1).sum(-1).astype(np.float32)
        else:  # topq
            M, H, W = A.shape
            flat = A.reshape(M, -1)
            k = max(1, int(round(H * W * float(args.q))))
            idx = np.argpartition(flat, -k, axis=1)[:, -k:]
            topq_sum = np.take_along_axis(flat, idx, axis=1).sum(axis=1)
            e = topq_sum.astype(np.float32)

        e = e * msk  # 죽은 슬롯 제거
        # 상위 keep_frac 슬롯 인덱스
        keep = max(1, int(round(len(e) * np.clip(args.keep_frac, 0.0, 1.0))))
        idx_keep = np.argsort(-e)[:keep]

        # 프로토타입 차원에 맞게 슬롯 피처 생성
        try:
            F_all = build_feature_match_dim(S, XY, target_dim=d, xy_weight=args.xy_weight)  # (M,d)
        except Exception:
            # 차원 안 맞으면 스킵
            continue

        Fk = F_all[idx_keep]      # (m',d)
        Ak = A[idx_keep]          # (m',H,W)
        XYk = XY[idx_keep]        # (m',2)

        # 업샘플 히트맵 준비(28x28)
        if "A_upsampled" in z:
            Aup = z["A_upsampled"].astype(np.float32)  # (M,28,28)
            Ak_up = Aup[idx_keep]
            # normalize 0..1 per slot
            Ak_up = (Ak_up - Ak_up.min(axis=(1,2), keepdims=True)) / (Ak_up.max(axis=(1,2), keepdims=True) - Ak_up.min(axis=(1,2), keepdims=True) + 1e-8)
        else:
            Ak_up = []
            for a in Ak:
                up = upsample_2x(a)
                up = up - up.min()
                up = up / (up.max() + 1e-8)
                Ak_up.append(up)
            Ak_up = np.stack(Ak_up, 0)  # (m',28,28)

        inbox.append((y, img, Fk, Ak_up, XYk))

    # 3) 프로토타입별 우선큐 (타일에는 여러 슬롯을 겹쳐 렌더)
    ins = [ [] for _ in range(K) ]   # (score, tiebreaker, image)
    outs = [ [] for _ in range(K) ]
    counter = 0

    for (y, img, Fk, Ak_up, XYk) in tqdm(inbox, desc="assign", disable=not args.progress):
        if Fk.shape[-1] != d or Fk.shape[0] == 0:
            continue

        # 모든 슬롯 ↔ 모든 프로토타입 거리
        dists = np.linalg.norm(Fk[:, None, :] - C[None, :, :], axis=-1)  # (m',K)
        nn = np.argmin(dists, axis=1)          # (m',)
        dd = dists[np.arange(Fk.shape[0]), nn] # (m',)

        # 각 프로토타입마다 이 샘플에서 매칭된 슬롯들 모으기
        for k in np.unique(nn):
            sel = np.where(nn == k)[0]
            if sel.size == 0:
                continue
            # 이 프로토타입과 가장 가까운 슬롯들 Top-K
            order = sel[np.argsort(dd[sel])]              # 가까운 순
            pick = order[:max(1, int(args.slots_topk))]   # 최소 1

            # 히트맵 합성
            heats = [Ak_up[i] for i in pick]
            gray = (img.astype(np.float32) / 255.0)
            tile = overlay_sum(gray, heats)

            # XY 점도 표시(시각적 근거)
            pts = []
            for i in pick:
                cx = int(np.clip(XYk[i, 0] * 28, 0, 27))
                cy = int(np.clip(XYk[i, 1] * 28, 0, 27))
                pts.append((cx, cy))
            tile = draw_points(tile, pts, color=(0, 255, 255))

            # 랭킹 점수: 이 프로토타입과의 최소 거리(작을수록 좋음)
            score = -float(dd[pick[0]])
            entry = (score, counter, tile); counter += 1

            heap = ins[k] if (y == int(C_cls[k])) else outs[k]
            if len(heap) < args.topN:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    # 4) 저장
    kept = 0
    for k in range(K):
        tiles_in  = [t[2] for t in sorted(ins[k],  key=lambda x: -x[0])]
        tiles_out = [t[2] for t in sorted(outs[k], key=lambda x: -x[0])]
        if len(tiles_in) == 0 and len(tiles_out) == 0:
            continue
        od = os.path.join(args.out_dir, f"c{int(C_cls[k]):02d}_k{k:04d}")
        os.makedirs(od, exist_ok=True)
        if HAS_IMGIO and len(tiles_in) > 0:
            n = len(tiles_in); rows = (n + 7) // 8
            H, W, _ = tiles_in[0].shape
            canvas = np.zeros((rows*(H+2)+2, 8*(W+2)+2, 3), np.uint8)
            y = 2; x = 2; col = 0
            for im in tiles_in:
                canvas[y:y+H, x:x+W] = im
                col += 1; x += W+2
                if col == 8:
                    col = 0; x = 2; y += H+2
            imwrite(os.path.join(od, "IN.png"), canvas)
        if HAS_IMGIO and len(tiles_out) > 0:
            n = len(tiles_out); rows = (n + 7) // 8
            H, W, _ = tiles_out[0].shape
            canvas = np.zeros((rows*(H+2)+2, 8*(W+2)+2, 3), np.uint8)
            y = 2; x = 2; col = 0
            for im in tiles_out:
                canvas[y:y+H, x:x+W] = im
                col += 1; x += W+2
                if col == 8:
                    col = 0; x = 2; y += H+2
            imwrite(os.path.join(od, "OUT.png"), canvas)
        kept += 1

    print(f"[n_inout_montage] saved → {args.out_dir}")
    print(f"[n_inout_montage] kept {kept}/{K} prototypes")
    

if __name__ == "__main__":
    main()
