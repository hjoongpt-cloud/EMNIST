# src_n/tools/n_slot_majority_audit.py
import os, json, argparse, heapq, itertools, csv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from src_m.tools.m_train import build_model, seed_all  # 기존 build_model/seed_all 재사용

try:
    from imageio import imwrite
    HAS_IMGIO = True
except Exception:
    HAS_IMGIO = False


# ----------------------------
# Data
# ----------------------------
def get_loader(split="test", batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), subset=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.EMNIST(root="./data", split="balanced", train=(split == "train"), download=True, transform=tf)
    if subset: ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def topq_mask_per_channel(t: torch.Tensor, q: float) -> torch.Tensor:
    """ t: (B,M,H,W) >=0 → 채널별 상위 q 비율 True """
    if q <= 0:  return torch.zeros_like(t, dtype=torch.bool)
    if q >= 1:  return torch.ones_like(t, dtype=torch.bool)
    B, M, H, W = t.shape
    flat = t.reshape(B, M, -1)
    k = max(1, int(round((H * W) * q)))
    k = min(k, H * W)
    idx = torch.topk(flat, k=k, dim=2).indices
    m = torch.zeros_like(flat, dtype=torch.bool)
    m.scatter_(2, idx, True)
    return m.view_as(t)


def color_overlay(gray_01: np.ndarray, heat_01: np.ndarray) -> np.ndarray:
    """ gray_01, heat_01: (H,W) in [0,1] """
    H, W = gray_01.shape
    base = np.stack([gray_01, gray_01, gray_01], axis=-1)
    base[..., 0] = np.clip(base[..., 0] + 0.7 * heat_01, 0, 1)
    return (base * 255).astype(np.uint8)


# ----------------------------
# Safe heap helpers (tie-safe)
# ----------------------------
_uid_counter = itertools.count()  # 전역 증가 id

def safe_heappush(heap, score: float, image: np.ndarray):
    uid = next(_uid_counter)
    # (score가 기준, tie시 uid로 안정 비교)
    heapq.heappush(heap, (float(score), uid, image))

def safe_heapreplace(heap, score: float, image: np.ndarray):
    uid = next(_uid_counter)
    heapq.heapreplace(heap, (float(score), uid, image))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)

    ap.add_argument("--q_act", type=float, default=0.02, help="슬롯별 활성 상위 비율")
    ap.add_argument("--q_need", type=float, default=0.02, help="슬롯별 필요성(양그라드) 상위 비율")

    ap.add_argument("--maj_frac", type=float, default=0.33, help="다수 공동활성 임계(슬롯수의 비율)")
    ap.add_argument("--min_cut", type=int, default=1, help="소수 공동활성 임계(절대값)")

    ap.add_argument("--top_tiles", type=int, default=48, help="슬롯별 몽타주 타일 수(각 세트)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load config
    if args.config.endswith(".json"):
        import json as _json
        with open(args.config, "r") as f:
            cfg = _json.load(f)
    else:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    mean = tuple(cfg.get("normalize", {}).get("mean", [0.1307]))
    std = tuple(cfg.get("normalize", {}).get("std", [0.3081]))

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    trunk.to(device).eval()
    head.to(device).eval()

    loader = get_loader(args.split, args.batch_size, args.num_workers, mean, std, args.subset)

    # probe M, Hm, Wm
    with torch.no_grad():
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        Z0, aux0 = trunk(x0)
        A0 = aux0["A_maps"]  # (B,M,Hm,Wm)
        M = A0.size(1)
        Hm, Wm = A0.size(2), A0.size(3)

    # accumulators
    act_count = np.zeros(M, dtype=np.int64)
    crowd_hits = np.zeros(M, dtype=np.int64)
    solo_hits = np.zeros(M, dtype=np.int64)
    pos_grad_hits = np.zeros(M, dtype=np.int64)

    heaps_MAJ_over = [[] for _ in range(M)]  # (score, uid, img)
    heaps_MIN_need = [[] for _ in range(M)]  # (score, uid, img)

    all_maj_counts = []
    all_need_counts = []

    pbar = tqdm(loader, desc="slot-audit")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        # for montage gray
        with torch.no_grad():
            x_vis = (x.detach().cpu().numpy() * np.array(std)[None, :, None, None] + np.array(mean)[None, :, None, None])
            x_vis = np.clip(x_vis, 0, 1)[:, 0]  # (B,H,W) in [0,1]

        # forward
        x.requires_grad_(True)
        Z, aux = trunk(x)
        A = aux["A_maps"]  # (B,M,Hm,Wm), should carry grad wrt x
        logits = head(Z)   # (B,C)

        # need = ReLU(∂logit_y / ∂A)
        sel = logits[torch.arange(x.size(0)), y].sum()
        gA = torch.autograd.grad(sel, A, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if gA is None:
            # 일부 구조에서 A가 선택 로짓에 직접 기여하지 않을 수도 있으므로 0으로 대체
            gA = torch.zeros_like(A)
        need = F.relu(gA)         # (B,M,Hm,Wm)
        act = F.relu(A)           # (B,M,Hm,Wm)

        # top-q masks
        m_need = topq_mask_per_channel(need, args.q_need)  # (B,M,Hm,Wm) bool
        m_act = topq_mask_per_channel(act, args.q_act)     # (B,M,Hm,Wm) bool

        # majority / minority counts over slots-per-pixel
        maj_count = m_act.sum(dim=1)      # (B,Hm,Wm)
        need_count = m_need.sum(dim=1)    # (B,Hm,Wm)
        all_maj_counts.append(maj_count.detach().cpu().numpy().reshape(-1))
        all_need_counts.append(need_count.detach().cpu().numpy().reshape(-1))

        maj_thresh = max(1, int(round(args.maj_frac * M)))
        is_crowd = (maj_count >= maj_thresh)    # many slots active at pixel
        is_solo = (maj_count <= args.min_cut)   # very few slots active

        # per-slot stats
        with torch.no_grad():
            m_act_c = m_act.detach().cpu()
            m_need_c = m_need.detach().cpu()
            crowd_c = is_crowd.detach().cpu()
            solo_c = is_solo.detach().cpu()
            for m in range(M):
                am = m_act_c[:, m]  # (B,Hm,Wm)
                n_act = int(am.sum().item())
                act_count[m] += n_act
                if n_act > 0:
                    crowd_hits[m] += int((am & crowd_c).sum().item())
                    solo_hits[m] += int((am & solo_c).sum().item())
                    pos_grad_hits[m] += int((am & m_need_c[:, m]).sum().item())

        # upsample for overlay
        act_u = F.interpolate(act, scale_factor=2, mode="bilinear", align_corners=False)   # (B,M,28,28)
        need_u = F.interpolate(need, scale_factor=2, mode="bilinear", align_corners=False) # (B,M,28,28)

        # collect tiles
        for b in range(x.size(0)):
            gimg = x_vis[b]  # (28,28)

            # A) crowd but NOT needed: (crowd) & (need_count == 0)
            mA = (is_crowd[b] & (need_count[b] == 0))
            idxs = mA.nonzero(as_tuple=False)
            if idxs.numel():
                # 해당 위치에서 가장 강한 act 슬롯 기준으로 저장
                for t in idxs.split(1, dim=0)[:128]:
                    i, j = int(t[0, 0]), int(t[0, 1])
                    m = int(torch.argmax(act[b, :, i, j]).item())
                    score = float(act[b, m, i, j].item())
                    heat = act_u[b, m].detach().cpu().numpy()
                    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                    img = color_overlay(gimg, heat)
                    if HAS_IMGIO:
                        heap = heaps_MAJ_over[m]
                        if len(heap) < args.top_tiles:
                            safe_heappush(heap, score, img)
                        else:
                            if score > heap[0][0]:
                                safe_heapreplace(heap, score, img)

            # B) solo BUT needed: (solo) & (need_count >= 1)
            mB = (is_solo[b] & (need_count[b] >= 1))
            idxs = mB.nonzero(as_tuple=False)
            if idxs.numel():
                for t in idxs.split(1, dim=0)[:128]:
                    i, j = int(t[0, 0]), int(t[0, 1])
                    m = int(torch.argmax(need[b, :, i, j]).item())
                    score = float(need[b, m, i, j].item())
                    heat = need_u[b, m].detach().cpu().numpy()
                    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                    img = color_overlay(gimg, heat)
                    if HAS_IMGIO:
                        heap = heaps_MIN_need[m]
                        if len(heap) < args.top_tiles:
                            safe_heappush(heap, score, img)
                        else:
                            if score > heap[0][0]:
                                safe_heapreplace(heap, score, img)

    # ------------- save -------------
    os.makedirs(args.out_dir, exist_ok=True)

    # histograms (majority / needcount)
# ---- 전역 분포 히스토그램 저장 ----
    if HAS_IMGIO and (len(all_maj_counts) > 0 or len(all_need_counts) > 0):
        # 안전하게 numpy로 병합
        maj  = np.concatenate(all_maj_counts, axis=0) if len(all_maj_counts)  else np.array([], dtype=np.int64)
        need = np.concatenate(all_need_counts, axis=0) if len(all_need_counts) else np.array([], dtype=np.int64)

        def save_bar_hist(arr, out_path):
            if arr.size == 0:
                return
            max_k = int(arr.max())
            bins = np.arange(0, max_k + 2)
            hist, _ = np.histogram(arr, bins=bins)

            # 아주 가벼운 바차트 캔버스(이미지)
            H = 200
            W = max(1, len(hist) * 4)
            canvas = np.zeros((H, W, 3), np.uint8)
            if hist.max() > 0:
                h = (hist / hist.max() * (H - 1)).astype(int)
                for i, v in enumerate(h):
                    canvas[H - v:, i * 4 : i * 4 + 3, :] = (255, 255, 255)
            imwrite(out_path, canvas)

        save_bar_hist(maj,  os.path.join(args.out_dir, "hist_majority.png"))
        save_bar_hist(need, os.path.join(args.out_dir, "hist_needcount.png"))


    # per-slot CSV + montages
    with open(os.path.join(args.out_dir, "slot_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "act_count", "crowd_hits", "solo_hits", "crowding_idx", "solitude_idx", "pos_grad_frac"])
        for m in range(M):
            crowding_idx = (crowd_hits[m] / act_count[m]) if act_count[m] > 0 else 0.0
            solitude_idx = (solo_hits[m] / act_count[m]) if act_count[m] > 0 else 0.0
            pos_frac = (pos_grad_hits[m] / act_count[m]) if act_count[m] > 0 else 0.0
            w.writerow([m, act_count[m], crowd_hits[m], solo_hits[m],
                        f"{crowding_idx:.6f}", f"{solitude_idx:.6f}", f"{pos_frac:.6f}"])

            # save montages
            for tag, heap in [("MAJ_over", heaps_MAJ_over[m]), ("MIN_need", heaps_MIN_need[m])]:
                if len(heap) == 0 or not HAS_IMGIO:
                    continue
                tiles = [t[2] for t in sorted(heap, key=lambda x: -x[0])]  # sort by score desc
                n = len(tiles)
                rows = (n + 7) // 8
                H, W, _ = tiles[0].shape
                canvas = np.zeros((rows * (H + 2) + 2, 8 * (W + 2) + 2, 3), np.uint8)
                yy, xx, col = 2, 2, 0
                for img in tiles:
                    canvas[yy : yy + H, xx : xx + W] = img
                    col += 1
                    xx += W + 2
                    if col == 8:
                        col = 0
                        xx = 2
                        yy += H + 2
                od = os.path.join(args.out_dir, f"slot_{m:03d}")
                os.makedirs(od, exist_ok=True)
                imwrite(os.path.join(od, f"{tag}.png"), canvas)

    print(f"[slot_majority_audit] saved → {args.out_dir}")


if __name__ == "__main__":
    main()
