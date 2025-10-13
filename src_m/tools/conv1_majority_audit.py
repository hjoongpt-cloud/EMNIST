# src_m/tools/conv1_majority_audit.py
import os, json, argparse, heapq
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as tvutils
from tqdm import tqdm

from src_m.tools.m_train import build_model, seed_all

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from imageio import imwrite
    HAS_IMGIO = True
except Exception:
    HAS_IMGIO = False


# ---------------------------
# Loader
# ---------------------------
def get_loader(split="test", batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), subset=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.EMNIST(root="./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    if subset: ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# ---------------------------
# Receptive Field → 입력 패치 좌표
# ---------------------------
def rf_patch_xy(i, j, ksz, stride, pad, dilation):
    di, dj = dilation
    kh, kw = ksz
    sh, sw = stride
    ph, pw = ( (kh-1)*di + 1, (kw-1)*dj + 1 )
    y0 = i*sh - pad[0]
    x0 = j*sw - pad[1]
    return int(x0), int(y0), int(pw), int(ph)


def safe_crop(img, x0, y0, w, h):
    H, W = img.shape[-2:]
    x1, y1 = max(0,x0), max(0,y0)
    x2, y2 = min(W, x0+w), min(H, y0+h)
    if x2<=x1 or y2<=y1:
        return img[..., H//2:H//2+1, W//2:W//2+1]
    return img[..., y1:y2, x1:x2]


def make_montage(patches, nrow=8):
    if len(patches)==0:
        return None
    grid = tvutils.make_grid(torch.cat(patches, dim=0), nrow=nrow, padding=1)
    img = (grid*255.0).clamp(0,255).byte().cpu().numpy().transpose(1,2,0)
    return img


def topq_mask_per_channel(t, q):
    # t: (B,K,H,W) >=0
    # 각 채널(K)별로 상위 q 비율을 True
    if q <= 0:  return torch.zeros_like(t, dtype=torch.bool)
    if q >= 1:  return torch.ones_like(t, dtype=torch.bool)
    B,K,H,W = t.shape
    t2 = t.reshape(B,K,-1)
    k = max(1, int(round((H*W)*q)))
    idx = torch.topk(t2, k=min(k, H*W), dim=2).indices
    m = torch.zeros_like(t2, dtype=torch.bool)
    m.scatter_(2, idx, True)
    return m.view_as(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--split", default="test", choices=["train","test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)

    # 임계 퍼센트(채널별)
    ap.add_argument("--q_act",   type=float, default=0.02, help="활성 상위 비율로 act mask 생성")
    ap.add_argument("--q_need",  type=float, default=0.02, help="양(+) 기여 그라디언트 상위 비율로 need mask 생성")

    # majority/minority 기준(공동활성 수 K중 몇 개 이상/이하)
    ap.add_argument("--maj_frac", type=float, default=0.33, help="다수공동활성 기준(예: 0.33*K 이상)")
    ap.add_argument("--min_cut",  type=int,   default=1,    help="소수공동활성 기준(예: 1개 이하)")

    ap.add_argument("--top_tiles", type=int, default=48,    help="각 세트 몽타주 타일 최소 수")
    ap.add_argument("--top_imgs",  type=int, default=40,    help="discordance 이미지 그리드 수(각 세트)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- cfg & model
    if args.config.endswith(".json"):
        import json as _json
        with open(args.config, "r") as f: cfg = _json.load(f)
    else:
        import yaml
        with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    mean = tuple(cfg.get("normalize",{}).get("mean", [0.1307]))
    std  = tuple(cfg.get("normalize",{}).get("std",  [0.3081]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    trunk.to(device).eval()
    head.to(device).eval()

    # ---- conv1 찾기
    conv1 = None
    for m in trunk.modules():
        if isinstance(m, nn.Conv2d):
            conv1 = m
            break
    assert conv1 is not None, "Conv2d(conv1)를 찾지 못했습니다."
    ksz     = conv1.kernel_size
    stride  = conv1.stride
    pad     = conv1.padding
    dilation= conv1.dilation
    K = conv1.weight.size(0)

    loader = get_loader(args.split, args.batch_size, args.num_workers, mean, std, args.subset)

    # ---- hook
    saved_fmap = {}
    def hook(module, inp, out):
        saved_fmap["fmap"] = out
    h = conv1.register_forward_hook(hook)

    # ---- 컨테이너
    # 필터별 지표
    crowd_count = np.zeros(K, dtype=np.int64)   # 다수공동 활성에서 해당 필터도 활성
    solo_count  = np.zeros(K, dtype=np.int64)   # 소수/고립 활성에서 해당 필터도 활성
    act_count   = np.zeros(K, dtype=np.int64)   # 전체 활성 수
    pos_grad_in_act = np.zeros(K, dtype=np.int64)  # 활성 중 pos-grad 비율 계산용

    # 타일 힙
    heaps_MAJ_over = [ [] for _ in range(K) ]  # 다수공동 & need 낮음
    heaps_MIN_need = [ [] for _ in range(K) ]  # 소수/고립 & need 높음

    # 전역 분포 수집
    all_maj_counts = []
    all_need_counts = []

    # 디스코던스 이미지 후보(원본 이미지 저장)
    #   High-MAJ / Low-margin  vs  Low-MAJ / High-margin
    discord_pool = []  # (score, sign, img_tensor) sign=+1(highMAJ/lowMargin), -1(lowMAJ/highMargin)
    max_pool = args.top_imgs*3

    pbar = tqdm(loader, desc="maj-min-audit")
    for x, y in pbar:
        x = x.to(device).requires_grad_(True)
        y = y.to(device)

        # 시각화용 복원된 이미지
        with torch.no_grad():
            x_vis = (x.detach().cpu().numpy() * np.array(std)[None,:,None,None] + np.array(mean)[None,:,None,None])
            x_vis = np.clip(x_vis, 0, 1)  # (B,1,H,W)

        # FWD
        saved_fmap.clear()
        Z, _ = trunk(x)
        fmap = saved_fmap["fmap"]                # (B,K,H1,W1)
        logits = head(Z)                         # (B,C)
        B, _, H1, W1 = fmap.shape

        # 필요성(양+ grad) & 활성(>0) 정규화
        sel = logits[torch.arange(B), y].sum()
        grads = torch.autograd.grad(sel, fmap, retain_graph=False, create_graph=False)[0]
        need = F.relu(grads)                     # (B,K,H1,W1)
        act  = F.relu(fmap)                      # (B,K,H1,W1)

        m_need = topq_mask_per_channel(need, args.q_need)
        m_act  = topq_mask_per_channel(act,  args.q_act)
        m_should = m_need  # “필요성”만으로도 충분, 원하면 m_need&m_content로 바꿀 수 있음

        # majority/minority count (픽셀별)
        maj_count  = m_act.sum(dim=1)           # (B,H1,W1)  몇 개 필터가 동시에 켜졌는가
        need_count = m_need.sum(dim=1)          # (B,H1,W1)
        all_maj_counts.append(maj_count.detach().cpu().reshape(-1))
        all_need_counts.append(need_count.detach().cpu().reshape(-1))

        # 이미지 레벨 통계(중앙값 공동활성 수 + 마진)
        with torch.no_grad():
            maj_med = torch.median(maj_count.float().view(B,-1), dim=1).values  # (B,)
            top2 = torch.topk(logits, k=2, dim=1).values
            margins = top2[:,0]-top2[:,1]   # 정답 확신도 근사
            # 케이스 A: 공동활성 높으나 마진 낮음 (오답/불확실) → +1
            scoreA = ( maj_med / max(1, K) ) - torch.sigmoid(margins)
            # 케이스 B: 공동활성 낮으나 마진 높음 → -1
            scoreB = torch.sigmoid(margins) - ( maj_med / max(1, K) )
            for b in range(B):
                img = torch.from_numpy(x_vis[b,0]).unsqueeze(0)  # (1,H,W) float
                # Pool 관리(상위만 유지)
                for sc, sign in [(float(scoreA[b]), +1), (float(scoreB[b]), -1)]:
                    if len(discord_pool) < max_pool:
                        heapq.heappush(discord_pool, (sc, sign, img))
                    else:
                        if sc > discord_pool[0][0]:
                            heapq.heapreplace(discord_pool, (sc, sign, img))

        # 필터별 crowd/solo/pos-grad
        maj_thresh = max(1, int(round(args.maj_frac*K)))
        is_crowd = (maj_count >= maj_thresh)   # (B,H1,W1)
        is_solo  = (maj_count <= args.min_cut)

        # pos-grad in active?
        pos_grad = m_need  # 양(+) 기여로 정의

        # 채널 루프 (가벼운 통계만)
        with torch.no_grad():
            m_act_cpu  = m_act.detach().cpu()
            pos_cpu    = pos_grad.detach().cpu()
            crowd_cpu  = is_crowd.detach().cpu()
            solo_cpu   = is_solo.detach().cpu()

            for k in range(K):
                act_k = m_act_cpu[:,k]  # (B,H1,W1)
                n_act = int(act_k.sum().item())
                act_count[k] += n_act
                if n_act > 0:
                    crowd_count[k]  += int((act_k & crowd_cpu).sum().item())
                    solo_count[k]   += int((act_k & solo_cpu).sum().item())
                    pos_grad_in_act[k] += int((act_k & pos_cpu[:,k]).sum().item())

        # 타일 수집: MAJ_over / MIN_need
        need_cpu = need.detach().cpu()
        act_cpu  = act.detach().cpu()
        for b in range(B):
            img = torch.from_numpy(x_vis[b,0]).unsqueeze(0)  # (1,H,W)
            # 대표 필터를 정해야 함: 해당 위치에서 act가 가장 큰 필터를 선택
            act_b = act_cpu[b]         # (K,H1,W1)
            need_b= need_cpu[b]        # (K,H1,W1)
            # A) Majority Over: is_crowd & need_count==0 (혹은 낮은 분위수)
            mA = (is_crowd[b] & (need_count[b]==0))
            idxs = mA.nonzero(as_tuple=False)
            if idxs.numel():
                # 각 위치에서 최대 act filter
                ixs = idxs.split(1, dim=0)
                for t in ixs[:128]:  # 배치당 과도 누적 방지
                    i,j = int(t[0,0]), int(t[0,1])
                    k = int(torch.argmax(act_b[:,i,j]).item())
                    score = float(act_b[k,i,j].item())
                    x0,y0,w,h = rf_patch_xy(i,j, ksz, stride, pad, dilation)
                    patch = safe_crop(img, x0,y0,w,h)
                    patch = F.interpolate(patch.unsqueeze(0), size=(28,28), mode="bilinear", align_corners=False)[0]
                    heap = heaps_MAJ_over[k]
                    if len(heap) < args.top_tiles:
                        heapq.heappush(heap, (score, patch))
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, (score, patch))

            # B) Minority Need: is_solo & need_count high (여기서는 >=1)
            mB = (is_solo[b] & (need_count[b]>=1))
            idxs = mB.nonzero(as_tuple=False)
            if idxs.numel():
                ixs = idxs.split(1, dim=0)
                for t in ixs[:128]:
                    i,j = int(t[0,0]), int(t[0,1])
                    k = int(torch.argmax(need_b[:,i,j]).item())
                    score = float(need_b[k,i,j].item())
                    x0,y0,w,h = rf_patch_xy(i,j, ksz, stride, pad, dilation)
                    patch = safe_crop(img, x0,y0,w,h)
                    patch = F.interpolate(patch.unsqueeze(0), size=(28,28), mode="bilinear", align_corners=False)[0]
                    heap = heaps_MIN_need[k]
                    if len(heap) < args.top_tiles:
                        heapq.heappush(heap, (score, patch))
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, (score, patch))

    h.remove()

    # ---------------------------
    # 저장
    # ---------------------------
    outd = args.out_dir
    os.makedirs(outd, exist_ok=True)

    # 전역 분포 그림
    all_maj = torch.cat([torch.from_numpy(a) for a in all_maj_counts], dim=0).numpy() if len(all_maj_counts)>0 else np.array([])
    all_need= torch.cat([torch.from_numpy(a) for a in all_need_counts], dim=0).numpy() if len(all_need_counts)>0 else np.array([])

    if HAS_PLT:
        if all_maj.size>0:
            plt.figure()
            plt.hist(all_maj, bins=range(0,K+2), align="left")
            plt.xlabel("#active filters per pixel"); plt.ylabel("count"); plt.title("Majority Count Distribution")
            plt.tight_layout(); plt.savefig(os.path.join(outd, "hist_majority_count.png")); plt.close()

        if all_need.size>0:
            plt.figure()
            maxn = min(K, int(all_need.max())+1)
            plt.hist(all_need, bins=range(0,maxn+2), align="left")
            plt.xlabel("#needed filters per pixel"); plt.ylabel("count"); plt.title("Need Count Distribution")
            plt.tight_layout(); plt.savefig(os.path.join(outd, "hist_need_count.png")); plt.close()

    # 디스코던스 이미지 그리드 저장
    if HAS_IMGIO and len(discord_pool)>0:
        pool_sorted = sorted(discord_pool, key=lambda t: -t[0])
        hi_maj_low_margin = [t[2] for t in pool_sorted if t[1]==+1][:args.top_imgs]
        low_maj_hi_margin = [t[2] for t in pool_sorted if t[1]==-1][:args.top_imgs]
        for tag, lst in [("hiMAJ_lowMargin", hi_maj_low_margin), ("lowMAJ_hiMargin", low_maj_hi_margin)]:
            if len(lst)>0:
                grid = tvutils.make_grid(torch.stack(lst, dim=0), nrow=8, padding=1)
                img = (grid*255.0).clamp(0,255).byte().cpu().numpy().transpose(1,2,0)
                imwrite(os.path.join(outd, f"discord_{tag}.png"), img)

    # 필터별 타일 & 요약
    crowding_idx = np.zeros(K, dtype=np.float32)
    solitude_idx = np.zeros(K, dtype=np.float32)
    pos_grad_frac= np.zeros(K, dtype=np.float32)

    for k in range(K):
        if act_count[k] > 0:
            crowding_idx[k] = crowd_count[k] / act_count[k]
            solitude_idx[k] = solo_count[k]  / act_count[k]
            pos_grad_frac[k]= pos_grad_in_act[k] / act_count[k]
        else:
            crowding_idx[k] = solitude_idx[k] = pos_grad_frac[k] = 0.0

        od = os.path.join(outd, f"filter_{k:03d}")
        os.makedirs(od, exist_ok=True)

        for tag, heap in [("MAJ_over", heaps_MAJ_over[k]), ("MIN_need", heaps_MIN_need[k])]:
            patches = [t[1] for t in sorted(heap, key=lambda x: -x[0])]
            img = make_montage(patches, nrow=8)
            if img is not None and HAS_IMGIO:
                imwrite(os.path.join(od, f"{tag}.png"), img)

        with open(os.path.join(od, "metrics.json"), "w") as f:
            json.dump({
                "filter": int(k),
                "act_count": int(act_count[k]),
                "crowd_hits": int(crowd_count[k]),
                "solo_hits":  int(solo_count[k]),
                "crowding_idx": float(crowding_idx[k]),
                "solitude_idx": float(solitude_idx[k]),
                "pos_grad_frac": float(pos_grad_frac[k]),
            }, f, indent=2)

    # 전체 요약 CSV
    import csv
    with open(os.path.join(outd, "filter_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filter","act_count","crowd_hits","solo_hits","crowding_idx","solitude_idx","pos_grad_frac"])
        for k in range(K):
            w.writerow([k, act_count[k], crowd_count[k], solo_count[k],
                        f"{crowding_idx[k]:.6f}", f"{solitude_idx[k]:.6f}", f"{pos_grad_frac[k]:.6f}"])

    print(f"[conv1_majority_audit] saved → {outd}")
    print(" - hist_majority_count.png / hist_need_count.png (전역 분포)")
    print(" - discord_hiMAJ_lowMargin.png / discord_lowMAJ_hiMargin.png (정확도 불일치 이미지)")
    print(" - filter_xxx/MAJ_over.png & MIN_need.png + metrics.json (양쪽 타일/지표)")
    print(" - filter_summary.csv (필터별 crowding/solitude/방향성 요약)")
    

if __name__ == "__main__":
    main()
