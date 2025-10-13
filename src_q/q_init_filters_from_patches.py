# =============================================
# FILE: src_q/q_init_filters_from_patches.py
# =============================================
# EMNIST에서 saliency 기반 9x9 패치를 추출 → MiniBatchKMeans로 클러스터 → 초기 conv1 필터(.npy) 저장
# 또한 패치/필터 그리드 시각화를 함께 저장합니다.

import os, math, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_grid_img(t: torch.Tensor, nrow: int = 16, normalize: bool = True, pad_value: float = 0.5):
    # t: (N,1,H,W)
    return make_grid(t, nrow=nrow, normalize=normalize, pad_value=pad_value)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _extract_patches_from_image(img: torch.Tensor, per_image: int, patch: int, q_keep: float, nms_k: int):
    """
    img: (1,H,W) in [0,1]
    returns: list of (1,patch,patch) tensors
    """
    assert img.ndim == 3 and img.size(0) == 1
    sal = img.abs()  # 간단 saliency
    # NMS via max-pooling
    pad = nms_k // 2
    pooled = F.max_pool2d(sal, kernel_size=nms_k, stride=1, padding=pad)
    maxima = (sal >= pooled - 1e-7)  # local maxima mask
    thr = torch.quantile(sal.flatten(), torch.tensor(q_keep, device=sal.device))
    mask = maxima & (sal >= thr)
    ys, xs = torch.nonzero(mask[0], as_tuple=True)
    if ys.numel() == 0:
        return []
    scores = sal[0, ys, xs]
    idx = torch.argsort(scores, descending=True)[:per_image]
    ys, xs = ys[idx], xs[idx]

    # extract patches
    rad = patch // 2
    img_padded = F.pad(img, (rad, rad, rad, rad), mode="reflect")
    patches = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, x0 = y + rad, x + rad
        crop = img_padded[:, y0 - rad:y0 + rad + 1, x0 - rad:x0 + rad + 1]
        patches.append(crop)
    return patches


def collect_patches(dl, max_images: int, per_image: int, patch: int, q_keep: float, nms_k: int):
    patches = []
    seen = 0
    for img, _ in dl:
        img = img
        for b in range(img.size(0)):
            pt = _extract_patches_from_image(img[b], per_image, patch, q_keep, nms_k)
            patches.extend(pt)
            seen += 1
            if seen >= max_images:
                break
        if seen >= max_images:
            break
    if len(patches) == 0:
        return torch.empty(0,1,patch,patch)
    return torch.stack(patches, dim=0)  # (N,1,p,p)


def whiten_per_patch(x: torch.Tensor):
    # x: (N,1,p,p)
    N = x.size(0)
    x = x.view(N, -1)
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) + 1e-6
    x = (x - m) / s
    p = int((x.size(1))**0.5)
    return x.view(N,1,p,p)


def kmeans_filters(patches: torch.Tensor, k: int):
    # patches: (N,1,p,p)
    N, _, p, _ = patches.shape
    X = patches.view(N, -1).cpu().numpy()
    if not _HAVE_SK:
        raise RuntimeError("scikit-learn이 필요합니다: pip install scikit-learn")
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=min(8192, max(256, k*10)), n_init=5, max_no_improvement=50, verbose=0)
    mbk.fit(X)
    centers = mbk.cluster_centers_.astype(np.float32).reshape(k, 1, p, p)
    # 각 필터를 zero-mean / unit-norm으로 정규화(시각화 및 초기 안정성 향상)
    W = centers
    W = W - W.mean(axis=(2,3), keepdims=True)
    norms = np.sqrt((W**2).sum(axis=(2,3), keepdims=True)) + 1e-6
    W = W / norms
    return W  # (K,1,p,p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="출력 .npy (예: outputs/filters_init_k192_9.npy)")
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--per_image", type=int, default=6)
    ap.add_argument("--q_keep", type=float, default=0.95)
    ap.add_argument("--nms_k", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    set_seed(args.seed)
    out_path = Path(args.out)
    _ensure_dir(out_path.parent)

    tfm = transforms.Compose([
        transforms.ToTensor(),  # (H,W)->(1,H,W) in [0,1]
    ])
    ds = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    patches = collect_patches(dl, args.max_images, args.per_image, args.patch, args.q_keep, args.nms_k)
    if patches.numel() == 0:
        raise RuntimeError("패치를 하나도 모으지 못했습니다. 하이퍼파라미터(q_keep, nms_k, per_image)를 조정하세요.")

    patches_w = whiten_per_patch(patches)
    # 시각화: 랜덤 패치 그리드
    grid_patches = _to_grid_img(patches_w[:256], nrow=16)
    save_image(grid_patches, str(out_path.parent / "viz_patches_grid.png"))

    W = kmeans_filters(patches_w, args.k)
    np.save(str(out_path), W)

    # 시각화: 필터 그리드
    W_t = torch.from_numpy(W)
    grid_filters = _to_grid_img(W_t[:256], nrow=16, normalize=True)
    save_image(grid_filters, str(out_path.parent / "viz_filters_grid.png"))

    print(f"[save] filters: {out_path}  shape={W.shape}")
    print(f"[viz] patches: {out_path.parent/'viz_patches_grid.png'}")
    print(f"[viz] filters: {out_path.parent/'viz_filters_grid.png'}")

if __name__ == "__main__":
    main()
