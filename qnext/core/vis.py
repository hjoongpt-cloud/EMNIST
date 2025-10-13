# qnext/core/vis.py
import math
import torch
import torchvision.utils as vutils

@torch.no_grad()
def save_filters_grid(W, out_path, nrow=None, padding=1):
    """
    conv1 가중치 W:[K,1,H,W]를 타일로 시각화해서 out_path에 저장.
    각 필터를 개별적으로 0~1로 정규화하여 대비 확보.
    """
    W = W.detach().float().cpu()
    if W.dim() != 4 or W.size(1) != 1:
        raise ValueError(f"Expected weight shape [K,1,H,W], got {tuple(W.shape)}")
    K = W.size(0)
    # per-filter min-max 정규화
    w = W.clone()
    w = w - w.amin(dim=(2,3), keepdim=True)
    denom = w.amax(dim=(2,3), keepdim=True)
    w = w / denom.clamp_min(1e-6)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(K)))
    grid = vutils.make_grid(w, nrow=nrow, padding=padding)  # [1,H*,W*]
    vutils.save_image(grid, out_path)
