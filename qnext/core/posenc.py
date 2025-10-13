# qnext/core/posenc.py
import torch
import math
import torch.nn.functional as F

@torch.no_grad()
def fixed_sincos_2d(H, W, num_pairs=16, device="cpu"):
    """
    중심 원점, x/y ∈ [-1,1], 지수 스케일 주파수로 64D 생성(= num_pairs*4, x/y 각각 sin,cos)
    return: [64, H, W]
    """
    # 좌표망
    jy, ix = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    x = (ix - (W-1)/2) / ((W-1)/2)
    y = (jy - (H-1)/2) / ((H-1)/2)

    # 주파수 세트(지수적)
    # 범위를 넓게: 1, 2, 4, ..., 2^(num_pairs-1)
    omegas = torch.tensor([2.0**k for k in range(num_pairs)], device=device)

    # 각 축에 대해 sin/cos 생성
    feats = []
    for w in omegas:
        feats += [torch.sin(w * x), torch.cos(w * x)]
    for w in omegas:
        feats += [torch.sin(w * y), torch.cos(w * y)]

    P = torch.stack(feats, dim=0)  # [4*num_pairs, H, W] = [64,H,W] if num_pairs=16
    # per-location 정규화
    P = F.normalize(P.view(P.size(0), -1), p=2, dim=0).view_as(P)
    return P  # [64,H,W]
