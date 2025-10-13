
import torch
import torch.nn as nn
import torch.nn.functional as F

def slot_diversity_loss(S_slots, eps=1e-6):
    """
    Encourage slots to be different: sum_{m!=n} cos^2(S_m, S_n)
    S_slots: (B, M, D)
    """
    B, M, D = S_slots.shape
    S_n = F.normalize(S_slots, dim=-1)
    C = torch.einsum('bmd,bnd->bmn', S_n, S_n)  # (B,M,N)
    mask = 1 - torch.eye(M, device=S_slots.device)
    L = (C**2 * mask).sum(dim=(1,2)) / (M*(M-1)+eps)
    return L.mean()

def coverage_entropy_loss(A_maps, target_entropy=2.0, weight=1.0):
    """
    Make per-slot attention not too sharp / not too flat.
    A_maps: (B, M, H, W)
    """
    B, M, H, W = A_maps.shape
    N = H*W
    P = A_maps.view(B, M, N) + 1e-8
    P = P / P.sum(-1, keepdim=True)
    H_cur = -(P * P.log()).sum(-1)  # (B,M)
    if not isinstance(target_entropy, torch.Tensor):
        tgt = torch.full_like(H_cur, float(target_entropy))
    else:
        tgt = target_entropy.to(H_cur.device, H_cur.dtype)
        if tgt.dim()==0: tgt = tgt.expand_as(H_cur)
    return weight * (H_cur - tgt).abs().mean()

def cutpaste_count_consistency(A_maps_aug, A_maps_base, margin_ratio=0.02):
    """
    Encourage slot attention mass to increase when we paste a small stroke.
    A_maps_*: (B, M, H, W)
    """
    B,M,H,W = A_maps_base.shape
    base_mass = A_maps_base.view(B,M,-1).sum(-1)  # (B,M)
    aug_mass  = A_maps_aug.view(B,M,-1).sum(-1)   # (B,M)
    delta = aug_mass - base_mass
    margin = margin_ratio * (H*W)
    loss = torch.clamp(margin - delta, min=0.0).mean()
    return loss
