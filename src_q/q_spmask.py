# -*- coding: utf-8 -*-
# src_q/q_spmask.py
import itertools
from typing import List, Tuple
import torch

def grid_boxes(H: int, W: int, gh: int, gw: int) -> List[Tuple[int,int,int,int]]:
    """Divide HxW into ghxgw cells and return inclusive boxes (x0,y0,x1,y1).
       Special-case 14x14@3x3 to match legacy: edges [0,5,10,14] (5-5-4)."""
    if H == 14 and W == 14 and gh == 3 and gw == 3:
        edges = [0, 5, 10, 14]
        boxes = []
        for r in range(3):
            for c in range(3):
                x0, x1 = edges[c], edges[c+1]-1
                y0, y1 = edges[r], edges[r+1]-1
                boxes.append((x0,y0,x1,y1))
        return boxes

    boxes = []
    for r in range(gh):
        for c in range(gw):
            x0 = int(round(c * (W/float(gw))))
            y0 = int(round(r * (H/float(gh))))
            x1 = int(round((c+1)*(W/float(gw)))) - 1
            y1 = int(round((r+1)*(H/float(gh)))) - 1
            boxes.append((x0,y0,x1,y1))
    return boxes

def comb4_indices():
    """All 9C4 combinations = 126 items, each is tuple of 4 grid cell indices (0..8)."""
    return list(itertools.combinations(range(9), 4))  # length 126

@torch.no_grad()
def build_comb4_base_masks(H: int, W: int, device=None):
    """Return (base_masks[M0,H,W], comb_spec(list of tuples), boxes(list of boxes))."""
    boxes = grid_boxes(H, W, 3, 3)
    combs = comb4_indices()  # len 126
    M0 = len(combs)
    base = torch.zeros(M0, H, W, device=device)
    for m, comb in enumerate(combs):
        for g in comb:
            x0,y0,x1,y1 = boxes[g]
            base[m, y0:y1+1, x0:x1+1] = 1.0
    return base, combs, boxes

def replicate_masks(base_masks: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat base masks along M dimension."""
    if repeats <= 1:
        return base_masks
    return base_masks.repeat(repeats, 1, 1)

@torch.no_grad()
def random_drop1_from_comb4(
    A_bmhw: torch.Tensor,               # (B,M,H,W) masked maps (AFTER comb4 mask)
    comb_spec,                          # list of tuples len=126; each tuple has 4 cell indices
    boxes,                              # list of 9 grid boxes
    drop_prob: float = 1.0,             # prob per (b,m) to actually drop a cell (default 1.0)
    group_idx: torch.Tensor = None,     # (M,) base index per slot (0..125), else fallback m % 126
) -> torch.Tensor:
    """Slow but stable: for each (b,m) we sample one of the 4 cells in its comb and zero it out.
       Implemented with small python loops (accuracy-first). Used only in training when enabled."""
    if drop_prob <= 0.0:
        return A_bmhw
    B, M, H, W = A_bmhw.shape
    device = A_bmhw.device

    # decide which (b,m) to drop
    do_drop = (torch.rand(B, M, device=device) < float(drop_prob))
    if not bool(do_drop.any()):
        return A_bmhw

    # select which of 4 cells to drop per (b,m)
    which = torch.randint(0, 4, (B, M), device=device)

    out = A_bmhw.clone()  # we will modify

    for m in range(M):
        base_idx = int(group_idx[m].item()) if group_idx is not None else (m % 126)
        comb = comb_spec[base_idx]  # 4 cell ids

        for k in range(4):
            # all batch rows that will drop the k-th cell of this comb
            sel_b = (do_drop[:, m] & (which[:, m] == k)).nonzero(as_tuple=True)[0]
            if sel_b.numel() == 0:
                continue
            g = comb[k]
            x0,y0,x1,y1 = boxes[g]
            out[sel_b, m, y0:y1+1, x0:x1+1] = 0.0

    return out
