import torch
import itertools


def _grid_14_splits():
    """
    Partition 14 into 3 bins with widths (5,5,4):
    edges: [0,5,10,14]
    """
    edges = [0, 5, 10, 14]
    bins = [(edges[i], edges[i+1]) for i in range(3)]  # (start, end)
    return bins


def build_grid3_boxes_14x14():
    """Return list of 9 boxes for 14x14 grid, each as (x0,y0,x1,y1), 0-indexed inclusive.
    Uses row-major order: r0c0, r0c1, r0c2, r1c0, ...
    """
    xs = _grid_14_splits()
    ys = _grid_14_splits()
    boxes = []
    for r in range(3):
        for c in range(3):
            x0, x1 = xs[c]
            y0, y1 = ys[r]
            boxes.append((x0, y0, x1 - 1, y1 - 1))  # inclusive end
    return boxes  # len=9


def build_slot_union_masks_14(MH=14, MW=14, device="cpu"):
    """
    Build fixed slot union masks for 3x3 grid on 14x14 with (5,5,4) splits.
    - 9 grid cells -> choose pairs (i<j) => 36 slots
    Returns:
      masks_mhw: (36, 14, 14) float {0,1}
      pairs: list of (i,j) pairs length 36
      grid_boxes: list of 9 grid boxes [(x0,y0,x1,y1)]
    """
    grid_boxes = build_grid3_boxes_14x14()
    pairs = list(itertools.combinations(range(9), 4))  # 36
    masks = torch.zeros((len(pairs), MH, MW), dtype=torch.float32, device=device)
    for m, comb in enumerate(pairs):         # comb는 길이 4의 튜플
        for idx in comb:
            x0, y0, x1, y1 = grid_boxes[idx]
            masks[m, y0:y1+1, x0:x1+1] = 1.0
    return masks, pairs, grid_boxes

def repeat_masks_with_groups(masks, combos, repeat=1):
    """
    각 위치조합(하나의 '로케이션 그룹')을 repeat번 복제.
    리턴:
      masks_r: (126*repeat, 14, 14)
      group_id: (126*repeat,)  # 원래 조합의 인덱스(0..125)
      local_rank: (126*repeat,) # 각 그룹 내 복제 번호(0..repeat-1)
      combos: 그대로 전달
    """
    if repeat <= 1:
        B = masks.shape[0]
        group_id = torch.arange(B, dtype=torch.long, device=masks.device)
        local_rank = torch.zeros(B, dtype=torch.long, device=masks.device)
        return masks, group_id, local_rank, combos
    masks_r = masks.repeat_interleave(repeat, dim=0)
    B = masks.shape[0]
    group_id = torch.arange(B, dtype=torch.long, device=masks.device).repeat_interleave(repeat)
    local_rank = torch.stack([torch.arange(repeat, device=masks.device) for _ in range(B)], dim=0).reshape(-1)
    return masks_r, group_id, local_rank, combos
def upsample_mask_14_to_28(mask_mhw):
    """(M,14,14) -> (M,28,28) nearest-neighbor upsample for visualization."""
    import torch.nn.functional as F
    return F.interpolate(mask_mhw.unsqueeze(0), size=(28, 28), mode="nearest").squeeze(0)