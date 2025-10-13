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
    pairs = list(itertools.combinations(range(9), 2))  # 36
    masks = torch.zeros((len(pairs), MH, MW), dtype=torch.float32, device=device)
    for m, (i, j) in enumerate(pairs):
        for idx in (i, j):
            x0, y0, x1, y1 = grid_boxes[idx]
            masks[m, y0:y1+1, x0:x1+1] = 1.0
    return masks, pairs, grid_boxes


def upsample_mask_14_to_28(mask_mhw):
    """(M,14,14) -> (M,28,28) nearest-neighbor upsample for visualization."""
    import torch.nn.functional as F
    return F.interpolate(mask_mhw.unsqueeze(0), size=(28, 28), mode="nearest").squeeze(0)