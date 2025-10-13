# src_n/tools/n_iou_utils.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def topq_mask_per_channel(a_bmhw: torch.Tensor, q: float) -> torch.Tensor:
    """
    a_bmhw: (B,M,H,W) >=0
    returns: bool mask (B,M,H,W) where each (B,M,*) keeps top-q fraction pixels
    """
    if q <= 0:  # keep none
        return torch.zeros_like(a_bmhw, dtype=torch.bool)
    if q >= 1:
        return torch.ones_like(a_bmhw, dtype=torch.bool)
    B,M,H,W = a_bmhw.shape
    flat = a_bmhw.reshape(B,M,-1)
    k = max(1, int(round(q * (H*W))))
    idx = torch.topk(flat, k=min(k, H*W), dim=-1).indices  # (B,M,k)
    m = torch.zeros_like(flat, dtype=torch.bool)
    m.scatter_(2, idx, True)
    return m.view(B,M,H,W)

@torch.no_grad()
def binary_iou(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    m1, m2: bool or {0,1} float of shape (..., H, W)
    returns IoU over last two dims
    """
    m1 = m1.bool(); m2 = m2.bool()
    inter = (m1 & m2).sum(dim=(-2,-1)).float()
    union = (m1 | m2).sum(dim=(-2,-1)).float().clamp(min=1e-6)
    return inter / union

@torch.no_grad()
def greedy_diverse_topk(scores_bm: torch.Tensor,
                        A_bmhw: torch.Tensor,
                        k: int,
                        q_iou: float = 0.1,
                        iou_thr: float = 0.30) -> torch.Tensor:
    """
    Per sample, select up to k slots using greedy NMS based on IoU of top-q masks.
    scores_bm: (B,M) non-negative
    A_bmhw:    (B,M,H,W) >=0
    returns: bool mask (B,M) with True for selected slots
    """
    B,M = scores_bm.shape
    sel = torch.zeros_like(scores_bm, dtype=torch.bool)
    # precompute masks
    masks = topq_mask_per_channel(A_bmhw, q_iou)  # (B,M,H,W)
    # sort scores desc
    order = torch.argsort(scores_bm, dim=1, descending=True)  # (B,M)

    for b in range(B):
        picked = []
        for j in range(M):
            m = int(order[b,j].item())
            ok = True
            for pm in picked:
                iou = binary_iou(masks[b, m:m+1], masks[b, pm:pm+1])[0]
                if float(iou) >= iou_thr:
                    ok = False
                    break
            if ok:
                sel[b, m] = True
                picked.append(m)
                if len(picked) >= k:
                    break
    return sel

def pairwise_iou_mean(masks_bmhw: torch.Tensor, sel_bm: torch.Tensor) -> torch.Tensor:
    """
    Mean IoU among selected masks per sample.
    masks_bmhw: bool (B,M,H,W)
    sel_bm    : bool (B,M)
    returns: (B,) mean IoU over pairs (upper triangle), 0 if <2 selected.
    """
    B,M,H,W = masks_bmhw.shape
    out = []
    for b in range(B):
        idx = sel_bm[b].nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            out.append(torch.zeros((), device=masks_bmhw.device))
            continue
        mb = masks_bmhw[b, idx]  # (K,H,W)
        # pairwise IoU
        K = mb.size(0)
        ious = []
        for i in range(K):
            for j in range(i+1, K):
                ious.append(binary_iou(mb[i:i+1], mb[j:j+1])[0])
        if len(ious)==0:
            out.append(torch.zeros((), device=masks_bmhw.device))
        else:
            out.append(torch.stack(ious).mean())
    return torch.stack(out)  # (B,)
