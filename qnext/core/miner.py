# qnext/core/miner.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def logits_to_margin(logits, y):
    # true - best_rival
    true = logits[torch.arange(logits.size(0)), y]
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), y] = -1e9
    rival, _ = masked.max(dim=1)
    return true - rival  # (B,)

@torch.no_grad()
def mine_hard_anchors(model, dl, device, margin_lo=-2.0, margin_hi=-0.2,
                      extreme_cut=-4.0, max_samples=None):
    """
    train 예측으로 앵커 마이닝. -2<=margin<0 중심, margin<-6 극단 오답은 제외.
    return: dict(tensor indices per class), plus flat index list
    """
    model.eval()
    idx_all, margins_all, y_all = [], [], []
    base = 0
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            logits = model(x)
        m = logits_to_margin(logits, y)
        # 보관
        bs = x.size(0)
        idx_all.append(torch.arange(base, base+bs, device=device))
        margins_all.append(m)
        y_all.append(y)
        base += bs
    idx = torch.cat(idx_all); margins = torch.cat(margins_all); y_all = torch.cat(y_all)

    mask_core = (margins >= margin_lo) & (margins < margin_hi)
    mask_extreme = (margins < extreme_cut)
    mask = mask_core & (~mask_extreme)
    sel = idx[mask]
    if max_samples is not None and sel.numel() > max_samples:
        perm = torch.randperm(sel.numel(), device=device)[:max_samples]
        sel = sel[perm]

    # 클래스별 버킷
    buckets = {}
    for c in y_all.unique().tolist():
        c = int(c)
        m_c = mask & (y_all == c)
        buckets[c] = idx[m_c].detach().cpu()

    return sel.detach().cpu(), buckets, margins.detach().cpu()

@torch.no_grad()
def topk_nearest_negatives(emb, y, k_near=5):
    """
    임베딩(평균 풀링 등)으로 클래스가 다른 근접 음성 인덱스 반환.
    emb: [N, D], y:[N]
    return: list[Tensor] length N, each (k_near,)
    """
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.t()  # (N,N)
    sim.fill_diagonal_(-1.0)
    # 같은 클래스는 배제
    same = (y[:, None] == y[None, :])
    sim = sim.masked_fill(same, -1.0)
    topk = sim.topk(k=k_near, dim=1).indices  # (N, k_near)
    return topk
