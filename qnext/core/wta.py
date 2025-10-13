import torch

def _soft_topk_weights(x, k, tau):
    C = x.size(1)
    k = min(int(k), C) if k is not None else 1
    scores = x / max(tau, 1e-6)
    topk = torch.topk(scores, k=k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk, True)
    masked = scores.masked_fill(~mask, float('-inf'))
    w = torch.softmax(masked, dim=1)
    return w

def _soft_top1_weights(x, tau):
    scores = x / max(tau, 1e-6)
    return torch.softmax(scores, dim=1)

def _hard_top1_weights(x):
    idx = torch.argmax(x, dim=1, keepdim=True)
    w = torch.zeros_like(x)
    w.scatter_(1, idx, 1.0)
    return w

def apply_wta(feat, mode="soft_top1", tau=0.7, k=1, straight_through=True):
    assert feat.dim()==4, "feat must be (B,C,H,W)"
    score = torch.abs(feat)
    if mode in (None, "none"):
        w = torch.ones_like(score[:, :1])
        out = feat
        idx = torch.argmax(score, dim=1)
    elif mode == "soft_top1":
        w = _soft_top1_weights(score, tau)
        out = feat * w
        idx = torch.argmax(w, dim=1)
    elif mode == "soft_topk":
        w = _soft_topk_weights(score, k=k, tau=tau)
        out = feat * w
        idx = torch.argmax(w, dim=1)
    elif mode == "hard_top1":
        onehot = _hard_top1_weights(score)
        if straight_through:
            soft = _soft_top1_weights(score, tau)
            out = feat * (onehot + soft - soft.detach())
        else:
            out = feat * onehot
        w = onehot
        idx = torch.argmax(w, dim=1)
    else:
        raise ValueError(f"Unknown WTA mode: {mode}")
    aux = {"w": w, "winner_idx": idx}
    return out, aux
