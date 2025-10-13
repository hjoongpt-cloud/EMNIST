import torch
import torch.nn as nn
import itertools

def _grid_slices(H, W, gh=3, gw=3):
    hs = [int(round(i*H/gh)) for i in range(gh+1)]
    ws = [int(round(j*W/gw)) for j in range(gw+1)]
    rects = []
    for i in range(gh):
        for j in range(gw):
            rects.append((hs[i], hs[i+1], ws[j], ws[j+1]))
    return rects

def _choose_9C4_masks(H, W, gh=3, gw=3):
    rects = _grid_slices(H,W,gh,gw)
    masks = []
    for comb in itertools.combinations(range(len(rects)), 4):
        m = torch.zeros(H,W, dtype=torch.float32)
        for idx in comb:
            y0,y1,x0,x1 = rects[idx]
            m[y0:y1, x0:x1] = 1.0
        s = m.sum().clamp_min(1.0)
        m = m / s
        masks.append(m)
    return torch.stack(masks, dim=0)

def jitter_masks(masks, pixels=1, alpha=0.1):
    if pixels <= 0 and alpha <= 0: return masks
    S,H,W = masks.size()
    out = masks.clone()
    if pixels>0:
        dy = torch.randint(-pixels, pixels+1, (S,))
        dx = torch.randint(-pixels, pixels+1, (S,))
        for i in range(S):
            m = masks[i]
            y0 = max(0, -dy[i]); y1 = min(H, H - dy[i]) 
            x0 = max(0, -dx[i]); x1 = min(W, W - dx[i])
            tgt = out[i]*0
            tgt[y0+dy[i]:y1+dy[i], x0+dx[i]:x1+dx[i]] = m[y0:y1, x0:x1]
            out[i] = tgt
    if alpha>0:
        out = (1-alpha)*masks + alpha*out
    s = out.sum(dim=(1,2), keepdim=True).clamp_min(1e-6)
    return out / s

class Slots(nn.Module):
    def __init__(self, H=28, W=28, scheme="9C4"):
        super().__init__()
        self.H=H; self.W=W; self.scheme=scheme
        if scheme=="9C4":
            M = _choose_9C4_masks(H,W,3,3)
        else:
            raise NotImplementedError
        self.register_buffer("masks", M)
        self.S = M.size(0)

    def embed(self, feats, jitter_px=0, jitter_alpha=0.0, slotdrop_p=0.0, training=False):
        B,D,H,W = feats.shape
        assert H==self.H and W==self.W
        M = self.masks
        if training:
            if jitter_px>0 or jitter_alpha>0:
                M = jitter_masks(M, pixels=jitter_px, alpha=jitter_alpha)
            if slotdrop_p>0:
                keep = (torch.rand(M.size(0), device=feats.device) > slotdrop_p).float().view(-1,1,1)
                M = M.to(feats.device) * keep + (1.0 - keep) * 0.0
        feats_ = feats.unsqueeze(1)
        M_ = M.unsqueeze(0).unsqueeze(2)
        weighted = feats_ * M_
        return weighted.sum(dim=(-1,-2))
