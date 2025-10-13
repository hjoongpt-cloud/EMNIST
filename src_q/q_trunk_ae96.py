# Slot trunk integrated with AE96 front-end
# Front-end: Conv9x9(K=96) -> GELU -> soft top-3 WTA (after warmup) -> 1x1 proj -> 64 x (H,W)
# Slot head: comb4 over 3x3 grid (9 choose 4 = 126 slots), per-slot MLP -> class logits
# Aggregator: probability-sum (default) or alternatives

from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from q_frontend_ae96 import FrontEndAE96

# ---------------- Masks (3x3 grid) ----------------

def grid_slices(H: int, W: int, gh: int = 3, gw: int = 3) -> List[Tuple[slice, slice]]:
    ys = [round(i * H / gh) for i in range(gh + 1)]
    xs = [round(i * W / gw) for i in range(gw + 1)]
    cells = []
    for i in range(gh):
        for j in range(gw):
            cells.append((slice(ys[i], ys[i+1]), slice(xs[j], xs[j+1])))
    return cells  # len = gh*gw


def build_cell_masks(H: int, W: int, gh: int = 3, gw: int = 3, device=None, dtype=torch.float32):
    cells = grid_slices(H, W, gh, gw)
    masks = []
    for (ys, xs) in cells:
        m = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        m[..., ys, xs] = 1.0
        m = m / (m.sum() + 1e-6)
        masks.append(m)  # normalized per-cell mask
    return masks  # list of (1,1,H,W)


def build_comb_masks(cell_masks: List[torch.Tensor], r: int = 4):
    # Sum r cells, then normalize across spatial dims
    combs = list(combinations(range(len(cell_masks)), r))
    masks = []
    for idxs in combs:
        m = torch.zeros_like(cell_masks[0])
        for i in idxs:
            m = m + cell_masks[i]
        m = m / (m.sum() + 1e-6)
        masks.append(m)
    return masks  # list of (1,1,H,W)

# ---------------- Slot Head ----------------

class SlotHead(nn.Module):
    def __init__(self, d=64, num_classes=47, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, f):  # (B,S,d)
        B,S,D = f.shape
        return self.mlp(f.view(B*S, D)).view(B, S, -1)  # (B,S,C)

# ---------------- Trunk Model ----------------

class SlotTrunkAE96(nn.Module):
    def __init__(self, num_classes=47, K=96, stride=2, topk=3, tau=0.7, wta_warmup=5,
                 init_conv1: str = '', init_proj: str = '', proj_keep_idx: str = '',
                 gh: int = 3, gw: int = 3, comb_r: int = 4,
                 aggregator: str = 'prob_sum', d=64, hidden=128, dropout=0.1):
        super().__init__()
        assert stride in (1,2)
        self.enc = FrontEndAE96(K=K, d=d, stride=stride, topk=topk, tau=tau,
                                 wta_warmup=wta_warmup, init_conv1=init_conv1,
                                 init_proj=init_proj, init_keep_idx=proj_keep_idx)
        self.num_classes = num_classes
        self.aggregator = aggregator
        self.gh, self.gw, self.comb_r = gh, gw, comb_r
        self.head = SlotHead(d=d, num_classes=num_classes, hidden=hidden, dropout=dropout)
        self.register_buffer('_masks_ready', torch.tensor(0, dtype=torch.uint8), persistent=False)
        self._masks: torch.Tensor = None  # (S,1,H,W)

    @torch.no_grad()
    def _build_masks_if_needed(self, H: int, W: int, device, dtype):
        if self._masks_ready.item() == 1:
            return
        cell_masks = build_cell_masks(H, W, self.gh, self.gw, device=device, dtype=dtype)
        comb_masks = build_comb_masks(cell_masks, r=self.comb_r)  # list of (1,1,H,W)
        self._masks = torch.cat(comb_masks, dim=0)  # (S,1,H,W)
        self._masks_ready.fill_(1)

    def set_epoch(self, ep: int):
        self.enc.set_epoch(ep)

    @torch.no_grad()
    def renorm_conv1(self):
        self.enc.renorm_conv1()

    def forward(self, x):  # x: (B,1,28,28)
        h = self.enc(x)                      # (B,64,H,W)
        B,D,H,W = h.shape
        if self._masks_ready.item() == 0:
            self._build_masks_if_needed(H, W, h.device, h.dtype)
        S = self._masks.shape[0]
        m = self._masks                      # (S,1,H,W)
        # Pool per slot
        ms = m.unsqueeze(0)                  # (1,S,1,H,W)
        h_exp = h.unsqueeze(1)               # (B,1,D,H,W)
        denom = (ms).sum(dim=(3,4)).clamp_min(1e-6)  # (1,S,1)
        num = (h_exp * ms).sum(dim=(3,4))            # (B,S,D)
        f = num / denom                        # (B,S,D)
        # Slot head -> logits per slot
        logits_s = self.head(f)               # (B,S,C)
        # Aggregate slots
        if self.aggregator == 'prob_sum':
            probs = logits_s.softmax(dim=-1)  # (B,S,C)
            probs_mean = probs.mean(dim=1)    # (B,C)
            return probs_mean.log()           # log-prob-like
        elif self.aggregator == 'logit_mean':
            return logits_s.mean(dim=1)
        elif self.aggregator == 'prob_max':
            probs = logits_s.softmax(dim=-1).max(dim=1)[0]
            return probs.log()
        else:
            # default to mean of logits
            return logits_s.mean(dim=1)
