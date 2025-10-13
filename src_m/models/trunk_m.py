
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .slot_pool import SlotPool

class TrunkM(nn.Module):
    """
    Trunk with stride-2 conv1 (28x28 -> 14x14 tokens), per-location channel top-k (8),
    channel-embedding to D, 2D sinusoidal PE, 1-layer MHA, then SlotPool.
    """
    def __init__(self,
                 proj_in_channels=150,
                 proj_out_dim=32,
                 num_heads=4,
                 conv1_stride=2,
                 global_topk_ratio=0.08,
                 k_ch=10,
                 slot_M=12,
                 pretrained_filter_path=None,
                 freeze_conv1=False,
                 slot_aggregate='proj32',
                 H=14, W=14):
        super().__init__()
        self.C1 = proj_in_channels
        self.D = proj_out_dim
        self.num_heads = num_heads
        self.conv1 = nn.Conv2d(1, self.C1, kernel_size=9, stride=conv1_stride, padding=4, bias=False)
        # load optional pretrained conv1 filters (NumPy)
        if pretrained_filter_path is not None:
            try:
                import numpy as _np
                arr = _np.load(pretrained_filter_path)
                arr = arr.astype('float32')
                if arr.ndim == 3 and arr.shape == (self.C1, 9, 9):
                    w = torch.from_numpy(arr).unsqueeze(1)  # (C1,1,9,9)
                    with torch.no_grad():
                        self.conv1.weight.copy_(w)
                    print(f"[TrunkM] Loaded conv1 filters from {pretrained_filter_path} shape={arr.shape}")
                elif arr.ndim == 4 and arr.shape == (self.C1, 1, 9, 9):
                    w = torch.from_numpy(arr)
                    with torch.no_grad():
                        self.conv1.weight.copy_(w)
                    print(f"[TrunkM] Loaded conv1 filters from {pretrained_filter_path} shape={arr.shape}")
                else:
                    print(f"[TrunkM] WARN: conv1 filter shape mismatch {arr.shape}; expected (C1,9,9) or (C1,1,9,9)")
            except Exception as e:
                print(f"[TrunkM] WARN: failed to load conv1 filters: {e}")
        if freeze_conv1:
            for p in self.conv1.parameters():
                p.requires_grad = False
            print("[TrunkM] conv1 frozen (requires_grad=False)")

        self.global_topk_ratio = global_topk_ratio
        self.k_ch = k_ch

        self.channel_embed = nn.Parameter(torch.randn(self.C1, self.D) * 0.02)
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)

        self.self_attn = nn.MultiheadAttention(self.D, self.num_heads, batch_first=True)

        # Slot pooling on 14x14
        self.slot_pool = SlotPool(D=self.D, M=slot_M, H=H, W=W, n_heads=self.num_heads, aggregate=slot_aggregate)

        # expose out_dim for the head
        if slot_aggregate == 'proj32' or slot_aggregate == 'mean':
            self.out_dim = self.D
        else:
            self.out_dim = self.D * slot_M

        self.H, self.W = H, W

    def _pos2d(self, h, w, device):
        dim = self.D
        d2 = dim // 2
        if self.pos_cache.numel() != h * w * dim:
            pe = torch.zeros(h, w, dim, device=device)
            # y part
            div_y = torch.exp(torch.arange(0, d2, 2, device=device) * -(math.log(10000.0) / d2))
            pos_y = torch.arange(h, device=device).unsqueeze(1)                       # (H,1)
            pe_y = torch.zeros(h, d2, device=device)
            pe_y[:, 0::2] = torch.sin(pos_y * div_y)                                  # (H, d2/2)
            pe_y[:, 1::2] = torch.cos(pos_y * div_y)
            pe[:, :, :d2] = pe_y.unsqueeze(1).expand(h, w, d2)                         # -> (H,W,d2)
            # x part
            div_x = torch.exp(torch.arange(0, d2, 2, device=device) * -(math.log(10000.0) / d2))
            pos_x = torch.arange(w, device=device).unsqueeze(1)                       # (W,1)
            pe_x = torch.zeros(w, d2, device=device)
            pe_x[:, 0::2] = torch.sin(pos_x * div_x)
            pe_x[:, 1::2] = torch.cos(pos_x * div_x)
            pe[:, :, d2:] = pe_x.unsqueeze(0).expand(h, w, d2)                         # -> (H,W,d2)
            self.pos_cache = pe.view(h * w, dim)
        return self.pos_cache

    def forward(self, x):
        # x: (B,1,28,28)
        A = F.relu(self.conv1(x))  # (B, C1, 14, 14) with stride=2
        B, C, H, W = A.shape

        # global top-k masking per sample
        total = A.numel() // B
        k_global = max(1, int(math.ceil(self.global_topk_ratio * total)))
        A_flat = A.view(B, -1)
        thresh, _ = torch.kthvalue(A_flat, A_flat.shape[1] - k_global + 1, dim=1, keepdim=True)
        mask_flat = A_flat >= thresh
        A_global = (A_flat * mask_flat).view(B, C, H, W).to(A.dtype)

        # per-location channel top-k
        v = A_global.permute(0,2,3,1)  # (B,H,W,C)
        topv, topi = torch.topk(v, self.k_ch, dim=-1)   # (B,H,W,k_ch)
        logits = torch.zeros_like(v)
        logits.scatter_(-1, topi, topv)                 # keep only top-k channels

        # sample-wise normalize by global max
        flat_logits = logits.reshape(B, -1)
        global_max, _ = flat_logits.max(dim=1, keepdim=True)
        denom = global_max.clone()
        denom[denom == 0] = 1.0
        normed_flat = flat_logits / denom
        zero_mask = (global_max == 0)
        normed_flat = normed_flat.masked_fill(zero_mask.expand_as(normed_flat), 0.0)
        sparse_weights = normed_flat.view_as(logits)  # (B,H,W,C1)

        # channel embedding -> loc_emb (B,H,W,D) + 2D PE
        loc_emb = torch.einsum('bhwc,cd->bhwd', sparse_weights, self.channel_embed)  # (B,H,W,D)
        pe = self._pos2d(H, W, loc_emb.device).view(H, W, self.D)
        loc_emb = loc_emb + pe.unsqueeze(0)
        seq = loc_emb.view(B, H*W, self.D)  # (B,N,D)

        # self-attn context
        attn_out,_ = self.self_attn(seq, seq, seq)  # (B,N,D)

        # slot pooling
        Z, A_maps, head_energy, S_slots, topk_mass = self.slot_pool(attn_out)  # Z: (B,out_dim)
        aux = dict(A_maps=A_maps, head_energy=head_energy, sparse_weights=sparse_weights, top_idx=topi, S_slots=S_slots, topk_mass=topk_mass)
        return Z, aux
