import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrunkH(nn.Module):
    """
    conv1 → sparse activation (global top-k + local top-2 with proper fallback) →
    channel embedding projection + 2D positional encoding → self-attn → mean pool.

    Returns:
      pooled: (B, D) feature
      sparse_weights: (B, H, W, C)
      top2_idx: (B, H, W, 2)
    """
    def __init__(self, cfg):
        super().__init__()
        in_ch = 1
        C1 = cfg.model.proj_in_channels
        self.D = cfg.model.proj_out_dim
        ks1 = getattr(cfg.model, "conv1_kernel_size", 9)
        pad1 = ks1 // 2

        self.conv1 = nn.Conv2d(in_ch, C1, kernel_size=ks1, padding=pad1, bias=False)

        pf_path = cfg.model.pretrained_filter_path
        if pf_path:
            filters = np.load(pf_path)
            if filters.ndim == 3:
                filters = filters[:, None, :, :]
            self.conv1.weight.data.copy_(torch.from_numpy(filters).to(self.conv1.weight.dtype))
            if cfg.model.freeze_conv1:
                for p in self.conv1.parameters():
                    p.requires_grad = False

        self.global_topk_ratio = getattr(cfg.model, "global_topk_ratio", 0.0005)
        self.channel_embed = nn.Parameter(torch.randn(C1, self.D))
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)
        self.attn = nn.MultiheadAttention(self.D, cfg.model.num_heads, batch_first=True)

    def _2d_positional_encoding(self, h, w, device):
        dim = self.D
        if self.pos_cache.numel() != h * w * dim:
            pe = torch.zeros(h, w, dim, device=device)
            y = torch.arange(h, device=pe.device).unsqueeze(1)
            x = torch.arange(w, device=pe.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, device=pe.device) * -(math.log(10000.0) / dim))
            pe[..., 0::2] = torch.sin(y * div_term)
            pe[..., 1::2] = torch.cos(x * div_term)
            self.pos_cache = pe.view(h * w, dim)
        return self.pos_cache  # (h*w, dim)

    def forward(self, x):
        B, _, H, W = x.shape
        A = F.relu(self.conv1(x))  # (B, C, H, W)

        # Global top-k masking
        total = A.numel()
        k = max(1, int(math.ceil(self.global_topk_ratio * total)))
        #flat = A.abs().view(-1)
        flat = A.view(-1)
        if k >= flat.numel():
            mask_global = torch.ones_like(A, dtype=torch.bool)
        else:
            threshold, _ = torch.kthvalue(flat, flat.numel() - k + 1)
            #mask_global = A.abs() >= threshold  # (B,C,H,W)
            mask_global = A >= threshold  # (B,C,H,W)
        A_global = A * mask_global.to(A.dtype)

        # Prepare v for per-location handling
        v = A_global.permute(0, 2, 3, 1)  # (B,H,W,C)
        device = v.device
        B_, H_, W_, C = v.shape

        # Count nonzero per location
        #nonzero_mask = (v.abs() > 0)  # (B,H,W,C)
        nonzero_mask = (v > 0)  # (B,H,W,C)
        nonzero_count = nonzero_mask.sum(dim=-1)  # (B,H,W)

        # Initialize logits with -inf so softmax zeroes out others
        neg_inf = -1e9
        logits = torch.full_like(v, neg_inf)  # (B,H,W,C)

        # Case >=2 nonzeros: top-2
        top2_vals, top2_idx = torch.topk(v, 2, dim=-1)  # (B,H,W,2)
        ge2_mask = (nonzero_count >= 2)  # (B,H,W)
        if ge2_mask.any():
            B_idx = torch.arange(B_, device=device)[:, None, None]
            H_idx = torch.arange(H_, device=device)[None, :, None]
            W_idx = torch.arange(W_, device=device)[None, None, :]
            B_idx = B_idx.expand(B_, H_, W_)
            H_idx = H_idx.expand(B_, H_, W_)
            W_idx = W_idx.expand(B_, H_, W_)
            for slot in range(2):
                idx = top2_idx[..., slot]  # (B,H,W)
                val = torch.gather(v, -1, idx.unsqueeze(-1)).squeeze(-1)  # (B,H,W)
                # apply only where >=2
                mask_loc = ge2_mask
                logits[mask_loc, idx[mask_loc]] = val[mask_loc]

        # Case exactly 1 nonzero: use that one
        single_mask = (nonzero_count == 1)
        if single_mask.any():
            # find the index with the single nonzero
            idx_single = v.masked_fill(~nonzero_mask, float('-inf')).argmax(dim=-1)  # (B,H,W)
            val_single = torch.gather(v, -1, idx_single.unsqueeze(-1)).squeeze(-1)  # (B,H,W)
            logits[single_mask, idx_single[single_mask]] = val_single[single_mask]

        # Case zero nonzero: leave logits as all -inf

        # Softmax per location
        sparse_weights = torch.softmax(logits, dim=-1)  # (B,H,W,C)
        # Zero out locations that were completely empty to avoid NaNs or tiny nonzero due to numerical
        empty_mask = (nonzero_count == 0)[..., None]  # (B,H,W,1)
        sparse_weights = sparse_weights.masked_fill(empty_mask, 0.0)

        # Embed locations: expected channel embed
        loc_emb = torch.einsum('bhwc,cd->bhwd', sparse_weights, self.channel_embed)  # (B,H,W,D)

        # Add positional encoding
        pe = self._2d_positional_encoding(H_, W_, device)  # (H*W, D)
        pe = pe.view(H_, W_, self.D)  # (H,W,D)
        loc_emb = loc_emb + pe.unsqueeze(0)  # (B,H,W,D)

        # Self-attention
        seq = loc_emb.view(B_, H_ * W_, self.D)  # (B, H*W, D)
        attn_out, _ = self.attn(seq, seq, seq)  # (B, H*W, D)
        pooled = attn_out.mean(dim=1)  # (B, D)

        return pooled, sparse_weights, top2_idx
