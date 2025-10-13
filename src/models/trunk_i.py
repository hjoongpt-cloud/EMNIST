import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrunkI(nn.Module):
    """
    conv1 → sparse activation (global top-k + local top-2 with fallback via topk) →
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
        flat = A.view(-1)
        if k >= flat.numel():
            mask_global = torch.ones_like(A, dtype=torch.bool)
        else:
            threshold, _ = torch.kthvalue(flat, flat.numel() - k + 1)
            mask_global = A >= threshold  # (B,C,H,W)
        A_global = A * mask_global.to(A.dtype)

        # Prepare for per-location channel handling
        v = A_global.permute(0, 2, 3, 1)  # (B,H,W,C)
        device = v.device
        B_, H_, W_, C = v.shape

        # Count nonzero per location (used only for knowing empties)
        nonzero_mask = (v > 0)  # (B,H,W,C)
        nonzero_count = nonzero_mask.sum(dim=-1)  # (B,H,W)

        # Initialize logits with zeros; will fill top-2 values (others stay zero)
        logits = torch.zeros_like(v)  # (B,H,W,C)

        # Top-2 per location (covers single or multi nonzero automatically)
        #top2_vals, top2_idx = torch.topk(v, 2, dim=-1)  # (B,H,W,2)
        top2_vals, top2_idx = torch.topk(v, 3, dim=-1)  # (B,H,W,2)
        logits = torch.zeros_like(v)  # (B,H,W,C)

        # assign both top-2 slots; second slot may be zero if only one nonzero exists
        b_idx, h_idx, w_idx = torch.nonzero(torch.ones(B_, H_, W_, device=device, dtype=torch.bool), as_tuple=True)
        for slot in range(2):
            idx = top2_idx[..., slot]  # (B,H,W)
            val = torch.gather(v, -1, idx.unsqueeze(-1)).squeeze(-1)  # (B,H,W)
            c_idx = idx[b_idx, h_idx, w_idx]
            vals = val[b_idx, h_idx, w_idx]
            logits[b_idx, h_idx, w_idx, c_idx] = vals

        # Zero out locations that had no nonzero to avoid introducing spurious (they are already zero)
        # Global max over entire feature map per sample
        flat_logits = logits.reshape(B_, -1)

        # global max per sample, but avoid divide-by-zero by clamping
        global_max, _ = flat_logits.max(dim=1, keepdim=True)  # (B,1)
        denom = global_max.clone()
        denom[denom == 0] = 1.0  # so we don't divide by zero
        normed_flat = flat_logits / denom  # safe broadcast division

        # zero out positions where original global_max was zero (all zeros case)
        zero_mask = (global_max == 0)  # (B,1)
        normed_flat = normed_flat.masked_fill(zero_mask.expand_as(normed_flat), 0.0)

        # reshape back
        sparse_weights = normed_flat.view_as(logits)  # (B,H,W,C)

        # Embed locations
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
