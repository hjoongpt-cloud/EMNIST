import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TrunkProj(nn.Module):
    """
    Projection-based trunk:
      - conv1 (with optional pretrained filters + freeze)
      - spatial projection
      - dropout → 2D positional encoding → self-attention → mean pool
    """
    def __init__(self, cfg):
        super().__init__()
        C1 = cfg.model.proj_in_channels
        D = cfg.model.proj_out_dim
        ks = getattr(cfg.model, 'conv1_kernel_size', 9)

        # 1) conv1 정의
        self.conv1 = nn.Conv2d(1, C1, kernel_size=ks, bias=False)

        # 2) pretrained filter 로드 및 freeze 처리
        path = cfg.model.pretrained_filter_path
        if path:
            w = np.load(path)  # 예상 shape: (C1, 1, ks, ks)
            self.conv1.weight.data.copy_(torch.from_numpy(w))
            if cfg.model.freeze_conv1:
                for p in self.conv1.parameters():
                    p.requires_grad = False

        # 3) spatial projection layer
        self.proj = nn.Linear(C1, D)
        self.proj_dropout = nn.Dropout(cfg.model.proj_dropout)

        # 4) positional encoding cache
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)

        # 5) self-attention
        self.attn = nn.MultiheadAttention(D, cfg.model.num_heads, batch_first=True)

    def _2d_positional_encoding(self, h, w):
        total_dim = self.pos_cache.numel()
        if total_dim != h * w * self.proj.out_features:
            dim = self.proj.out_features
            pe = torch.zeros(h, w, dim, device=self.conv1.weight.device)
            y = torch.arange(h, device=pe.device).unsqueeze(1)
            x = torch.arange(w, device=pe.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2, device=pe.device) *
                            -(math.log(10000.0) / dim))
            pe[..., 0::2] = torch.sin(y * div)
            pe[..., 1::2] = torch.cos(x * div)
            self.pos_cache = pe.view(h * w, dim)
        return self.pos_cache

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.conv1(x))            # (B, C1, H', W')
        b, c, h, w = x.shape

        # spatial → sequence
        seq = x.view(b, c, h * w).permute(0, 2, 1)  # (B, H'*W', C1)

        # linear projection + dropout
        seq = self.proj(seq)                     # (B, H'*W', D)
        seq = self.proj_dropout(seq)

        # add positional encoding
        pe = self._2d_positional_encoding(h, w)
        seq = seq + pe.unsqueeze(0)

        # self-attention + mean pooling
        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(dim=1)             # (B, D)
        return pooled