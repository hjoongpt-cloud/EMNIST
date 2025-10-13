# src/models/trunk.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trunk(nn.Module):
    """
    conv1 -> conv2 -> dropout -> attn -> mean pool
    """
    def __init__(self, embed_dim, num_heads,
                 conv1_channels=55,
                 conv2_dropout=0.0,
                 conv2corr_lambda=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=9)
        self.conv2 = nn.Conv2d(conv1_channels, embed_dim, kernel_size=3)
        self.dropout2 = nn.Dropout(conv2_dropout)
        self.conv2corr_lambda = conv2corr_lambda
        self.register_buffer('pos_cache', torch.zeros(0), persistent=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def conv2corr_penalty(self):
        if self.conv2corr_lambda == 0.0:
            return 0.0
        w = self.conv2.weight
        filters = w.view(w.size(0), -1)
        normed = filters / (filters.norm(dim=1, keepdim=True) + 1e-6)
        corr = normed @ normed.T
        off = corr - torch.eye(corr.size(0), device=corr.device)
        return self.conv2corr_lambda * (off**2).sum()

    def _2d_positional_encoding(self, h, w):
        if self.pos_cache.numel() != h * w * self.conv2.out_channels:
            dim = self.conv2.out_channels
            pe = torch.zeros(h, w, dim, device=self.conv2.weight.device)
            y = torch.arange(h, device=pe.device).unsqueeze(1)
            x = torch.arange(w, device=pe.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2, device=pe.device) *
                            -(math.log(10000.0) / dim))
            pe[..., 0::2] = torch.sin(y * div)
            pe[..., 1::2] = torch.cos(x * div)
            self.pos_cache = pe.view(h * w, dim)
        return self.pos_cache

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        penalty = self.conv2corr_penalty()

        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)
        pe = self._2d_positional_encoding(h, w)
        seq = seq + pe.unsqueeze(0)

        attn_out, _ = self.attn(seq, seq, seq)
        pooled = attn_out.mean(dim=1)
        return pooled, penalty
