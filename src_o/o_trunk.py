#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OTrunk(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, d_ff=256, conv1_filters=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff

        assert conv1_filters is not None, "conv1_filters numpy array required"
        k = int(conv1_filters.shape[-1])

        # 학습과 동일: stride=2, same padding -> 28x28 -> 14x14
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_filters.shape[0],
            kernel_size=k,
            stride=2,
            padding=k // 2,
            bias=False,
        )
        with torch.no_grad():
            self.conv1.weight.copy_(torch.from_numpy(conv1_filters))
        for p in self.conv1.parameters():
            p.requires_grad_(False)  # 동결

        self.proj = nn.Conv2d(conv1_filters.shape[0], d_model, kernel_size=1, bias=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # 학습과 동일 (pre-norm)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # 학습과 동일: pos emb 사용 안 함
        self.register_buffer("pos_emb", None, persistent=False)
        self.token_ln = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # x: (B,1,28,28)
        f = self.conv1(x)                  # (B,C,H,W) ; H=W=14 (stride=2,pad=k//2 가정)
        f_relu = F.relu(f, inplace=False)  # heat용
        f_gelu = F.gelu(f)                 # 토큰용

        # heat map: 채널 평균 (양수 에너지)
        heat = torch.mean(f_relu, dim=1)   # (B,H,W)

        # 토큰화
        t = self.proj(f_gelu)              # (B,d,H,W)
        B, d, H, W = t.shape
        t = t.permute(0, 2, 3, 1).reshape(B, H * W, self.d_model)  # (B, H*W, d)

        # Transformer (pre-norm) -> post-encoder LN (train과 동일)
        t = self.encoder(t)                # (B, N, d)
        t = self.token_ln(t)

        return t, {"heat_map": heat}
