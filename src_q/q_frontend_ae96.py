# Front-end encoder: Conv9x9(K), GELU, soft top-3 WTA, 1x1 proj -> 64 dims (stride=2 by default)
# Supports loading AE-derived initial conv1 filters (.npy), proj_down (.npz), and optional keep_idx slicing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrontEndAE96(nn.Module):
    def __init__(self, K=96, d=64, stride=2, topk=3, tau=0.7, wta_warmup=5,
                 init_conv1: str = '', init_proj: str = '', init_keep_idx: str = ''):
        super().__init__()
        self.K, self.d, self.stride = K, d, stride
        self.topk, self.tau, self.wta_warmup = topk, tau, wta_warmup
        self._cur_epoch = 0
        pad = 4
        self.conv1 = nn.Conv2d(1, K, kernel_size=9, stride=stride, padding=pad, bias=False)
        self.enc_act = nn.GELU()
        self.proj_down = nn.Conv2d(K, d, kernel_size=1, bias=True)

        # Init from files if provided
        if init_conv1:
            W = np.load(init_conv1)
            if W.ndim == 3:  # (K,9,9)
                W = W[:, None, :, :]
            assert W.shape == (K,1,9,9), f'conv1 init shape must be (K,1,9,9); got {W.shape}'
            with torch.no_grad():
                self.conv1.weight.copy_(torch.from_numpy(W).to(self.conv1.weight.dtype))
        if init_proj:
            npz = np.load(init_proj)
            Wp, bp = npz['W'], npz['b']
            # normalize shapes
            if Wp.ndim == 2:
                Wp = Wp[:, :, None, None]
            if bp.ndim > 1:
                bp = bp.reshape(-1)
            # auto-slice using keep_idx if channels don't match
            if Wp.shape[1] != K:
                if init_keep_idx:
                    keep = np.load(init_keep_idx)
                    assert keep.shape[0] == K, f'keep_idx length {keep.shape[0]} != K={K}'
                    Wp = Wp[:, keep, :, :]
                else:
                    raise AssertionError(f'proj init channels {Wp.shape[1]} != K={K}. Provide init_keep_idx to slice.')
            assert Wp.shape == (self.d, K, 1, 1) and bp.shape == (self.d,), \
                   f'proj init shapes mismatch: W={Wp.shape}, b={bp.shape}, expected {(self.d, K, 1, 1)} and {(self.d,)}'
            with torch.no_grad():
                self.proj_down.weight.copy_(torch.from_numpy(Wp).to(self.proj_down.weight.dtype))
                self.proj_down.bias.copy_(torch.from_numpy(bp).to(self.proj_down.bias.dtype))
        self.renorm_conv1()

    @torch.no_grad()
    def renorm_conv1(self):
        W = self.conv1.weight.view(self.K, -1)
        n = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.conv1.weight.copy_((W / n).view_as(self.conv1.weight))

    def set_epoch(self, ep: int):
        self._cur_epoch = int(ep)

    @staticmethod
    def _wta_soft_topk(a: torch.Tensor, k: int = 3, tau: float = 0.7) -> torch.Tensor:
        B,C,H,W = a.shape
        flat = a.view(B,C,-1)
        p = (flat / tau).softmax(dim=1)
        topv, topi = torch.topk(p, k, dim=1)
        mask = torch.zeros_like(p).scatter_(1, topi, topv)
        g = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
        return (flat * g).view(B,C,H,W)

    def forward(self, x: torch.Tensor, use_wta: bool = True) -> torch.Tensor:
        a = self.enc_act(self.conv1(x))
        if use_wta and (self._cur_epoch >= self.wta_warmup) and self.topk > 0:
            a = self._wta_soft_topk(a, k=self.topk, tau=self.tau)
        h = self.proj_down(a)  # (B,64,H',W')
        return h
