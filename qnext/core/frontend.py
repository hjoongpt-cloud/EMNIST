# qnext/core/frontend.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from .wta import apply_wta
from .posenc import fixed_sincos_2d
import torch.nn.functional as F

class Frontend(nn.Module):
    def __init__(self, K=150, D=64, enc_act="gelu",
                 wta_mode="soft_top1", wta_tau=0.7, wta_k=1,
                 use_2dpe=False, pe_pairs=16, pe_alpha=0.5):
        super().__init__()
        self.D = D
        self.enc_act = enc_act
        self.wta_mode = wta_mode; self.wta_tau = wta_tau; self.wta_k = wta_k
        self.use_2dpe = bool(use_2dpe)
        self.pe_pairs = int(pe_pairs)
        self.pe_alpha = float(pe_alpha)

        self.conv1 = nn.Conv2d(1, K, kernel_size=9, padding=4, bias=False)
        self.register_buffer("channel_mask", torch.ones(1, self.conv1.out_channels, 1, 1))
        self.proj_down = nn.Conv2d(self.conv1.out_channels, D, kernel_size=1, bias=True)
        self.K = self.conv1.out_channels

        self.act = nn.ReLU() if enc_act.lower()=="relu" else nn.GELU()

    @torch.no_grad()
    def renorm_conv1(self, eps=1e-6):
        W = self.conv1.weight
        W_flat = W.view(W.size(0), -1)
        W_flat = F.normalize(W_flat, p=2, dim=1, eps=eps)
        self.conv1.weight.copy_(W_flat.view_as(W))

    def _ensure_sync(self, device, K_cur):
        if (self.channel_mask.size(1) != K_cur) or (self.channel_mask.device != device):
            self.register_buffer("channel_mask", torch.ones(1, K_cur, 1, 1, device=device))
            self.K = K_cur
        if (self.proj_down.in_channels != K_cur) or (self.proj_down.weight.device != device):
            old = self.proj_down
            new = nn.Conv2d(K_cur, old.out_channels, kernel_size=1, bias=True).to(device)
            with torch.no_grad():
                k_copy = min(old.in_channels, new.in_channels)
                if k_copy > 0:
                    new.weight[:, :k_copy] = old.weight[:, :k_copy].to(device)
                if old.bias is not None:
                    new.bias.copy_(old.bias.to(device))
            self.proj_down = new

    def forward(self, x):
        # x: [B,1,H,W]
        device = x.device
        a = self.conv1(x)  # [B,K,H,W]
        self._ensure_sync(device, a.size(1))
        a = a * self.channel_mask
        a = self.act(a)
        a_wta, aux = apply_wta(a, mode=self.wta_mode, tau=self.wta_tau, k=self.wta_k)
        z = self.proj_down(a_wta)  # [B,D,H,W]

        if self.use_2dpe:
            B, D, H, W = z.shape
            P = fixed_sincos_2d(H, W, num_pairs=self.pe_pairs, device=z.device)  # [64,H,W]
            if P.size(0) != D:
                # 64->D로 맞추기 위한 1x1 고정 선형 투사(가중치 고정 랜덤)
                # reproducible fixed random (registered as buffer)
                if not hasattr(self, "pe_proj"):
                    torch.manual_seed(1234)
                    Wp = torch.randn(D, P.size(0), 1, 1, device=z.device)
                    Wp = F.normalize(Wp.view(D, -1), p=2, dim=1).view(D, P.size(0), 1, 1)
                    self.register_buffer("pe_proj", Wp)
                P = F.conv2d(P.unsqueeze(0), self.pe_proj).squeeze(0)  # [D,H,W]
            # normalize & sum & renorm
            z = F.normalize(z, p=2, dim=1) + self.pe_alpha * F.normalize(P.unsqueeze(0).expand(B,-1,-1,-1), p=2, dim=1)
            z = F.normalize(z, p=2, dim=1)

        return z, {"wta": aux, "a": a, "a_wta": a_wta}

    # === 아래 load/resize 유틸은 기존과 동일(필요한 부분만 유지) ===
    @torch.no_grad()
    def _resize_channels(self, K_new:int, D_keep:int=None):
        device = self.conv1.weight.device
        old_K = self.conv1.out_channels
        old_D = self.proj_down.out_channels
        D_new = old_D if D_keep is None else int(D_keep)
        conv1_new = nn.Conv2d(1, K_new, kernel_size=9, padding=4, bias=False).to(device)
        proj_new  = nn.Conv2d(K_new, D_new, kernel_size=1, bias=True).to(device)
        with torch.no_grad():
            k_copy = min(old_K, K_new)
            if k_copy > 0:
                conv1_new.weight[:k_copy] = self.conv1.weight[:k_copy]
                proj_new.weight[:, :k_copy] = self.proj_down.weight[:, :k_copy]
            if self.proj_down.bias is not None:
                proj_new.bias.copy_(self.proj_down.bias[:D_new])
        self.conv1 = conv1_new
        self.proj_down = proj_new
        self.register_buffer("channel_mask", torch.ones(1, K_new, 1, 1, device=device))
        self.K = K_new

    @torch.no_grad()
    def load_conv1_from_npy(self, path):
        W = torch.tensor(np.load(path)).float()  # [K_file, 1, 9, 9]
        K_file = int(W.shape[0])
        # K가 다르면 먼저 구조를 맞춘다
        if self.conv1.out_channels != K_file:
            self._resize_channels(K_new=K_file)
        # 이제 모양이 일치하므로 복사
        self.conv1.weight.copy_(W)


    @torch.no_grad()
    def load_proj_from_npz(self, path):
        data = np.load(path)
        W = torch.tensor(data["weight"]).float()
        b = torch.tensor(data["bias"]).float() if "bias" in data else None
        D_file, K_file = int(W.shape[0]), int(W.shape[1])
        if (self.proj_down.in_channels != K_file) or (self.proj_down.out_channels != D_file):
            device = self.proj_down.weight.device
            new = nn.Conv2d(K_file, D_file, kernel_size=1, bias=True).to(device)
            self.proj_down = new
            self.register_buffer("channel_mask", torch.ones(1, K_file, 1, 1, device=device))
            self.K = K_file
        self.proj_down.weight.copy_(W)
        if b is not None:
            self.proj_down.bias.copy_(b)

    @torch.no_grad()
    def load_from_ae_dir(self, ae_dir, only_conv1=False, only_proj=False):
        ae_dir = Path(ae_dir)
        loaded = {"conv1": False, "proj": False}
        conv_path = ae_dir / "conv1.npy"
        proj_path = ae_dir / "proj_down.npz"

        # conv 먼저 로드해서 K를 확정한다 (proj는 그 다음에 맞춤)
        if not only_proj and conv_path.exists():
            # conv1 로드시 _resize_channels로 K 자동 동기화
            self.load_conv1_from_npy(str(conv_path))
            loaded["conv1"] = True

        if not only_conv1 and proj_path.exists():
            # proj_down 로드시 in_channels/out_channels 자동 동기화
            self.load_proj_from_npz(str(proj_path))
            loaded["proj"] = True

        return loaded

