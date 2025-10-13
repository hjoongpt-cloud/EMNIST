# -*- coding: utf-8 -*-
# src_q/q_trunk.py
import math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .q_spmask import build_comb4_base_masks, replicate_masks, random_drop1_from_comb4

def _eps(dtype: torch.dtype):
    return 1e-4 if dtype == torch.float16 else 1e-8

class SlotClassifier(nn.Module):
    def __init__(self,
                 filters_path: str,
                 num_slots_base: int = 126,
                 repeats_per_pos: int = 3,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_classes: int = 47,
                 # experiments
                 exp_wq: bool = True,
                 exp_gate: bool = True,
                 exp_head_bias: bool = True,
                 tau_intra: float = 0.7,
                 mask_drop1: bool = False):
        super().__init__()
        self.num_slots_base = int(num_slots_base)
        self.repeats_per_pos = int(repeats_per_pos)
        self.num_slots = self.num_slots_base * self.repeats_per_pos
        self.d_model = d_model
        self.num_classes = num_classes

        self.exp_wq = bool(exp_wq)          # per-repeat query adapters (Wq)
        self.exp_gate = bool(exp_gate)      # intra-group gating
        self.exp_head_bias = bool(exp_head_bias)  # per-repeat class bias
        self.tau_intra = float(tau_intra)
        self.mask_drop1 = bool(mask_drop1)

        # trunk
        self.conv1 = nn.Conv2d(1, 150, kernel_size=9, stride=2, padding=4, bias=False)
        self._load_conv1(filters_path)
        self.proj = nn.Conv2d(150, d_model, kernel_size=1, bias=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.token_ln = nn.LayerNorm(d_model)

        # slots
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, d_model) * (1.0 / math.sqrt(d_model)))

        if self.exp_wq:
            self.query_adapters = nn.ModuleList([nn.Linear(d_model, d_model, bias=True)
                                                 for _ in range(self.repeats_per_pos)])
        else:
            self.query_adapters = None

        if self.exp_gate:
            self.gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        else:
            self.gate_head = None

        if self.exp_head_bias:
            self.head_bias = nn.Parameter(torch.zeros(self.repeats_per_pos, num_classes))
        else:
            self.head_bias = None

        self.classifier = nn.Linear(d_model, num_classes)

        # lazy buffers
        self.register_buffer("_masks_base", None, persistent=False)  # (126,H,W)
        self._comb_spec = None
        self._boxes = None
        self.register_buffer("_masks_mhw", None, persistent=False)  # (M,H,W)
        self.register_buffer("_repeat_idx", None, persistent=False)  # (M,)
        self.register_buffer("_group_idx", None, persistent=False)   # (M,)

    def _load_conv1(self, path: str):
        w = np.load(path)  # (150,1,9,9)
        w = torch.from_numpy(w).float()
        if w.ndim != 4 or w.shape != (150,1,9,9):
            raise RuntimeError(f"filters shape mismatch: got {tuple(w.shape)}; need (150,1,9,9)")
        with torch.no_grad():
            self.conv1.weight.copy_(w)

    def _ensure_masks(self, H: int, W: int, device):
        if (self._masks_base is None) or (self._masks_base.shape[-2:] != (H, W)) or (self._masks_base.device != device):
            base, combs, boxes = build_comb4_base_masks(H, W, device=device)  # (126,H,W)
            self._masks_base = base
            self._comb_spec = combs
            self._boxes = boxes

        masks = replicate_masks(self._masks_base, self.repeats_per_pos)[:self.num_slots]
        self._masks_mhw = masks

        ridx = torch.arange(self.num_slots, device=device) % self.repeats_per_pos
        gidx = torch.arange(self.num_slots, device=device) // self.repeats_per_pos
        self._repeat_idx = ridx
        self._group_idx = gidx  # base index per slot (0..125)

    @torch.no_grad()
    def _check_stats_once(self, x: torch.Tensor):
        y = self.conv1(x)
        return dict(conv1_mean=y.mean().item(), conv1_std=y.std().item())

    def forward(self, x: torch.Tensor, use_spmask: bool, spmask_grid: int, spmask_assign: str, tau_p: float):
        B = x.size(0)
        h = F.gelu(self.conv1(x))          # (B,150,14,14)
        h = self.proj(h)                   # (B,D,14,14)
        H, W = int(h.size(2)), int(h.size(3))

        # ensure masks before anything that depends on indices
        self._ensure_masks(H, W, x.device)

        tokens = h.permute(0,2,3,1).reshape(B, H*W, self.d_model)
        tokens = self.encoder(tokens)
        tokens = self.token_ln(tokens)
        t = F.normalize(tokens, dim=-1)    # (B,N,D)
        dtype_t = t.dtype

        # per-slot queries (+ optional per-repeat adapters)
        q = self.slot_queries  # (M,D)  — fp32 파라미터
        if self.exp_wq:
            ridx = self._repeat_idx  # (M,)
            # ✅ q2를 t와 동일 dtype/디바이스로 미리 만들기
            q2 = torch.empty_like(q, device=q.device, dtype=t.dtype)
            for r in range(self.repeats_per_pos):
                sel = (ridx == r)
                if sel.any():
                    # ✅ adapter 입력/출력 모두 t.dtype로 강제
                    out = self.query_adapters[r](q[sel].to(t.dtype)).to(t.dtype)
                    q2[sel] = out
            q = q2
        else:
            q = q.to(t.dtype)

        q = F.normalize(q, dim=-1)


        # attention over pixels
        A_logits = torch.einsum("md,bnd->bmn", q, t) / math.sqrt(self.d_model)  # (B,M,N)
        A_slot   = torch.softmax(A_logits, dim=2).view(B, self.num_slots, H, W)

        # spatial mask (comb4-round) + renorm
        if use_spmask and spmask_grid == 3 and spmask_assign == "round":
            A_masked = A_slot * self._masks_mhw.unsqueeze(0)   # (B,M,H,W)
            if self.training and self.mask_drop1:
                A_masked = random_drop1_from_comb4(
                    A_masked, self._comb_spec, self._boxes,
                    drop_prob=1.0, group_idx=self._group_idx
                )
            mass_raw = A_masked.float().flatten(2).sum(-1).clamp_min(_eps(A_masked.dtype))   # (B,M)
            A_eff = (A_masked.float() / mass_raw.view(B, self.num_slots, 1, 1)).to(A_slot.dtype)
        else:
            mass_raw = A_slot.float().flatten(2).sum(-1).clamp_min(_eps(A_slot.dtype))
            A_eff = A_slot

        # pooled embeddings
        A_flat = A_eff.view(B, self.num_slots, H*W)
        S = torch.bmm(A_flat, t)           # (B,M,D)
        S = F.normalize(S, dim=-1)

        # per-slot logits (+ optional per-repeat head bias)
        slot_logits = self.classifier(S)   # (B,M,C)
        if self.exp_head_bias:
            bias_mc = self.head_bias[self._repeat_idx].to(slot_logits.dtype).to(slot_logits.device)
            slot_logits = slot_logits + bias_mc.unsqueeze(0)

        # global slot weights P from mass residual vs expectation
        if use_spmask and spmask_grid == 3 and spmask_assign == "round":
            N_px = float(H * W)
            area_in = self._masks_mhw.float().flatten(1).sum(-1).to(mass_raw.device)   # (M,)
            exp_mass = (area_in / N_px).view(1, self.num_slots)
        else:
            exp_mass = mass_raw.new_zeros((1, self.num_slots))
        z_base = mass_raw.float() - exp_mass
        z = (z_base - z_base.mean(dim=1, keepdim=True)) / max(1e-6, float(tau_p))
        P_global = torch.softmax(z, dim=1)  # (B,M)

        # optional intra-group gating (3-way per position)
        if self.exp_gate:
            g_logit = self.gate_head(S).squeeze(-1)     # (B,M)
            R = self.repeats_per_pos
            G = self.num_slots // R
            P_intra = []
            for g in range(G):
                s = slice(g*R, (g+1)*R)
                p = torch.softmax(g_logit[:, s] / max(1e-6, float(self.tau_intra)), dim=1)
                P_intra.append(p)
            P_intra = torch.cat(P_intra, dim=1)         # (B,M)
        else:
            P_intra = torch.ones_like(P_global)

        P = (P_global * P_intra)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logits = (slot_logits * P.unsqueeze(-1)).sum(dim=1)

        aux = {
            "A_maps": A_eff,
            "A_maps_raw": A_slot,
            "feat_hw": t.view(B, H, W, self.d_model).detach(),
            "S_slots": S.detach(),
            "slot_prob": P.detach(),
            "P_intra": P_intra.detach(),
            "slot_logits": slot_logits.detach()
        }
        return logits, aux
