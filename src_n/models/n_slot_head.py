# src_n/models/n_slot_head.py
import json
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# utilities
# -----------------------------
def build_feat_for_dim(S_slots: torch.Tensor, XY: torch.Tensor, target_dim: int, xy_weight: float) -> torch.Tensor:
    """
    S_slots: (B, M, Dslot) or (M, Dslot)
    XY     : (B, M, 2)     or (M, 2)
    return : (..., target_dim)  where target_dim ∈ {Dslot, 2, Dslot+2}
    """
    if S_slots.dim() == 2:   # (M,D)
        S_slots = S_slots.unsqueeze(0)
        XY = XY.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    Dslot = S_slots.size(-1)
    if target_dim == Dslot:        # S만
        out = F.normalize(S_slots, dim=-1)
    elif target_dim == 2:          # XY만
        out = F.normalize(XY, dim=-1)
    elif target_dim == Dslot + 2:  # S+XY
        s  = F.normalize(S_slots, dim=-1)
        xy = F.normalize(XY, dim=-1)
        out = torch.cat([s, xy * xy_weight], dim=-1)
    else:
        raise ValueError(f"Unsupported target_dim={target_dim} (Dslot={Dslot})")

    if squeeze_back:
        out = out.squeeze(0)
    return out


def make_slot_prob(E_raw: Optional[torch.Tensor], A_maps: torch.Tensor, slot_mask: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """
    E_raw: (B,M) if available else None
    A_maps: (B,M,H,W)  fallback source
    slot_mask: (M,)
    return: (B,M) probability per slot (masked, rows sum to 1)
    """
    B, M = A_maps.size(0), A_maps.size(1)
    mask = slot_mask.view(1, M).to(A_maps.device)

    if (E_raw is not None) and (E_raw.ndim == 2) and (E_raw.shape[1] == M):
        e = E_raw.detach().float()
    else:
        # fallback = mass
        e = A_maps.flatten(2).sum(-1)  # (B,M)

    e = e * mask
    # temperature-softmax on alive slots only
    z = (e - e.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    p = torch.softmax(z, dim=1) * mask
    s = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p / s


# -----------------------------
# prototypes
# -----------------------------
class Proto(nn.Module):
    def __init__(self, centers: torch.Tensor, learn_centers: bool = False):
        """
        centers: (Kc, d) or (d,) -> store as (Kc, d)
        """
        super().__init__()
        if centers.ndim == 1:
            centers = centers.unsqueeze(0)
        if learn_centers:
            self.centers = nn.Parameter(centers.float())    # (Kc, d) (옵션: 미세조정 허용)
        else:
            self.register_buffer("centers", centers.float())# (Kc, d)
        Kc = centers.size(0)
        self.psi = nn.Parameter(torch.zeros(Kc, dtype=torch.float32))  # 센터 중요도

    @property
    def dim(self) -> int:
        return 0 if self.centers.numel() == 0 else int(self.centers.size(1))

    def score(self, F_active: torch.Tensor, beta: float) -> torch.Tensor:
        """
        F_active: (B, n_active, d) or (n_active, d) or (d,)
        return  : (B,) or scalar tensor  — 슬롯기반 클래스 점수
        """
        C = self.centers
        if C.numel() == 0:
            return torch.zeros((), device=F_active.device) if F_active.dim()==1 else torch.zeros(F_active.size(0), device=F_active.device)

        if F_active.dim() == 1:         # (d,)
            F_active = F_active.unsqueeze(0).unsqueeze(0)   # (1,1,d)
        elif F_active.dim() == 2:       # (n_active,d)
            F_active = F_active.unsqueeze(0)                # (1,n_active,d)
        # else (B,n_active,d)

        B, N, d = F_active.shape
        diff  = F_active[:, :, None, :] - C[None, None, :, :]   # (B,N,Kc,d)
        dist2 = (diff ** 2).sum(dim=-1)                         # (B,N,Kc)
        sim   = torch.exp(-beta * dist2)                        # (B,N,Kc)

        w = torch.softmax(self.psi, dim=0)                      # (Kc,)
        score_per_slot = (sim * w[None, None, :]).sum(dim=2)    # (B,N)
        return score_per_slot.mean(dim=1)                       # (B,)


class SlotBank(nn.Module):
    def __init__(self, per_class_centers: Dict[int, List[torch.Tensor]], C: int, learn_centers: bool = False):
        super().__init__()
        banks = []
        dims = []
        for c in range(C):
            vecs = per_class_centers.get(c, [])
            if len(vecs) > 0:
                centers = torch.stack(vecs, dim=0)  # (Kc, d)
                banks.append(Proto(centers, learn_centers=learn_centers))
                dims.append(centers.size(1))
            else:
                banks.append(Proto(torch.empty(0, 0), learn_centers=False))
                dims.append(0)
        self.banks = nn.ModuleList(banks)
        # 차원 체크(전 클래스 동일 가정)
        self._dim = max(dims) if len(dims) else 0

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, F_active: torch.Tensor, beta: float) -> torch.Tensor:
        """
        F_active: (B, n_active, d)
        return  : (B, C)  각 클래스 bank 점수
        """
        B = F_active.size(0)
        out = []
        for p in self.banks:
            if p.dim == 0:
                out.append(torch.zeros(B, device=F_active.device))
            else:
                out.append(p.score(F_active, beta))
        return torch.stack(out, dim=1)  # (B,C)


# -----------------------------
# SlotHead — 베이스 로짓과 가중합
# -----------------------------
class SlotHead(nn.Module):
    def __init__(self,
                 per_class_centers: Dict[int, List[torch.Tensor]],
                 C: int,
                 beta: float = 3.0,
                 xy_weight: float = 1.0,
                 tau: float = 0.7,
                 feature_dim_hint: Optional[int] = None,
                 learn_alpha: bool = True,
                 learn_centers: bool = False,
                 slot_scale: float = 1.0,
                 device: Optional[torch.device] = None):
        """
        per_class_centers: {cid: [Tensor(d), ...]}
        C: #classes
        """
        super().__init__()
        device = device or torch.device("cpu")

        # 프로토뱅크
        self.bank = SlotBank(per_class_centers, C=C, learn_centers=learn_centers).to(device)
        self.beta = beta
        self.xy_weight = xy_weight
        self.tau = tau
        self.C = C
        self.slot_scale = nn.Parameter(torch.tensor(float(slot_scale), dtype=torch.float32), requires_grad=True)

        # alpha: sigmoid(alpha_param)로 base/slot 혼합
        if learn_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # 0.0 -> alpha=0.5 초기
        else:
            self.register_buffer("alpha_param", torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _load_per_class(proto_json: str, device: torch.device) -> Dict[int, List[torch.Tensor]]:
        with open(proto_json, "r") as f:
            data = json.load(f)
        per_class = data.get("per_class", {})
        out: Dict[int, List[torch.Tensor]] = {}
        for k, block in per_class.items():
            cid = int(k)
            vecs: List[torch.Tensor] = []
            if isinstance(block, dict):
                mus = block.get("mu") or block.get("center") or block.get("centers")
                if mus is not None:
                    arr = np.asarray(mus, dtype=np.float32)
                    if arr.ndim == 1:
                        vecs.append(torch.tensor(arr, device=device))
                    else:
                        for v in arr:
                            vecs.append(torch.tensor(v, device=device))
                else:
                    inner = block.get("protos") or block.get("clusters") or block.get("items") or []
                    if isinstance(inner, list):
                        for p in inner:
                            if isinstance(p, dict):
                                mv = p.get("mu") or p.get("center") or p.get("centers")
                            else:
                                mv = p
                            if mv is None: continue
                            vecs.append(torch.tensor(np.asarray(mv, np.float32), device=device))
            elif isinstance(block, list):
                for p in block:
                    mv = p.get("mu") if isinstance(p, dict) else p
                    if mv is None: continue
                    vecs.append(torch.tensor(np.asarray(mv, np.float32), device=device))
            else:
                vecs.append(torch.tensor(np.asarray(block, np.float32), device=device))
            out[cid] = vecs
        return out

    @classmethod
    def from_json(cls, proto_json: str, C: int, **kwargs):
        device = kwargs.get("device", torch.device("cpu"))
        per_class = cls._load_per_class(proto_json, device=device)
        return cls(per_class, C=C, **kwargs).to(device)

    def forward(self,
                base_logits: torch.Tensor,   # (B,C)
                aux: dict,                   # must contain A_maps:(B,M,H,W), slot_mask:(M,), optionally S_slots:(B,M,D), XY:(B,M,2), head_energy:(B,M)
                feature_mode: str = "s+xy",
                xy_weight: Optional[float] = None,
                ) -> torch.Tensor:
        """
        return fused logits: alpha*base + (1-alpha)*slot_logits
        """
        A_maps: torch.Tensor = aux["A_maps"]          # (B,M,H,W)
        slot_mask: torch.Tensor = aux.get("slot_mask", torch.ones(A_maps.size(1), device=A_maps.device))
        E_raw: Optional[torch.Tensor] = aux.get("head_energy", None)
        B, M = A_maps.size(0), A_maps.size(1)

        # slot probs (differentiable gate)
        p = make_slot_prob(E_raw, A_maps, slot_mask, tau=self.tau)   # (B,M)

        # features
        # S_slots (if provided) & XY (if provided via COM from A_maps)
        S_slots: Optional[torch.Tensor] = aux.get("S_slots", None)   # (B,M,D) optional
        XY: Optional[torch.Tensor]      = aux.get("XY", None)        # (B,M,2) optional
        if XY is None:
            # compute COM from A_maps (detach to avoid heavy grads? keep grad to A if wanted)
            B_, M_, H, W = A_maps.shape
            yy, xx = torch.meshgrid(torch.arange(H, device=A_maps.device),
                                    torch.arange(W, device=A_maps.device), indexing="ij")
            mass = A_maps.flatten(2).sum(dim=2) + 1e-8     # (B,M)
            cx = (A_maps * xx).flatten(2).sum(dim=2) / mass
            cy = (A_maps * yy).flatten(2).sum(dim=2) / mass
            XY = torch.stack([cx / (W - 1 + 1e-8), cy / (H - 1 + 1e-8)], dim=2)  # (B,M,2)

        # pick target dim from bank
        d = self.bank.dim
        if d == 0:
            slot_logits = torch.zeros_like(base_logits)
        else:
            xyw = self.xy_weight if xy_weight is None else float(xy_weight)
            if (feature_mode == "s") and (S_slots is None):
                # fallback to XY only if S missing
                feature_mode = "xy"

            if feature_mode == "s":
                F_full = build_feat_for_dim(S_slots, XY, d, xyw)         # (B,M,d)
            elif feature_mode == "xy":
                F_full = build_feat_for_dim(torch.zeros_like(XY[..., :1]).repeat(1,1,d-2) if d>2 else XY, XY, d, xyw)
            else:
                # s+xy; if S_slots None, will raise; guard:
                if S_slots is None:
                    # emulate s=0 then concat XY
                    s_dummy = torch.zeros(B, M, max(1, d-2), device=A_maps.device)
                    F_full = build_feat_for_dim(s_dummy, XY, d, xyw)
                else:
                    F_full = build_feat_for_dim(S_slots, XY, d, xyw)     # (B,M,d)

            # 확률 p로 가중 평균(소프트 top-p 느낌). (B,n_active,d)로 변환
            # 여기서는 모든 슬롯을 확률로 가중해 모으되, 수치안정 위해 p>0인 것만 취사
            # 간단 구현: p를 normalized weight로 쓰지 않고, (B,M,1)*F_full → 슬롯 축 평균
            # 은근히 약해질 수 있어, 은닉에서 n_active를 유지하기 위해 샘플별로 상위 슬롯만 뽑아 사용
            # 상위 슬롯 수 = max(1, round(M*0.5))  (원하면 인자로 뺄 수 있음)
            k = max(1, int(round(0.5 * M)))
            idx = torch.topk(p, k=k, dim=1).indices                                  # (B,k)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)                         # (B,k,d)
            F_active = F_full.gather(dim=1, index=gather_idx)                        # (B,k,d)

            # 클래스별 proto bank 점수
            slot_scores = self.bank(F_active, beta=self.beta)                        # (B,C)
            slot_logits = self.slot_scale * slot_scores

        alpha = torch.sigmoid(self.alpha_param)
        fused = alpha * base_logits + (1.0 - alpha) * slot_logits
        return fused
