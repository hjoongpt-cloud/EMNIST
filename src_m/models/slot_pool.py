
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelPosBias2D(nn.Module):
    """
    Head-specific relative position bias for 2D grids.
    Produces a (n_heads, M, N) tensor given query coordinates (M,2) and key coords (N,2).
    """
    def __init__(self, H: int, W: int, n_heads: int):
        super().__init__()
        self.H, self.W, self.nh = H, W, n_heads
        self.bias = nn.Parameter(torch.zeros(n_heads, 2*H-1, 2*W-1))

    @torch.no_grad()
    def reset_parameters(self, nearby_pos_boost: float = 0.0, far_neg: float = 0.0):
        nn.init.zeros_(self.bias)
        if nearby_pos_boost != 0.0 or far_neg != 0.0:
            # heuristic init: boost near neighbors, penalize far
            cy, cx = self.H-1, self.W-1
            for h in range(self.nh):
                b = torch.zeros_like(self.bias[h])
                for dy in range(-self.H+1, self.H):
                    for dx in range(-self.W+1, self.W):
                        r = math.sqrt(float(dy*dy + dx*dx))
                        val = nearby_pos_boost * math.exp(- (r**2) / (2*(max(self.H,self.W)/6.0)**2)) - far_neg*(r/(max(self.H,self.W)))
                        b[dy+cy, dx+cx] = val
                self.bias[h].copy_(b)

    def forward(self, q_coords: torch.Tensor, k_coords: torch.Tensor):
        """
        q_coords: (M,2), k_coords: (N,2) -> returns (n_heads, M, N)
        """
        dy = q_coords[:, None, 0] - k_coords[None, :, 0]  # (M,N)
        dx = q_coords[:, None, 1] - k_coords[None, :, 1]
        # round to nearest bins
        idx_y = (dy.round() + (self.H - 1)).clamp(0, 2*self.H - 2).long()
        idx_x = (dx.round() + (self.W - 1)).clamp(0, 2*self.W - 2).long()
        B = self.bias[:, idx_y, idx_x]  # (nh, M, N)
        return B


class SlotPool(nn.Module):
    """
    Cross-attention slot pooling with head-specific relative position bias.
    - Input keys/values: token sequence T (B, N, D) from your Trunk after self-attention.
    - Learnable slot queries: M x D
    - Output:
        Z: (B, M*D) or a projected (B, D) if aggregate='proj32' or 'mean'
        A_maps: (B, M, H, W) attention maps (averaged across heads) for diagnostics
        head_energy: (B, M, n_heads) head-wise attention mass (sum over N)
        S_slots: (B, M, D) per-slot embeddings (useful for diversity loss)
    """
    def __init__(self, D=32, M=12, H=14, W=14, n_heads=4, aggregate='proj32', topk_ratio = 0.05):
        super().__init__()
        assert D % n_heads == 0, "D must be divisible by n_heads"
        self.D, self.M, self.H, self.W, self.nh = D, M, H, W, n_heads
        self.aggregate = aggregate  # 'concat' | 'proj32' | 'mean'
        Dh = D // n_heads
        self.Q = nn.Parameter(torch.randn(M, D) * 0.02)
        self.topk_ratio = topk_ratio
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)

        # coords buffers for keys (grid HxW) and slot anchors (spread)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        self.register_buffer("k_coords", torch.stack([yy.flatten(), xx.flatten()], dim=-1).float())  # (N,2)

        # roughly spread initial slot coordinates on the grid (helps early stability)
        cy = torch.linspace(0, H-1, int(math.sqrt(M))+1)
        cx = torch.linspace(0, W-1, int(math.sqrt(M))+1)
        grid = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=-1).reshape(-1,2)
        if grid.shape[0] >= M:
            init_q = grid[:M]
        else:
            # fallback: tile center
            center = torch.tensor([[H/2.0, W/2.0]]).repeat(M,1)
            init_q = center
        self.register_buffer("q_coords", init_q.float())  # (M,2)

        self.rpb = RelPosBias2D(H, W, n_heads=n_heads)

        # aggregators
        if self.aggregate == 'proj32':
            self.proj = nn.Linear(M*D, D)
        elif self.aggregate == 'mean':
            # no params
            pass

    def forward(self, T):
        """
        T: (B, N=H*W, D)
        returns: Z, A_maps, head_energy, S_slots
        """
        B, N, D = T.shape
        Dh = D // self.nh
        Q = self.q_proj(self.Q)                       # (M, D)
        K = self.k_proj(T)                            # (B, N, D)
        V = self.v_proj(T)                            # (B, N, D)

        # reshape per-head
        Qh = Q.view(self.M, self.nh, Dh).transpose(0,1)             # (nh, M, Dh)
        Kh = K.view(B, N, self.nh, Dh).permute(0,2,1,3)             # (B, nh, N, Dh)
        Vh = V.view(B, N, self.nh, Dh).permute(0,2,1,3)             # (B, nh, N, Dh)

        # content logits

        # --- content logits: QK^T/√d + relative bias ---
        # Qh: (nh, M, Dh), Kh: (B, nh, N, Dh)
        content = torch.einsum('hmd,bhnd->bhmn', Qh, Kh) / math.sqrt(Dh)  # (B, nh, M, N)
        rpb = self.rpb(self.q_coords, self.k_coords)                      # (nh, M, N)
        logits = content + rpb.unsqueeze(0)                               # (B, nh, M, N)

        # --- attention ---
        A = torch.softmax(logits, dim=-1)                                 # (B, nh, M, N)
        A_avg = A.mean(dim=1)                                  # (B, M, N)
        topk = max(1, int(0.05 * A_avg.shape[-1]))             # 상위 5% (원하면 yaml로)
        topk_mass = A_avg.topk(topk, dim=-1).values.sum(-1) 
        # --- head-wise sharpness (1 - normalized entropy) ---
        P  = A.clamp_min(1e-12)
        Hn = -(P * P.log()).sum(dim=-1)                                   # (B, nh, M)
        Hn = Hn / math.log(P.shape[-1])                                    # normalize by log(N)
        head_energy = (1.0 - Hn).permute(0, 2, 1)                         # (B, M, nh), higher = sharper

        # --- slot embeddings: concat heads -> (B, M, D) ---
        # (A @ Vh): (B, nh, M, Dh)  using batched matmul on the last two dims
        S_heads = (A @ Vh)                                                # (B, nh, M, Dh)
        S = S_heads.transpose(1, 2).contiguous().view(B, self.M, D)       # (B, M, nh*Dh=D)

        # --- aggregate slots to Z ---
        if self.aggregate == 'concat':
            Z = S.reshape(B, self.M * D)
        elif self.aggregate == 'proj32':
            Z = self.proj(S.reshape(B, self.M * D))
        elif self.aggregate == 'mean':
            Z = S.mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregate: {self.aggregate}")

        # --- diagnostics: head-avg attention map ---
        A_maps = A.mean(1).view(B, self.M, self.H, self.W)                # (B, M, H, W)

        return Z, A_maps, head_energy, S, topk_mass



