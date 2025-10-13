
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ClassifierHead(nn.Module):
    """
    Two modes:
      - concat: Linear(M*D -> C)
      - proj32: Linear(M*D -> 32) -> Linear(32 -> C)
    Supports group-lasso penalty per slot block for automatic slot pruning.
    """
    def __init__(self, mode: str, M: int, D: int, num_classes: int):
        super().__init__()
        assert mode in ("concat", "proj32", "mean_proj32")
        self.mode = mode
        self.M, self.D = M, D
        if mode == "concat":
            self.fc = nn.Linear(M*D, num_classes)
            self.proj = None
        elif mode == "proj32":
            self.proj = nn.Linear(M*D, 32)
            self.fc = nn.Linear(32, num_classes)
        elif mode == "mean_proj32":
            # for Trunk that outputs mean(D); keep interface
            self.proj = nn.Linear(D, 32)
            self.fc = nn.Linear(32, num_classes)

        # slot mask for pruning (1: alive, 0: pruned)
        self.register_buffer("slot_mask", torch.ones(M))

    def forward(self, Z):
        if self.mode == "concat" or self.mode == "proj32":
            # apply slot mask as feature gating (block-wise)
            B = Z.shape[0]
            Z_view = Z.view(B, self.M, self.D)
            Z_view = Z_view * self.slot_mask.view(1, self.M, 1)  # zero pruned slots
            Z_flat = Z_view.reshape(B, self.M * self.D)
            if self.mode == "concat":
                return self.fc(Z_flat)
            else:
                h = self.proj(Z_flat)
                return self.fc(h)
        else:
            # mean_proj32: Z is already (B,D)
            h = self.proj(Z)
            return self.fc(h)

    def group_lasso_penalty(self, lambda_gl: float = 1e-3):
        """
        Compute group lasso on slot blocks (based on *input* weights).
        """
        if self.mode == "concat":
            W = self.fc.weight  # (C, M*D)
            # partition into M blocks of size D
            norms = []
            for m in range(self.M):
                block = W[:, m*self.D:(m+1)*self.D]
                norms.append(block.norm(p=2))
            return lambda_gl * torch.stack(norms).sum()
        elif self.mode == "proj32":
            W = self.proj.weight  # (32, M*D)
            norms = []
            for m in range(self.M):
                block = W[:, m*self.D:(m+1)*self.D]
                norms.append(block.norm(p=2))
            return lambda_gl * torch.stack(norms).sum()
        else:
            return torch.tensor(0.0, device=self.fc.weight.device)

    @torch.no_grad()
    def prune_by_threshold(self, threshold: float):
        """
        Mark slots with small group norms as pruned (mask to 0). No hard reparam of layers yet.
        """
        if self.mode == "concat":
            W = self.fc.weight  # (C, M*D)
            norms = []
            for m in range(self.M):
                block = W[:, m*self.D:(m+1)*self.D]
                norms.append(block.norm(p=2).item())
        elif self.mode == "proj32":
            W = self.proj.weight
            norms = []
            for m in range(self.M):
                block = W[:, m*self.D:(m+1)*self.D]
                norms.append(block.norm(p=2).item())
        else:
            norms = [1.0]*self.M
        norms = torch.tensor(norms, device=self.slot_mask.device)
        self.slot_mask = (norms >= threshold).float()
        return norms.cpu().numpy()
