# src/models/expert_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with optional top-k routing.
    """
    def __init__(self, input_dim, num_experts, expert_hidden, expert_out, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_out)
            ) for _ in range(num_experts)
        ])

        # gating network (single Sequential)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_experts)
        )

    def forward(self, x):
        if self.num_experts == 1:
            # fast path: single expert but return deterministic gate p=(1,)
            y = self.experts[0](x)
            p = torch.ones(x.size(0), 1, device=x.device)
            return y, p

        gate_logits = self.gate(x)  # (B, E)
        if self.top_k < self.num_experts:
            topv, topi = torch.topk(gate_logits, self.top_k, dim=1)
            top_p = F.softmax(topv, dim=1)          # (B, k)
            p = torch.zeros_like(gate_logits).scatter_(1, topi, top_p)
        else:
            p = F.softmax(gate_logits, dim=1)       # (B, E)

        out = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B,E,O)
        y = torch.sum(p.unsqueeze(-1) * out, dim=1)
        return y, p

    def forward_all(self, x):
        if self.num_experts == 1:
            out = self.experts[0](x).unsqueeze(1)           # [B,1,D_out]
            p = torch.ones(x.size(0), 1, device=x.device)   # [B,1]
            gate_logit = torch.zeros(x.size(0), 1, device=x.device)  # [B,1]
            return out, p, gate_logit
        gate_logit = self.gate(x)                           # [B,E]
        p = F.softmax(gate_logit, dim=-1)
        outs = [exp(x) for exp in self.experts]             # list of [B,D_out]
        outs_all = torch.stack(outs, dim=1)                 # [B,E,D_out]
        return outs_all, p, gate_logit
    
class LoadBalanceLoss(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.E = num_experts

    def forward(self, p):
        if p is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        pi = p.mean(dim=0)
        ideal = 1.0 / self.E
        return torch.sum((pi - ideal) ** 2)


class EntropyLoss(nn.Module):
    def forward(self, p):
        if p is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        ent = -torch.sum(p * torch.log(p + 1e-9), dim=1)
        return ent.mean()
