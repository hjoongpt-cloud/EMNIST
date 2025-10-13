# src/models/expert_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with optional top-k routing.
    """
    def __init__(self, input_dim, num_experts, expert_hidden, expert_out, top_k=1,use_input_bias=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_input_bias = use_input_bias
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
        self.expert_input_bias = nn.Parameter(torch.zeros(num_experts, input_dim))
        nn.init.normal_(self.expert_input_bias, mean=0.0, std=0.02)
        self.stopgrad_gate = False            # if True, no grad flows through p
        self.gate_temperature = 1.0           # softmax temperature
        self.gate_mix_alpha = 1.0             # blend: alpha*current + (1-alpha)*teacher
        self.gate_teacher = None
    def _apply_experts(self, x):  # x: [B, D]
        outs = []
        if self.use_input_bias:
            for e, expert in enumerate(self.experts):
                outs.append(expert(x + self.expert_input_bias[e]))
        else:
            for expert in self.experts:
                outs.append(expert(x))
        return torch.stack(outs, dim=1)  # [B, E, D_out]
    def forward(self, x):
        if self.num_experts == 1:
            # fast path: single expert but return deterministic gate p=(1,)
            y = self.experts[0](x + self.expert_input_bias[0] if self.use_input_bias else x)
            p = torch.ones(x.size(0), 1, device=x.device)
            return y, p

        gate_logits = self.gate(x)  # (B, E)
        
        
        T = max(self.gate_temperature, 1e-6)
        p_cur = F.softmax(gate_logits / T, dim=1)

        # --- optional teacher blend (no grad through teacher) ---
        if self.gate_teacher is not None and self.gate_mix_alpha < 1.0:
            with torch.no_grad():
                p_t = F.softmax(self.gate_teacher(x) / T, dim=1)
            p = self.gate_mix_alpha * p_cur + (1.0 - self.gate_mix_alpha) * p_t
        else:
            p = p_cur

        if self.top_k < self.num_experts:
            topv, topi = torch.topk(p, self.top_k, dim=1)
            p_sparse = torch.zeros_like(p).scatter_(1, topi, topv)
            p = p_sparse / (p_sparse.sum(dim=1, keepdim=True) + 1e-9)


        outs_all = self._apply_experts(x)  # (B,E,O)
        p_used = p.detach() if self.stopgrad_gate else p
        y = torch.sum(p_used.unsqueeze(-1) * outs_all, dim=1)
        return y, p




    def forward_all(self, x):
        if self.num_experts == 1:
            out = self.experts[0](x).unsqueeze(1)           # [B,1,D_out]
            p = torch.ones(x.size(0), 1, device=x.device)   # [B,1]
            gate_logit = torch.zeros(x.size(0), 1, device=x.device)  # [B,1]
            return out, p, gate_logit
        gate_logit = self.gate(x)
        T = max(self.gate_temperature, 1e-6)
        p = F.softmax(gate_logit / T, dim=-1)
        outs = [exp(x) for exp in self.experts]             # list of [B,D_out]
        outs_all = self._apply_experts(x)                 # [B,E,D_out]
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
