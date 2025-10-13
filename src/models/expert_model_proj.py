import torch
import torch.nn as nn
from src.models.trunk_proj import TrunkProj
from src.models.expert_moe import MoELayer, LoadBalanceLoss, EntropyLoss

class ExpertModelProj(nn.Module):
    """
    Combines TrunkProj with MoE layer and classifier, plus losses.
    """
    def __init__(self, cfg):
        super().__init__()
        self.trunk = TrunkProj(cfg)
        D = cfg.model.proj_out_dim
        # MoE layer
        self.moe = MoELayer(
            input_dim=D,
            num_experts=cfg.model.num_experts,
            expert_hidden=cfg.model.expert_hidden,
            expert_out=D,
            top_k=cfg.model.top_k
        )
        # classifier
        self.classifier = nn.Linear(D, cfg.model.num_classes)
        # losses

    def forward(self, x):
        features = self.trunk(x)
        logits, p = self.moe(features)
        logits = self.classifier(logits)
        return logits, p

    def forward_all_expert_logits(self, x):
        features = self.trunk(x)
        outs_all, p, gate_logits = self.moe.forward_all(features)
        B, E, D = outs_all.shape
        logits_all = self.classifier(outs_all.view(B*E, D)).view(B, E, -1)
        return logits_all, p, gate_logits
    def conv2corr_penalty(self):
        # projection 기반 실험에선 conv2 페널티가 없으므로 항상 0을 반환
        return torch.tensor(0.0, device=next(self.parameters()).device)