import torch
import torch.nn as nn
from src.models.trunk_i import TrunkI
from src.models.expert_moe import MoELayer, LoadBalanceLoss, EntropyLoss

class ExpertModelI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.trunk = TrunkI(cfg)
        self.D = cfg.model.proj_out_dim

        # MoE layer
        self.moe = MoELayer(
            input_dim=self.D,
            num_experts=cfg.model.num_experts,
            expert_hidden=cfg.model.expert_hidden,
            expert_out=self.D,
            top_k=cfg.model.top_k
        )
        self.classifier = nn.Linear(self.D, cfg.model.num_classes)

        # losses (original interface: pass num_experts, apply lambda externally)
        self.lb_loss = LoadBalanceLoss(cfg.model.num_experts)
        self.ent_loss = EntropyLoss()

        # channel embedding (already handled inside TrunkH for feature, but keep if needed)
        # NOTE: TrunkH already projects sparse → D, so ExpertModelH does not need its own channel_embed here.

    def forward_features(self, x):
        pooled, _, _ = self.trunk(x)  # use only pooled feature for MoE input
        return pooled  # (B, D)

    def forward(self, x):
        feat = self.forward_features(x)
        moe_out, p = self.moe(feat)
        logits = self.classifier(moe_out)
        return logits, p

    def forward_all_expert_logits(self, x):
        feat = self.forward_features(x)
        outs_all, p, gate_logits = self.moe.forward_all(feat)
        B, E, D = outs_all.shape
        logits_all = self.classifier(outs_all.view(B * E, D)).view(B, E, -1)
        return logits_all, p, gate_logits

    def conv2corr_penalty(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
