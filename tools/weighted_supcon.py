# tools/weighted_supcon.py
import torch, torch.nn as nn, torch.nn.functional as F

class WeightedSupCon(nn.Module):
    def __init__(self, temperature=0.07, semi_hard_margin=None):
        super().__init__()
        self.t = temperature
        self.m = semi_hard_margin

    def forward(self, feats, labels, anchor_w=None, cluster_id=None):
        """
        feats: [B, V, D]  (V=2 권장)
        labels: [B]
        anchor_w: [B] or None (budget-normalized outside or inside)
        cluster_id: [C]→gid 매핑을 labels에 적용할 수 있도록 tensor/list 길이>=max(labels)+1
        """
        device = feats.device
        B, V, D = feats.shape
        z = F.normalize(feats.reshape(B*V, D), dim=1)               # [BV,D]
        y = labels.view(B,1).expand(B,V).reshape(B*V)               # [BV]
        same = torch.eq(y.unsqueeze(0), y.unsqueeze(1)).to(device)  # [BV,BV]
        eye = torch.eye(B*V, device=device).bool()
        pos_mask = same & (~eye)

        # cluster-aware neg weight
        if cluster_id is not None:
            # gid per base sample (repeat for V views)
            gid = torch.tensor(cluster_id, device=device, dtype=torch.long)
            gid_y = gid[y]
            same_cluster = torch.eq(gid_y.unsqueeze(0), gid_y.unsqueeze(1))  # [BV,BV]
            # in-cluster negs: 1.3, out-cluster negs: 1.0
            neg_w = torch.where(same_cluster & (~same), torch.full_like(same_cluster, 1.3, dtype=torch.float),
                                torch.ones_like(same_cluster, dtype=torch.float))
        else:
            neg_w = torch.ones(B*V, B*V, device=device)

        # sim & log_prob
        sim = (z @ z.t()) / self.t
        sim = sim.masked_fill(eye, -1e9)
        if self.m is not None:
            # semi-hard: only keep negatives within margin (optional)
            with torch.no_grad():
                mx, _ = sim.max(dim=1, keepdim=True)
            keep = (mx - sim) <= self.m
            neg_w = neg_w * keep.float()

        log_prob = F.log_softmax(sim, dim=1)            # [BV,BV]

        # positive term (unweighted; standard)
        pos_log = (log_prob * pos_mask.float()).sum(dim=1)
        pos_cnt = pos_mask.float().sum(dim=1).clamp_min(1.0)
        loss_per_anchor = - pos_log / pos_cnt           # [BV]

        # apply anchor weights on one view per sample to avoid double count
        anchors = torch.arange(0, B*V, V, device=device)
        loss = loss_per_anchor[anchors]                 # [B]

        if anchor_w is not None:
            w = anchor_w.to(device)
            w = w / (w.mean() + 1e-9)                  # budget-preserving
            loss = loss * w

        return loss.mean()
