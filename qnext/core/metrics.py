# qnext/core/metrics.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def batch_slot_cossim(slot_feats_bsd):
    """
    slot_feats_bsd: [B, S, D] (배치의 슬롯 임베딩)
    반환:
      M_avg: (S,S) 평균 cos-sim 행렬 (cpu tensor)
      stats: dict(offdiag_mean, offdiag_p90, offdiag_p95, offdiag_p99)
    """
    if slot_feats_bsd is None:
        return None, {}

    if slot_feats_bsd.ndim != 3:
        return None, {}

    B, S, D = slot_feats_bsd.shape
    if B == 0 or S == 0:
        return None, {}

    z = F.normalize(slot_feats_bsd.reshape(B * S, D), dim=1)  # [B*S, D]
    Z = z.reshape(B, S, D)                                    # [B, S, D]

    sims = []
    for b in range(B):
        M = Z[b] @ Z[b].T                                     # [S, S]
        sims.append(M)
    M_avg = torch.stack(sims, 0).mean(0)                      # [S, S]

    off = M_avg[~torch.eye(S, dtype=torch.bool, device=M_avg.device)]
    if off.numel() == 0:
        stats = {"offdiag_mean": 0.0, "offdiag_p90": 0.0, "offdiag_p95": 0.0, "offdiag_p99": 0.0}
        return M_avg.detach().cpu(), stats

    off_sorted, _ = torch.sort(off)
    def kth(p):
        k = max(0, min(off_sorted.numel()-1, int(round(p * (off_sorted.numel()-1)))))
        return float(off_sorted[k].item())

    stats = {
        "offdiag_mean": float(off.mean().item()),
        "offdiag_p90":  kth(0.90),
        "offdiag_p95":  kth(0.95),
        "offdiag_p99":  kth(0.99),
    }
    return M_avg.detach().cpu(), stats
