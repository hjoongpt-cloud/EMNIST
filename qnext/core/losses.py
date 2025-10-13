# qnext/core/losses.py
import torch
import torch.nn.functional as F

def margin_penalty(logits, y, m_star=0.5, weight=0.3):
    """
    L = weight * mean( relu(m_star - (logit_true - best_rival)) )
    """
    true = logits[torch.arange(logits.size(0), device=logits.device), y]
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), y] = -1e9
    rival, _ = masked.max(dim=1)
    margin = true - rival
    pen = torch.relu(m_star - margin)
    return weight * pen.mean(), margin.detach()

def supcon_loss(z, y, temp=0.15, eps=1e-9):
    """
    supervised contrastive (평범한 버전, 배치 내 같은 클래스 긍정, 나머지 부정)
    z:[B,D], y:[B]
    """
    z = F.normalize(z, p=2, dim=1)
    sim = z @ z.t() / max(temp, 1e-6)  # (B,B)
    mask_pos = (y[:, None] == y[None, :]).to(z.dtype)
    mask_self = torch.eye(z.size(0), device=z.device, dtype=z.dtype)
    mask_pos = mask_pos - mask_self  # self 제외

    # log-softmax
    logits = sim - torch.max(sim, dim=1, keepdim=True).values
    exp_logits = torch.exp(logits) * (1 - mask_self)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    # positives 평균
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + eps)
    loss = -mean_log_prob_pos.mean()
    return loss
