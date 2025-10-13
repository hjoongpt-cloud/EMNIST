# qnext/core/slot_tools.py
import torch

@torch.no_grad()
def slot_scores(delta_acc, delta_margin, energy, w=(0.5, 0.3, 0.2)):
    """
    delta_acc/delta_margin/energy: [S] Tensor (값이 클수록 중요/유익)
    """
    # 표준화
    def z(x):
        m = x.mean(); s = x.std().clamp_min(1e-6)
        return (x - m) / s
    z_da = z(delta_acc); z_dm = z(delta_margin); z_en = z(energy)
    w1, w2, w3 = w
    score = w1*z_da + w2*z_dm + w3*z_en
    return score

@torch.no_grad()
def pick_top_slots(score, pct=0.15, cap=0.30, min_k=4):
    S = score.numel()
    k = max(min_k, int(round(S * pct)))
    k = min(k, int(round(S * cap)))
    topk = score.topk(k=k, largest=True).indices
    return topk

@torch.no_grad()
def find_duplicate_slots(sim_matrix, thresh=0.95, max_reinit_ratio=0.2):
    """
    sim_matrix: [S,S] (cosine sim). 대각 제외 후 thresh 이상인 클러스터의 인덱스 집합 반환.
    """
    S = sim_matrix.size(0)
    dup = (sim_matrix > thresh).float()
    dup.fill_diagonal_(0.0)
    groups = []
    used = torch.zeros(S, dtype=torch.bool, device=sim_matrix.device)
    for i in torch.argsort(dup.sum(dim=1), descending=True):
        i = int(i)
        if used[i]: continue
        group = (dup[i] > 0).nonzero(as_tuple=False).view(-1).tolist()
        if len(group) == 0: continue
        group = list(set(group + [i]))
        for g in group: used[g] = True
        groups.append(group)
    # 재초기화 대상 수 제한
    total = sum(len(g)-1 for g in groups)  # 각 그룹에서 1개 유지 가정
    limit = int(S * max_reinit_ratio)
    return groups[:], min(total, limit)
