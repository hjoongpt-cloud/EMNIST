# qnext/core/patch_init.py
import torch, math
import torch.nn.functional as F

@torch.no_grad()
def extract_patches(dl, max_images=20000, patch_size=9, stride=1, energy_thresh=0.1, device="cuda"):
    """train dataloader에서 1채널 이미지를 받아 9x9 패치들 추출 후 [N, 81] 반환"""
    ps = patch_size
    patches = []
    seen = 0
    for x, _ in dl:
        x = x.to(device)
        # x: [B,1,H,W]
        # unfold: [B, C*ps*ps, L]
        uf = F.unfold(x, kernel_size=ps, stride=stride)  # [B, 1*ps*ps, L]
        uf = uf.transpose(1, 2).reshape(-1, ps*ps)       # [B*L, 81]
        # 에너지(ℓ2노름) 필터
        e = torch.norm(uf, dim=1)
        keep = e > energy_thresh
        uf = uf[keep]
        patches.append(uf)
        seen += x.size(0)
        if seen >= max_images: break
    if len(patches)==0:
        return torch.empty(0, ps*ps, device=device)
    P = torch.cat(patches, dim=0)  # [N,81]
    # mean-center + ℓ2 normalize
    P = P - P.mean(dim=1, keepdim=True)
    P = F.normalize(P, p=2, dim=1, eps=1e-6)
    return P

@torch.no_grad()
def spherical_kmeans(P, K, iters=20, retries=2, seed=42):
    """코사인 거리 기반 kmeans. P:[N,D](ℓ2정규화), 반환 C:[K,D] (정규화됨)"""
    if P.numel()==0:
        raise RuntimeError("No patches for kmeans")
    torch.manual_seed(seed)
    N, D = P.shape
    best_centers, best_inertia = None, float("inf")
    for r in range(retries):
        # kmeans++ 간단 초기화
        idx = torch.randint(0, N, (1,), device=P.device)
        centers = P[idx].clone()  # [1,D]
        for _ in range(1, K):
            sim = torch.mm(P, centers.t())                      # [N,k]
            dist = (1 - sim.clamp(-1,1))                        # 코사인 거리 ~ (1-cos)
            dmin, _ = dist.min(dim=1)
            probs = (dmin + 1e-9) / (dmin.sum() + 1e-9)
            new_idx = torch.multinomial(probs, 1)
            centers = torch.cat([centers, P[new_idx]], dim=0)   # [k+1,D]
        # 반복
        for _ in range(iters):
            sim = torch.mm(P, centers.t())                      # [N,K]
            assign = sim.argmax(dim=1)                          # [N]
            # 각 클러스터 평균(코사인 정규화)
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (assign==k)
                if mask.any():
                    c = P[mask].mean(dim=0)
                    c = F.normalize(c, p=2, dim=0, eps=1e-6)
                    new_centers[k] = c
                else:
                    # 빈 클러스터는 랜덤 패치로 재시작
                    ridx = torch.randint(0, N, (1,), device=P.device)
                    new_centers[k] = P[ridx]
            centers = new_centers
        # inertia(1-cos) 합계
        sim = torch.mm(P, centers.t())
        dist = (1 - sim.clamp(-1,1))
        inertia = dist.min(dim=1)[0].sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.clone()
    return best_centers  # [K,D]

@torch.no_grad()
def centers_to_conv1(centers, conv1, patch_size=9):
    """centers:[K, ps*ps] -> conv1.weight:[K,1,ps,ps]로 복사 + ℓ2 정규화"""
    K, D = centers.shape
    ps = patch_size
    assert D == ps*ps, f"Center dim {D} != {ps*ps}"
    W = centers.view(K, 1, ps, ps)
    # 필터 ℓ2 정규화
    W = F.normalize(W.view(K, -1), p=2, dim=1, eps=1e-6).view_as(W)
    conv1.weight.copy_(W)
    return K
