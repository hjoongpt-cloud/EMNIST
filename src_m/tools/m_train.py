import os, argparse, json, random, math, yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from src_m.models.trunk_m import TrunkM
from src_m.models.clf_head import ClassifierHead
from src_m.losses.slot_aux import slot_diversity_loss, coverage_entropy_loss
from src_m.losses.group_lasso import group_lasso_penalty_from_head


# ----------------------------
# Utils
# ----------------------------
def as_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def as_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_loaders(batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), subset=None):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=tf)
    test  = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=tf)
    if subset is not None:
        train = torch.utils.data.Subset(train, list(range(subset)))
        test  = torch.utils.data.Subset(test, list(range(min(max(1, subset//5), len(test)))))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def build_model(cfg):
    mcfg = cfg["model"]
    trunk = TrunkM(
        proj_in_channels=mcfg.get("proj_in_channels",150),
        proj_out_dim=mcfg.get("proj_out_dim",32),
        num_heads=mcfg.get("num_heads",4),
        conv1_stride=mcfg.get("conv1_stride",2),
        global_topk_ratio=mcfg.get("global_topk_ratio",0.08),
        k_ch=mcfg.get("k_ch",8),
        slot_M=mcfg.get("slot_M",12),
        slot_aggregate=mcfg.get("slot_aggregate","proj32"),
        pretrained_filter_path=mcfg.get("pretrained_filter_path", None),
        freeze_conv1=mcfg.get("freeze_conv1", False),
        H=14, W=14
    )
    # head 입력 모드: concat이면 (B, M*D), proj32/mean이면 (B, D)
    agg = mcfg.get("slot_aggregate", "proj32")
    mode = "concat" if agg == "concat" else ("proj32" if agg == "proj32" else "mean_proj32")
    head = ClassifierHead(mode=mode, M=mcfg.get("slot_M",12), D=mcfg.get("proj_out_dim",32), num_classes=47)
    return trunk, head

@torch.no_grad()
def evaluate(trunk, head, loader, device):
    trunk.eval(); head.eval()
    correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        Z, aux = trunk(x)
        logits = head(Z)
        pred = logits.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct/total

def usage_gini_from_topkmass(topk_mass: torch.Tensor, slot_mask: torch.Tensor = None) -> float:
    """
    topk_mass: (B, M) 또는 (M,)  — slot_pool에서 평균한 상위 p% 토큰 질량.
    """
    if topk_mass.dim() == 2:
        tm = topk_mass.mean(dim=0)  # (M,)
    else:
        tm = topk_mass
    if slot_mask is not None:
        tm = tm * slot_mask.to(tm.device).float()
    arr = tm.detach().cpu().numpy()
    s = arr.sum()
    if s <= 0:
        return 0.0
    arr = arr / s
    arr_sorted = np.sort(arr)
    cum = np.cumsum(arr_sorted)
    gini = 1.0 - 2.0 * np.trapz(cum, dx=1.0/len(arr))
    return float(gini)

def compute_slot_div_mean(S_slots: torch.Tensor) -> float:
    # S_slots: (B, M, D) → (M, D) 정규화 후 상삼각 코사인 제곱 평균
    S = F.normalize(S_slots.mean(dim=0), dim=-1)   # (M,D)
    sim = (S @ S.t())                              # (M,M)
    vals = torch.triu(sim, diagonal=1)
    vals = vals[vals.abs() > 0]
    return float((vals**2).mean().item()) if vals.numel() else 0.0


# ----------------------------
# Train One Seed
# ----------------------------
def train_one_seed(cfg, out_dir, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)

    optcfg = cfg.get("opt", {})
    trcfg  = cfg.get("train", {})

    # 강제 캐스팅
    epochs = as_int(optcfg.get("epochs", trcfg.get("epochs", 10)), 10)
    bs     = as_int(optcfg.get("batch_size", trcfg.get("batch_size", 256)), 256)
    lr     = as_float(optcfg.get("lr", 1e-3), 1e-3)
    nw     = as_int(trcfg.get("num_workers", 2), 2)

    subset = trcfg.get("subset", None)
    if isinstance(subset, str):
        subset = None if subset.lower() in ("none", "null", "") else as_int(subset, None)

    # pruning 하이퍼
    pr = cfg.get("pruning", {})
    enable_gl        = bool(pr.get("enable_group_lasso", True))
    lambda_gl        = as_float(pr.get("lambda_gl", 1e-3), 1e-3)
    warmup_epochs    = as_int(pr.get("warmup_epochs", 2), 2)
    prune_check_every= as_int(pr.get("prune_check_every", 1), 1)
    final_tune_epochs= as_int(pr.get("final_tune_epochs", 3), 3)
    ft_start_epoch   = epochs - final_tune_epochs + 1      # 프루닝 직후 첫 파인튜닝 에폭
    head_only_epochs = min(2, final_tune_epochs) 
    # 사후 프루닝 설정
    apply_during    = bool(pr.get("apply_during_train", False))    # 학습 중 프루닝 비활성(default)
    post_keep_ratio = as_float(pr.get("post_keep_ratio", 0.7), 0.7)
    post_score      = pr.get("post_score", "norm")  # "norm" or "norm_plus_usage"

    # slot losses
    sl = cfg.get("slot_losses", {})
    lambda_div      = as_float(sl.get("lambda_div", 0.10), 0.10)
    lambda_cov      = as_float(sl.get("lambda_cov", 0.05), 0.05)
    target_entropy  = as_float(sl.get("target_entropy", 2.0), 2.0)

    # normalize
    mean = tuple(as_float(v, 0.1307) for v in cfg.get("normalize",{}).get("mean",[0.1307]))
    std  = tuple(as_float(v, 0.3081) for v in cfg.get("normalize",{}).get("std",[0.3081]))

    best=0.0
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"resolved_hparams.json"),"w") as f:
        json.dump(dict(epochs=epochs,batch_size=bs,lr=lr,seed=seed,mean=mean,std=std), f, indent=2)

    train_loader, test_loader = get_loaders(batch_size=bs, num_workers=nw, mean=mean, std=std, subset=subset)
    trunk, head = build_model(cfg)
    trunk.to(device); head.to(device)

    params = list(trunk.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 사용량 EMA (slot 중요도 추정; topk_mass 필요)
    M_slots = cfg["model"].get("slot_M", 12)
    usage_ema = torch.zeros(M_slots, device=device)
    ema_alpha = 0.9

    log_path = os.path.join(out_dir, "train_metrics.jsonl")
    did_post_prune = False

    for epoch in range(1, epochs+1):
        trunk.train(); head.train()
        
        if epoch >= ft_start_epoch:
            # 파인튜닝 구간 안이면: 몇 번째 파인튜닝 에폭인지 계산 (1..final_tune_epochs)
            ft_pos = epoch - ft_start_epoch + 1
            head_only = (ft_pos <= head_only_epochs)
            for p in trunk.parameters():
                p.requires_grad = (not head_only)   # 앞 1~2에폭: trunk 고정 / 이후: trunk 풀기
        else:
            # 파인튜닝 전: 평소대로 trunk 학습
            for p in trunk.parameters():
                p.requires_grad = True
        epoch_loss=0.0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            Z, aux = trunk(x)
            logits = head(Z)

            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            L = ce
            if lambda_div>0: L += lambda_div * slot_diversity_loss(aux["S_slots"])
            if lambda_cov>0: L += coverage_entropy_loss(aux["A_maps"], target_entropy=target_entropy, weight=lambda_cov)

            # GL은 final_tune_epochs 구간에서는 자동 OFF (아래 조건)
            if enable_gl and (epoch > warmup_epochs) and (epoch <= epochs-final_tune_epochs):
                L += group_lasso_penalty_from_head(head, lambda_gl=lambda_gl)

            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            epoch_loss += L.item()

        sched.step()
        acc = evaluate(trunk, head, test_loader, device)
        print(f"[seed {seed}] epoch {epoch:02d} | loss {epoch_loss/len(train_loader):.4f} | acc {acc*100:.2f}%")

        # (A) 학습 중 프루닝이 필요한 경우에만 수행 (기본 False)
        if apply_during and enable_gl and (epoch > warmup_epochs) and (epoch <= epochs-final_tune_epochs) and ((epoch-1)%max(1,prune_check_every)==0):
            W = head.fc.weight if head.mode == "concat" else head.proj.weight
            M, D = head.M, head.D
            norms = np.array([ W[:, m*D:(m+1)*D].norm(p=2).item() for m in range(M) ])
            # 간단: 하위 10% 컷 (원하면 percentile도 YAML로 뺄 수 있음)
            pct = 10
            thr = float(np.percentile(norms, pct))
            head.prune_by_threshold(thr)
            alive = int(head.slot_mask.sum().item())
            print(f"  prune_check(during): alive={alive}/{M} | thr={thr:.3e} "
                  f"| norms[min/med/max]={norms.min():.3e}/{np.median(norms):.3e}/{norms.max():.3e}")

        # (B) 모니터링 (매 에폭)
        slot_mask = getattr(head, "slot_mask", None)
        try:
            # topk_mass가 있으면 그 기반으로 usage_gini (권장 경로)
            if "topk_mass" in aux:
                usage_gini = usage_gini_from_topkmass(aux["topk_mass"], slot_mask=slot_mask)
                # EMA 업데이트
                tm = aux["topk_mass"].detach()
                if tm.dim() == 2:
                    tm = tm.mean(dim=0)
                tm = tm / (tm.sum() + 1e-8)
                usage_ema = ema_alpha*usage_ema + (1-ema_alpha)*tm.to(device)
            else:
                # fallback: A_maps 전체 질량 기반(덜 민감)
                A_maps = aux["A_maps"]
                if slot_mask is not None:
                    A_maps = A_maps * slot_mask.view(1,-1,1,1)
                P = A_maps.view(A_maps.size(0), A_maps.size(1), -1).sum(dim=-1).mean(dim=0)  # (M,)
                P = P / (P.sum() + 1e-8)
                arr = P.detach().cpu().numpy()
                arr_sorted = np.sort(arr); cum = np.cumsum(arr_sorted)
                usage_gini = float(1.0 - 2.0*np.trapz(cum, dx=1.0/len(arr)))

            divm  = compute_slot_div_mean(aux["S_slots"])
            sharp = float(aux["head_energy"].detach().mean().item())
        except Exception:
            usage_gini, divm, sharp = float('nan'), float('nan'), float('nan')

        alive_now = int(getattr(head, "slot_mask", torch.ones(head.M, device=logits.device)).sum().item())
        print(f"  monitor: usage_gini={usage_gini:.3f} | slot_div_mean={divm:.3f} | head_sharp={sharp:.3f} | alive={alive_now}/{head.M}")

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "acc": float(acc),
                "loss": float(epoch_loss/len(train_loader)),
                "usage_gini": usage_gini,
                "slot_div_mean": divm,
                "head_sharp_mean": sharp,
                "alive_slots": alive_now
            }) + "\n")

        # (C) 마지막 3에폭 직전에 "단 한 번" 프루닝 → 이후 파인튜닝
        if (not did_post_prune) and (epoch == epochs - final_tune_epochs):
            W = head.fc.weight if head.mode == "concat" else head.proj.weight
            M, D = head.M, head.D
            with torch.no_grad():
                norms_t = torch.stack([ W[:, m*D:(m+1)*D].norm(p=2) for m in range(M) ]).to(device)  # (M,)

            if post_score == "norm_plus_usage" and usage_ema is not None:
                u = usage_ema / (usage_ema.sum() + 1e-8)
                score = norms_t / (norms_t.max() + 1e-8) + u
            else:
                score = norms_t

            k_keep = max(2, int(round(M * post_keep_ratio)))
            keep_idx = torch.topk(score, k_keep).indices
            mask = torch.zeros(M, device=device); mask[keep_idx] = 1.0

            # 헤드 마스크 적용 및 블록 제로
            head.slot_mask.data = mask
            with torch.no_grad():
                for m in range(M):
                    if mask[m] < 0.5:
                        if head.mode == "concat":
                            W[:, m*D:(m+1)*D].zero_()
                        else:
                            W[:, m*D:(m+1)*D].zero_()

            did_post_prune = True
            alive_final = int(mask.sum().item())
            print(f"[post-prune] alive_slots={alive_final}/{M} (keep_ratio={post_keep_ratio:.2f}, score={post_score})")

        # best 모델 저장
        if acc>best:
            best=acc
            torch.save(dict(trunk=trunk.state_dict(), head=head.state_dict()),
                       os.path.join(out_dir, "best.pt"))

    with open(os.path.join(out_dir,"report.json"),"w") as f:
        json.dump({"best_acc": best}, f, indent=2)
    print(f"[seed {seed}] best acc: {best*100:.2f}%  -> {out_dir}/best.pt")
    return best


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", type=str, required=True)
    ap.add_argument("-o","--out_root", type=str, default="outputs/M")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seeds = cfg.get("seeds", [42])
    run_name = os.path.splitext(os.path.basename(args.config))[0]
    out_root = os.path.join(args.out_root, run_name)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root,"config_merged.yaml"),"w") as f:
        yaml.safe_dump(cfg, f)

    all_res = {}
    for s in seeds:
        out_dir = os.path.join(out_root, f"seed_{s}")
        best = train_one_seed(cfg, out_dir, s)
        all_res[str(s)] = best

    with open(os.path.join(out_root,"summary.json"),"w") as f:
        json.dump({k: float(v) for k,v in all_res.items()}, f, indent=2)
    print("Done. Summary:", all_res)

if __name__ == "__main__":
    main()
