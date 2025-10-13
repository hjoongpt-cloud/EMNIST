# =============================================
# FILE: src_q/q_pretrain_filters.py  (updated)
# =============================================
# - Robust filter visualization (clipped z-score)
# - Per-class AE recon save (3 per class): individual PNGs + grids
# - Training stabilization:
#     * Separate WD for conv1 / others / gate (gate WD=0)
#     * Optional per-epoch per-filter renorm
#     * Optional top-1 warmup (disable top-1 for first N epochs)
# - Pruning: keep_ratio parameter (default 0.5) instead of hard half
# - Importance formula tunable: choose components (usage/gate/l2)
#
# Usage (unchanged flags still work):
#   --wd_conv1 1e-3 --weight_decay 1e-4 --renorm_conv1 1 --top1_warmup 1
#   --keep_ratio 0.5 --imp_use_usage 1 --imp_use_gate 1 --imp_use_l2 0
#
import os, argparse, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_filter_grid_robust(W: torch.Tensor, path: Path, nrow: int = 16, clip_std: float = 2.0):
    """
    Robust per-filter z-score -> clip -> min-max to [0,1]
    W: (K,1,9,9)
    """
    Wn = W.detach().float()
    K = Wn.size(0)
    Wn = Wn.view(K, -1)
    m = Wn.mean(dim=1, keepdim=True)
    s = Wn.std(dim=1, keepdim=True).clamp_min(1e-6)
    Wn = (Wn - m) / s
    Wn = Wn.clamp(-clip_std, clip_std)
    # now min-max per filter
    W_min = Wn.min(dim=1, keepdim=True)[0]
    W_max = Wn.max(dim=1, keepdim=True)[0]
    Wn = (Wn - W_min) / (W_max - W_min + 1e-6)
    Wn = Wn.view(K, 1, 9, 9)
    grid = make_grid(Wn, nrow=nrow, pad_value=0.5)
    save_image(grid, str(path))


class AEConv9(nn.Module):
    def __init__(self, k_init: int, d_model: int = 64, use_top1: bool = True):
        super().__init__()
        self.k = k_init
        self.use_top1 = use_top1
        self.conv1 = nn.Conv2d(1, self.k, kernel_size=9, padding=4, bias=False)
        # 게이트 파라미터(채널별). 초기 0 → sigmoid=0.5
        self.gate_logit = nn.Parameter(torch.zeros(self.k))
        # 1x1 병목
        self.proj_down = nn.Conv2d(self.k, d_model, kernel_size=1, bias=True)
        self.proj_up = nn.Conv2d(d_model, self.k, kernel_size=1, bias=True)
        # 사용률 EMA(등록 형태로 유지)
        self.register_buffer("usage_ema", torch.full((self.k,), 1.0/self.k))

    def forward(self, x, force_no_top1: bool = False):
        # x: (B,1,H,W)
        z = self.conv1(x)  # (B,K,H,W)
        a = F.relu(z)
        gate = torch.sigmoid(self.gate_logit).view(1, -1, 1, 1)
        a = a * gate

        apply_top1 = (self.use_top1 and not force_no_top1)
        if apply_top1:
            # top-1 mask per pixel
            idx = a.argmax(dim=1, keepdim=True)  # (B,1,H,W)
            mask = torch.zeros_like(a).scatter_(1, idx, 1.0)
            a = a * mask
            # 사용률 추정: 배치 내 채널 비율
            with torch.no_grad():
                ch_count = mask.sum(dim=(0,2,3))  # (K,)
                total = mask.size(0) * mask.size(2) * mask.size(3) + 1e-6
                p = ch_count / total
                self.usage_ema.mul_(0.99).add_(p * 0.01)

        h = F.relu(self.proj_down(a))
        a2 = F.relu(self.proj_up(h))  # (B,K,H,W)
        # tied weights decoder
        x_hat = F.conv_transpose2d(a2, self.conv1.weight, padding=4)  # (B,1,H,W)
        aux = {"gate": gate.detach(), "usage": self.usage_ema.detach(), "act": a.detach()}
        return x_hat, aux


@torch.no_grad()
def run_recon_and_save(model: AEConv9, dl, device, out_dir: Path, postfix: str = "", per_class: int = 3, save_individual: bool = True):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir / "class_samples")
    tf_inputs, tf_recons, labels = [], [], []
    picked = {i:0 for i in range(47)}  # EMNIST balanced has 47 classes
    for img, y in dl:
        img = img.to(device)
        y = y.cpu().numpy()
        x_hat, _ = model(img)
        for b in range(img.size(0)):
            c = int(y[b])
            if picked.get(c, 0) < per_class:
                tf_inputs.append(img[b].cpu())
                tf_recons.append(x_hat[b].cpu())
                labels.append(c)
                picked[c] = picked.get(c, 0)+1
        if all(v >= per_class for v in picked.values()):
            break
    if len(tf_inputs) == 0:
        return
    inputs = torch.stack(tf_inputs)
    recons = torch.stack(tf_recons)
    # Save per-class individual PNGs
    if save_individual:
        per_class_count = {i:0 for i in range(47)}
        for i in range(inputs.size(0)):
            c = int(labels[i])
            if per_class_count[c] < per_class:
                cls_dir = out_dir / "class_samples" / f"class_{c:02d}"
                _ensure_dir(cls_dir)
                save_image(inputs[i], str(cls_dir / f"input_{per_class_count[c]:02d}{postfix}.png"), normalize=True)
                save_image(recons[i], str(cls_dir / f"recon_{per_class_count[c]:02d}{postfix}.png"), normalize=True)
                per_class_count[c] += 1
    # Grids (ordered by class)
    order = np.argsort(np.array(labels))
    inputs = inputs[order]
    recons = recons[order]
    grid_in = make_grid(inputs, nrow=per_class, normalize=True)
    grid_out = make_grid(recons, nrow=per_class, normalize=True)
    save_image(grid_in, str(out_dir / f"class_samples_grid_input{postfix}.png"))
    save_image(grid_out, str(out_dir / f"class_samples_grid_recon{postfix}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--init_filters", type=str, default="", help="(선택) 1단계 .npy 경로; 주면 conv1 초기화 & k 갱신")
    ap.add_argument("--k_init", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--wd_conv1", type=float, default=1e-3, help="conv1 전용 WD (scale 폭주 방지)")
    ap.add_argument("--amp", type=int, default=1)
    # regularization
    ap.add_argument("--lambda_gate_l1", type=float, default=1e-3)
    ap.add_argument("--lambda_bal", type=float, default=1e-3)
    ap.add_argument("--use_top1", type=int, default=1)
    ap.add_argument("--top1_warmup", type=int, default=0, help="초기 N epoch 동안 top-1 비활성 (collapse 방지)")
    ap.add_argument("--renorm_conv1", type=int, default=1, help="에폭마다 conv1 per-filter renorm")
    # pruning
    ap.add_argument("--auto_prune", type=int, default=1)
    ap.add_argument("--min_keep", type=int, default=32)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--imp_alpha_usage", type=float, default=1.0)
    ap.add_argument("--imp_beta_gate", type=float, default=1.0)
    ap.add_argument("--imp_gamma_l2", type=float, default=0.0, help="기본 0: L2는 중요도에서 제외")
    ap.add_argument("--imp_use_usage", type=int, default=1)
    ap.add_argument("--imp_use_gate", type=int, default=1)
    ap.add_argument("--imp_use_l2", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Dataset
    tfm = transforms.ToTensor()
    ds_tr = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    ds_te = datasets.EMNIST(args.data_root, split="balanced", train=False, download=True, transform=tfm)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # init filters
    init_W = None
    k_from_file = None
    if args.init_filters:
        init_W = np.load(args.init_filters)
        if init_W.ndim == 3:  # (K,9,9) -> (K,1,9,9)
            init_W = init_W[:, None, :, :]
        assert init_W.ndim == 4 and init_W.shape[1:] == (1,9,9), f"init_filters shape must be (K,1,9,9) or (K,9,9); got {init_W.shape}"
        k_from_file = int(init_W.shape[0])

    k = k_from_file if k_from_file is not None else int(args.k_init)
    model = AEConv9(k_init=k, d_model=args.d_model, use_top1=bool(args.use_top1)).to(device)
    if init_W is not None:
        with torch.no_grad():
            w = torch.from_numpy(init_W).float()
            model.conv1.weight.copy_(w)
        save_filter_grid_robust(model.conv1.weight.detach().cpu(), out_dir / "viz_conv1_filters_init.png", nrow=16)
        print(f"[init] conv1 initialized from {args.init_filters} (K={k})")

    # Optim: separate param groups
    params_conv1 = list(model.conv1.parameters())
    params_gate = [model.gate_logit]  # WD=0
    params_rest = [p for n,p in model.named_parameters() if (not n.startswith("conv1.")) and (n != "gate_logit")]
    opt = torch.optim.AdamW([
        {"params": params_conv1, "weight_decay": args.wd_conv1},
        {"params": params_rest, "weight_decay": args.weight_decay},
        {"params": params_gate, "weight_decay": 0.0},
    ], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and (device.type=="cuda"))

    def balanced_reg(p):
        # 균형 사용률 정규화: (p - 1/K)^2 평균
        K = p.numel()
        target = torch.full_like(p, 1.0 / K)
        return F.mse_loss(p, target)

    # Train loop
    for ep in range(1, args.epochs+1):
        model.train()
        total_l1 = 0.0
        total = 0
        for img, _ in dl_tr:
            img = img.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
                # warmup: force_no_top1 during first N epochs
                force_no_top1 = (ep <= args.top1_warmup)
                x_hat, aux = model(img, force_no_top1=force_no_top1)
                rec = F.l1_loss(x_hat, img)
                gate = torch.sigmoid(model.gate_logit)
                reg_gate = gate.abs().mean()
                reg_bal = balanced_reg(model.usage_ema)
                loss = rec + args.lambda_gate_l1 * reg_gate + args.lambda_bal * reg_bal
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # Optional per-epoch per-filter renorm (to unit Frobenius norm)
        if args.renorm_conv1:
            with torch.no_grad():
                W = model.conv1.weight.data
                Kc = W.size(0)
                W = W.view(Kc, -1)
                n = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
                W = (W / n).view_as(model.conv1.weight.data)
                model.conv1.weight.data.copy_(W)

        # simple logging
        avg = float(rec.detach().mean().cpu()) if isinstance(rec, torch.Tensor) else 0.0
        first8 = model.usage_ema.detach().cpu().numpy()[:8]
        print(f"[ep {ep}] l1≈{avg:.4f}  usage(first8)={np.array2string(first8, precision=3)}")

    # Save ALL filters after train
    W_all = model.conv1.weight.detach().cpu().numpy()
    np.save(str(out_dir / f"filters_emnist_pretrained_k{W_all.shape[0]}_9.npy"), W_all)
    save_filter_grid_robust(model.conv1.weight.detach().cpu(), out_dir / "viz_conv1_filters_all.png", nrow=16)

    # Importance & pruning
    gate = torch.sigmoid(model.gate_logit.detach()).cpu().numpy()  # (K,)
    usage = model.usage_ema.detach().cpu().numpy()  # (K,)
    l2 = np.sqrt((W_all**2).sum(axis=(1,2,3)))  # (K,)
    importance = np.ones_like(l2)
    if args.imp_use_usage: importance *= (usage ** args.imp_alpha_usage)
    if args.imp_use_gate:  importance *= (gate  ** args.imp_beta_gate)
    if args.imp_use_l2:    importance *= (l2    ** args.imp_gamma_l2)
    order = np.argsort(-importance)
    keep = max(args.min_keep, int(round(len(order) * args.keep_ratio)))
    keep_idx = np.sort(order[:keep])

    W_pruned = W_all[keep_idx]
    np.save(str(out_dir / f"filters_emnist_pretrained_k{W_pruned.shape[0]}_9_pruned.npy"), W_pruned)
    np.save(str(out_dir / "filters_keep_indices.npy"), keep_idx)
    np.save(str(out_dir / "usage_ema.npy"), usage)
    np.save(str(out_dir / "gate_sigma.npy"), gate)
    np.save(str(out_dir / "l2norm.npy"), l2)
    np.save(str(out_dir / "importance.npy"), importance)

    # 시각화: pruned filters
    save_filter_grid_robust(torch.from_numpy(W_pruned), out_dir / "viz_conv1_filters_pruned.png", nrow=16)


    # 재생성 시각화 (클래스별 3개) — also save individual PNGs per class
    # 프루닝된 out_channels에 맞추어 conv1 모듈 자체를 재구성해야 함 (weight copy_만 하면 shape mismatch 발생)
    with torch.no_grad():
        Kp = W_pruned.shape[0]
        device_ = model.conv1.weight.device
        dtype_ = model.conv1.weight.dtype
        new_conv1 = nn.Conv2d(1, Kp, kernel_size=9, padding=4, bias=False).to(device_)
        new_conv1.weight.data.copy_(torch.from_numpy(W_pruned).to(device=device_, dtype=dtype_))
        model.conv1 = new_conv1
        model.gate_logit = nn.Parameter(torch.zeros(Kp, device=device_))
        model.k = Kp
        model.proj_down = nn.Conv2d(model.k, model.proj_down.out_channels, kernel_size=1, bias=True).to(device_)
        model.proj_up = nn.Conv2d(model.proj_down.out_channels, model.k, kernel_size=1, bias=True).to(device_)
        model.usage_ema = torch.full((model.k,), 1.0/model.k, device=device_)

    dl_te_local = DataLoader(datasets.EMNIST(args.data_root, split="balanced", train=False, download=True, transform=transforms.ToTensor()),
                             batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    run_recon_and_save(model, dl_te_local, device, out_dir, postfix="_pruned", per_class=3, save_individual=True)

    print("[save] ALL: ", out_dir / f"filters_emnist_pretrained_k{W_all.shape[0]}_9.npy")
    print("[save] PRUNED: ", out_dir / f"filters_emnist_pretrained_k{W_pruned.shape[0]}_9_pruned.npy")
    print("[viz ] init/all/pruned: ", out_dir / "viz_conv1_filters_init.png", out_dir / "viz_conv1_filters_all.png", out_dir / "viz_conv1_filters_pruned.png")
