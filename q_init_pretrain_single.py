# =============================================
# FILE: q_init_pretrain_single.py
# =============================================
# End-to-end pipeline in ONE FILE:
#   1) Extract salient 9x9 patches from EMNIST → MiniBatchKMeans → initial conv1 filters (.npy)
#   2) Pretrain small AE with conv1 initialized from step1 → robust pruning → visualizations
#   3) Save per-class (3 each) AE recon images (individual PNGs + grids)
#
# Run example:
#   python q_init_pretrain_single.py \
#     --k 192 --patch 9 --per_image 6 --q_keep 0.95 --nms_k 3 --max_images 30000 \
#     --out_root outputs/Q_single \
#     --epochs 20 --batch_size 512 --d_model 64 \
#     --lr 3e-4 --weight_decay 1e-4 --wd_conv1 1e-3 \
#     --lambda_gate_l1 1e-3 --lambda_bal 1e-3 \
#     --use_top1 1 --top1_warmup 1 --renorm_conv1 1 \
#     --auto_prune 1 --keep_ratio 0.5 --min_keep 64 \
#     --imp_use_usage 1 --imp_use_gate 1 --imp_use_l2 0
#
# Notes:
# - Requires: torch, torchvision, scikit-learn
# - This file avoids any PYTHONPATH/-m import confusion; run it directly.
# =============================================
import os, sys, math, random, argparse, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ---------- Utilities ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_grid_img(t: torch.Tensor, nrow: int = 16, normalize: bool = True, pad_value: float = 0.5):
    # t: (N,1,H,W)
    return make_grid(t, nrow=nrow, normalize=normalize, pad_value=pad_value)


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
    W_min = Wn.min(dim=1, keepdim=True)[0]
    W_max = Wn.max(dim=1, keepdim=True)[0]
    Wn = (Wn - W_min) / (W_max - W_min + 1e-6)
    Wn = Wn.view(K, 1, 9, 9)
    grid = make_grid(Wn, nrow=nrow, pad_value=0.5)
    save_image(grid, str(path))


# ---------- Step 1: Patch extraction & KMeans ----------
def _extract_patches_from_image(img: torch.Tensor, per_image: int, patch: int, q_keep: float, nms_k: int):
    """
    img: (1,H,W) in [0,1]
    returns: list of (1,patch,patch) tensors
    """
    assert img.ndim == 3 and img.size(0) == 1
    sal = img.abs()  # simple saliency
    pad = nms_k // 2
    pooled = F.max_pool2d(sal, kernel_size=nms_k, stride=1, padding=pad)
    maxima = (sal >= pooled - 1e-7)  # local maxima mask
    thr = torch.quantile(sal.flatten(), torch.tensor(q_keep, device=sal.device))
    mask = maxima & (sal >= thr)
    ys, xs = torch.nonzero(mask[0], as_tuple=True)
    if ys.numel() == 0:
        return []
    scores = sal[0, ys, xs]
    idx = torch.argsort(scores, descending=True)[:per_image]
    ys, xs = ys[idx], xs[idx]

    # extract patches
    rad = patch // 2
    img_padded = F.pad(img, (rad, rad, rad, rad), mode="reflect")
    patches = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, x0 = y + rad, x + rad
        crop = img_padded[:, y0 - rad:y0 + rad + 1, x0 - rad:x0 + rad + 1]
        patches.append(crop)
    return patches


def collect_patches(dl, max_images: int, per_image: int, patch: int, q_keep: float, nms_k: int):
    patches = []
    seen = 0
    for img, _ in dl:
        for b in range(img.size(0)):
            pt = _extract_patches_from_image(img[b], per_image, patch, q_keep, nms_k)
            patches.extend(pt)
            seen += 1
            if seen >= max_images:
                break
        if seen >= max_images:
            break
    if len(patches) == 0:
        return torch.empty(0,1,patch,patch)
    return torch.stack(patches, dim=0)  # (N,1,p,p)


def whiten_per_patch(x: torch.Tensor):
    # x: (N,1,p,p)
    N = x.size(0)
    x = x.view(N, -1)
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) + 1e-6
    x = (x - m) / s
    p = int((x.size(1))**0.5)
    return x.view(N,1,p,p)


def kmeans_filters(patches: torch.Tensor, k: int):
    # patches: (N,1,p,p)
    N, _, p, _ = patches.shape
    X = patches.view(N, -1).cpu().numpy()
    if not _HAVE_SK:
        raise RuntimeError("scikit-learn이 필요합니다: pip install scikit-learn")
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=min(8192, max(256, k*10)), n_init=5, max_no_improvement=50, verbose=0)
    mbk.fit(X)
    centers = mbk.cluster_centers_.astype(np.float32).reshape(k, 1, p, p)
    # zero-mean & unit-norm per filter
    W = centers
    W = W - W.mean(axis=(2,3), keepdims=True)
    norms = np.sqrt((W**2).sum(axis=(2,3), keepdims=True)) + 1e-6
    W = W / norms
    return W  # (K,1,p,p)


# ---------- Step 2: AE + pruning ----------
class AEConv9(nn.Module):
    def __init__(self, k_init: int, d_model: int = 64, use_top1: bool = True):
        super().__init__()
        self.k = k_init
        self.use_top1 = use_top1
        self.conv1 = nn.Conv2d(1, self.k, kernel_size=9, padding=4, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(self.k))
        self.proj_down = nn.Conv2d(self.k, d_model, kernel_size=1, bias=True)
        self.proj_up = nn.Conv2d(d_model, self.k, kernel_size=1, bias=True)
        self.register_buffer("usage_ema", torch.full((self.k,), 1.0/self.k))

    def forward(self, x, force_no_top1: bool = False):
        z = self.conv1(x)  # (B,K,H,W)
        a = F.relu(z)
        gate = torch.sigmoid(self.gate_logit).view(1, -1, 1, 1)
        a = a * gate

        apply_top1 = (self.use_top1 and not force_no_top1)
        if apply_top1:
            idx = a.argmax(dim=1, keepdim=True)  # (B,1,H,W)
            mask = torch.zeros_like(a).scatter_(1, idx, 1.0)
            a = a * mask
            with torch.no_grad():
                ch_count = mask.sum(dim=(0,2,3))  # (K,)
                total = mask.size(0) * mask.size(2) * mask.size(3) + 1e-6
                p = ch_count / total
                self.usage_ema.mul_(0.99).add_(p * 0.01)

        h = F.relu(self.proj_down(a))
        a2 = F.relu(self.proj_up(h))  # (B,K,H,W)
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


# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    # Step-1 (init filters)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--per_image", type=int, default=6)
    ap.add_argument("--q_keep", type=float, default=0.95)
    ap.add_argument("--nms_k", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=30000)

    # Common
    ap.add_argument("--out_root", type=str, default="outputs/Q_single")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")

    # Step-2 (pretrain)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--wd_conv1", type=float, default=1e-3, help="conv1 전용 WD (scale 폭주 방지)")
    ap.add_argument("--amp", type=int, default=1)
    # regularization
    ap.add_argument("--lambda_gate_l1", type=float, default=1e-3)
    ap.add_argument("--lambda_bal", type=float, default=1e-3)
    ap.add_argument("--use_top1", type=int, default=1)
    ap.add_argument("--top1_warmup", type=int, default=0, help="초기 N epoch 동안 top-1 비활성")
    ap.add_argument("--renorm_conv1", type=int, default=1, help="에폭마다 conv1 per-filter renorm")
    # pruning
    ap.add_argument("--auto_prune", type=int, default=1)
    ap.add_argument("--min_keep", type=int, default=32)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--imp_alpha_usage", type=float, default=1.0)
    ap.add_argument("--imp_beta_gate", type=float, default=1.0)
    ap.add_argument("--imp_gamma_l2", type=float, default=0.0, help="기본 0: L2 제외")
    ap.add_argument("--imp_use_usage", type=int, default=1)
    ap.add_argument("--imp_use_gate", type=int, default=1)
    ap.add_argument("--imp_use_l2", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_root)
    _ensure_dir(out_root)

    # ---- Step-1: init filters ----
    init_dir = out_root / f"filters_init_k{args.k}_p{args.patch}"
    _ensure_dir(init_dir)
    init_npy = init_dir / f"filters_init_k{args.k}_{args.patch}.npy"

    print("[Step-1] Collect EMNIST patches...", flush=True)
    tfm = transforms.ToTensor()
    ds = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    patches = collect_patches(dl, args.max_images, args.per_image, args.patch, args.q_keep, args.nms_k)
    if patches.numel() == 0:
        raise RuntimeError("패치를 하나도 모으지 못했습니다. q_keep/nms_k/per_image를 조정하세요.")
    patches_w = whiten_per_patch(patches)
    save_image(_to_grid_img(patches_w[:256], nrow=16), str(init_dir / "viz_patches_grid.png"))

    print(f"[Step-1] KMeans (k={args.k})...", flush=True)
    W_init = kmeans_filters(patches_w, args.k)
    np.save(str(init_npy), W_init)
    save_image(_to_grid_img(torch.from_numpy(W_init[:256]), nrow=16, normalize=True), str(init_dir / "viz_filters_grid.png"))
    print(f"[Step-1] Saved filters to {init_npy}, shape={W_init.shape}", flush=True)

    # ---- Step-2: pretrain ----
    pre_dir = out_root / f"pretrain_k{args.k}_p{args.patch}"
    _ensure_dir(pre_dir)

    print("[Step-2] Build AE...", flush=True)
    k = int(W_init.shape[0])
    model = AEConv9(k_init=k, d_model=args.d_model, use_top1=bool(args.use_top1)).to(device)
    with torch.no_grad():
        w = torch.from_numpy(W_init).float()
        model.conv1.weight.copy_(w)
    save_filter_grid_robust(model.conv1.weight.detach().cpu(), pre_dir / "viz_conv1_filters_init.png", nrow=16)
    print(f"[init] conv1 initialized (K={k})", flush=True)

    tfm = transforms.ToTensor()
    ds_tr = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    ds_te = datasets.EMNIST(args.data_root, split="balanced", train=False, download=True, transform=tfm)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optim with separate WD
    params_conv1 = list(model.conv1.parameters())
    params_gate = [model.gate_logit]
    params_rest = [p for n,p in model.named_parameters() if (not n.startswith("conv1.")) and (n != "gate_logit")]
    opt = torch.optim.AdamW([
        {"params": params_conv1, "weight_decay": args.wd_conv1},
        {"params": params_rest, "weight_decay": args.weight_decay},
        {"params": params_gate, "weight_decay": 0.0},
    ], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and (device.type=="cuda"))

    def balanced_reg(p):
        K = p.numel()
        target = torch.full_like(p, 1.0 / K)
        return F.mse_loss(p, target)

    print("[Step-2] Train...", flush=True)
    for ep in range(1, args.epochs+1):
        model.train()
        total_l1 = 0.0
        total = 0
        for img, _ in dl_tr:
            img = img.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
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

            total_l1 += rec.item() * img.size(0)
            total += img.size(0)

        if args.renorm_conv1:
            with torch.no_grad():
                W = model.conv1.weight.data
                Kc = W.size(0)
                W = W.view(Kc, -1)
                n = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
                W = (W / n).view_as(model.conv1.weight.data)
                model.conv1.weight.data.copy_(W)

        avg = total_l1 / max(1, total)
        print(f"[ep {ep}] train_l1={avg:.4f} usage(first8)={model.usage_ema.detach().cpu().numpy()[:8]}", flush=True)

    # Save ALL filters after train
    W_all = model.conv1.weight.detach().cpu().numpy()
    np.save(str(pre_dir / f"filters_emnist_pretrained_k{W_all.shape[0]}_9.npy"), W_all)
    save_filter_grid_robust(model.conv1.weight.detach().cpu(), pre_dir / "viz_conv1_filters_all.png", nrow=16)

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
    np.save(str(pre_dir / f"filters_emnist_pretrained_k{W_pruned.shape[0]}_9_pruned.npy"), W_pruned)
    np.save(str(pre_dir / "filters_keep_indices.npy"), keep_idx)
    np.save(str(pre_dir / "usage_ema.npy"), usage)
    np.save(str(pre_dir / "gate_sigma.npy"), gate)
    np.save(str(pre_dir / "l2norm.npy"), l2)
    np.save(str(pre_dir / "importance.npy"), importance)

    # Visualize pruned filters
    save_filter_grid_robust(torch.from_numpy(W_pruned), pre_dir / "viz_conv1_filters_pruned.png", nrow=16)

    # Rebuild conv1 to pruned shape & run classwise recon saves
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

    run_recon_and_save(model, dl_te, device, pre_dir, postfix="_pruned", per_class=3, save_individual=True)

    print("[done] outputs:", str(out_root), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] Exception:", repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
