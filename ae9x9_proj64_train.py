# ============================================================
# FILE: ae9x9_proj64_train.py (KMEANS init + WTA top-k + pruning + stride=2 fix + FT + Hardest-per-class eval)
# ============================================================
# End-to-end AE training in ONE file, with:
#  - Optional KMeans conv1 init from salient 9x9 patches
#  - Front-end: 9x9 conv -> GELU -> 1x1 proj to 64 dims (per location)
#  - Position-wise top-k (WTA) gating (default: top-3, soft)
#  - Decoders: tied9 | untied9 | unet_lite
#  - Optional pruning (tied9/untied9) + optional fine-tuning
#  - stride=2 output size fix (output_padding=stride-1)
#  - NEW: Hardest-per-class evaluation & visualization (top-K by recon loss)
#
# Quickstart (KMeans + tied9, stride=2, top-3 WTA, hardest-per-class eval):
#   python ae9x9_proj64_train.py \
#     --out_dir outputs/ae64_k192_s2_tied9_wta3 \
#     --do_kmeans 1 --K 192 --per_image 6 --q_keep 0.95 --nms_k 3 --max_images 30000 \
#     --decoder tied9 --stride 2 \
#     --epochs 20 --batch_size 512 \
#     --lr 3e-4 --weight_decay 1e-4 --wd_conv1 1e-3 \
#     --edge_loss 0.1 --renorm_conv1 1 --amp 1 \
#     --wta_mode soft --topk 3 --tau 0.5 --wta_warmup 0 \
#     --hard_eval 1 --hard_k 3
#
# Prune + short finetune + hardest eval (tied9/untied9):
#   ... --prune 1 --keep_ratio 0.5 --min_keep 64 \
#       --finetune_after_prune 3 --ft_lr 1e-4 --hard_eval 1 --hard_k 3
#
# Iterative prune (3 steps to 0.5) + hardest eval:
#   ... --prune 1 --iterative_prune 1 --prune_steps 3 --keep_ratio 0.5 \
#       --ft_step_epochs 2 --ft_lr 1e-4 --hard_eval 1 --hard_k 3
#
# Requirements: torch, torchvision, (optional for KMeans) scikit-learn
# ============================================================

import os, sys, math, argparse, random, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# --- Try scikit-learn (KMeans); only needed if --do_kmeans 1 ---
try:
    from sklearn.cluster import MiniBatchKMeans
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

# ------------------- Utils -------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def grid_img(t: torch.Tensor, nrow: int = 16, normalize: bool = True, pad_value: float = 0.5):
    return make_grid(t, nrow=nrow, normalize=normalize, pad_value=pad_value)


def save_filter_grid_robust(W: torch.Tensor, path: Path, nrow: int = 16, clip_std: float = 2.0):
    """Per-filter z-score -> clip -> min-max -> grid save."""
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


# ------------------- KMEANS INIT (Step-0) -------------------

def _extract_patches_from_image(img: torch.Tensor, per_image: int, patch: int, q_keep: float, nms_k: int):
    assert img.ndim == 3 and img.size(0) == 1
    sal = img.abs()
    pad = nms_k // 2
    pooled = F.max_pool2d(sal, kernel_size=nms_k, stride=1, padding=pad)
    maxima = (sal >= pooled - 1e-7)
    thr = torch.quantile(sal.flatten(), torch.tensor(q_keep, device=sal.device))
    mask = maxima & (sal >= thr)
    ys, xs = torch.nonzero(mask[0], as_tuple=True)
    if ys.numel() == 0:
        return []
    scores = sal[0, ys, xs]
    idx = torch.argsort(scores, descending=True)[:per_image]
    ys, xs = ys[idx], xs[idx]

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
            patches.extend(_extract_patches_from_image(img[b], per_image, patch, q_keep, nms_k))
            seen += 1
            if seen >= max_images:
                break
        if seen >= max_images:
            break
    if len(patches) == 0:
        return torch.empty(0,1,patch,patch)
    return torch.stack(patches, dim=0)


def whiten_per_patch(x: torch.Tensor):
    N = x.size(0)
    x = x.view(N, -1)
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) + 1e-6
    x = (x - m) / s
    p = int((x.size(1))**0.5)
    return x.view(N,1,p,p)


def kmeans_filters(patches: torch.Tensor, k: int):
    if not _HAVE_SK:
        raise RuntimeError("scikit-learn이 필요합니다: pip install scikit-learn")
    N, _, p, _ = patches.shape
    X = patches.view(N, -1).cpu().numpy()
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=min(8192, max(256, k*10)), n_init=5,
                          max_no_improvement=50, verbose=0)
    mbk.fit(X)
    centers = mbk.cluster_centers_.astype(np.float32).reshape(k, 1, p, p)
    W = centers
    W = W - W.mean(axis=(2,3), keepdims=True)
    norms = np.sqrt((W**2).sum(axis=(2,3), keepdims=True)) + 1e-6
    W = W / norms
    return W  # (K,1,9,9)


# ------------------- AE Model -------------------
class AE9x9Proj64(nn.Module):
    def __init__(self, K=192, d=64, stride=1, decoder="tied9",
                 topk=3, wta_mode="soft", tau=0.5, wta_warmup=0):
        super().__init__()
        self.K, self.d, self.stride, self.decoder = K, d, stride, decoder
        self.topk, self.wta_mode, self.tau, self.wta_warmup = topk, wta_mode, tau, wta_warmup
        self._cur_epoch = 0  # updated from train loop
        pad = 4
        # Encoder
        self.conv1 = nn.Conv2d(1, K, kernel_size=9, stride=stride, padding=pad, bias=False)
        self.enc_act = nn.GELU()
        self.proj_down = nn.Conv2d(K, d, kernel_size=1, bias=True)
        # Decoder variants
        if decoder in ("tied9", "untied9"):
            self.proj_up = nn.Conv2d(d, K, kernel_size=1, bias=True)
            if decoder == "untied9":
                self.deconv9 = nn.ConvTranspose2d(K, 1, kernel_size=9, stride=stride, padding=pad,
                                                  output_padding=(stride-1), bias=True)
        elif decoder == "unet_lite":
            self.proj_up = nn.Conv2d(d, K, kernel_size=1, bias=True)
            ch = K
            self.dec_block = nn.Sequential(
                nn.Conv2d(K*2, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, 1, 1)
            )
        else:
            raise ValueError("decoder must be one of {'tied9','untied9','unet_lite'}")

    def apply_wta(self, a: torch.Tensor) -> torch.Tensor:
        if self.topk <= 0 or self.wta_mode == "none" or (self._cur_epoch < self.wta_warmup):
            return a
        B, K, H, W = a.shape
        a_flat = a.view(B, K, -1)
        if self.wta_mode == "soft":
            p = (a_flat / self.tau).softmax(dim=1)  # (B,K,HW)
            if self.topk == 1:
                g = p
            else:
                topv, topi = torch.topk(p, self.topk, dim=1)
                mask = torch.zeros_like(p).scatter_(1, topi, topv)
                g = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
        elif self.wta_mode == "hard":
            topv, topi = torch.topk(a_flat, self.topk, dim=1)
            hard = torch.zeros_like(a_flat).scatter_(1, topi, 1.0)
            p = (a_flat / self.tau).softmax(dim=1)
            g = (hard - p).detach() + p  # straight-through
        else:
            return a
        return (a_flat * g).view(B, K, H, W)

    def forward(self, x, return_feats=False):
        a = self.enc_act(self.conv1(x))   # (B,K,H',W')
        a = self.apply_wta(a)
        z = a
        h = self.proj_down(z)             # (B,64,H',W')
        if self.decoder == "tied9":
            kfeat = self.proj_up(h)
            x_hat = F.conv_transpose2d(kfeat, self.conv1.weight, stride=self.stride, padding=4,
                                       output_padding=(self.stride-1))
        elif self.decoder == "untied9":
            kfeat = self.proj_up(h)
            x_hat = self.deconv9(kfeat)
        else:
            kfeat = self.proj_up(h)
            z_up = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False) if self.stride==2 else z
            x_hat = self.dec_block(torch.cat([kfeat, z_up], dim=1))
        return (x_hat, {"z": z, "h": h}) if return_feats else x_hat

    @torch.no_grad()
    def renorm_conv1(self):
        W = self.conv1.weight.view(self.K, -1)
        n = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.conv1.weight.copy_((W / n).view_as(self.conv1.weight))


def sobel_edges(x):
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-6)


def ae_loss_per_sample(x_hat, x, w_edge=0.0):
    l1 = (x_hat - x).abs().view(x.size(0), -1).mean(dim=1)  # (B,)
    if w_edge > 0:
        e1 = (sobel_edges(x_hat) - sobel_edges(x)).abs().view(x.size(0), -1).mean(dim=1)
        return l1 + w_edge * e1
    return l1


def ae_loss(x_hat, x, w_edge=0.0):
    # batch mean
    return ae_loss_per_sample(x_hat, x, w_edge=w_edge).mean()


@torch.no_grad()
def save_recon_grids(model, dl, device, out_dir: Path, postfix: str = "", per_class: int = 3):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    picked = {i:0 for i in range(47)}  # EMNIST balanced
    inputs, recons, labels = [], [], []
    for img, y in dl:
        img = img.to(device)
        x_hat = model(img)
        y = y.cpu().numpy()
        for b in range(img.size(0)):
            c = int(y[b])
            if picked.get(c, 0) < per_class:
                inputs.append(img[b].cpu())
                recons.append(x_hat[b].cpu())
                labels.append(c)
                picked[c] = picked.get(c, 0) + 1
        if all(v >= per_class for v in picked.values()):
            break
    if not inputs:
        return
    order = np.argsort(np.array(labels))
    inputs = torch.stack(inputs)[order]
    recons = torch.stack(recons)[order]
    save_image(grid_img(inputs, nrow=per_class, normalize=True), str(out_dir / f"class_grid_input{postfix}.png"))
    save_image(grid_img(recons, nrow=per_class, normalize=True), str(out_dir / f"class_grid_recon{postfix}.png"))


@torch.no_grad()
def save_hardest_recon_grids(model, dl, device, out_dir: Path, postfix: str = "", per_class_k: int = 3, w_edge: float = 0.0):
    """Pick top-K hardest samples per class by recon loss and save input/recon grids.
       Also writes a CSV with per-class mean loss and top-K losses.
    """
    import csv
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # top-K buffers per class
    KCL = 47  # EMNIST balanced
    tops = {c: [] for c in range(KCL)}  # list of (loss, x_cpu, xhat_cpu)
    sums = np.zeros(KCL, dtype=np.float64)
    cnts = np.zeros(KCL, dtype=np.int64)

    for img, y in dl:
        img = img.to(device)
        x_hat = model(img)
        losses = ae_loss_per_sample(x_hat, img, w_edge=w_edge).detach().cpu()
        y = y.cpu().numpy()
        for b in range(img.size(0)):
            c = int(y[b])
            loss_b = float(losses[b].item())
            sums[c] += loss_b; cnts[c] += 1
            buf = tops[c]
            sample = (loss_b, img[b].cpu(), x_hat[b].cpu())
            if len(buf) < per_class_k:
                buf.append(sample)
            else:
                # replace smallest if current is larger
                min_i = min(range(per_class_k), key=lambda i: buf[i][0])
                if loss_b > buf[min_i][0]:
                    buf[min_i] = sample

    # build tensors sorted by class & loss desc within class
    inputs, recons, labels = [], [], []
    for c in range(KCL):
        if len(tops[c]) == 0:
            continue
        tops[c].sort(key=lambda t: -t[0])
        for (loss_b, x_cpu, xhat_cpu) in tops[c]:
            inputs.append(x_cpu)
            recons.append(xhat_cpu)
            labels.append(c)
    if len(inputs) == 0:
        return

    inputs = torch.stack(inputs)
    recons = torch.stack(recons)
    # each class contributes up to per_class_k, so nrow = per_class_k
    save_image(grid_img(inputs, nrow=per_class_k, normalize=True), str(out_dir / f"class_grid_input_hard{postfix}.png"))
    save_image(grid_img(recons, nrow=per_class_k, normalize=True), str(out_dir / f"class_grid_recon_hard{postfix}.png"))

    # CSV report
    csv_path = out_dir / f"hardest_per_class{postfix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count", "mean_loss", "topK_losses_desc"])
        for c in range(KCL):
            mean = (sums[c] / max(1, cnts[c])) if cnts[c] > 0 else 0.0
            top_losses = [f"{t[0]:.6f}" for t in sorted(tops[c], key=lambda t: -t[0])]
            writer.writerow([c, int(cnts[c]), f"{mean:.6f}", ";".join(top_losses)])


# ------------------- Pruning helpers -------------------
@torch.no_grad()
def measure_usage(model: AE9x9Proj64, dl, device):
    model.eval()
    K = model.K
    s = torch.zeros(K, device=device)
    n = 0
    for x, _ in dl:
        x = x.to(device)
        a = model.enc_act(model.conv1(x))
        a = model.apply_wta(a)
        s += a.mean(dim=(0,2,3))
        n += 1
    if n == 0:
        return torch.ones(K, device=device) / K
    s /= n
    return s


def prune_model_tied_or_untied(model: AE9x9Proj64, keep_idx: np.ndarray):
    device = next(model.parameters()).device
    keep_idx_t = torch.from_numpy(keep_idx).to(device)
    Kp = int(keep_idx.shape[0])

    new_conv1 = nn.Conv2d(1, Kp, kernel_size=9, stride=model.stride, padding=4, bias=False).to(device)
    new_conv1.weight.data.copy_(model.conv1.weight.data[keep_idx_t])
    model.conv1 = new_conv1

    old_pd = model.proj_down
    new_pd = nn.Conv2d(Kp, old_pd.out_channels, kernel_size=1, bias=True).to(device)
    new_pd.weight.data.copy_(old_pd.weight.data[:, keep_idx_t, :, :])
    new_pd.bias.data.copy_(old_pd.bias.data)
    model.proj_down = new_pd

    old_pu = model.proj_up
    new_pu = nn.Conv2d(old_pu.in_channels, Kp, kernel_size=1, bias=True).to(device)
    new_pu.weight.data.copy_(old_pu.weight.data[keep_idx_t, :, :, :])
    new_pu.bias.data.copy_(old_pu.bias.data[keep_idx_t])
    model.proj_up = new_pu

    if model.decoder == "untied9":
        old_dc = model.deconv9
        new_dc = nn.ConvTranspose2d(Kp, 1, kernel_size=9, stride=model.stride, padding=4,
                                    output_padding=(model.stride-1), bias=True).to(device)
        new_dc.weight.data.copy_(old_dc.weight.data[keep_idx_t])
        new_dc.bias.data.copy_(old_dc.bias.data)
        model.deconv9 = new_dc

    model.K = Kp
    return model


# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    # Out & data
    ap.add_argument("--out_dir", type=str, default="outputs/ae64")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)

    # Model/Train
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--stride", type=int, default=1, choices=[1,2])
    ap.add_argument("--decoder", type=str, default="tied9", choices=["tied9","untied9","unet_lite"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--wd_conv1", type=float, default=1e-3)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--edge_loss", type=float, default=0.0)
    ap.add_argument("--renorm_conv1", type=int, default=1)

    # WTA(top-k) options
    ap.add_argument("--wta_mode", type=str, default="soft", choices=["none","soft","hard"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--wta_warmup", type=int, default=0)

    # Init options
    ap.add_argument("--do_kmeans", type=int, default=0, help="1: run KMeans and init conv1 with it")
    ap.add_argument("--init_filters", type=str, default="", help=".npy (K,1,9,9) or (K,9,9)")

    # KMeans hyperparams (if --do_kmeans 1)
    ap.add_argument("--per_image", type=int, default=6)
    ap.add_argument("--q_keep", type=float, default=0.95)
    ap.add_argument("--nms_k", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=30000)
    ap.add_argument("--patch", type=int, default=9)

    # Pruning (single-shot)
    ap.add_argument("--prune", type=int, default=0, help="1 to enable post-training pruning")
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--min_keep", type=int, default=64)
    ap.add_argument("--imp_alpha_usage", type=float, default=1.0)
    ap.add_argument("--imp_gamma_l2", type=float, default=1.0)

    # Finetune after prune / Iterative prune options
    ap.add_argument("--finetune_after_prune", type=int, default=0, help="epochs to fine-tune after final prune")
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--iterative_prune", type=int, default=0)
    ap.add_argument("--prune_steps", type=int, default=3)
    ap.add_argument("--ft_step_epochs", type=int, default=2)

    # Hardest-per-class eval
    ap.add_argument("--hard_eval", type=int, default=0)
    ap.add_argument("--hard_k", type=int, default=3)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Data
    tfm = transforms.ToTensor()
    ds_tr = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    ds_te = datasets.EMNIST(args.data_root, split="balanced", train=False, download=True, transform=tfm)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # ---------- Step-0: optional KMeans ----------
    W_init = None
    if args.do_kmeans:
        if not _HAVE_SK:
            raise RuntimeError("--do_kmeans=1 requires scikit-learn. pip install scikit-learn")
        print(f"[KMEANS] Collect patches (patch={args.patch}, per_image={args.per_image}, q_keep={args.q_keep})", flush=True)
        dl_patch = DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        patches = collect_patches(dl_patch, args.max_images, args.per_image, args.patch, args.q_keep, args.nms_k)
        if patches.numel() == 0:
            raise RuntimeError("패치를 하나도 모으지 못했습니다. q_keep/nms_k/per_image를 조정하세요.")
        patches_w = whiten_per_patch(patches)
        save_image(grid_img(patches_w[:256], nrow=16), str(out_dir / "viz_patches_grid.png"))
        print(f"[KMEANS] Fit MiniBatchKMeans (K={args.K})", flush=True)
        W_init = kmeans_filters(patches_w, args.K)
        np.save(str(out_dir / f"filters_init_k{args.K}_9.npy"), W_init)
        save_image(grid_img(torch.from_numpy(W_init[:256]), nrow=16, normalize=True), str(out_dir / "viz_filters_grid.png"))
        print(f"[KMEANS] Saved filters: {out_dir / f'filters_init_k{args.K}_9.npy'}", flush=True)
    elif args.init_filters:
        W_init = np.load(args.init_filters)
        if W_init.ndim == 3:
            W_init = W_init[:, None, :, :]
        assert W_init.shape == (args.K,1,9,9), f"init_filters shape must be (K,1,9,9) or (K,9,9); got {W_init.shape}"
        print(f"[INIT] Loaded conv1 filters from {args.init_filters}", flush=True)

    # ---------- Build model ----------
    model = AE9x9Proj64(K=args.K, d=64, stride=args.stride, decoder=args.decoder,
                        topk=args.topk, wta_mode=args.wta_mode, tau=args.tau, wta_warmup=args.wta_warmup).to(device)

    # Apply init
    if W_init is not None:
        with torch.no_grad():
            model.conv1.weight.copy_(torch.from_numpy(W_init).to(model.conv1.weight.device).to(model.conv1.weight.dtype))
    else:
        model.renorm_conv1()
    save_filter_grid_robust(model.conv1.weight.detach().cpu(), out_dir / "viz_conv1_init.png", nrow=16)

    # ---------- Optim ----------
    params_conv1 = list(model.conv1.parameters())
    params_rest = [p for n,p in model.named_parameters() if not n.startswith("conv1.")]
    opt = torch.optim.AdamW([
        {"params": params_conv1, "weight_decay": args.wd_conv1},
        {"params": params_rest, "weight_decay": args.weight_decay},
    ], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and (device.type=="cuda"))

    # ---------- Train ----------
    print("[TRAIN] start", flush=True)
    for ep in range(1, args.epochs+1):
        model._cur_epoch = ep
        model.train()
        total, total_l = 0, 0.0
        for x, _ in dl_tr:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
                x_hat = model(x)
                loss = ae_loss(x_hat, x, w_edge=args.edge_loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_l += loss.item() * x.size(0); total += x.size(0)
        if args.renorm_conv1:
            model.renorm_conv1()
        print(f"[ep {ep}] train_loss={total_l/max(1,total):.4f}", flush=True)

    # Save weights and conv1
    torch.save(model.state_dict(), str(out_dir / "ae9x9_proj64.pt"))
    np.save(str(out_dir / f"conv1_filters_k{args.K}_9.npy"), model.conv1.weight.detach().cpu().numpy())
    save_filter_grid_robust(model.conv1.weight.detach().cpu(), out_dir / "viz_conv1_trained.png", nrow=16)

    # Recon grids (pre-prune)
    save_recon_grids(model, dl_te, device, out_dir, postfix=f"_{args.decoder}_s{args.stride}", per_class=3)
    if args.hard_eval:
        save_hardest_recon_grids(model, dl_te, device, out_dir,
                                 postfix=f"_{args.decoder}_s{args.stride}",
                                 per_class_k=args.hard_k, w_edge=args.edge_loss)

    # ---------- Optional Pruning ----------
    def compute_importance_and_keep(model, keep_ratio):
        usage = measure_usage(model, dl_te, device).detach().cpu().numpy()  # (K,)
        W_all = model.conv1.weight.detach().cpu().numpy()                   # (K,1,9,9)
        l2 = np.sqrt((W_all**2).sum(axis=(1,2,3)))
        importance = (usage ** args.imp_alpha_usage) * (l2 ** args.imp_gamma_l2)
        order = np.argsort(-importance)
        keep = max(args.min_keep, int(round(len(order) * keep_ratio)))
        keep_idx = np.sort(order[:keep])
        return keep_idx, W_all, usage, l2, importance

    if args.prune:
        if model.decoder == "unet_lite":
            print("[PRUNE] unet_lite pruning is not implemented in this script; skipping.")
        else:
            if args.iterative_prune:
                ratios = np.linspace(1.0, args.keep_ratio, args.prune_steps+1)[1:]
                for i, r in enumerate(ratios, 1):
                    print(f"[PRUNE-STEP {i}/{len(ratios)}] target keep_ratio={r:.3f}")
                    keep_idx, W_all, usage, l2, importance = compute_importance_and_keep(model, r)
                    np.save(str(out_dir / f"prune_step{i}_keep_indices.npy"), keep_idx)
                    prune_model_tied_or_untied(model, keep_idx)
                    if args.ft_step_epochs > 0:
                        print(f"[FT] step finetune {args.ft_step_epochs} epochs @ lr={args.ft_lr}")
                        ft_opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)
                        for ep in range(args.ft_step_epochs):
                            model.train(); total=0; total_l=0.0
                            for x,_ in dl_tr:
                                x=x.to(device)
                                ft_opt.zero_grad(set_to_none=True)
                                with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
                                    x_hat = model(x)
                                    loss = ae_loss(x_hat, x, w_edge=args.edge_loss)
                                loss.backward(); ft_opt.step()
                                total_l += loss.item()*x.size(0); total+=x.size(0)
                            print(f"  [ft ep {ep+1}] loss={total_l/max(1,total):.4f}")
                save_recon_grids(model, dl_te, device, out_dir, postfix=f"_{args.decoder}_s{args.stride}_pruned", per_class=3)
                if args.hard_eval:
                    save_hardest_recon_grids(model, dl_te, device, out_dir,
                                             postfix=f"_{args.decoder}_s{args.stride}_pruned",
                                             per_class_k=args.hard_k, w_edge=args.edge_loss)
            else:
                print("[PRUNE] single-shot pruning...", flush=True)
                keep_idx, W_all, usage, l2, importance = compute_importance_and_keep(model, args.keep_ratio)
                np.save(str(out_dir / "prune_keep_indices.npy"), keep_idx)
                np.save(str(out_dir / "prune_usage.npy"), usage)
                np.save(str(out_dir / "prune_l2.npy"), l2)
                np.save(str(out_dir / "prune_importance.npy"), importance)
                save_filter_grid_robust(torch.from_numpy(W_all[keep_idx]), out_dir / "viz_conv1_filters_pruned.png", nrow=16)
                prune_model_tied_or_untied(model, keep_idx)
                if args.finetune_after_prune > 0:
                    print(f"[FT] finetune after prune {args.finetune_after_prune} epochs @ lr={args.ft_lr}")
                    ft_opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)
                    for ep in range(args.finetune_after_prune):
                        model.train(); total=0; total_l=0.0
                        for x,_ in dl_tr:
                            x=x.to(device)
                            ft_opt.zero_grad(set_to_none=True)
                            with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
                                x_hat = model(x)
                                loss = ae_loss(x_hat, x, w_edge=args.edge_loss)
                            loss.backward(); ft_opt.step()
                            total_l += loss.item()*x.size(0); total+=x.size(0)
                        print(f"  [ft ep {ep+1}] loss={total_l/max(1,total):.4f}")
                save_recon_grids(model, dl_te, device, out_dir, postfix=f"_{args.decoder}_s{args.stride}_pruned", per_class=3)
                if args.hard_eval:
                    save_hardest_recon_grids(model, dl_te, device, out_dir,
                                             postfix=f"_{args.decoder}_s{args.stride}_pruned",
                                             per_class_k=args.hard_k, w_edge=args.edge_loss)
                W_pruned = model.conv1.weight.detach().cpu().numpy()
                np.save(str(out_dir / f"conv1_filters_k{W_pruned.shape[0]}_9_pruned.npy"), W_pruned)

    print("[done]", out_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
