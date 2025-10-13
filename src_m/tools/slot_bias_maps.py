# src_m/tools/slot_bias_maps.py
import os, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

from src_m.tools.m_train import build_model, seed_all

def get_loader(split="test", batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), subset=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.EMNIST(root="./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    if subset is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def slot_prob_from(A, slot_mask, tau=0.7, raw=None):
    """
    A: (B,M,H,W)  mass fallback
    slot_mask: (M,)
    raw: optional (B,M) head_energy
    return: (B,M) row-stochastic prob over alive slots
    """
    B, M = A.size(0), A.size(1)
    mask = slot_mask.view(1, M).to(A.device)
    if (raw is not None) and (raw.ndim == 2) and (raw.shape[1] == M):
        e = raw.detach().float()
    else:
        e = A.flatten(2).sum(-1)  # mass
    e = e * mask
    z = (e - e.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    p = torch.softmax(z, dim=1) * mask
    s = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p / s

def save_triplet(out_path, bias, mean, delta):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, im, title in zip(axes, [bias, mean, delta], ["bias", "mean", "delta=mean-bias"]):
        imn = im - im.min()
        imn = imn / (imn.max() + 1e-8)
        ax.imshow(imn, cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--respect_slot_mask", action="store_true")
    ap.add_argument("--only_correct", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.7, help="temperature for slot probability")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # cfg
    if args.config.endswith(".json"):
        with open(args.config, "r") as f: cfg = json.load(f)
    else:
        import yaml
        with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    mean = tuple(cfg.get("normalize",{}).get("mean",[0.1307]))
    std  = tuple(cfg.get("normalize",{}).get("std", [0.3081]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    trunk.to(device).eval(); head.to(device).eval()

    loader = get_loader(args.split, args.batch_size, args.num_workers, mean, std, args.subset)

    # probe once for sizes
    with torch.no_grad():
        x0,_ = next(iter(loader))
        x0 = x0.to(device)
        Z0, aux0 = trunk(x0)
        A0 = aux0["A_maps"]         # (B,M,H,W)
        M = A0.size(1); Hm, Wm = A0.size(2), A0.size(3)

    slot_mask = getattr(head, "slot_mask", torch.ones(M, device=device)).detach().float().to(device)
    if not args.respect_slot_mask:
        slot_mask = torch.ones_like(slot_mask)

    # -----------------------------
    # 1) Bias maps: forward with blank input (0 tensor)
    # -----------------------------
    bias_sum = torch.zeros(M, Hm, Wm, device=device)
    reps = 4
    B0 = min(args.batch_size, 256)
    for _ in range(reps):
        x_blank = torch.zeros(B0, x0.size(1), x0.size(2), x0.size(3), device=device)
        Zb, auxb = trunk(x_blank)
        Ab = auxb["A_maps"] * slot_mask.view(1, M, 1, 1)
        bias_sum += Ab.sum(dim=0)

    bias_map = (bias_sum / (reps * B0)).detach().cpu()

    # -----------------------------
    # 2) Mean maps on real data (optional: only correct)
    # -----------------------------
    mean_sum = torch.zeros(M, Hm, Wm, device=device)
    mass_bias = bias_map.view(M, -1).sum(dim=1)  # for later
    mass_mean = torch.zeros(M, device=device)

    n_samp = 0
    pbar = tqdm(loader, desc="accum(mean)")
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        x.requires_grad_(False)
        with torch.no_grad():
            Z, aux = trunk(x)
            A = aux["A_maps"] * slot_mask.view(1, M, 1, 1)   # (B,M,H,W)
            logits = head(Z)
            pred = logits.argmax(dim=1)

            if args.only_correct:
                keep = (pred == y)
                if keep.sum() == 0:
                    continue
                A = A[keep]

        mean_sum += A.sum(dim=0)
        mass_mean += A.flatten(2).sum(dim=(0,2))
        n_samp += A.size(0)

    mean_map = (mean_sum / max(1, n_samp)).detach().cpu()
    mass_mean = mass_mean.detach().cpu()

    # -----------------------------
    # 3) Delta & summary
    # -----------------------------
    delta_map = (mean_map - bias_map).clamp_min(0)
    bias_mass = bias_map.view(M, -1).sum(dim=1)
    delta_mass = delta_map.view(M, -1).sum(dim=1)
    bias_frac = (bias_mass / (bias_mass + delta_mass + 1e-8)).numpy()

    # 저장
    import csv
    with open(os.path.join(args.out_dir, "slot_bias_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot","bias_mass","mean_mass","bias_fraction"])
        for m in range(M):
            w.writerow([m, float(bias_mass[m].item()), float(mass_mean[m].item()), f"{float(bias_frac[m]):.6f}"])

    for m in range(M):
        out_png = os.path.join(args.out_dir, f"slot_{m:03d}", "bias_mean_delta.png")
        save_triplet(out_png, bias_map[m].numpy(), mean_map[m].numpy(), delta_map[m].numpy())

    print(f"[slot_bias_maps] saved → {args.out_dir}")
    print(f"[slot_bias_maps] slots={M}, images_used={n_samp}, bias_avg_mass={bias_mass.mean().item():.4f}")

if __name__ == "__main__":
    main()
