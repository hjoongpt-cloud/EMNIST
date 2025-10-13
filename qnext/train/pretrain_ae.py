import argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from qnext.core.vis import save_filters_grid
import argparse, torch
from qnext.core.data import get_dataloaders
from qnext.core.frontend import Frontend
from qnext.core.patch_init import extract_patches, spherical_kmeans, centers_to_conv1

class AE(nn.Module):
    def __init__(self, frontend: Frontend):
        super().__init__()
        self.enc = frontend
        D = self.enc.D; K = self.enc.K
        self.proj_up = nn.Conv2d(D, K, kernel_size=1, bias=True)
        self.deconv9 = nn.ConvTranspose2d(K, 1, kernel_size=9, padding=4, bias=False)
        with torch.no_grad():
            self.deconv9.weight.copy_(self.enc.conv1.weight)

    def forward(self, x):
        z, aux = self.enc(x)
        a_hat = self.proj_up(z)
        x_hat = self.deconv9(a_hat)
        return x_hat, aux

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--enc_act", type=str, default="relu")
    ap.add_argument("--wta_mode", type=str, default="none")
    ap.add_argument("--wta_tau", type=float, default=0.7)
    ap.add_argument("--out_dir", type=str, default="outputs/ae")
    ap.add_argument("--lr", type=float, default=1e-3)
    # === 패치 기반 초기화 옵션 ===
    ap.add_argument("--init_patches", type=int, default=0, help="1이면 train에서 패치 추출로 conv1 초기화")
    ap.add_argument("--patch_max_images", type=int, default=20000)
    ap.add_argument("--patch_size", type=int, default=9)
    ap.add_argument("--patch_stride", type=int, default=1)
    ap.add_argument("--patch_energy_thresh", type=float, default=0.1)
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--kmeans_retries", type=int, default=2)
    ap.add_argument("--kmeans_seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, _, _, _ = get_dataloaders(data_root=args.data_root)

    # AE 모델 구성
    fe = Frontend(K=150, D=64, enc_act=args.enc_act, wta_mode=args.wta_mode, wta_tau=args.wta_tau, wta_k=1).to(device)

    # === 패치 기반 초기화 ===
    if args.init_patches:
        print("[PATCH-INIT] extracting patches...")
        P = extract_patches(dl_train, max_images=args.patch_max_images,
                            patch_size=args.patch_size, stride=args.patch_stride,
                            energy_thresh=args.patch_energy_thresh, device=device)
        print(f"[PATCH-INIT] patches: {P.shape}")
        K_target = fe.conv1.out_channels  # 현재 K에 맞춰 클러스터
        print(f"[PATCH-INIT] spherical kmeans K={K_target} iters={args.kmeans_iters}")
        centers = spherical_kmeans(P, K=K_target, iters=args.kmeans_iters,
                                   retries=args.kmeans_retries, seed=args.kmeans_seed)  # [K, ps*ps]
        centers_to_conv1(centers, fe.conv1, patch_size=args.patch_size)
        # mask/K 동기화
        if fe.channel_mask.shape[1] != K_target:
            fe.register_buffer("channel_mask", torch.ones(1, K_target, 1, 1, device=device))
        fe.K = K_target
        print("[PATCH-INIT] conv1 initialized from patch centers.")
    ae = AE(fe).to(device)

    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs+1):
        ae.train(); running = 0.0; total=0
        for x,_ in dl_train:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = ae(x)
            loss = mse(x_hat, x)
            loss.backward(); opt.step()
            running += float(loss.item()) * x.size(0); total += x.size(0)
        print(f"[AE] ep{ep} recon_loss={(running/max(1,total)):.4f}")
        with torch.no_grad(): ae.enc.renorm_conv1()
    torch.save(ae.state_dict(), f"{args.out_dir}/ae_last.pt")
    save_filters_grid(ae.enc.conv1.weight, f"{args.out_dir}/filters_grid.png")
    import numpy as np
    np.save(f"{args.out_dir}/conv1.npy", ae.enc.conv1.weight.detach().cpu().numpy())
    np.savez(f"{args.out_dir}/proj_down.npz",
             weight=ae.enc.proj_down.weight.detach().cpu().numpy(),
             bias=ae.enc.proj_down.bias.detach().cpu().numpy())
    print(f"[AE] saved to {args.out_dir}")

if __name__ == "__main__":
    main()
