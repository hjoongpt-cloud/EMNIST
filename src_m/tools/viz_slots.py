# src_m/tools/viz_slots.py
import os, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms

from src_m.tools.m_train import build_model, seed_all

def block_l2_heatmap(head, save_png):
    # class-slot 블록 L2
    mode = head.mode
    M, D, C = head.M, head.D, head.num_classes
    W = head.fc.weight if mode=="concat" else head.proj.weight  # (C, M*D)
    W = W.detach().cpu()
    H = torch.zeros(C, M)
    for m in range(M):
        blk = W[:, m*D:(m+1)*D]  # (C,D)
        H[:,m] = torch.linalg.vector_norm(blk, ord=2, dim=1)
    H = H.numpy()

    plt.figure(figsize=(0.35*M+2, 0.35*C+2))
    plt.imshow(H, aspect="auto")
    plt.colorbar(); plt.xlabel("slot m"); plt.ylabel("class c"); plt.title("L2-norm per (class,slot) block")
    plt.tight_layout()
    plt.savefig(save_png, dpi=160); plt.close()
    print("[viz] saved", save_png)

def head_energy_hist(dump_dir, save_png):
    # dump에서 head_energy 읽어 분포 그림
    import glob
    import numpy as np
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    vals=[]
    for f in tqdm(files, desc="reading E"):
        z = np.load(f)
        if "head_energy" in z:
            vals.append(z["head_energy"].astype(np.float32))
    if len(vals)==0:
        print("no head_energy in dumps"); return
    X = np.concatenate(vals,0).ravel()
    plt.figure(figsize=(6,3))
    plt.hist(X, bins=80)
    plt.title("head_energy distribution"); plt.tight_layout()
    plt.savefig(save_png, dpi=160); plt.close()
    print("[viz] saved", save_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f: cfg = json.load(f) if args.config.endswith(".json") else __import__("yaml").safe_load(f)
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    head.to(device).eval()

    block_l2_heatmap(head, os.path.join(args.out_dir, "block_l2_heatmap.png"))
    head_energy_hist(args.dump_dir, os.path.join(args.out_dir, "head_energy_hist.png"))

if __name__ == "__main__":
    seed_all(42)
    main()
