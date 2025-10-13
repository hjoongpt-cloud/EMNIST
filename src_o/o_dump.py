#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt, extract_slots_with_queries

def get_loader(split="test", batch_size=256, num_workers=2, subset=None, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=tf)
        if split == "train":
            ds = full
        else:
            n_total = len(full); n_val = int(round(n_total * float(val_ratio))); n_tr = n_total - n_val
            g = torch.Generator().manual_seed(seed)
            tr_ds, val_ds = random_split(full, [n_tr, n_val], generator=g)
            ds = val_ds
    else:
        ds = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=tf)
    if subset is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--spmask_grid", type=int, default=3)
    ap.add_argument("--spmask_assign", type=str, default="round")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filters = np.load(args.conv1_filters)
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = int(meta.get("d_model", 64)); nhead = int(meta.get("nhead", 4)); num_layers = int(meta.get("num_layers", 2))
    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=256, conv1_filters=filters).to(device).eval()
    slot_q = load_slot_queries_from_ckpt(args.ckpt, device)

    loader = get_loader(args.split, args.batch_size, args.num_workers, args.subset, args.val_ratio, args.val_seed)

    ctr = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[dump:{args.split}]"):
            x = x.to(device)
            tokens, aux = trunk(x)  # tokens:(B,196,d)
            A, P, S, A_raw = extract_slots_with_queries(tokens, slot_q, use_pair_mask=True,
                                                        tau_p=args.tau, grid=args.spmask_grid, assign=args.spmask_assign,
                                                        heat_map_bhw=aux.get("heat_map", None))
            B = x.size(0)
            for b in range(B):
                out = {
                    "A_upsampled": A[b].cpu().numpy(),  # (M,14,14)
                    "slot_prob":   P[b].cpu().numpy(),  # (M,)
                    "S_slots":     S[b].cpu().numpy(),  # (M,d)
                    "image":       x[b].cpu().numpy(),  # (1,28,28)
                    "clazz":       int(y[b].item()),
                    "A_raw":       A_raw[b].cpu().numpy(),
                }
                np.savez_compressed(os.path.join(args.out_dir, f"{ctr:06d}.npz"), **out)
                ctr += 1

if __name__ == "__main__":
    main()
