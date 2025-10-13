#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np
from tqdm import tqdm
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt

@torch.no_grad()
def round_pair_mask(M, H, W, device):
    # 14x14, 3x3 grid → 36개 페어를 슬롯 인덱스에 round-배정
    xs, ys = [0,5,10,14], [0,5,10,14]
    cells=[]
    for gy in range(3):
        for gx in range(3):
            x0,x1=xs[gx],xs[gx+1]; y0,y1=ys[gy],ys[gy+1]
            m=torch.zeros(H,W,device=device); m[y0:y1,x0:x1]=1
            cells.append(m)
    pairs=[]
    for i in range(9):
        for j in range(i+1,9):
            pairs.append(torch.maximum(cells[i],cells[j]))
    idx=torch.arange(M,device=device)%len(pairs)
    return torch.stack([pairs[i] for i in idx.tolist()],0)  # (M,H,W)

@torch.no_grad()
def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full = datasets.EMNIST("./data", split="balanced", train=True, download=True, transform=tf)
        n=len(full); nv=int(round(n*val_ratio)); nt=n-nv
        g=torch.Generator().manual_seed(seed)
        tr,va=random_split(full,[nt,nv],generator=g)
        ds = tr if split=="train" else va
    else:
        ds = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--proto_json", required=False)  # 선택: 정확도까지 보고 싶으면 프로토 주기
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk 구성이 train과 동일해야 함 (stride=2/GELU/norm_first 등)
    filters=np.load(args.conv1_filters)
    meta=torch.load(args.ckpt, map_location="cpu").get("meta",{})
    d_model=int(meta.get("d_model",64)); nhead=int(meta.get("nhead",4)); nl=int(meta.get("num_layers",1))
    trunk=OTrunk(d_model=d_model, nhead=nhead, num_layers=nl, d_ff=256, conv1_filters=filters).to(device).eval()

    # slot queries
    q = load_slot_queries_from_ckpt(args.ckpt, device)  # (M,D)

    # data
    loader = get_loader(args.split, bs=256, nw=2)

    H=W=14
    hist_bins=np.linspace(0,1,51)
    stats={"P_mass":{"hist":np.zeros(50), "maxs":[]},
           "P_lse":{"hist":np.zeros(50), "maxs":[]}}

    with torch.no_grad():
        for x,y in tqdm(loader, desc="[check-trainP]"):
            x=x.to(device)
            tok,_=trunk(x)                          # (B,196,D)
            B,N,D=tok.shape; M=q.size(0)

            # train과 동일: per-slot softmax over pixels
            logits = torch.einsum("bnd,md->bmn", F.normalize(tok,dim=-1), q) / np.sqrt(D)  # (B,M,N)
            A_slot = torch.softmax(logits, dim=2).view(B,M,H,W)                             # (B,M,H,W)

            # mask 적용 + 슬롯별 재정규화 (train 주석과 동일 가정)
            mask = round_pair_mask(M,H,W,device).unsqueeze(0)                               # (1,M,H,W)
            A_eff = A_slot * mask
            A_eff = A_eff / A_eff.flatten(2).sum(-1).clamp_min(1e-8).view(B,M,1,1)         # per-slot sum=1

            # P_mass (warmup 이후 규칙)
            mass = A_eff.flatten(2).sum(-1)                                                # ≡1
            z = (mass - mass.mean(dim=1, keepdim=True)) / max(1e-6, float(args.tau))
            P_mass = torch.softmax(z, dim=1)

            # P_lse (warmup 이전 규칙)
            s = torch.logsumexp(logits, dim=2)
            z = (s - s.mean(dim=1, keepdim=True)) / max(1e-6, float(args.tau))
            P_lse = torch.softmax(z, dim=1)

            # 통계
            for tag, P in [("P_mass", P_mass), ("P_lse", P_lse)]:
                mx = P.max(dim=1).values.detach().cpu().numpy()
                stats[tag]["maxs"].extend(mx.tolist())
                cts,_ = np.histogram(P.detach().cpu().numpy().reshape(-1), bins=hist_bins)
                stats[tag]["hist"] += cts

    # 저장
    out = {k: {"P_max_mean": float(np.mean(v["maxs"])),
               "P_max_median": float(np.median(v["maxs"])),
               "hist": v["hist"].tolist()}
           for k,v in stats.items()}
    with open(os.path.join(args.out_dir, "trainP_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__=="__main__":
    main()
