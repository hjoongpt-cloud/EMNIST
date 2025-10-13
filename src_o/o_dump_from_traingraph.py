#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, json, torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt

@torch.no_grad()
def round_pair_mask(M, H, W, device):
    xs,ys=[0,5,10,14],[0,5,10,14]
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
def dump_like_train(trunk, q, loader, out_dir, tau=0.7):
    os.makedirs(out_dir, exist_ok=True)
    H=W=14; ctr=0
    for x,y in tqdm(loader, desc="[dump-train-graph]"):
        x=x.to(next(trunk.parameters()).device)
        tok,_=trunk(x)                            # (B,196,D)
        B,N,D=tok.shape; M=q.size(0)
        # per-pixel softmax(슬롯축)
        logits=torch.einsum("bnd,md->bnm", F.normalize(tok,dim=-1), q)/np.sqrt(D)
        A_raw=torch.softmax(logits,dim=2).permute(0,2,1).contiguous().view(B,M,H,W)
        # round pair mask (훈련 가정)
        mask=round_pair_mask(M,H,W,tok.device).unsqueeze(0)
        A_masked=A_raw*mask                         # 정규화 X
        # P: 질량 기반
        mass=A_masked.flatten(2).sum(-1)            # (B,M)
        z=(mass-mass.mean(dim=1,keepdim=True))/max(1e-6,float(tau))
        P=torch.softmax(z,dim=1)
        # S: 평균 후 L2
        t=F.normalize(tok,dim=-1).view(B,H*W,D)
        S=torch.bmm(A_masked.view(B,M,H*W), t)
        S=S/mass.clamp_min(1e-8).unsqueeze(-1)
        S=F.normalize(S,dim=-1)

        for b in range(B):
            np.savez_compressed(
                os.path.join(out_dir, f"{ctr:06d}.npz"),
                A_upsampled=A_masked[b].cpu().numpy(),
                slot_prob=P[b].cpu().numpy(),
                S_slots=S[b].cpu().numpy(),
                image=x[b].cpu().numpy(),
                clazz=int(y[b].item()),
                A_raw=A_raw[b].cpu().numpy(),
            )
            ctr+=1

def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full=datasets.EMNIST("./data",split="balanced",train=True,download=True,transform=tf)
        n=len(full); nv=int(round(n*val_ratio)); nt=n-nv
        g=torch.Generator().manual_seed(seed)
        tr,va=random_split(full,[nt,nv],generator=g)
        ds=tr if split=="train" else va
    else:
        ds=datasets.EMNIST("./data",split="balanced",train=False,download=True,transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.7)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filters=np.load(args.conv1_filters)
    meta=torch.load(args.ckpt,map_location="cpu").get("meta",{})
    d_model=int(meta.get("d_model",64)); nhead=int(meta.get("nhead",4)); nl=int(meta.get("num_layers",2))
    trunk=OTrunk(d_model=d_model,nhead=nhead,num_layers=nl,d_ff=256,conv1_filters=filters).to(device).eval()
    q=load_slot_queries_from_ckpt(args.ckpt, device)
    loader=get_loader(args.split, bs=args.batch_size, nw=args.num_workers)

    dump_like_train(trunk, q, loader, args.out_dir, tau=args.tau)

if __name__=="__main__":
    main()
