#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np
from tqdm import tqdm
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt

# ---------- prototypes ----------
def load_protos(path, device):
    with open(path, "r") as f:
        data = json.load(f)
    per = data.get("per_class", data)
    C_list, y_list = [], []
    items = per.items() if isinstance(per, dict) else [("0", per)]
    for c, v in items:
        cid = int(c)
        if isinstance(v, dict) and "mu" in v:
            vecs = v["mu"]
        else:
            vecs = v
        if isinstance(vecs, (list, tuple)) and len(vecs) and isinstance(vecs[0], (list, tuple, np.ndarray)):
            for u in vecs:
                C_list.append(np.asarray(u, np.float32))
                y_list.append(cid)
        else:
            C_list.append(np.asarray(vecs, np.float32))
            y_list.append(cid)
    if not C_list:
        raise RuntimeError("no prototypes parsed")
    C = torch.from_numpy(np.stack(C_list, 0)).float().to(device)
    C = F.normalize(C, dim=1)
    C_cls = torch.tensor(y_list, device=device, dtype=torch.long)
    return C, C_cls

# ---------- slot maps ----------
@torch.no_grad()
def per_pixel_softmax_over_slots(tokens, q, H, W):
    # tokens:(B,N,D), q:(M,D) -> A:(B,M,H,W) with softmax over slot-axis per pixel
    B, N, D = tokens.shape
    logits = torch.einsum("bnd,md->bnm", F.normalize(tokens, dim=-1), q) / np.sqrt(D)
    A = torch.softmax(logits, dim=2).permute(0,2,1).contiguous().view(B, q.size(0), H, W)
    return A

@torch.no_grad()
def per_slot_softmax_over_pixels(tokens, q, H, W):
    # tokens:(B,N,D), q:(M,D) -> A:(B,M,H,W) with softmax over pixel-axis per slot
    B, N, D = tokens.shape
    logits = torch.einsum("bnd,md->bmn", F.normalize(tokens, dim=-1), q) / np.sqrt(D)  # (B,M,N)
    A = torch.softmax(logits, dim=2).view(B, q.size(0), H, W)
    return A

# ---------- round pair mask (3x3, 36 pairs) ----------
@torch.no_grad()
def round_pair_mask(M, H, W, device):
    assert H == 14 and W == 14, "assume 14x14 feature grid"
    xs, ys = [0,5,10,14], [0,5,10,14]
    cells = []
    for gy in range(3):
        for gx in range(3):
            x0, x1 = xs[gx], xs[gx+1]
            y0, y1 = ys[gy], ys[gy+1]
            mask = torch.zeros(H, W, device=device)
            mask[y0:y1, x0:x1] = 1.0
            cells.append(mask)
    pairs = []
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append(torch.maximum(cells[i], cells[j]))
    P = len(pairs)  # 36
    idx = torch.arange(M, device=device) % P
    return torch.stack([pairs[i] for i in idx.tolist()], 0)  # (M,H,W)

@torch.no_grad()
def apply_mask(A, assign="round"):
    # A:(B,M,H,W)
    B, M, H, W = A.shape
    mask_mhw = round_pair_mask(M, H, W, A.device)
    return A * mask_mhw.unsqueeze(0)

# ---------- mass/P & slot embeddings ----------
@torch.no_grad()
def mass_P(A_masked, tau):
    # A_masked:(B,M,H,W)
    mass = A_masked.flatten(2).sum(-1)  # (B,M)
    z = (mass - mass.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    return torch.softmax(z, dim=1)      # (B,M)

@torch.no_grad()
def slot_embed(tokens, A_masked, avg=True):
    # tokens:(B,N,D), A_masked:(B,M,H,W)
    B, N, D = tokens.shape
    Bb, M, H, W = A_masked.shape
    assert Bb == B, "batch size mismatch between tokens and A_masked"
    t = F.normalize(tokens, dim=-1).view(B, H*W, D)          # (B,HW,D)
    A_flat = A_masked.view(B, M, H*W)                        # (B,M,HW)
    S = torch.bmm(A_flat, t)                                 # (B,M,D)
    if avg:
        mass = A_masked.flatten(2).sum(-1).clamp_min(1e-8).unsqueeze(-1)  # (B,M,1)
        S = S / mass
    return F.normalize(S, dim=-1)

@torch.no_grad()
def evi_wsum_map(S, P, C, C_cls):
    # S:(B,M,D), P:(B,M), C:(K,D), C_cls:(K,)
    sim = torch.einsum("bmd,kd->bmk", S, C)  # (B,M,K)
    labels_sorted = sorted(int(x) for x in torch.unique(C_cls).tolist())
    Cp = len(labels_sorted)
    neg = -1e9 * sim.new_ones(())
    evi = sim.new_zeros(S.size(0), S.size(1), Cp)
    for j, lab in enumerate(labels_sorted):
        mask = (C_cls == lab).view(1,1,-1)
        s = sim.masked_fill(~mask, neg)
        evi[:,:,j] = s.max(dim=2).values
    w = P / P.sum(dim=1, keepdim=True).clamp_min(1e-8)
    logits = torch.einsum("bmc,bm->bc", evi, w)  # (B,Cp)
    pred_idx = logits.argmax(dim=1)              # indices in labels_sorted
    # map indices back to actual labels
    mapped = torch.tensor([labels_sorted[int(i)] for i in pred_idx.cpu().tolist()], device=pred_idx.device)
    return mapped, logits

# ---------- data ----------
def get_loader(split="val", bs=256, nw=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if split in ("train","val"):
        full = datasets.EMNIST("./data", split="balanced", train=True, download=True, transform=tf)
        n = len(full); nv = int(round(n*val_ratio)); nt = n - nv
        g = torch.Generator().manual_seed(seed)
        tr, va = random_split(full, [nt, nv], generator=g)
        ds = tr if split=="train" else va
    else:
        ds = datasets.EMNIST("./data", split="balanced", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk & queries
    filters = np.load(args.conv1_filters)
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = int(meta.get("d_model", 64)); nhead = int(meta.get("nhead", 4)); nl = int(meta.get("num_layers", 2))
    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=nl, d_ff=256, conv1_filters=filters).to(device).eval()
    q = load_slot_queries_from_ckpt(args.ckpt, device)

    # prototypes
    C, C_cls = load_protos(args.proto_json, device)

    # loader
    loader = get_loader("val", bs=256, nw=2)

    stats = {"V_train":{}, "V_pxslot":{}, "V_slotsum":{}}
    hist_bins = np.linspace(0, 1, 51)
    corr = []

    def upd(tag, P, acc):
        mx = P.max(dim=1).values.detach().cpu().numpy()
        stats[tag].setdefault("max_list", []).extend(mx.tolist())
        cts, _ = np.histogram(P.detach().cpu().numpy().reshape(-1), bins=hist_bins)
        stats[tag]["hist"] = stats[tag].get("hist", np.zeros_like(hist_bins[:-1])) + cts
        stats[tag]["acc_sum"] = stats[tag].get("acc_sum", 0.0) + float(acc[0])
        stats[tag]["n"] = stats[tag].get("n", 0) + int(acc[1])

    with torch.no_grad():
        for x, y in tqdm(loader, desc="[probe]"):
            x = x.to(device); y = y.to(device)
            tok, _ = trunk(x)                      # (B,196,D)
            B, N, D = tok.shape
            H = W = int(np.sqrt(N))

            # V-train (의도): per-pixel softmax over slots + round mask + mass→P + avg S
            A1 = per_pixel_softmax_over_slots(tok, q, H, W)
            M1 = apply_mask(A1, "round")           # no renorm
            P1 = mass_P(M1, args.tau)
            S1 = slot_embed(tok, M1, avg=True)
            pred1, _ = evi_wsum_map(S1, P1, C, C_cls)
            acc1 = ((pred1 == y).sum().item(), y.numel())

            # V-pxslot (동일 경로, sanity)
            A2 = A1; M2 = M1; P2 = P1
            S2 = S1
            pred2, _ = evi_wsum_map(S2, P2, C, C_cls)
            acc2 = ((pred2 == y).sum().item(), y.numel())

            # V-slotsum: per-slot softmax over pixels + round mask + mass→P
            A3 = per_slot_softmax_over_pixels(tok, q, H, W)
            M3 = apply_mask(A3, "round")
            P3 = mass_P(M3, args.tau)
            S3 = slot_embed(tok, M3, avg=True)
            pred3, _ = evi_wsum_map(S3, P3, C, C_cls)
            acc3 = ((pred3 == y).sum().item(), y.numel())

            upd("V_train", P1, acc1)
            upd("V_pxslot", P2, acc2)
            upd("V_slotsum", P3, acc3)

            # P correlation
            p1 = P1.flatten().cpu().numpy(); p2 = P2.flatten().cpu().numpy(); p3 = P3.flatten().cpu().numpy()
            corr.append([
                float(np.corrcoef(p1, p2)[0,1]),
                float(np.corrcoef(p1, p3)[0,1]),
                float(np.corrcoef(p2, p3)[0,1]),
            ])

    # summary
    out = {}
    for k,v in stats.items():
        arr = np.array(v["max_list"], dtype=np.float32)
        out[k] = {
            "P_max_mean": float(arr.mean() if arr.size else 0.0),
            "P_max_median": float(np.median(arr) if arr.size else 0.0),
            "acc": float(v["acc_sum"]/max(1, v["n"])),
            "hist": v["hist"].tolist(),
        }
    out["corr_mean"] = {
        "train_pxslot": float(np.mean([c[0] for c in corr])) if corr else 0.0,
        "train_slotsum": float(np.mean([c[1] for c in corr])) if corr else 0.0,
        "pxslot_slotsum": float(np.mean([c[2] for c in corr])) if corr else 0.0,
    }
    with open(os.path.join(args.out_dir, "probe_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
