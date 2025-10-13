# src_m/tools/proto_fuse.py
import os, json, argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ===================== 공통 유틸 =====================

def build_feat_for_dim(S_slots: torch.Tensor, XY: torch.Tensor, target_dim: int, xy_weight: float) -> torch.Tensor:
    """
    S_slots: (B?,M,Dslot) or (M,Dslot) or None
    XY     : (B?,M,2)
    return : (..., target_dim)
    """
    if target_dim == 2:
        return F.normalize(XY, dim=-1)
    if S_slots is None:
        raise ValueError("Bank expects S or S+XY, but S_slots is None. Re-dump with --save_s_slots.")
    Dslot = S_slots.size(-1)
    if target_dim == Dslot:
        return F.normalize(S_slots, dim=-1)
    if target_dim == Dslot + 2:
        s  = F.normalize(S_slots, dim=-1)
        xy = F.normalize(XY, dim=-1)
        return torch.cat([s, xy * xy_weight], dim=-1)
    raise ValueError(f"Unsupported target_dim={target_dim} (Dslot={Dslot})")


def load_dump_dir(dump_dir: str, device: torch.device):
    paths = sorted(glob(os.path.join(dump_dir, "*.npz")))
    if not paths: raise FileNotFoundError(dump_dir)

    # infer shapes
    N = len(paths); C=None; Mmax=0; Dslot=None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "logits" in z:
            C = C or int(z["logits"].shape[-1])
        Mi = int(z["slot_mask"].shape[0])
        Mmax = max(Mmax, Mi)
        if "S_slots" in z:
            Dslot = Dslot or int(z["S_slots"].shape[1])
    if C is None:
        raise KeyError("logits not found in dumps (needed for base scores).")

    Y_np    = np.empty(N, np.int64)
    base_np = np.empty((N, C), np.float32)
    XY_np   = np.zeros((N, Mmax, 2), np.float32)
    P_np    = np.zeros((N, Mmax),    np.float32)
    M_np    = np.zeros((N, Mmax),    np.float32)
    S_np    = None
    if Dslot is not None:
        S_np = np.zeros((N, Mmax, Dslot), np.float32)

    for i,p in enumerate(paths):
        z = np.load(p, allow_pickle=True)
        Y_np[i]    = int(z["clazz"])
        base_np[i] = z["logits"].astype(np.float32)
        XY_i = z["XY"].astype(np.float32)
        P_i  = z["slot_prob"].astype(np.float32) if "slot_prob" in z else z["energy_norm"].astype(np.float32)
        msk  = z["slot_mask"].astype(np.float32)
        Mi = XY_i.shape[0]
        XY_np[i,:Mi] = XY_i; P_np[i,:Mi] = P_i; M_np[i,:Mi] = msk
        if S_np is not None and "S_slots" in z:
            S_np[i,:Mi] = z["S_slots"].astype(np.float32)

    Y    = torch.from_numpy(Y_np).long().to(device)
    base = torch.from_numpy(base_np).float().to(device)
    XY   = torch.from_numpy(XY_np).float().to(device)
    P    = torch.from_numpy(P_np).float().to(device)
    Msk  = torch.from_numpy(M_np).float().to(device)
    S    = torch.from_numpy(S_np).float().to(device) if S_np is not None else None
    return Y, base, (S, XY, P, Msk), Mmax, C, (0 if S is None else S.size(-1))


def load_proto_json(path, device, max_class_protos=None):
    with open(path,"r") as f: data = json.load(f)
    per = data.get("per_class", {})
    out = {}
    for k, block in per.items():
        cid = int(k)
        vecs = []
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                arr = np.asarray(mus, np.float32)
                if arr.ndim == 1: vecs.append(torch.tensor(arr, dtype=torch.float32, device=device))
                else:
                    for v in arr: vecs.append(torch.tensor(v, dtype=torch.float32, device=device))
        elif isinstance(block, list):
            for p in block:
                mu = p.get("mu") if isinstance(p, dict) else p
                if mu is None: continue
                vecs.append(torch.tensor(np.asarray(mu, np.float32), dtype=torch.float32, device=device))
        else:
            vecs.append(torch.tensor(np.asarray(block, np.float32), dtype=torch.float32, device=device))
        if (max_class_protos is not None) and len(vecs)>max_class_protos:
            vecs = vecs[:max_class_protos]
        out[cid] = vecs
    return out


# ===================== 모델 =====================

class Proto(nn.Module):
    def __init__(self, centers: torch.Tensor):
        super().__init__()
        if centers.ndim == 1: centers = centers.unsqueeze(0)
        self.register_buffer("centers", centers.float())  # (Kc, d)
        self.psi = nn.Parameter(torch.zeros(centers.size(0), dtype=torch.float32))

    @property
    def dim(self):
        return 0 if self.centers.numel()==0 else int(self.centers.size(1))

    def score(self, S_active: torch.Tensor, beta: float) -> torch.Tensor:
        if self.centers.numel()==0:
            return torch.zeros((), device=S_active.device)
        if S_active.ndim == 1: S_active = S_active.unsqueeze(0)
        diff  = S_active[:,None,:] - self.centers[None,:,:]    # (n,Kc,d)
        dist2 = (diff**2).sum(-1)
        sim   = torch.exp(-beta * dist2)                       # (n,Kc)
        w = torch.softmax(self.psi, dim=0)                     # (Kc,)
        return (sim * w[None,:]).sum(dim=1).mean()             # scalar


class Bank(nn.Module):
    def __init__(self, protos):
        super().__init__()
        self.protos = nn.ModuleList(protos)

    @property
    def dim(self):
        for p in self.protos:
            if p.dim>0: return p.dim
        return 0

    def score(self, S_active: torch.Tensor, beta: float) -> torch.Tensor:
        if len(self.protos)==0: return torch.zeros((), device=S_active.device)
        vals = [p.score(S_active, beta) for p in self.protos]
        return torch.stack(vals).max()


class ProtoMixer(nn.Module):
    def __init__(self, per_class_centers, C, alpha, beta, xy_weight, device):
        super().__init__()
        self.C = C
        self.beta = beta
        self.xy_weight = xy_weight
        # learnable alpha (logit)
        a0 = max(1e-4, min(1.0-1e-4, float(alpha)))
        self.alpha_param = nn.Parameter(torch.tensor(np.log(a0/(1.0-a0)), dtype=torch.float32))

        banks = []
        for c in range(C):
            vecs = per_class_centers.get(c, [])
            protos = []
            if len(vecs)>0:
                centers = torch.stack(vecs, dim=0)  # (Kc,d)
                protos = [Proto(centers)]
            banks.append(Bank(protos))
        self.banks = nn.ModuleList(banks)
        self.to(device)

    def _score_one(self, S_i, top_p: float):
        S_slots = S_i["S_slots"]  # (M,D) or None
        XY      = S_i["XY"]       # (M,2)
        P       = S_i["P"]        # (M,)
        mask    = S_i["slot_mask"]# (M,)

        score_vec = P * mask
        vals, idx = torch.sort(score_vec, descending=True)
        csum = torch.cumsum(vals, dim=0)
        k = max(1, int((csum <= (top_p * (score_vec.sum() + 1e-8))).sum().item()))
        active_idx = idx[:k]

        scores = []
        for bank in self.banks:
            d = bank.dim
            if d == 0:
                scores.append(torch.zeros((), device=score_vec.device))
                continue
            S_feat = build_feat_for_dim(S_slots, XY, d, self.xy_weight)   # (M,d)
            S_active = S_feat[active_idx]                                 # (k,d)
            scores.append(bank.score(S_active, self.beta))
        return torch.stack(scores)  # (C,)

    def fuse_batch(self, base_b, S_tuple, top_p: float):
        S_slots_b, XY_b, P_b, mask_b = S_tuple
        B = base_b.size(0)
        outs = []
        alpha = torch.sigmoid(self.alpha_param)
        for i in range(B):
            S_i = {
                "S_slots":  None if S_slots_b is None else S_slots_b[i],
                "XY":       XY_b[i],
                "P":        P_b[i],
                "slot_mask":mask_b[i],
            }
            s_i = self._score_one(S_i, top_p=top_p)  # (C,)
            outs.append(alpha * base_b[i] + (1.0 - alpha) * s_i)
        return torch.stack(outs, dim=0)


# ===================== Train / Eval / Save =====================

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()

def train_mixer(model: ProtoMixer, Y, base, S_tuple, train_top_p: float, epochs: int, lr: float, bs: int = 256):
    S_slots, XY, P, M = S_tuple
    ds = TensorDataset(base, *(x for x in [S_slots, XY, P, M] if x is not None), Y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train(); tot=0.0; n=0
        for batch in dl:
            if S_slots is None:
                base_b, XY_b, P_b, M_b, y_b = batch
                S_tup = (None, XY_b, P_b, M_b)
            else:
                base_b, S_b, XY_b, P_b, M_b, y_b = batch
                S_tup = (S_b, XY_b, P_b, M_b)
            fused = model.fuse_batch(base_b, S_tup, top_p=train_top_p)
            loss = F.cross_entropy(fused, y_b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * y_b.size(0); n += y_b.size(0)
        print(f"[learn] epoch {ep} | loss {tot/max(1,n):.4f}")

@torch.no_grad()
def eval_mixer(model: ProtoMixer, Y, base, S_tuple, top_p: float, bs: int = 512):
    S_slots, XY, P, M = S_tuple
    ds = TensorDataset(base, *(x for x in [S_slots, XY, P, M] if x is not None), Y)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    base_acc, fused_acc, n = 0.0, 0.0, 0
    for batch in dl:
        if S_slots is None:
            base_b, XY_b, P_b, M_b, y_b = batch
            S_tup = (None, XY_b, P_b, M_b)
        else:
            base_b, S_b, XY_b, P_b, M_b, y_b = batch
            S_tup = (S_b, XY_b, P_b, M_b)
        base_acc  += accuracy_from_logits(base_b, y_b) * y_b.size(0)
        fused_b    = model.fuse_batch(base_b, S_tup, top_p=top_p)
        fused_acc += accuracy_from_logits(fused_b, y_b) * y_b.size(0)
        n += y_b.size(0)
    return base_acc/n, fused_acc/n

@torch.no_grad()
def save_bank(model: ProtoMixer, path: str):
    bank = {
        "alpha": float(torch.sigmoid(model.alpha_param).item()),
        "beta": model.beta,
        "xy_weight": model.xy_weight,
        "C": model.C,
        "classes": []
    }
    for c, cb in enumerate(model.banks):
        entry = {"class": c, "protos": []}
        for p in cb.protos:
            entry["protos"].append({
                "centers": p.centers.detach().cpu().numpy().tolist(),
                "psi": p.psi.detach().cpu().numpy().tolist()
            })
        bank["classes"].append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(bank, path)
    print(f"[saved] bank -> {path}")


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--xy_weight", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--train_top_p", type=float, default=1.0)
    ap.add_argument("--learn_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_class_protos", type=int, default=None)
    ap.add_argument("--save_bank", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Y, base, S_tuple, Mmax, C, Dslot = load_dump_dir(args.dump_dir, device=device)
    per_class_centers = load_proto_json(args.proto_json, device=device, max_class_protos=args.max_class_protos)

    mixer = ProtoMixer(per_class_centers, C=C, alpha=args.alpha, beta=args.beta, xy_weight=args.xy_weight, device=device)

    base_acc, fused_acc = eval_mixer(mixer, Y, base, S_tuple, top_p=args.top_p)
    print(f"n={Y.numel()} | base_acc={base_acc*100:.2f}% | fused_acc={fused_acc*100:.2f}% | Δ={(fused_acc-base_acc)*100:+.2f}")

    if args.learn_epochs > 0:
        train_mixer(mixer, Y, base, S_tuple, train_top_p=args.train_top_p, epochs=args.learn_epochs, lr=args.lr, bs=args.batch_size)
        base_acc, fused_acc = eval_mixer(mixer, Y, base, S_tuple, top_p=args.top_p)
        print(f"[after learn] base_acc={base_acc*100:.2f}% | fused_acc={fused_acc*100:.2f}% | Δ={(fused_acc-base_acc)*100:+.2f}")

    if args.save_bank:
        save_bank(mixer, args.save_bank)


if __name__ == "__main__":
    main()
