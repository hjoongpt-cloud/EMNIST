#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt, extract_slots_with_queries, \
                             load_prototypes_json, build_class_index, per_slot_evidence_compact

def get_loaders(batch_size=256, num_workers=2, val_ratio=0.1, seed=42):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=tf)
    test_ds    = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=tf)
    n_total = len(full_train); n_val = int(round(n_total*val_ratio)); n_tr = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    tr_ds, val_ds = random_split(full_train, [n_tr, n_val], generator=g)
    train_loader = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes  = int(full_train.targets.max().item()) + 1
    return train_loader, val_loader, test_loader, num_classes

def aggregate_logits(evi_bmc, P_bm, method="wsum", slot_topk=5, beta=1.2):
    """
    evi_bmc: (B,M,Cp), P_bm: (B,M)
    """
    if method not in ("wsum","softk"):
        raise ValueError(f"agg {method} not supported in this script (use wsum/softk)")
    if P_bm.dim() == 1: P_bm = P_bm.view(1, -1)
    B, M, Cp = evi_bmc.shape
    if method == "wsum":
        w = P_bm / P_bm.sum(dim=1, keepdim=True).clamp_min(1e-8)         # (B,M)
    else:  # softk (+ optional top-k)
        if slot_topk and slot_topk > 0 and slot_topk < M:
            topv, topi = torch.topk(P_bm, int(slot_topk), dim=1)
            mask = torch.zeros_like(P_bm); mask.scatter_(1, topi, 1.0)
            q = P_bm * mask
        else:
            q = P_bm
        z = (q - q.mean(dim=1, keepdim=True)) / max(1e-6, float(beta))
        w = torch.softmax(z, dim=1)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bmc,bm->bc", evi_bmc, w)

def train_epoch(trunk, opt, loader, device, slot_q, C, C_cls, labels_sorted, args, use_amp=True):
    trunk.train()
    run_loss, n = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pbar = tqdm(loader, desc="train")
    for x, y in pbar:
        x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            tokens, aux = trunk(x)  # (B,196,D)
            A, P, S, _ = extract_slots_with_queries(tokens, slot_q, True,
                                                    tau_p=args.tau, grid=args.spmask_grid, assign=args.spmask_assign)
        with torch.cuda.amp.autocast(enabled=False):
            evi = per_slot_evidence_compact(S, C, C_cls, labels_sorted,
                                            class_reduce=args.class_reduce, proto_tau=args.proto_tau)  # (B,M,Cp)
            logits = aggregate_logits(evi, P, method=args.agg, slot_topk=args.slot_topk, beta=args.beta)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trunk.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update()
        run_loss += float(loss.item()); n += 1
        pbar.set_postfix(loss=f"{run_loss/max(1,n):.4f}")
    return run_loss/max(1,n)

@torch.no_grad()
def evaluate(trunk, loader, device, slot_q, C, C_cls, labels_sorted, args):
    trunk.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and bool(args.amp))):
            tokens, aux = trunk(x)
            A, P, S, _ = extract_slots_with_queries(tokens, slot_q, True,
                                                    tau_p=args.tau, grid=args.spmask_grid, assign=args.spmask_assign)
        with torch.cuda.amp.autocast(enabled=False):
            evi = per_slot_evidence_compact(S, C, C_cls, labels_sorted,
                                            class_reduce=args.class_reduce, proto_tau=args.proto_tau)
            logits = aggregate_logits(evi, P, method=args.agg, slot_topk=args.slot_topk, beta=args.beta)
            pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item()); total += int(y.numel())
    return 100.0 * correct / max(1, total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_o_train", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--agg", type=str, default="wsum")
    ap.add_argument("--slot_topk", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--proto_tau", type=float, default=0.4)
    ap.add_argument("--class_reduce", choices=["lse","max"], default="lse")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--spmask_grid", type=int, default=3)
    ap.add_argument("--spmask_assign", type=str, default="round")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filters = np.load(args.conv1_filters)
    meta = torch.load(args.ckpt_o_train, map_location="cpu").get("meta", {})
    d_model = int(meta.get("d_model", 64)); nhead = int(meta.get("nhead", 4)); num_layers = int(meta.get("num_layers", 2))
    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=256, conv1_filters=filters).to(device)

    C, C_cls, _ = load_prototypes_json(args.proto_json, device=device, filter_zero_proto=True)
    labels_sorted, label_to_col, col_to_label = build_class_index(C_cls)

    slot_q = load_slot_queries_from_ckpt(args.ckpt_o_train, device)

    opt = optim.AdamW([p for p in trunk.parameters() if p.requires_grad], lr=args.lr)

    train_loader, val_loader, test_loader, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, val_ratio=args.val_ratio, seed=args.seed
    )

    best = -1.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(trunk, opt, train_loader, device, slot_q, C, C_cls, labels_sorted, args, use_amp=bool(args.amp))
        val_acc = evaluate(trunk, val_loader, device, slot_q, C, C_cls, labels_sorted, args)
        print(f"[ep {ep:02d}] loss={tr_loss:.4f} | val={val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save({"model": trunk.state_dict(), "meta": meta}, os.path.join(args.out_dir, "best_with_slots.pt"))
            print(f"[save] best_with_slots.pt (val={best:.2f}%)")

    # final test
    best_ckpt = torch.load(os.path.join(args.out_dir, "best_with_slots.pt"), map_location="cpu")
    trunk.load_state_dict(best_ckpt["model"], strict=False)
    test_acc = evaluate(trunk, test_loader, device, slot_q, C, C_cls, labels_sorted, args)
    with open(os.path.join(args.out_dir, "final_test.json"), "w") as f:
        import json; json.dump({"test_acc": float(test_acc)}, f, indent=2)
    print(f"[final] test_acc={test_acc:.2f}%")

if __name__ == "__main__":
    main()
