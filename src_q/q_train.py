# -*- coding: utf-8 -*-
# src_q/q_train.py
#
# Triplicate-per-position with optional:
# - per-repeat query adapters (exp_wq)
# - intra-group gating (exp_gate)
# - per-repeat head bias (exp_head_bias)
# - comb4 mask with optional random drop-1 cell (mask_drop1, slow but accurate)
import os, math, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .q_trunk import SlotClassifier

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True"
)

def seed_all(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model: SlotClassifier, loader: DataLoader, device: torch.device,
             use_spmask: bool, spmask_grid: int, spmask_assign: str, tau_p: float):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits, _ = model(x, use_spmask=use_spmask, spmask_grid=spmask_grid,
                          spmask_assign=spmask_assign, tau_p=tau_p)
        pred = logits.argmax(dim=1)
        correct += int((pred==y).sum().item())
        total   += int(y.numel())
    return 100.0 * correct / max(1,total)

def get_loaders(batch_size=512, num_workers=8):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.EMNIST(root="./data", split="balanced", train=True,  download=True, transform=tf)
    test_ds  = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), prefetch_factor=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers>0), prefetch_factor=4)
    num_classes = int(train_ds.targets.max().item()) + 1
    return train_loader, test_loader, num_classes

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters_path", required=True, help="path to 9x9 conv1 filters .npy (150,1,9,9)")
    ap.add_argument("--out_dir", required=True)

    # slots
    ap.add_argument("--num_slots_base", type=int, default=126)
    ap.add_argument("--repeats_per_pos", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)

    # train
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # mask & temps
    ap.add_argument("--warmup_epochs", type=int, default=5, help="epochs with no spatial mask")
    ap.add_argument("--tau", type=float, default=0.7, help="global slot weight temp")
    ap.add_argument("--tau_intra", type=float, default=0.7, help="intra-group gate temp")

    # experiments toggles
    ap.add_argument("--exp_wq", type=int, default=1)
    ap.add_argument("--exp_gate", type=int, default=1)
    ap.add_argument("--exp_head_bias", type=int, default=1)
    ap.add_argument("--mask_drop1", type=int, default=0)  # slow but sometimes helps

    # optim
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--freeze_conv1_epochs", type=int, default=30)
    ap.add_argument("--amp", type=int, default=0)

    args = ap.parse_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = get_loaders(args.batch_size, args.num_workers)

    model = SlotClassifier(
        filters_path=args.filters_path,
        num_slots_base=args.num_slots_base,
        repeats_per_pos=args.repeats_per_pos,
        d_model=args.d_model, nhead=args.nhead,
        num_classes=num_classes,
        exp_wq=bool(args.exp_wq),
        exp_gate=bool(args.exp_gate),
        exp_head_bias=bool(args.exp_head_bias),
        tau_intra=args.tau_intra,
        mask_drop1=bool(args.mask_drop1),
    ).to(device)

    # freeze conv1 first
    conv1_params = list(model.conv1.parameters())
    for p in conv1_params: p.requires_grad_(False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type=="cuda")  # (warn: deprecated API string, but ok)

    # metric file
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f: f.write("epoch,train_loss,test_acc,P_ent,maxP\n")

    x0, _ = next(iter(train_loader))
    stats = model._check_stats_once(x0[:8].to(device))
    print(f"[sanity] conv1 stats: {stats}")

    best = -1.0
    for ep in range(1, args.epochs+1):
        model.train()

        # unfreeze conv1 after N epochs (slower but can improve)
        if ep == args.freeze_conv1_epochs + 1:
            for p in conv1_params: p.requires_grad_(True)
            optim.add_param_group({"params": conv1_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay})
            print("[info] conv1 unfrozen (+added to optimizer)")

        use_spmask = (ep > args.warmup_epochs)
        tau_p = args.tau

        run_loss = run_Pent = run_maxP = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type=="cuda"):
                logits, aux = model(x, use_spmask=use_spmask, spmask_grid=3, spmask_assign="round", tau_p=tau_p)
                loss = F.cross_entropy(logits.float(), y)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optim); scaler.update()

            P = aux["slot_prob"]; M = P.size(1)
            Pent = (-(P.clamp_min(1e-8).log() * P).sum(dim=1) / math.log(float(M))).mean()
            maxP = P.max(dim=1).values.mean()

            run_loss += float(loss.item())
            run_Pent += float(Pent.item())
            run_maxP += float(maxP.item())
            n_batches += 1

        train_loss = run_loss / max(1, n_batches)
        P_ent = run_Pent / max(1, n_batches)
        maxP  = run_maxP / max(1, n_batches)

        acc = evaluate(model, test_loader, device,
                       use_spmask=use_spmask, spmask_grid=3, spmask_assign="round", tau_p=tau_p)

        print(f"[ep {ep:02d}] loss={train_loss:.4f} | acc={acc:.2f}% | P_ent={P_ent:.3f} | maxP={maxP:.3f} | spmask={int(use_spmask)}")

        with open(csv_path, "a") as f:
            f.write(f"{ep},{train_loss:.6f},{acc:.4f},{P_ent:.6f},{maxP:.6f}\n")

        if acc > best:
            best = acc
            ckpt = {
                "model": model.state_dict(),
                "meta": {
                    "epoch": ep,
                    "num_slots_base": args.num_slots_base,
                    "repeats_per_pos": args.repeats_per_pos,
                    "num_slots": int(args.num_slots_base * args.repeats_per_pos),
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "tau": args.tau,
                    "tau_intra": args.tau_intra,
                    "warmup_epochs": args.warmup_epochs,
                    "filters_path": args.filters_path,
                    "exp_wq": int(bool(args.exp_wq)),
                    "exp_gate": int(bool(args.exp_gate)),
                    "exp_head_bias": int(bool(args.exp_head_bias)),
                    "mask_drop1": int(bool(args.mask_drop1)),
                    "freeze_conv1_epochs": args.freeze_conv1_epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                }
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"[save] best.pt (acc={best:.2f}%)")

if __name__ == "__main__":
    train()
