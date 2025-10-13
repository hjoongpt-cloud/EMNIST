# ============================================================
# FILE: ae9x9_proj64_linear_eval.py
# PURPOSE: Use AE front-end as a frozen (or partially frozen) encoder and
#          train a simple classifier head on top (EMNIST balanced).
#          - Loads AE checkpoint (preferred) or conv1 filters (.npy)
#          - Keeps WTA (top-k) gating consistent with AE
#          - Several head options (GAP linear, token-avg, 1x1+GAP, attn-pool)
#          - Hardest-per-class analysis by classification loss (top-K)
#          - Default stride=2, top-3 WTA
# ============================================================

import os, sys, math, argparse, random, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# ------------------- Utils -------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def grid_img(t: torch.Tensor, nrow: int = 16, normalize: bool = True, pad_value: float = 0.5):
    return make_grid(t, nrow=nrow, normalize=normalize, pad_value=pad_value)

# ------------------- Encoder (same front-end as AE) -------------------
class Encoder9x9Proj64(nn.Module):
    def __init__(self, K=192, d=64, stride=2, topk=3, wta_mode="soft", tau=0.5, wta_warmup=0):
        super().__init__()
        self.K, self.d, self.stride = K, d, stride
        self.topk, self.wta_mode, self.tau, self.wta_warmup = topk, wta_mode, tau, wta_warmup
        self._cur_epoch = 999999  # classification에서는 보통 warmup 없이 바로 적용
        pad = 4
        self.conv1 = nn.Conv2d(1, K, kernel_size=9, stride=stride, padding=pad, bias=False)
        self.enc_act = nn.GELU()
        self.proj_down = nn.Conv2d(K, d, kernel_size=1, bias=True)

    @torch.no_grad()
    def renorm_conv1(self):
        W = self.conv1.weight.view(self.K, -1)
        n = W.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.conv1.weight.copy_((W / n).view_as(self.conv1.weight))

    def apply_wta(self, a: torch.Tensor) -> torch.Tensor:
        if self.topk <= 0 or self.wta_mode == "none" or (self._cur_epoch < self.wta_warmup):
            return a
        B, K, H, W = a.shape
        a_flat = a.view(B, K, -1)
        if self.wta_mode == "soft":
            p = (a_flat / self.tau).softmax(dim=1)
            if self.topk == 1:
                g = p
            else:
                topv, topi = torch.topk(p, self.topk, dim=1)
                mask = torch.zeros_like(p).scatter_(1, topi, topv)
                g = mask / (mask.sum(dim=1, keepdim=True) + 1e-6)
        elif self.wta_mode == "hard":
            topv, topi = torch.topk(a_flat, self.topk, dim=1)
            hard = torch.zeros_like(a_flat).scatter_(1, topi, 1.0)
            p = (a_flat / self.tau).softmax(dim=1)
            g = (hard - p).detach() + p
        else:
            return a
        return (a_flat * g).view(B, K, H, W)

    def forward(self, x):
        a = self.enc_act(self.conv1(x))      # (B,K,H',W') -> H'=W'=14 (stride=2)
        a = self.apply_wta(a)
        h = self.proj_down(a)                # (B,64,H',W')
        return h

# ------------------- Heads -------------------
class GAPLinearHead(nn.Module):
    def __init__(self, d=64, num_classes=47):
        super().__init__()
        self.fc = nn.Linear(d, num_classes)
    def forward(self, h):
        # h: (B,d,H,W)
        x = h.mean(dim=(2,3))
        return self.fc(x)

class TokenAvgLinearHead(nn.Module):
    def __init__(self, d=64, num_classes=47):
        super().__init__()
        self.fc = nn.Linear(d, num_classes)
    def forward(self, h):
        B,d,H,W = h.shape
        x = h.view(B,d,-1).mean(dim=2)
        return self.fc(x)

class Conv1x1GapHead(nn.Module):
    def __init__(self, d=64, d_cls=64, num_classes=47):
        super().__init__()
        self.p1 = nn.Conv2d(d, d_cls, 1)
        self.act = nn.GELU()
        self.fc = nn.Linear(d_cls, num_classes)
    def forward(self, h):
        z = self.act(self.p1(h))
        x = z.mean(dim=(2,3))
        return self.fc(x)

class AttnPoolHead(nn.Module):
    def __init__(self, d=64, num_classes=47):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d))
        self.fc = nn.Linear(d, num_classes)
    def forward(self, h):
        # h: (B,d,H,W) -> (B,d,HW)
        B,d,H,W = h.shape
        x = h.view(B,d,-1)
        # attn weights: softmax over HW using dot(q, x)
        q = self.query.view(1,d,1)
        logits = (x * q).sum(dim=1)              # (B,HW)
        w = logits.softmax(dim=1).unsqueeze(1)   # (B,1,HW)
        pooled = (x * w).sum(dim=2)              # (B,d)
        return self.fc(pooled)

HEADS = {
    "gap_linear": GAPLinearHead,
    "token_avg_linear": TokenAvgLinearHead,
    "conv1x1_gap_linear": Conv1x1GapHead,
    "attnpool_linear": AttnPoolHead,
}

# ------------------- Model wrapper -------------------
class EncClassifier(nn.Module):
    def __init__(self, enc: Encoder9x9Proj64, head_name: str = "gap_linear", num_classes: int = 47):
        super().__init__()
        self.enc = enc
        Head = HEADS[head_name]
        self.head = Head(d=enc.d, num_classes=num_classes)
    def forward(self, x):
        h = self.enc(x)
        return self.head(h)

# ------------------- Hardest-per-class picker -------------------
@torch.no_grad()
def save_hardest_classification(model, dl, device, out_dir: Path, K: int = 3, smooth: float = 0.0, postfix: str = ""):
    import csv
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none', label_smoothing=smooth)
    KCL = 47
    tops = {c: [] for c in range(KCL)}  # list of (loss, x, y_true, y_pred, p_pred)
    counts = np.zeros(KCL, dtype=np.int64)

    for x,y in dl:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss_b = CE(logits, y).detach().cpu().numpy()
        probs = logits.softmax(dim=1)
        p_pred, y_pred = probs.max(dim=1)
        for i in range(x.size(0)):
            c = int(y[i].item()); counts[c] += 1
            entry = (float(loss_b[i]), x[i].cpu(), int(y[i].item()), int(y_pred[i].item()), float(p_pred[i].item()))
            buf = tops[c]
            if len(buf) < K:
                buf.append(entry)
            else:
                j = min(range(K), key=lambda t: buf[t][0])
                if entry[0] > buf[j][0]:
                    buf[j] = entry

    # sort and save grids
    inputs = []; labels = []
    pred_lbl = []; pred_conf = []
    for c in range(KCL):
        tops[c].sort(key=lambda t: -t[0])
        for (ls, x_cpu, y_true, y_hat, p) in tops[c]:
            inputs.append(x_cpu)
            labels.append(y_true)
            pred_lbl.append(y_hat)
            pred_conf.append(p)

    if len(inputs) == 0:
        return

    inputs = torch.stack(inputs)
    save_image(grid_img(inputs, nrow=K, normalize=True), str(out_dir / f"cls_hard_inputs{postfix}.png"))

    # CSV with details
    with open(out_dir / f"cls_hard_meta{postfix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","picked","y_pred","p_pred","loss"])
        idx = 0
        for c in range(KCL):
            for k in range(min(K, len(tops[c]))):
                ls, x_cpu, y_true, y_hat, p = tops[c][k]
                w.writerow([c, idx, y_hat, f"{p:.6f}", f"{ls:.6f}"])
                idx += 1

# ------------------- Train/Eval -------------------

def accuracy_from_logits(logits, y):
    _, pred = logits.max(dim=1)
    correct = (pred == y).sum().item()
    return correct / max(1, y.numel())

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total, correct = 0, 0
    for x,y in dl:
        x=x.to(device); y=y.to(device)
        logits = model(x)
        _, pred = logits.max(dim=1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct / max(1,total)

# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs/cls_eval")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)

    # Encoder config (match AE)
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--stride", type=int, default=2, choices=[1,2])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--wta_mode", type=str, default="soft", choices=["none","soft","hard"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--wta_warmup", type=int, default=0)

    # Init from AE
    ap.add_argument("--ae_ckpt", type=str, default="", help="path to ae9x9_proj64.pt (preferred)")
    ap.add_argument("--init_filters", type=str, default="", help="optional .npy (K,1,9,9) or (K,9,9)")

    # Classifier head
    ap.add_argument("--head", type=str, default="gap_linear", choices=list(HEADS.keys()))
    ap.add_argument("--d_cls", type=int, default=64, help="only for conv1x1_gap_linear")

    # Freeze mode
    ap.add_argument("--freeze_mode", type=str, default="linear", choices=["none","linear","head_plus_proj"],
                    help="none: tune all; linear: freeze encoder (conv1+proj_down); head_plus_proj: freeze conv1 only")

    # Train hyperparams
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--amp", type=int, default=1)

    # Eval options
    ap.add_argument("--hard_eval", type=int, default=1)
    ap.add_argument("--hard_k", type=int, default=3)

    # Resume/Eval-only
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--eval_only", type=int, default=0)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir); _ensure_dir(out_dir)

    # Data (EMNIST balanced)
    tfm = transforms.ToTensor()
    ds_tr = datasets.EMNIST(args.data_root, split="balanced", train=True, download=True, transform=tfm)
    ds_te = datasets.EMNIST(args.data_root, split="balanced", train=False, download=True, transform=tfm)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    # Build encoder
    enc = Encoder9x9Proj64(K=args.K, d=64, stride=args.stride, topk=args.topk,
                           wta_mode=args.wta_mode, tau=args.tau, wta_warmup=args.wta_warmup).to(device)

    # Init from AE checkpoint (preferred)
    if args.ae_ckpt and Path(args.ae_ckpt).exists():
        print(f"[INIT] Loading AE checkpoint from {args.ae_ckpt}")
        sd = torch.load(args.ae_ckpt, map_location="cpu")
        miss, unexp = enc.load_state_dict({k.replace('decoder','proj_up'):v for k,v in sd.items() if k.startswith('conv1') or k.startswith('proj_down')}, strict=False)
        # If strict=False, it will load conv1/proj_down and ignore others.
        enc.renorm_conv1()
    # Override conv1 from .npy if provided
    if args.init_filters:
        W = np.load(args.init_filters)
        if W.ndim == 3:
            W = W[:,None,:,:]
        assert W.shape[1:] == (1,9,9), f"init_filters must be (K,1,9,9) or (K,9,9), got {W.shape}"
        if W.shape[0] != args.K:
            print(f"[WARN] init_filters K={W.shape[0]} != args.K={args.K}. Rebuilding encoder with K={W.shape[0]}.")
            args.K = int(W.shape[0])
            enc = Encoder9x9Proj64(K=args.K, d=64, stride=args.stride, topk=args.topk,
                                   wta_mode=args.wta_mode, tau=args.tau, wta_warmup=args.wta_warmup).to(device)
        with torch.no_grad():
            enc.conv1.weight.copy_(torch.from_numpy(W).to(enc.conv1.weight.device).to(enc.conv1.weight.dtype))
        enc.renorm_conv1()

    # Build classifier
    model = EncClassifier(enc, head_name=args.head, num_classes=47).to(device)

    # Freeze modes
    if args.freeze_mode == "linear":
        for p in model.enc.parameters():
            p.requires_grad = False
        print("[FREEZE] encoder frozen (linear eval)")
    elif args.freeze_mode == "head_plus_proj":
        for p in model.enc.conv1.parameters():
            p.requires_grad = False
        print("[FREEZE] conv1 frozen; proj_down + head trainable")
    else:
        print("[FREEZE] none (full fine-tune)")

    # Optim
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and (device.type=="cuda"))
    CE = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Resume
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"]) ; opt.load_state_dict(ck["opt"])
        print(f"[RESUME] from {args.resume}")

    # Eval-only
    if args.eval_only:
        acc = evaluate(model, dl_te, device)
        print(f"[EVAL] test_acc={acc*100:.2f}%")
        if args.hard_eval:
            save_hardest_classification(model, dl_te, device, out_dir, K=args.hard_k, postfix="_test")
        return

    # Train loop
    best_acc = 0.0
    for ep in range(1, args.epochs+1):
        model.train()
        total, total_loss = 0, 0.0
        for x,y in dl_tr:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and (device.type=="cuda")):
                logits = model(x)
                loss = CE(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += y.numel(); total_loss += loss.item()*y.size(0)
        tr_loss = total_loss / max(1,total)
        te_acc = evaluate(model, dl_te, device)
        print(f"[ep {ep}] train_loss={tr_loss:.4f}  test_acc={te_acc*100:.2f}%")
        # checkpoint
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep}, str(out_dir / "last.pt"))
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ep": ep}, str(out_dir / "best.pt"))

    print(f"[DONE] best_test_acc={best_acc*100:.2f}%  -> {out_dir}")
    if args.hard_eval:
        save_hardest_classification(model, dl_te, device, out_dir, K=args.hard_k, postfix="_best")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
