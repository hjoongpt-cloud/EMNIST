# Train SlotTrunkAE96 (integrated with AE96 front-end) on EMNIST balanced
# - ToTensor only (no normalize)
# - soft top-3 WTA after warmup
# - conv1 freeze N epochs then unfreeze with lower LR
# - probability-sum aggregator by default

import argparse, os, sys, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from q_trunk_ae96 import SlotTrunkAE96

# ---------------- Utils ----------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def grid_img(t: torch.Tensor, nrow: int = 16, normalize: bool = True, pad_value: float = 0.5):
    return make_grid(t, nrow=nrow, normalize=normalize, pad_value=pad_value)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total, correct = 0, 0
    for x,y in dl:
        x=x.to(device); y=y.to(device)
        logits = model(x)
        if logits.shape[-1] == 47:
            pred = logits.argmax(dim=-1)
        else:
            pred = logits.softmax(dim=-1).argmax(dim=-1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct / max(1,total)

@torch.no_grad()
def save_hardest_classification(model, dl, device, out_dir: Path, K: int = 3, postfix: str = ""):
    import csv
    out_dir = Path(out_dir); _ensure_dir(out_dir)
    CE = nn.CrossEntropyLoss(reduction='none')
    KCL = 47
    tops = {c: [] for c in range(KCL)}
    for x,y in dl:
        x=x.to(device); y=y.to(device)
        logits = model(x)
        loss_b = CE(logits, y).detach().cpu().numpy()
        probs = logits.softmax(dim=1)
        p_pred, y_pred = probs.max(dim=1)
        for i in range(x.size(0)):
            c = int(y[i].item())
            entry = (float(loss_b[i]), x[i].cpu(), int(y_pred[i].item()), float(p_pred[i].item()))
            buf = tops[c]
            if len(buf) < K:
                buf.append(entry)
            else:
                j = min(range(K), key=lambda t: buf[t][0])
                if entry[0] > buf[j][0]:
                    buf[j] = entry
    # save grid
    imgs = []
    for c in range(KCL):
        tops[c].sort(key=lambda t: -t[0])
        for (ls, xi, yp, pp) in tops[c]:
            imgs.append(xi)
    if imgs:
        save_image(grid_img(torch.stack(imgs), nrow=K, normalize=True), str(Path(out_dir) / f"cls_hard_inputs{postfix}.png"))
    # meta csv
    with open(Path(out_dir) / f"cls_hard_meta{postfix}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class','rank','y_pred','p_pred','loss'])
        for c in range(KCL):
            for k in range(min(K, len(tops[c]))):
                ls, xi, yp, pp = tops[c][k]
                w.writerow([c,k,yp,f"{pp:.6f}",f"{ls:.6f}"])

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='outputs/q_slot_k96_s2_soft3_aeinit')
    ap.add_argument('--data_root', type=str, default='./data')
    ap.add_argument('--seed', type=int, default=42)

    # Front-end init
    ap.add_argument('--K', type=int, default=96)
    ap.add_argument('--stride', type=int, default=2, choices=[1,2])
    ap.add_argument('--init_conv1', type=str, required=True)
    ap.add_argument('--init_proj', type=str, required=True)
    ap.add_argument('--proj_keep_idx', type=str, default='', help='optional keep indices to slice proj if channel mismatch')

    # WTA
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--wta_warmup', type=int, default=5)

    # Slots
    ap.add_argument('--gh', type=int, default=3)
    ap.add_argument('--gw', type=int, default=3)
    ap.add_argument('--comb_r', type=int, default=4)
    ap.add_argument('--aggregator', type=str, default='prob_sum', choices=['prob_sum','logit_mean','prob_max'])

    # Head
    ap.add_argument('--head_hidden', type=int, default=128)
    ap.add_argument('--head_dropout', type=float, default=0.1)

    # Train
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--conv1_lr_mult', type=float, default=0.1)
    ap.add_argument('--freeze_conv1_epochs', type=int, default=8)
    ap.add_argument('--amp', type=int, default=1)

    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.out_dir); _ensure_dir(out_dir)

    # Data: ToTensor only (no normalize)
    tfm = transforms.ToTensor()
    ds_tr = datasets.EMNIST(args.data_root, split='balanced', train=True, download=True, transform=tfm)
    ds_te = datasets.EMNIST(args.data_root, split='balanced', train=False, download=True, transform=tfm)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = SlotTrunkAE96(num_classes=47, K=args.K, stride=args.stride, topk=args.topk, tau=args.tau, wta_warmup=args.wta_warmup,
                          init_conv1=args.init_conv1, init_proj=args.init_proj, proj_keep_idx=args.proj_keep_idx,
                          gh=args.gh, gw=args.gw, comb_r=args.comb_r,
                          aggregator=args.aggregator, d=64, hidden=args.head_hidden, dropout=args.head_dropout).to(device)

    # Freeze schedule
    for p in model.enc.conv1.parameters():
        p.requires_grad = False

    params_main = [p for n,p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_main, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(args.amp) and (device.type=='cuda'))
    CE = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs+1):
        model.set_epoch(ep)
        if ep == args.freeze_conv1_epochs:
            for p in model.enc.conv1.parameters():
                p.requires_grad = True
            opt.add_param_group({'params': model.enc.conv1.parameters(), 'lr': args.lr*args.conv1_lr_mult, 'weight_decay': args.weight_decay})
            print(f'[UNFREEZE] conv1 at epoch {ep}, lr_mult={args.conv1_lr_mult}')

        model.train(); total=0; total_loss=0.0
        for x,y in dl_tr:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=bool(args.amp) and (device.type=='cuda')):
                logits = model(x)
                loss = CE(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += y.numel(); total_loss += loss.item()*y.size(0)

        with torch.no_grad():
            model.renorm_conv1()

        acc = evaluate(model, dl_te, device)
        print(f'[ep {ep}] train_loss={total_loss/max(1,total):.4f}  test_acc={acc*100:.2f}%')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'ep': ep}, str(out_dir/'last.pt'))
        if acc > best:
            best = acc
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'ep': ep}, str(out_dir/'best.pt'))

    print(f'[DONE] best_test_acc={best*100:.2f}%  -> {out_dir}')
    save_hardest_classification(model, dl_te, device, out_dir, K=3, postfix='_best')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[FATAL]', repr(e), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
