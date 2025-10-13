# -*- coding: utf-8 -*-
# src_n/tools/n_dump.py
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from src_m.tools.m_train import build_model, seed_all

# ----------------------------- data -----------------------------
@torch.no_grad()
def get_loaders(batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), split="test", subset=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.EMNIST(root="./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    if subset is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(subset, len(ds)))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def upsample_like(a_mhw, out_hw=(28,28)):
    a = torch.from_numpy(a_mhw)[None]  # (1,M,H,W)
    a = F.interpolate(a, size=out_hw, mode="bilinear", align_corners=False)
    return a[0].cpu().numpy()

# ----------------------------- grid helpers -----------------------------
def _grid_boxes(H, W, gh, gw):
    """Return list of (x0,y0,x1,y1) for a gh x gw grid covering HxW."""
    boxes = []
    for r in range(gh):
        for c in range(gw):
            x0 = int(round(c * (W/float(gw))))
            y0 = int(round(r * (H/float(gh))))
            x1 = int(round((c+1)*(W/float(gw)))) - 1
            y1 = int(round((r+1)*(H/float(gh)))) - 1
            boxes.append((x0,y0,x1,y1))
    return boxes  # len = gh*gw

def _pair_list(n):
    """All unordered pairs (i<j) from range(n)."""
    lst = []
    for i in range(n):
        for j in range(i+1, n):
            lst.append((i,j))
    return lst  # len = nC2

def _mask_from_pair(H, W, boxes, pair):
    """Binary mask HW for union of two grid boxes."""
    x0a,y0a,x1a,y1a = boxes[pair[0]]
    x0b,y0b,x1b,y1b = boxes[pair[1]]
    m = np.zeros((H,W), dtype=np.float32)
    m[y0a:y1a+1, x0a:x1a+1] = 1.0
    m[y0b:y1b+1, x0b:x1b+1] = 1.0
    return m

def _auto_pick_pair_per_slot(A_mhw, boxes):
    """For each slot m, pick two boxes with largest mass (per image)."""
    M, H, W = A_mhw.shape
    G = len(boxes)
    # mass per grid cell
    mass = np.zeros((M, G), dtype=np.float32)
    for g,(x0,y0,x1,y1) in enumerate(boxes):
        mass[:, g] = A_mhw[:, y0:y1+1, x0:x1+1].reshape(M,-1).sum(axis=1)
    top2_idx = np.argsort(-mass, axis=1)[:, :2]  # (M,2)
    pairs = [(int(top2_idx[m,0]), int(top2_idx[m,1])) for m in range(M)]
    return pairs  # list of (g1,g2)

# ----------------------------- math helpers -----------------------------
def infer_xy_from_A(A_mhw: np.ndarray):
    """A: (M,H,W) -> (M,2) in [0,1] using center of mass."""
    M, H, W = A_mhw.shape
    xs = (np.arange(W) + 0.5) / max(1, W)
    ys = (np.arange(H) + 0.5) / max(1, H)
    Xg, Yg = np.meshgrid(xs, ys)
    XY = np.zeros((M, 2), dtype=np.float32)
    for m in range(M):
        a = A_mhw[m]
        mass = a.sum()
        if mass > 1e-8:
            cx = float((a * Xg).sum() / mass); cy = float((a * Yg).sum() / mass)
        else:
            cx, cy = 0.5, 0.5
        XY[m, 0] = cx; XY[m, 1] = cy
    return XY

def compute_slot_prob(A_eff, slot_mask, tau=0.7):
    """
    A_eff    : (B,M,H,W)  (already masked)
    slot_mask: (M,)
    return   : (B,M) row-softmax, masked & renorm
    """
    B, M, H, W = A_eff.shape
    mask = slot_mask.view(1, M).to(A_eff.device)
    e = A_eff.flatten(2).sum(-1) * mask  # mass
    z = (e - e.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    p = torch.softmax(z, dim=1) * mask
    p = p / (p.sum(dim=1, keepdim=True).clamp_min(1e-8))
    return p

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--respect_slot_mask", action="store_true")
    ap.add_argument("--save_upsampled", action="store_true")
    ap.add_argument("--save_s_slots", action="store_true")
    ap.add_argument("--tau", type=float, default=0.7)

    # slot prob score (kept)
    ap.add_argument("--slot_prob_source", choices=["energy","topq","grad"], default="energy",
                    help="energy(헤드 raw), topq(맵 상위q 합), grad(양의 그라드)")
    ap.add_argument("--q", type=float, default=0.02, help="topq용 비율(0~1)")
    ap.add_argument("--grad_target", choices=["pred","gt"], default="pred",
                    help="grad 모드일 때 어떤 로짓을 미분할지: 예측 or 정답")

    # NEW: spatial pair mask
    ap.add_argument("--spmask_enable", type=int, default=1)
    ap.add_argument("--spmask_grid", type=int, default=3, help="split 14x14 into gxg (e.g., 3)")
    ap.add_argument("--spmask_assign", choices=["auto","round"], default="auto",
                    help="auto: 슬롯별로 해당 이미지에서 mass가 큰 2칸 선택 / round: 고정 순환 할당")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            cfg = json.load(f)
        else:
            import yaml; cfg = yaml.safe_load(f)

    mean = tuple(cfg.get("normalize",{}).get("mean", [0.1307]))
    std  = tuple(cfg.get("normalize",{}).get("std",  [0.3081]))
    loader = get_loaders(args.batch_size, args.num_workers, mean, std, split=args.split, subset=args.subset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk, head = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    trunk.load_state_dict(ckpt["trunk"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    trunk.to(device).eval(); head.to(device).eval()

    M = head.M
    slot_mask = getattr(head, "slot_mask", torch.ones(M, device=device)).detach().float().to(device)
    slot_mask_np = slot_mask.cpu().numpy()
    alive = int(slot_mask.sum().item())
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(dict(M=int(M), alive=int(alive), split=args.split,
                       spmask_enable=int(args.spmask_enable),
                       spmask_grid=int(args.spmask_grid),
                       spmask_assign=str(args.spmask_assign)),
                  f, indent=2)

    # precompute grid boxes/pairs
    gh = gw = int(args.spmask_grid)
    boxes_28 = _grid_boxes(28, 28, gh, gw)
    all_pairs = _pair_list(len(boxes_28))

    idx_global = 0
    for x, y in tqdm(loader, desc="dump"):
        x = x.to(device); y = y.to(device)
        Z, aux = trunk(x)
        logits = head(Z)
        pred = logits.argmax(dim=1)

        # A_raw: (B,M,H,W)
        A_raw = aux["A_maps"]
        B, M_, H, W = A_raw.shape
        assert M_ == M

        # optional head energy (only for reference)
        E_raw = aux.get("head_energy", None)
        if isinstance(E_raw, torch.Tensor):
            E_raw = E_raw.detach()
            if E_raw.ndim == 2 and E_raw.shape[1] == M:
                pass
            elif E_raw.ndim == 2 and (M % E_raw.shape[1] == 0):
                E_raw = E_raw.repeat_interleave(M // E_raw.shape[1], dim=1)
            else:
                E_raw = None
        else:
            E_raw = None

        # === spatial pair mask ===
        region_bbox_all = []  # per-item later
        A_eff_list = []
        for b in range(B):
            A_b = A_raw[b].detach().cpu().numpy().astype(np.float32)  # (M,H,W)
            if args.spmask_enable:
                # decide pairs
                if args.spmask_assign == "auto":
                    pairs = _auto_pick_pair_per_slot(A_b, boxes_28)
                else:  # round
                    pairs = [all_pairs[m % len(all_pairs)] for m in range(M)]
                # build masks & apply
                A_eff_m = np.zeros_like(A_b)
                rb_m = np.zeros((M, 2, 4), dtype=np.int32)
                for m in range(M):
                    p = pairs[m]
                    rb_m[m,0] = np.array(boxes_28[p[0]], dtype=np.int32)
                    rb_m[m,1] = np.array(boxes_28[p[1]], dtype=np.int32)
                    msk = _mask_from_pair(H, W, boxes_28, p)
                    A_eff_m[m] = A_b[m] * msk
                region_bbox_all.append(rb_m)
                A_eff_list.append(torch.from_numpy(A_eff_m))
            else:
                # no gating
                rb_m = np.zeros((M, 2, 4), dtype=np.int32)
                # fallback: whole image boxes repeated
                rb_m[:,0] = np.array([0,0,W-1,H-1], dtype=np.int32)
                rb_m[:,1] = np.array([0,0,W-1,H-1], dtype=np.int32)
                region_bbox_all.append(rb_m)
                A_eff_list.append(torch.from_numpy(A_b))
        A_eff = torch.stack(A_eff_list, dim=0).to(A_raw.device)  # (B,M,H,W)

        # optional slot_mask from head
        if args.respect_slot_mask:
            A_eff = A_eff * slot_mask.view(1,M,1,1)

        # slot prob from masked A
        slot_prob = compute_slot_prob(A_eff, slot_mask, tau=args.tau)  # (B,M)

        # center of mass & masses
        A_np = A_eff.detach().cpu().numpy().astype(np.float32)  # (B,M,H,W)
        mass = A_eff.flatten(2).sum(-1).detach().cpu().numpy().astype(np.float32)  # (B,M)

        # S_slots(옵션)
        S_slots = None
        for key in ["S_slots","slots_S","slot_embed","slot_repr"]:
            if key in aux and isinstance(aux[key], torch.Tensor):
                t = aux[key].detach()
                if t.ndim==3 and t.shape[1]==M: S_slots = t; break

        # numpy conversions
        E_raw_np = None if E_raw is None else E_raw.detach().cpu().numpy().astype(np.float32)
        P_np     = slot_prob.detach().cpu().numpy().astype(np.float32)
        L_np     = logits.detach().cpu().numpy().astype(np.float32)
        Y_np     = y.detach().cpu().numpy().astype(np.int64)
        PRED_np  = pred.detach().cpu().numpy().astype(np.int64)
        imgs = (x.detach().cpu().numpy() * np.array(std)[None,:,None,None] + np.array(mean)[None,:,None,None])
        imgs = np.clip(imgs,0,1); imgs = (imgs[:,0]*255).astype(np.uint8)

        for i in range(B):
            XY = infer_xy_from_A(A_np[i])
            rec = {
                "id": int(idx_global),
                "clazz": int(Y_np[i]),
                "pred": int(PRED_np[i]),
                "logits": L_np[i],
                "A_maps": A_np[i],
                "XY": XY.astype(np.float32),
                "slot_prob": P_np[i],
                "slot_mass": mass[i],
                "slot_mask": slot_mask_np,
                "image": imgs[i],
                "region_bbox": region_bbox_all[i].astype(np.int32),  # (M,2,4)
            }
            if E_raw_np is not None:
                rec["energy_raw"] = E_raw_np[i]
            if S_slots is not None:
                rec["S_slots"] = S_slots[i].cpu().numpy().astype(np.float32)
            if args.save_upsampled:
                rec["A_upsampled"] = upsample_like(A_np[i])
            np.savez_compressed(os.path.join(args.out_dir, f"{idx_global:07d}.npz"), **rec)
            idx_global += 1

    print(f"[n_dump] saved {idx_global} items to {args.out_dir}. alive_slots={alive}/{M}")

if __name__ == "__main__":
    seed_all(42)
    main()
