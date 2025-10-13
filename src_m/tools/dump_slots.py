# src_m/tools/dump_slots.py
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from src_m.tools.m_train import build_model, seed_all

@torch.no_grad()
def get_loaders(batch_size=256, num_workers=2, mean=(0.1307,), std=(0.3081,), split="test", subset=None):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.EMNIST(root="./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    if subset is not None:
        N = min(subset, len(ds))
        ds = torch.utils.data.Subset(ds, list(range(N)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def upsample_like(a_mhw, out_hw=(28,28)):
    a = torch.from_numpy(a_mhw)[None]  # (1,M,H,W)
    a = F.interpolate(a, size=out_hw, mode="bilinear", align_corners=False)
    return a[0].cpu().numpy()

def infer_xy_from_A(A_mhw: np.ndarray):
    if isinstance(A_mhw, torch.Tensor):
        A_mhw = A_mhw.detach().float().cpu().numpy()
    M, H, W = A_mhw.shape
    xs = (np.arange(W) + 0.5) / W
    ys = (np.arange(H) + 0.5) / H
    Xg, Yg = np.meshgrid(xs, ys)  # (H,W)

    XY = np.zeros((M, 2), dtype=np.float32)
    for m in range(M):
        a = A_mhw[m]
        mass = a.sum()
        if mass > 1e-8:
            cx = float((a * Xg).sum() / mass)
            cy = float((a * Yg).sum() / mass)
        else:
            cx, cy = 0.5, 0.5
        XY[m, 0] = cx
        XY[m, 1] = cy
    return XY

def make_slot_prob(E_raw: torch.Tensor, A: torch.Tensor, slot_mask: torch.Tensor, tau: float = 0.7):
    """
    E_raw: (B,M) if available else None
    A    : (B,M,H,W)  fallback source
    slot_mask: (M,)
    return: (B,M) probability per slot (masked, rows sum to 1)
    """
    B, M = A.size(0), A.size(1)
    mask = slot_mask.view(1, M).to(A.device)

    if (E_raw is not None) and (E_raw.ndim == 2) and (E_raw.shape[1] == M):
        e = E_raw.detach().float()
    else:
        # fallback = mass
        e = A.flatten(2).sum(-1)  # (B,M)

    # mask negatives and dead slots
    e = e * mask
    # temperature-scaled softmax
    z = (e - e.mean(dim=1, keepdim=True)) / max(1e-6, float(tau))
    # if all zeros, fall back to uniform over alive slots
    if torch.all((z * mask) == 0):
        alive = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (mask / alive).float()
    p = torch.softmax(z, dim=1) * mask
    # renormalize on alive slots
    s = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p / s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--split", default="test", choices=["train","test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--respect_slot_mask", action="store_true")
    ap.add_argument("--save_upsampled", action="store_true")
    ap.add_argument("--save_s_slots", action="store_true")
    ap.add_argument("--tau", type=float, default=0.7, help="temperature for slot_prob")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            cfg = json.load(f)
        else:
            import yaml
            cfg = yaml.safe_load(f)

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

    meta = dict(M=int(M), alive=int(alive), split=args.split, respect_slot_mask=bool(args.respect_slot_mask))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)

    idx_global = 0
    for x, y in tqdm(loader, desc="dumping"):
        x = x.to(device); y = y.to(device)
        Z, aux = trunk(x)
        logits = head(Z)
        pred = logits.argmax(dim=1)

        A = aux["A_maps"].detach()                 # (B,M,H,W)
        B, M_, H, W = A.shape
        assert M_ == M

        E_raw = aux.get("head_energy", None)
        if isinstance(E_raw, torch.Tensor):
            E_raw = E_raw.detach()
            # try to coerce to (B,M)
            if E_raw.ndim == 2 and E_raw.shape[1] == M:
                pass
            elif E_raw.ndim == 2 and (M % E_raw.shape[1] == 0):
                g = M // E_raw.shape[1]
                E_raw = E_raw.repeat_interleave(g, dim=1)
            else:
                E_raw = None
        else:
            E_raw = None

        if args.respect_slot_mask:
            A = A * slot_mask.view(1, M, 1, 1)

        # mass-based energy (always available)
        mass = A.flatten(2).sum(-1)                # (B,M)
        # normalized energy used by 예전 코드(합=1)
        e = (mass * slot_mask.view(1, M)).clamp_min(0)
        sum_e = e.sum(dim=1, keepdim=True)
        alive_cnt = slot_mask.sum().clamp(min=1.0)
        e_norm = torch.where(sum_e > 0, e / (sum_e + 1e-8), slot_mask.view(1, M) / alive_cnt)

        # slot probability (권장 사용)
        slot_prob = make_slot_prob(E_raw, A, slot_mask, tau=args.tau)  # (B,M)

        # 선택 저장용 S_slots
        S_slots = None
        if args.save_s_slots:
            for key in ["S_slots", "slots_S", "slot_embed", "slot_repr"]:
                if key in aux and isinstance(aux[key], torch.Tensor):
                    t = aux[key].detach()
                    if t.ndim == 3 and t.shape[1] == M:
                        S_slots = t
                        break

        # numpy 변환
        A_np      = A.cpu().numpy().astype(np.float32)
        mass_np   = mass.cpu().numpy().astype(np.float32)
        E_raw_np  = None if E_raw is None else E_raw.cpu().numpy().astype(np.float32)
        E_norm_np = e_norm.cpu().numpy().astype(np.float32)
        P_np      = slot_prob.cpu().numpy().astype(np.float32)
        L_np      = logits.detach().cpu().numpy().astype(np.float32)
        Y_np      = y.detach().cpu().numpy().astype(np.int64)
        PRED_np   = pred.detach().cpu().numpy().astype(np.int64)
        imgs = (x.detach().cpu().numpy() * np.array(std)[None,:,None,None] + np.array(mean)[None,:,None,None])
        imgs = np.clip(imgs, 0, 1); imgs = (imgs[:,0]*255).astype(np.uint8)

        for i in range(B):
            XY = infer_xy_from_A(A_np[i])
            rec = {
                "id": int(idx_global),
                "clazz": int(Y_np[i]),
                "pred": int(PRED_np[i]),
                "logits": L_np[i],
                "A_maps": A_np[i],                    # (M,H,W)
                "XY": XY.astype(np.float32),         # (M,2)
                "slot_prob": P_np[i],                 # (M,)  ⬅️ 신규(권장)
                "energy_norm": E_norm_np[i],          # (M,)  합=1
                "slot_mass": mass_np[i],              # (M,)  sum of A
                "slot_mask": slot_mask_np,            # (M,)
                "image": imgs[i],
            }
            if E_raw_np is not None:
                rec["energy_raw"] = E_raw_np[i]       # 가능하면 원시값
            if S_slots is not None:
                rec["S_slots"] = S_slots[i].cpu().numpy().astype(np.float32)
            if args.save_upsampled:
                rec["A_upsampled"] = upsample_like(A_np[i])
            np.savez_compressed(os.path.join(args.out_dir, f"{idx_global:07d}.npz"), **rec)
            idx_global += 1

    print(f"[dump_slots] saved {idx_global} items to {args.out_dir}. alive_slots={alive}/{M}")

if __name__ == "__main__":
    seed_all(42)
    main()
