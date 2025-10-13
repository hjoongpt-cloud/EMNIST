# -*- coding: utf-8 -*-
# src_q/q_exp_mask_vs_acc.py
import os, json, math, argparse, re, warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True"
)

try:
    from .q_trunk import SlotClassifier
    from .q_spmask import build_comb4_base_masks, replicate_masks
except Exception:
    from q_trunk import SlotClassifier
    from q_spmask import build_comb4_base_masks, replicate_masks

def _parse_alt_name(name: str):
    if not name:
        return ("none", 0.0, 0)
    s = name.strip().lower()
    if s in ("none","base","psum"):
        return ("none", 0.0, 0)
    m = re.match(r"^lse_topk(\d+)_t(\d+(?:\.\d+)?)$", s)
    if m: return ("lse_topk", float(m.group(2)), int(m.group(1)))
    m = re.match(r"^lse_t(\d+(?:\.\d+)?)$", s)
    if m: return ("lse", float(m.group(1)), 0)
    m = re.match(r"^psharp(?:_tau)?(\d+(?:\.\d+)?)$", s)
    if m: return ("psharp", float(m.group(1)), 0)
    m = re.match(r"^topk(\d+)$", s)
    if m: return ("topk", 0.0, int(m.group(1)))
    if s == "drop1": return ("drop1", 0.0, 0)
    m = re.match(r"^consensus(\d+(?:\.\d+)?)$", s)
    if m: return ("consensus", float(m.group(1)), 0)
    raise ValueError(f"Unrecognized --switch_alt '{name}'")

def get_loader(batch_size=512, num_workers=8, split="test"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.EMNIST("./data", split="balanced", train=(split=="train"), download=True, transform=tf)
    dl_kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, pin_memory=True, persistent_workers=False)
    if num_workers>0: dl_kwargs["prefetch_factor"]=4
    loader = DataLoader(**dl_kwargs)
    n_classes = int(ds.targets.max().item()) + 1
    return loader, n_classes

@torch.no_grad()
def evaluate_mass_and_aggregators_ckpt(
    ckpt_path, out_json, batch_size=512, num_workers=8,
    tau=0.7, topk_list=(2,4,8), device_str=None,
    switch_margin=None, switch_alt=None
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, num_classes = get_loader(batch_size, num_workers)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    num_slots_base   = int(meta.get("num_slots_base", 126))
    repeats_per_pos  = int(meta.get("repeats_per_pos", 3))
    d_model          = int(meta.get("d_model", 64))
    nhead            = int(meta.get("nhead", 4))
    filters_path     = meta.get("filters_path", None)
    if filters_path is None:
        raise RuntimeError("filters_path(meta)가 없습니다.")

    model = SlotClassifier(
        filters_path=filters_path,
        num_slots_base=num_slots_base,
        repeats_per_pos=repeats_per_pos,
        d_model=d_model, nhead=nhead, num_classes=num_classes,
        exp_wq=bool(meta.get("exp_wq", 1)),
        exp_gate=bool(meta.get("exp_gate", 1)),
        exp_head_bias=bool(meta.get("exp_head_bias", 1)),
        tau_intra=float(meta.get("tau_intra", 0.7)),
        mask_drop1=bool(meta.get("mask_drop1", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    base = build_comb4_base_masks(14,14,device=device)[0]    # (126,14,14)
    masks_mhw = replicate_masks(base, repeats_per_pos)[:num_slots_base*repeats_per_pos]
    M, H_px, W_px = masks_mhw.shape
    N_px = float(H_px * W_px)
    area_in = masks_mhw.float().flatten(1).sum(-1)  # (M,)

    total = 0
    correct = {"Psum":0, "Top1":0, **{f"Top{k}":0 for k in topk_list}, "LSE":0}

    # margin-switch stats
    do_switch = (switch_margin is not None) and (switch_alt is not None)
    if do_switch:
        skind, sparam, sk = _parse_alt_name(switch_alt)
        hi_n = lo_n = 0
        hi_hits = lo_hits_base = lo_hits_alt = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        B = x.size(0)

        # raw maps/tokens (no spmask)
        _, aux = model(x, use_spmask=False, spmask_grid=3, spmask_assign="auto", tau_p=tau)
        A_raw = aux["A_maps_raw"]                 # (B,M,H,W)
        feat  = aux["feat_hw"]                    # (B,H,W,D)
        tokens = feat.view(B, H_px*W_px, d_model) # (B,N,D)

        # masked + renorm pooling
        A_masked = A_raw * masks_mhw.unsqueeze(0)
        mass = A_masked.float().flatten(2).sum(-1).clamp_min(1e-8)    # (B,M)
        A_eff = (A_masked.float() / mass.view(B, M, 1, 1)).to(A_raw.dtype)

        # slot embeddings / logits
        A_flat = A_eff.view(B, M, H_px*W_px)
        S = torch.bmm(A_flat, tokens)
        S = F.normalize(S, dim=-1)
        slot_logits = model.classifier(S)                              # (B,M,C)

        # P from mass residual vs expectation
        exp_mass = (area_in.to(mass.device) / N_px).view(1, M)
        z_base = mass.float() - exp_mass
        z = (z_base - z_base.mean(dim=1, keepdim=True)) / float(tau)
        P = torch.softmax(z, dim=1)                                    # (B,M)

        # aggregators
        logits_psum = (slot_logits * P.unsqueeze(-1)).sum(dim=1)
        top1_idx = P.argmax(dim=1)
        logits_top1 = slot_logits[torch.arange(B, device=device), top1_idx]
        logits_topk = {}
        for k in topk_list:
            kk = min(int(k), M)
            idx = torch.topk(P, kk, dim=1).indices
            gather = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1,-1,slot_logits.size(-1)))
            logits_topk[f"Top{kk}"] = gather.mean(dim=1)
        logits_lse = torch.logsumexp(slot_logits, dim=1)

        # predictions
        pred = {
            "Psum": logits_psum.argmax(1),
            "Top1": logits_top1.argmax(1),
            **{k: v.argmax(1) for k, v in logits_topk.items()},
            "LSE": logits_lse.argmax(1),
        }

        if do_switch:
            # margin on final logits (psum)
            t2, _ = torch.topk(logits_psum, k=2, dim=1)
            margin = (t2[:,0] - t2[:,1])
            keep_hi = (margin >= float(switch_margin))

            hi_n += int(keep_hi.sum().item())
            lo_n += int((~keep_hi).sum().item())
            hi_hits += int((pred["Psum"][keep_hi] == y[keep_hi]).sum().item())

            # low-confidence → alternative
            if skind == "lse":
                alt_pred = logits_lse.argmax(1)
            elif skind == "topk":
                k = max(1, min(sk, M))
                idx = torch.topk(P, k, dim=1).indices
                gather = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1,-1,slot_logits.size(-1)))
                alt_pred = gather.mean(dim=1).argmax(1)
            elif skind == "drop1":
                top_idx = P.argmax(dim=1, keepdim=True)
                P_mask = torch.ones_like(P)
                P_mask.scatter_(1, top_idx, 0.0)
                P2 = P * P_mask
                P2 = P2 / P2.sum(dim=1, keepdim=True).clamp_min(1e-8)
                alt_pred = (slot_logits * P2.unsqueeze(-1)).sum(dim=1).argmax(1)
            elif skind == "psharp":
                ttemp = max(1e-3, float(sparam))
                P2 = torch.softmax((z_base/ttemp), dim=1)
                alt_pred = (slot_logits * P2.unsqueeze(-1)).sum(dim=1).argmax(1)
            elif skind == "consensus":
                w = float(sparam)
                mix = (1.0/(1.0+w)) * logits_psum + (w/(1.0+w)) * logits_lse
                alt_pred = mix.argmax(1)
            elif skind == "lse_topk":
                ttemp = float(sparam); k = max(1, min(sk, M))
                lse = torch.logsumexp(slot_logits / ttemp, dim=1)
                idx = torch.topk(P, k, dim=1).indices
                gather = slot_logits.gather(1, idx.unsqueeze(-1).expand(-1,-1,slot_logits.size(-1)))
                topk_logits = gather.mean(dim=1)
                mix = 0.5*lse + 0.5*topk_logits
                alt_pred = mix.argmax(1)
            else:
                alt_pred = pred["Psum"]

            lo_hits_base += int((pred["Psum"][~keep_hi] == y[~keep_hi]).sum().item())
            lo_hits_alt  += int((alt_pred[~keep_hi]    == y[~keep_hi]).sum().item())

            pred_overall = pred["Psum"].clone()
            pred_overall[~keep_hi] = alt_pred[~keep_hi]
            correct["Psum"] += int((pred_overall == y).sum().item())
        else:
            for k in correct.keys():
                correct[k] += int((pred[k] == y).sum().item())

        total += B

    accs = {k: correct[k] / max(1, total) for k in correct}

    if do_switch:
        overall_acc = accs["Psum"]
        hi_acc = (hi_hits / max(1,hi_n)) if hi_n>0 else 0.0
        lo_base = (lo_hits_base / max(1,lo_n)) if lo_n>0 else 0.0
        lo_alt  = (lo_hits_alt  / max(1,lo_n)) if lo_n>0 else 0.0
        print(f"[switch] thr={switch_margin}, alt={switch_alt} | hi={hi_n/total*100:.2f}% acc={hi_acc*100:.2f}% | lo acc base={lo_base*100:.2f}% → alt={lo_alt*100:.2f}% | overall {overall_acc*100:.2f}%")

    out = {
        "acc": accs,
        "meta": {
            "num_slots_base": num_slots_base,
            "repeats_per_pos": repeats_per_pos,
            "num_slots": int(num_slots_base*repeats_per_pos),
            "d_model": d_model, "nhead": nhead,
            "tau": tau, "topk": list(topk_list),
            "M": M, "H_px": H_px, "W_px": W_px
        }
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("[exp_mask_vs_acc] saved ->", out_json)
    print("[acc] " + " | ".join([f"{k}={v*100:.2f}%" for k,v in accs.items()]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--topk", type=str, default="2,4,8")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--switch_margin", type=float, default=None)
    ap.add_argument("--switch_alt", type=str, default=None)
    args = ap.parse_args()

    topk_list = tuple(int(x) for x in args.topk.split(",") if x.strip())
    evaluate_mass_and_aggregators_ckpt(
        ckpt_path=args.ckpt, out_json=args.out_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        tau=args.tau, topk_list=topk_list, device_str=args.device,
        switch_margin=args.switch_margin, switch_alt=args.switch_alt
    )

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
