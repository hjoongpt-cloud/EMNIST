#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검증 체크리스트:
- 슬롯 정규성: A_raw/A_eff 슬롯별 합=1 편차
- P 분포: maxP 평균/중앙값, 히스토그램
- coverage@X: 슬롯 evidence에서 정답이 topX 안에 드는 비율
- 간단 정확도: wsum / maxslot / nearest-proto
- 라벨 공간: proto vs val labels
"""
import os, json, argparse, numpy as np
from tqdm import tqdm
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src_o.o_trunk import OTrunk
from src_o.slot_utils import load_slot_queries_from_ckpt, extract_slots_with_queries, \
                             load_prototypes_json, build_class_index, per_slot_evidence_compact

def get_loader_val(batch_size=256, num_workers=2, val_ratio=0.1, seed=123):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=tf)
    n_total = len(full); n_val = int(round(n_total * val_ratio)); n_tr = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    tr_ds, val_ds = random_split(full, [n_tr, n_val], generator=g)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--spmask_grid", type=int, default=3)
    ap.add_argument("--spmask_assign", type=str, default="round")
    ap.add_argument("--proto_tau", type=float, default=0.4)
    ap.add_argument("--topX", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trunk & slot_q
    import numpy as np
    filters = np.load(args.conv1_filters)
    meta = torch.load(args.ckpt, map_location="cpu").get("meta", {})
    d_model = int(meta.get("d_model", 64)); nhead = int(meta.get("nhead", 4)); num_layers = int(meta.get("num_layers", 2))
    trunk = OTrunk(d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=256, conv1_filters=filters).to(device).eval()
    slot_q = load_slot_queries_from_ckpt(args.ckpt, device)

    # prototypes
    C, C_cls, _ = load_prototypes_json(args.proto_json, device=device, filter_zero_proto=True)
    labels_sorted, label_to_col, col_to_label = build_class_index(C_cls)
    Cp = len(labels_sorted)

    # loader
    loader = get_loader_val(batch_size=args.batch_size, num_workers=args.num_workers,
                            val_ratio=args.val_ratio, seed=args.seed)

    # accumulators
    dev_Araw, dev_Aeff = [], []
    maxP_vals = []
    coverages = []
    correct_wsum = correct_maxslot = correct_np = 0
    total = 0
    seen_labels = set()

    # histogram bins
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist_bins = np.linspace(0, 1, 51)
    P_hist = np.zeros_like(hist_bins[:-1], dtype=np.int64)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="[verify]"):
            x = x.to(device); y = y.to(device)
            tokens, aux = trunk(x)
            A_eff, P, S, A_raw = extract_slots_with_queries(tokens, slot_q, True,
                                                            tau_p=args.tau, grid=args.spmask_grid, assign=args.spmask_assign)

            # slots sum-to-one deviation
            Araw_sum = A_raw.flatten(2).sum(-1)  # (B,M)
            Aeff_sum = A_eff.flatten(2).sum(-1)
            dev_Araw.append(torch.abs(Araw_sum - 1.0).mean().item())
            dev_Aeff.append(torch.abs(Aeff_sum - 1.0).mean().item())

            # P stats
            maxP = P.max(dim=1).values
            maxP_vals.extend(maxP.cpu().numpy().tolist())
            cts, _ = np.histogram(P.cpu().numpy().reshape(-1), bins=hist_bins)
            P_hist += cts

            # evidence
            evi = per_slot_evidence_compact(S, C, C_cls, labels_sorted, class_reduce="lse", proto_tau=args.proto_tau)  # (B,M,Cp)

            # coverage@X
            ranks = torch.argsort(evi, dim=2, descending=True)           # (B,M,Cp)
            cols = (ranks == y.view(-1,1,1)).nonzero(as_tuple=False)     # (K,3) [b,m,rank]
            y_rank = torch.full((x.size(0), evi.size(1)), fill_value=Cp, device=x.device, dtype=torch.long)
            if cols.numel() > 0:
                for b, m, r in cols:
                    y_rank[b, m] = r
            hit = (y_rank < min(args.topX, Cp)).float().mean(dim=1)       # per-sample coverage
            coverages.extend(hit.cpu().numpy().tolist())

            # simple accuracies
            # 1) wsum
            wsum_logits = torch.einsum("bmc,bm->bc", evi, P / P.sum(dim=1, keepdim=True).clamp_min(1e-8))
            pred_wsum = wsum_logits.argmax(dim=1)
            # 2) maxslot (슬롯 중 하나의 evidence 합이 최대인 클래스)
            maxslot_logits = evi.max(dim=1).values  # (B,Cp)
            pred_maxslot = maxslot_logits.argmax(dim=1)
            # 3) nearest-proto (각 슬롯의 최상 유사도를 클래스별로 모아 합산)
            best_per_class, _ = evi.max(dim=1)      # (B,Cp)
            pred_np = best_per_class.argmax(dim=1)

            # to original labels
            pred_wsum_lab  = torch.tensor([col_to_label[int(c)] for c in pred_wsum.cpu().tolist()], device=x.device)
            pred_maxslot_lab = torch.tensor([col_to_label[int(c)] for c in pred_maxslot.cpu().tolist()], device=x.device)
            pred_np_lab    = torch.tensor([col_to_label[int(c)] for c in pred_np.cpu().tolist()], device=x.device)

            correct_wsum    += int((pred_wsum_lab == y).sum().item())
            correct_maxslot += int((pred_maxslot_lab == y).sum().item())
            correct_np      += int((pred_np_lab == y).sum().item())
            total           += int(y.numel())

            for t in y.cpu().tolist(): seen_labels.add(int(t))

    # write reports
    with open(os.path.join(args.out_dir, "alignment_report.json"), "w") as f:
        json.dump({
            "A_raw_sum1_dev_mean": float(np.mean(dev_Araw)),
            "A_eff_sum1_dev_mean": float(np.mean(dev_Aeff)),
            "P_max_mean": float(np.mean(maxP_vals)),
            "P_max_median": float(np.median(maxP_vals)),
            "coverage@X_mean": float(np.mean(coverages)),
            "coverage@X_median": float(np.median(coverages)),
            "acc_wsum": 100.0 * correct_wsum / max(1, total),
            "acc_maxslot": 100.0 * correct_maxslot / max(1, total),
            "acc_nearest_proto": 100.0 * correct_np / max(1, total),
            "samples": int(total)
        }, f, indent=2)

    # P histogram
    centers = 0.5*(hist_bins[1:]+hist_bins[:-1])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.bar(centers, P_hist, width=centers[1]-centers[0])
    plt.xlabel("slot_prob"); plt.ylabel("count"); plt.title("P histogram (val)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "P_hist.png"))

    # label space report
    with open(os.path.join(args.out_dir, "label_space.json"), "w") as f:
        json.dump({
            "proto_classes": labels_sorted,
            "val_labels_seen": sorted(list(seen_labels)),
            "intersection_size": len(set(labels_sorted) & seen_labels),
            "missing_in_proto": sorted(list(seen_labels - set(labels_sorted))),
            "extra_in_proto": sorted(list(set(labels_sorted) - seen_labels))
        }, f, indent=2)

    print("[done] wrote reports to", args.out_dir)

if __name__ == "__main__":
    main()
