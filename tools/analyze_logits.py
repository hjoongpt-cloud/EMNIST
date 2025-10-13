#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse, os
from sklearn.calibration import calibration_curve

def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        conf = probs[mask].mean()
        acc  = (labels[mask]).mean()
        ece += np.abs(acc - conf) * mask.mean()
    return ece

def plot_reliability(probs, labels, out_path, n_bins=15):
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy='uniform')
    plt.figure()
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.xlabel('Confidence'); plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_margin_hist(margin, correct, out_path):
    plt.figure()
    plt.hist(margin[correct], bins=50, alpha=0.6, label='correct')
    plt.hist(margin[~correct], bins=50, alpha=0.6, label='wrong')
    plt.xlabel('Margin (logit1 - logit2)'); plt.ylabel('#samples')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_true_rank_heatmap(true_rank, labels, n_classes, out_path):
    # row: class, col: rank
    mat = np.zeros((n_classes, true_rank.max()+1), dtype=int)
    for r, c in zip(true_rank, labels):
        mat[c, r] += 1
    mat = mat / (mat.sum(axis=1, keepdims=True)+1e-9)
    plt.figure(figsize=(6,5))
    plt.imshow(mat, aspect='auto')
    plt.colorbar(); plt.xlabel('Rank'); plt.ylabel('Class')
    plt.title('True-rank distribution')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_oracle_scatter(ce_best, ce_chosen, out_path):
    plt.figure()
    plt.scatter(ce_best, ce_chosen, s=4, alpha=0.4)
    lim = [0, max(ce_best.max(), ce_chosen.max())]
    plt.plot(lim, lim, 'r--')
    plt.xlabel('Oracle CE'); plt.ylabel('Chosen CE')
    plt.title('Oracle vs Chosen CE')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="logits_test.npz path")
    ap.add_argument("--out_dir", default="logit_plots")
    ap.add_argument("--num_classes", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz)

    logits = data["logits"]           # [N,C]
    labels = data["labels"]           # [N]
    preds  = data["preds"]            # [N]
    probs  = (np.exp(logits) / np.exp(logits).sum(1, keepdims=True))  # softmax
    conf   = probs.max(1)

    # --- margin ---
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    margin = sorted_logits[:,0] - sorted_logits[:,1]

    # --- true-rank ---
    order = (-logits).argsort(1)
    true_rank = np.array([np.where(order[i]==labels[i])[0][0] for i in range(len(labels))])

    # --- logit gap (pred-true) ---
    gap = logits[np.arange(len(labels)), preds] - logits[np.arange(len(labels)), labels]

    correct = (preds == labels)

    # 1) 숫자 5개 출력
    ece = expected_calibration_error(conf, correct.astype(float))
    mean_margin_c = margin[correct].mean()
    mean_margin_w = margin[~correct].mean() if (~correct).any() else 0.0
    mean_true_rank = true_rank.mean()
    mean_gap_wrong = gap[~correct].mean() if (~correct).any() else 0.0

    print("=== Metrics ===")
    print(f"ECE: {ece:.4f}")
    print(f"Margin mean (correct): {mean_margin_c:.4f}")
    print(f"Margin mean (wrong):   {mean_margin_w:.4f}")
    print(f"True-rank mean:        {mean_true_rank:.2f}")
    print(f"Logit gap (wrong) mean:{mean_gap_wrong:.4f}")

    # oracle info (optional)
    if "ce_best" in data and "ce_chosen" in data:
        ce_best   = data["ce_best"]
        ce_chosen = data["ce_chosen"]
        misroute_rate = (data["best_e"] != data["chosen_e"]).mean()
        gate_regret   = (ce_chosen - ce_best).mean()
        print(f"Misroute rate:  {misroute_rate:.3f}")
        print(f"Gate regret CE: {gate_regret:.4f}")
    if "preds_top2" in data.files:
        preds_top2 = data["preds_top2"]
        acc_top2 = (preds_top2 == labels).mean()
        print(f"Top-2 mixture acc: {acc_top2:.4f}")
    # 2) 그림 4종
    plot_reliability(conf, correct.astype(int), os.path.join(args.out_dir, "reliability.png"))
    plot_margin_hist(margin, correct, os.path.join(args.out_dir, "margin_hist.png"))
    plot_true_rank_heatmap(true_rank, labels, args.num_classes,
                           os.path.join(args.out_dir, "true_rank_heatmap.png"))

    if "ce_best" in data:
        plot_oracle_scatter(data["ce_best"], data["ce_chosen"],
                            os.path.join(args.out_dir, "oracle_scatter.png"))

    print("Plots saved to", args.out_dir)

if __name__ == "__main__":
    main()
