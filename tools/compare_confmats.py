#!/usr/bin/env python3
"""
compare_confmats.py

- Router가 선택한 경로(chosen) vs Oracle-best 경로(oracle) 기준 Confusion Matrix 비교
- Regret(= CE_chosen - CE_best) 상위 top% 샘플들에 대해서도 CM/정확도 계산
- Expert별(oracle 기준) CM/정확도도 함께 저장
- 모든 결과를 PNG + 하나의 PDF(첫 페이지에 수치 요약) + JSON/CSV로 저장

필수: dump_logits.py 실행 시 --oracle로 만든 npz
      (필드: logits, labels, logits_all, best_e, chosen_e, ce_best, ce_chosen)

Usage
-----
python compare_confmats.py \
  --npz outputs/stage_E_deep/vanilla_analyze/seed_42/test_logits.npz \
  --num_classes 47 \
  --out_dir outputs/stage_E_deep/vanilla_analyze/seed_42/logit_plots \
  --top_percent 10
"""

import os
import argparse
import json
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REQ_KEYS = {
    "logits", "labels",
    "logits_all", "best_e", "chosen_e", "ce_best", "ce_chosen"
}

# ---------- helpers ----------
def ensure_fields(data, required):
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"npz missing fields: {missing}. Re-dump with --oracle.")

def make_cm(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

def plot_cm(cm, title):
    disp = ConfusionMatrixDisplay(cm, display_labels=range(cm.shape[0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, include_values=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def accuracy(y, p):
    return float((y == p).mean())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_percent", type=float, default=10.0,
                    help="Top %% of samples with largest regret (ce_chosen - ce_best)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.npz)
    ensure_fields(data, REQ_KEYS)

    labels     = data["labels"]              # [N]
    logits     = data["logits"]              # [N,C] chosen expert logits
    logits_all = data["logits_all"]          # [N,E,C]
    best_e     = data["best_e"]              # [N]
    chosen_e   = data["chosen_e"]            # [N]
    ce_best    = data["ce_best"]             # [N]
    ce_chosen  = data["ce_chosen"]           # [N]

    N, E, C = logits_all.shape
    num_classes = args.num_classes
    idx_all = np.arange(N)

    # predictions
    preds_chosen = logits.argmax(1)
    preds_oracle = logits_all[idx_all, best_e].argmax(1)

    # regret & top subset
    regret = ce_chosen - ce_best
    k = max(1, int(np.ceil(args.top_percent / 100.0 * N)))
    top_idx = np.argsort(regret)[-k:]  # largest regrets

    # ----- confusion matrices -----
    cm_chosen_global = make_cm(labels, preds_chosen, num_classes)
    cm_oracle_global = make_cm(labels, preds_oracle, num_classes)
    cm_chosen_top    = make_cm(labels[top_idx], preds_chosen[top_idx], num_classes)
    cm_oracle_top    = make_cm(labels[top_idx], preds_oracle[top_idx], num_classes)

    # per-expert (oracle split)
    per_exp_figs = []
    per_exp_stats = {}
    for e in range(E):
        sel = np.where(best_e == e)[0]
        if sel.size == 0:
            continue
        preds_e = logits_all[sel, e].argmax(1)
        cm_e = make_cm(labels[sel], preds_e, num_classes)
        fig_e = plot_cm(cm_e, f"Oracle CM (Expert {e})")
        per_exp_figs.append((e, fig_e))
        per_exp_stats[f"oracle_expert{e}_acc"] = accuracy(labels[sel], preds_e)
        per_exp_stats[f"oracle_expert{e}_N"]   = int(sel.size)

    # ----- stats -----
    misroute_rate = float((best_e != chosen_e).mean())
    gate_regret   = float(regret.mean())

    stats = {
        "N_total": int(N),
        "E": int(E),
        "top_percent": args.top_percent,
        "chosen_global_acc":  accuracy(labels, preds_chosen),
        "oracle_global_acc":  accuracy(labels, preds_oracle),
        f"chosen_top{int(args.top_percent)}_acc":  accuracy(labels[top_idx], preds_chosen[top_idx]),
        f"oracle_top{int(args.top_percent)}_acc":  accuracy(labels[top_idx], preds_oracle[top_idx]),
        "misroute_rate": misroute_rate,
        "gate_regret_ce": gate_regret,
        "regret_mean_top": float(regret[top_idx].mean()),
        "regret_max": float(regret.max()),
    }
    stats.update(per_exp_stats)

    # save stats json/csv
    with open(os.path.join(args.out_dir, "cm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out_dir, "cm_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)

    # ----- plot & save -----
    figs = []
    figs.append(plot_cm(cm_chosen_global, "Chosen (Router) - Global"))
    figs.append(plot_cm(cm_oracle_global, "Oracle - Global"))
    figs.append(plot_cm(cm_chosen_top,    f"Chosen - Top {int(args.top_percent)}% Regret"))
    figs.append(plot_cm(cm_oracle_top,    f"Oracle - Top {int(args.top_percent)}% Regret"))
    figs.extend([fig for _, fig in per_exp_figs])

    # PNG export
    names = [
        "cm_chosen_global.png",
        "cm_oracle_global.png",
        f"cm_chosen_top{int(args.top_percent)}p_regret.png",
        f"cm_oracle_top{int(args.top_percent)}p_regret.png",
    ]
    for fig, name in zip(figs[:4], names):
        fig.savefig(os.path.join(args.out_dir, name))
    for (e, fig) in per_exp_figs:
        fig.savefig(os.path.join(args.out_dir, f"cm_oracle_expert{e}.png"))

    # PDF export (first page = stats)
    pdf_path = os.path.join(args.out_dir, "confmats_all.pdf")
    with PdfPages(pdf_path) as pdf:
        # stats page
        fig_stats, ax = plt.subplots(figsize=(7, 5))
        ax.axis("off")
        txt_lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]
        ax.text(0.01, 0.99, "\n".join(txt_lines), va="top", family="monospace", fontsize=9)
        ax.set_title("ConfMat Stats", pad=10)
        pdf.savefig(fig_stats)
        plt.close(fig_stats)

        # each CM
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)
        for _, fig in per_exp_figs:
            # already closed above, so skip
            pass

    print("Saved:")
    print("  PDF :", pdf_path)
    print("  JSON:", os.path.join(args.out_dir, "cm_stats.json"))
    print("  CSV :", os.path.join(args.out_dir, "cm_stats.csv"))
    print("  PNGs:", *os.listdir(args.out_dir), sep="\n        ")


if __name__ == "__main__":
    main()
