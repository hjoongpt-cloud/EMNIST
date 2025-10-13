# qnext/train/train_cls.py
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn

from qnext.core.utils import set_seed, renorm_conv1, percentile
from qnext.core.data import get_dataloaders
from qnext.models.trunk import Trunk
from qnext.core.viz import save_margin_hist, save_slot_cossim_heatmap
from qnext.core.metrics import batch_slot_cossim
def piecewise_slotdrop(t, t_warm, t_peak, p_start, p_mid, p_end):
    # t ∈ [0,1]
    t = max(0.0, min(1.0, t))
    if t <= t_warm:
        # Warm-up: 낮게 유지(필요하면 p_start→조금 증가로 바꿔도 됨)
        return p_start
    elif t <= t_peak:
        # Diversity push: p_start → p_mid 선형 증가
        u = (t - t_warm) / max(1e-8, (t_peak - t_warm))
        return p_start * (1 - u) + p_mid * u
    else:
        # Stabilize: p_mid → p_end 선형 감소
        u = (t - t_peak) / max(1e-8, (1 - t_peak))
        return p_mid * (1 - u) + p_end * u

def evaluate(model, dl, device):
    model.eval()
    total, correct = 0, 0
    margins = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

            logit_y = logits.gather(1, y.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y.view(-1, 1), False)
            rival = logits.masked_fill(~mask, float('-inf')).amax(dim=1)
            margins.append((logit_y - rival).detach().cpu())
    margins = torch.cat(margins, dim=0) if margins else torch.tensor([])
    return correct / max(1, total), margins

def lin_decay(t, t0=0.0, t1=1.0, a=1.0, b=0.0):
    """선형 스케줄: t in [t0, t1] -> a*(1 - u) + b*u"""
    t = max(t0, min(t1, t))
    u = (t - t0) / max(1e-8, (t1 - t0))
    return a * (1.0 - u) + b * u

def main():
    ap = argparse.ArgumentParser()
    # 기본
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default="./data")

    # 인코더/WTA/어텐션
    ap.add_argument("--enc_act", type=str, default="gelu", choices=["relu", "gelu"])
    ap.add_argument("--wta_mode", type=str, default="soft_topk",
                    choices=["none", "soft_top1", "soft_topk", "hard_top1"])
    ap.add_argument("--wta_tau", type=float, default=0.7)    # 유지 (초기값, start/end가 있으면 무시됨)
    ap.add_argument("--wta_k", type=int, default=3)
    ap.add_argument("--attn_mode", type=str, default="slot", choices=["slot", "local"])
    ap.add_argument("--aggregator", type=str, default="logit_mean",
                    choices=["prob_mean", "logit_mean", "prob_max", "margin_switch"])
    ap.add_argument("--use_2dpe", type=int, default=1)
    ap.add_argument("--pe_pairs", type=int, default=16)
    ap.add_argument("--pe_alpha", type=float, default=0.5)

    # 출력/초기화
    ap.add_argument("--out_dir", type=str, default="outputs/main")
    ap.add_argument("--init_from_ae", type=str, default="")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=10)
    ap.add_argument("--only_load_conv1", action="store_true")
    ap.add_argument("--only_load_proj", action="store_true")

    # 슬롯 다양성(본 학습)
    ap.add_argument("--slotdrop_start", type=float, default=0.10)
    ap.add_argument("--slotdrop_mid", type=float, default=0.35)
    ap.add_argument("--slotdrop_end", type=float, default=0.22)
    ap.add_argument("--slotdrop_t_warm", type=float, default=0.30)
    ap.add_argument("--slotdrop_t_peak", type=float, default=0.80)
    # ap.add_argument("--slotdrop_p", type=float, default=0.35)           # 초기값(강)
    # ap.add_argument("--slotdrop_final", type=float, default=0.15)       # 최종값(약)
    # ap.add_argument("--slotdrop_warm_frac", type=float, default=0.3)    # 앞부분까지 선형 감소
    ap.add_argument("--jitter_px", type=int, default=2)
    ap.add_argument("--jitter_alpha", type=float, default=0.2)
    ap.add_argument("--diversity_enable", type=int, default=1)
    ap.add_argument("--diversity_lambda", type=float, default=0.01)

    # 드롭아웃(특징/헤드)
    ap.add_argument("--feature_dropout_p", type=float, default=0.25)
    ap.add_argument("--head_dropout_p", type=float, default=0.40)

    # WTA tau 스케줄
    ap.add_argument("--wta_tau_start", type=float, default=0.90)
    ap.add_argument("--wta_tau_end", type=float, default=0.70)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_train_early, dl_val_mine, dl_test = get_dataloaders(
        data_root=args.data_root, split_mode="remedial",
        val_ratio=0.1, early_ratio=0.1,
        batch_size=args.batch_size, num_workers=4, seed=args.seed
    )

    model = Trunk(
        enc_act=args.enc_act,
        wta_mode=args.wta_mode, wta_tau=args.wta_tau_start, wta_k=args.wta_k,
        attn_mode=args.attn_mode, aggregator=args.aggregator,
        init_from_ae=(args.init_from_ae if args.init_from_ae else None),
        only_load_conv1=args.only_load_conv1, only_load_proj=args.only_load_proj,
        # slotdrop_p는 생성자에서 넘기지 말고, 아래에서 런타임으로 주입합니다.
        jitter_px=args.jitter_px, jitter_alpha=args.jitter_alpha,
        diversity_enable=bool(args.diversity_enable), diversity_lambda=args.diversity_lambda,
        use_2dpe=bool(args.use_2dpe), pe_pairs=args.pe_pairs, pe_alpha=args.pe_alpha
    ).to(device)

    # 초기 slot drop 값을 런타임으로 설정
    if hasattr(model, "slotdrop_p"):
        model.slotdrop_p = float(args.slotdrop_start)


    # (선택적) 모델 내부 드롭아웃 훅 세팅
    if hasattr(model, "feature_dropout_p"):
        model.feature_dropout_p = args.feature_dropout_p
    else:
        print("[WARN] model has no attribute 'feature_dropout_p' (skipped).")

    if hasattr(model, "head_dropout_p"):
        model.head_dropout_p = args.head_dropout_p
    else:
        print("[WARN] model has no attribute 'head_dropout_p' (skipped).")

    # 인코더 동결
    if args.freeze_encoder_epochs > 0:
        for p in model.enc.parameters():
            p.requires_grad = False
        print(f"[Train] Encoder frozen for first {args.freeze_encoder_epochs} epochs.")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    CE = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,early_acc,val_mine_acc,test_acc,margin_p10,margin_p50,margin_p90,"
                "slot_offdiag_mean,slot_offdiag_p90,slot_offdiag_p95,slot_offdiag_p99\n")

    best = 0.0
    for ep in range(1, args.epochs + 1):
        # 스케줄: slotdrop, wta_tau
        t = ep / max(1, args.epochs)

        # slot-drop: 낮게 시작 → 피크까지 선형 증가 → 끝에 살짝 완화
        slotdrop_now = piecewise_slotdrop(
            t,
            args.slotdrop_t_warm, args.slotdrop_t_peak,
            args.slotdrop_start, args.slotdrop_mid, args.slotdrop_end
        )

        # WTA tau는 기존처럼 0.9→0.7 선형 감쇠 유지
        wta_tau_now = lin_decay(
            t, 0.0, 1.0,
            a=args.wta_tau_start, b=args.wta_tau_end
        )


        # 모델에 전달(존재하면)
        if hasattr(model, "slotdrop_p"):
            model.slotdrop_p = float(slotdrop_now)
        if hasattr(model, "enc") and hasattr(model.enc, "wta_tau"):
            model.enc.wta_tau = float(wta_tau_now)

        # 동결 해제 시점
        if args.freeze_encoder_epochs > 0 and ep == args.freeze_encoder_epochs + 1:
            for p in model.enc.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print(f"[Train] Encoder unfrozen at epoch {ep}.")

        # --- Train ---
        model.train()
        running = 0.0
        total = 0
        for x, y in dl_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = CE(logits, y)

                # 다양성 로스(옵션, 학습 전반 50%만)
                if args.diversity_enable and (t <= 0.5):
                    # forward_with_feats가 있으면 슬롯 특성으로 계산
                    if hasattr(model, "forward_with_feats"):
                        _, feats_bsd = model.forward_with_feats(x)
                        # cos-sim 기반 약한 다양성: 오프대각 평균 제곱(작게)
                        if feats_bsd is not None and feats_bsd.ndim == 3:
                            M_avg, _ = batch_slot_cossim(feats_bsd)
                            if M_avg is not None:
                                S = M_avg.size(0)
                                off = M_avg[~torch.eye(S, dtype=torch.bool)]
                                div_loss = (off ** 2).mean()
                                loss = loss + float(args.diversity_lambda) * div_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * y.size(0)
            total += y.size(0)

        with torch.no_grad():
            renorm_conv1(model)

        # --- Eval ---
        acc_early, _ = evaluate(model, dl_train_early, device)
        acc_val, margins = evaluate(model, dl_val_mine, device)
        m10 = percentile(margins, 10)
        m50 = percentile(margins, 50)
        m90 = percentile(margins, 90)

        # 슬롯 cos-sim (작은 배치로 스냅샷)
        model.eval()
        with torch.no_grad():
            try:
                x_early, _ = next(iter(dl_train_early))
            except StopIteration:
                x_early, _ = next(iter(dl_val_mine))
            x_early = x_early.to(device, non_blocking=True)
            if hasattr(model, "forward_with_feats"):
                _, feats_bsd = model.forward_with_feats(x_early)
            else:
                feats_bsd = None
            M_avg, sim_stats = batch_slot_cossim(feats_bsd)
            if M_avg is not None:
                save_slot_cossim_heatmap(
                    M_avg, str(out_dir / f"slot_cossim_e{ep:02d}.png"),
                    vmin=0.0, vmax=1.0, title="slot cos sim (avg)"
                )
            off_m  = sim_stats.get("offdiag_mean", 0.0)
            off_p90= sim_stats.get("offdiag_p90", 0.0)
            off_p95= sim_stats.get("offdiag_p95", 0.0)
            off_p99= sim_stats.get("offdiag_p99", 0.0)

        print(f"[E{ep:02d}] train={running/max(1,total):.4f} "
              f"early={acc_early*100:.2f} val_mine={acc_val*100:.2f} "
              f"m@50={m50:.3f} slot_offdiag_mean={off_m:.4f} "
              f"(slotdrop={slotdrop_now:.3f}, wta_tau={wta_tau_now:.3f})")

        with open(log_csv, "a") as f:
            f.write(f"{ep},{running/max(1,total):.6f},{acc_early:.6f},{acc_val:.6f},,"
                    f"{m10:.5f},{m50:.5f},{m90:.5f},"
                    f"{off_m:.6f},{off_p90:.6f},{off_p95:.6f},{off_p99:.6f}\n")

        # save last/best
        torch.save({"model": model.state_dict(), "ep": ep, "config": vars(args)},
                   str(out_dir / "last.pt"))
        if acc_early > best:
            best = acc_early
            torch.save({"model": model.state_dict(), "ep": ep, "config": vars(args)},
                       str(out_dir / "best.pt"))

    ckpt = torch.load(str(out_dir / "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc, test_margins = evaluate(model, dl_test, device)
    print(f"[TEST] acc={test_acc*100:.2f}%")
    save_margin_hist(test_margins, str(out_dir / "test_margin_hist.png"))
    with open(log_csv, "a") as f:
        f.write(f"{args.epochs},,,{test_acc:.6f},,,,\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "test_acc": float(test_acc),
            "m50": float(percentile(test_margins, 50)),
            "config": vars(args)
        }, f, indent=2)

if __name__ == "__main__":
    main()
