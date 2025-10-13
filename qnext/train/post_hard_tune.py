# qnext/train/post_hard_tune.py
import argparse, json, torch
from pathlib import Path

from qnext.models.trunk import Trunk
from qnext.core.data import get_dataloaders
from qnext.core.utils import set_seed
from qnext.core.miner import mine_hard_anchors
from qnext.core.samplers import ReplayBatcher
from qnext.core.losses import margin_penalty, supcon_loss


@torch.no_grad()
def eval_acc(model, dl, device):
    model.eval()
    tot = 0
    cor = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        cor += (pred == y).sum().item()
        tot += y.numel()
    return cor / max(1, tot)


def l2sp_reg(model, init_sd, weight=1e-4):
    if weight <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    reg = 0.0
    for n, p in model.named_parameters():
        if n not in init_sd:
            continue
        reg = reg + torch.sum((p - init_sd[n].to(p.device)) ** 2)
    return weight * reg


def get_batch_from_dataset(ds, idxs, device):
    """
    ds_full이 인덱싱 가능한 dataset이라고 가정.
    idxs가 텐서/리스트 인덱스일 때 (x,y) 텐서 묶음으로 반환.
    """
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()
    xs, ys = [], []
    for i in idxs:
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs, dim=0).to(device, non_blocking=True)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def split_params(model):
    """
    모듈별 파라미터 그룹을 깔끔히 분리해서 반환.
    반환: dict(keys: enc/slots/head/gating/agg), values: list of params
    """
    groups = {
        "enc": [],
        "slots": [],
        "head": [],
        "gating": [],
        "agg": [],
        "others": [],
    }

    named_to_bucket = []

    if hasattr(model, "enc"):
        groups["enc"].extend(list(model.enc.parameters()))
    if hasattr(model, "slots") and model.slots is not None:
        groups["slots"].extend(list(model.slots.parameters()))
    if hasattr(model, "attn") and model.attn is not None:
        groups["slots"].extend(list(model.attn.parameters()))
    if hasattr(model, "head") and model.head is not None:
        groups["head"].extend(list(model.head.parameters()))
    if hasattr(model, "gating") and model.gating is not None:
        groups["gating"].extend(list(model.gating.parameters()))
    if hasattr(model, "aggregator_params") and model.aggregator_params is not None:
        groups["agg"].extend(list(model.aggregator_params.parameters()))

    # others = 나머지 (중복 제외)
    all_in = set([id(p) for L in groups.values() for p in L])
    for p in model.parameters():
        if id(p) not in all_in:
            groups["others"].append(p)

    return groups


def build_optimizer(phase, groups, base_lr=1e-3, enc_scale=0.1, weight_decay=0.0):
    """
    phase: "p1_head", "p2_slots", "p3_all"
    enc_scale: p3에서 encoder에만 적용되는 저LR 스케일
    """
    assert phase in ("p1_head", "p2_slots", "p3_all")
    pg = []

    if phase == "p1_head":
        # Head만 업데이트, 나머지는 LR=0
        if len(groups["head"]) > 0:
            pg.append({"params": groups["head"], "lr": base_lr})
        for k in ["slots", "enc", "gating", "agg", "others"]:
            if len(groups[k]) > 0:
                pg.append({"params": groups[k], "lr": 0.0})

    elif phase == "p2_slots":
        # Slots(+head) 업데이트, encoder/gating/agg는 LR=0
        if len(groups["head"]) > 0:
            pg.append({"params": groups["head"], "lr": base_lr})
        if len(groups["slots"]) > 0:
            pg.append({"params": groups["slots"], "lr": base_lr})
        for k in ["enc", "gating", "agg", "others"]:
            if len(groups[k]) > 0:
                pg.append({"params": groups[k], "lr": 0.0})

    elif phase == "p3_all":
        # All 업데이트 + encoder만 저LR
        if len(groups["head"]) > 0:
            pg.append({"params": groups["head"], "lr": base_lr})
        if len(groups["slots"]) > 0:
            pg.append({"params": groups["slots"], "lr": base_lr})
        if len(groups["gating"]) > 0:
            pg.append({"params": groups["gating"], "lr": base_lr})
        if len(groups["agg"]) > 0:
            pg.append({"params": groups["agg"], "lr": base_lr})
        if len(groups["enc"]) > 0:
            pg.append({"params": groups["enc"], "lr": base_lr * enc_scale})
        if len(groups["others"]) > 0:
            pg.append({"params": groups["others"], "lr": base_lr})

    opt = torch.optim.Adam(pg, weight_decay=weight_decay)
    return opt


def forward_with_feats_safe(model, x):
    """
    (logits, feats) 반환을 시도; 없으면 안전 fallback
    - forward_with_feats(x)가 있으면 사용
    - return_aux=True 지원 시 aux에서 글로벌/슬롯 임베딩 사용
    - 최후 수단: logits를 임베딩처럼 사용
    """
    if hasattr(model, "forward_with_feats"):
        out = model.forward_with_feats(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out

    if hasattr(model, "forward"):
        out = model(x, return_aux=True) if "return_aux" in model.forward.__code__.co_varnames else model(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            logits, aux = out
            # slot_emb가 있으면 평균해서 임베딩으로 사용
            if isinstance(aux, dict):
                if "slot_emb" in aux and torch.is_tensor(aux["slot_emb"]):
                    # slot_emb: [B,S,D] → [B,D]
                    feats = aux["slot_emb"].mean(dim=1)
                    return logits, feats
            # 그 외엔 logits를 임베딩처럼 사용
            return logits, logits

    # 최후 수단
    logits = model(x)
    return logits, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="outputs/post")
    ap.add_argument("--epochs", type=int, default=5, help="epochs per cycle (total P1+P2+P3 <= epochs)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--replay_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)

    # schedules
    ap.add_argument("--margin_thresh_sched", type=str, default="0.3,0.5")   # per-cycle m*
    ap.add_argument("--lambda_margin_sched", type=str, default="0.2,0.3")   # per-cycle λ
    ap.add_argument("--supcon_temp", type=float, default=0.15)
    ap.add_argument("--supcon_weight_max", type=float, default=0.5)
    ap.add_argument("--warmup_frac", type=float, default=0.3)

    # phases (per cycle)
    ap.add_argument("--phase_epochs", type=str, default="2,3,2")            # P1,P2,P3
    ap.add_argument("--encoder_lr_scale", type=float, default=0.1)
    ap.add_argument("--l2sp_weight", type=float, default=1e-4)
    ap.add_argument("--base_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # (optional) cfg override / recovery
    ap.add_argument("--attn_mode", type=str, default=None, choices=[None, "slot", "local"])
    ap.add_argument("--use_2dpe", type=int, default=None)
    ap.add_argument("--pe_pairs", type=int, default=None)
    ap.add_argument("--pe_alpha", type=float, default=None)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- data --------
    dl_train, dl_train_early, dl_val_mine, dl_test = get_dataloaders(
        data_root=args.data_root, split_mode="remedial",
        batch_size=args.batch_size, num_workers=4, seed=args.seed
    )
    dl_val = dl_val_mine
    ds_full = dl_train.dataset

    # -------- model & ckpt --------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}

    attn_mode = args.attn_mode or cfg.get("attn_mode", "slot")
    enc_act = cfg.get("enc_act", "gelu")
    wta_mode = cfg.get("wta_mode", "soft_topk")
    wta_k = cfg.get("wta_k", 3)
    wta_tau = cfg.get("wta_tau", 0.7)

    use_2dpe = args.use_2dpe if args.use_2dpe is not None else cfg.get("use_2dpe", False)
    pe_pairs = args.pe_pairs if args.pe_pairs is not None else cfg.get("pe_pairs", 16)
    pe_alpha = args.pe_alpha if args.pe_alpha is not None else cfg.get("pe_alpha", 0.5)

    model = Trunk(
        enc_act=enc_act, wta_mode=wta_mode, wta_tau=wta_tau, wta_k=wta_k,
        attn_mode=attn_mode,
        use_2dpe=bool(use_2dpe), pe_pairs=int(pe_pairs), pe_alpha=float(pe_alpha),
    )

    sd = ckpt["model"]

    # --- K_eff 동기화 (conv1 채널 수 차이 방지) ---
    if "enc.conv1.weight" in sd and hasattr(model, "enc") and hasattr(model.enc, "_resize_channels"):
        K_eff = sd["enc.conv1.weight"].shape[0]
        model.enc._resize_channels(K_new=K_eff)
        if getattr(model.enc, "channel_mask", None) is not None and model.enc.channel_mask.shape[1] != K_eff:
            model.enc.register_buffer("channel_mask", torch.ones(1, K_eff, 1, 1))

    model.load_state_dict(sd)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True  # 과거 freeze로 꺼져있던 grad 복구
    torch.autograd.set_grad_enabled(True)  # 전역 grad 보증
    # L2-SP 기준 스냅샷
    init_sd = {n: p.detach().clone().cpu() for n, p in model.named_parameters()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    best_path = out / "post_best.pt"

    # schedules
    m_sched = [float(x) for x in args.margin_thresh_sched.split(",")]
    l_sched = [float(x) for x in args.lambda_margin_sched.split(",")]
    p1, p2, p3 = [int(x) for x in args.phase_epochs.split(",")]
    assert p1 + p2 + p3 <= args.epochs, "phase epochs exceed total epochs per cycle"

    best_val = -1.0

    def run_epochs(E, phase_name, m_star, lam_margin, sup_w_max, warmup_frac, opt, sampler):
        nonlocal best_val
        printed_dbg = False

        for ep in range(1, E + 1):
            torch.autograd.set_grad_enabled(True)
            model.train()
            progress = ep / max(1, E)
            sup_w = sup_w_max * max(0.0, min(1.0, (progress - warmup_frac) / max(1e-6, 1.0 - warmup_frac)))

            tot_loss = 0.0
            seen = 0

            for batch in sampler:
                # 배치가 인덱스인지, (x,y) 텐서인지 가드
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
                        x = batch[0].to(device, non_blocking=True)
                        y = batch[1].to(device, non_blocking=True)
                    else:
                        x, y = get_batch_from_dataset(ds_full, batch, device)
                elif torch.is_tensor(batch):
                    x, y = get_batch_from_dataset(ds_full, batch, device)
                else:
                    x, y = get_batch_from_dataset(ds_full, batch, device)

                opt.zero_grad(set_to_none=True)
                logits, feats = forward_with_feats_safe(model, x)

                if not printed_dbg:
                    print(f"[DBG] logits.requires_grad={logits.requires_grad}, feats.requires_grad={feats.requires_grad}")
                    printed_dbg = True

                ce = torch.nn.functional.cross_entropy(logits, y)
                Lm, _ = margin_penalty(logits, y, m_star=m_star, weight=lam_margin)
                zero = torch.zeros((), device=x.device)
                Lc = supcon_loss(feats, y, temp=args.supcon_temp) if sup_w_max > 0 else zero
                reg = l2sp_reg(model, init_sd, weight=args.l2sp_weight) if phase_name == "phase3" else zero

                loss = ce + Lm + (sup_w * Lc) + reg
                loss.backward()
                opt.step()

                tot_loss += float(loss.item()) * x.size(0)
                seen += x.size(0)

            # epoch 평가
            va = eval_acc(model, dl_val, device)
            te = eval_acc(model, dl_test, device)
            entry = {
                "phase": phase_name,
                "epoch": ep,
                "train_loss": tot_loss / max(1, seen),
                "val_acc": va,
                "test_acc": te,
                "m_star": m_star,
                "lambda_margin": lam_margin,
                "sup_w_max": sup_w_max,
                "sup_w_epoch": sup_w,
            }
            print(entry, flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # val-best 저장
            if va > best_val:
                best_val = va
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "attn_mode": attn_mode,
                            "enc_act": enc_act,
                            "wta_mode": wta_mode,
                            "wta_k": wta_k,
                            "wta_tau": wta_tau,
                            "use_2dpe": bool(use_2dpe),
                            "pe_pairs": int(pe_pairs),
                            "pe_alpha": float(pe_alpha),
                        },
                    },
                    str(best_path),
                )

    # -------- cycles --------
    for cyc in range(1, args.cycles + 1):
        print(f"\n=== CYCLE {cyc} / {args.cycles} ===", flush=True)

        # ---- 마이닝 (train 예측 -> 앵커: −2 ≤ margin < 0, 극단오답 제외) ----
        model.eval()
        anchors, buckets, margins_all = mine_hard_anchors(
            model, dl_train, device,
            margin_lo=-2.0, margin_hi=0.0, extreme_cut=-6.0
        )

        n_all = len(ds_full)
        all_idx = torch.arange(n_all)
        mask = torch.ones(n_all, dtype=torch.bool)
        mask[anchors] = False
        replay = all_idx[mask].tolist()
        anchors = anchors.tolist() if torch.is_tensor(anchors) else list(anchors)

        sampler = ReplayBatcher(
            anchors=anchors, replay=replay,
            batch_size=args.batch_size, ratio=args.replay_ratio, drop_last=True
        )

        # ---- per-cycle 스케줄 픽 ----
        m_star = m_sched[min(cyc - 1, len(m_sched) - 1)]
        lam_m = l_sched[min(cyc - 1, len(l_sched) - 1)]
        base_lr = args.base_lr

        # ---- Phase 1: head only (LR=base_lr / others=0) ----
        groups = split_params(model)
        opt = build_optimizer(
            phase="p1_head",
            groups=groups,
            base_lr=base_lr,
            enc_scale=args.encoder_lr_scale,
            weight_decay=args.weight_decay
        )
        run_epochs(p1, "phase1", m_star, lam_m * 0.5, 0.0, args.warmup_frac, opt, sampler)

        # ---- Phase 2: slots(+head) (encoder LR=0) ----
        groups = split_params(model)
        opt = build_optimizer(
            phase="p2_slots",
            groups=groups,
            base_lr=base_lr,
            enc_scale=args.encoder_lr_scale,
            weight_decay=args.weight_decay
        )
        run_epochs(p2, "phase2", m_star, lam_m, args.supcon_weight_max, args.warmup_frac, opt, sampler)

        # ---- Phase 3: all + encoder 저LR + L2-SP ----
        groups = split_params(model)
        opt = build_optimizer(
            phase="p3_all",
            groups=groups,
            base_lr=base_lr,
            enc_scale=args.encoder_lr_scale,
            weight_decay=args.weight_decay
        )
        run_epochs(p3, "phase3", m_star, lam_m, args.supcon_weight_max, args.warmup_frac, opt, sampler)

    # 최종 저장: val-best가 이미 best_path에 저장됨
    print(f"[POST] best checkpoint saved to {best_path}", flush=True)

    # 로그 요약 저장(옵션)
    summary = {
        "best_val_acc": float(best_val),
        "config": {
            "attn_mode": attn_mode,
            "enc_act": enc_act,
            "wta_mode": wta_mode,
            "wta_k": wta_k,
            "wta_tau": wta_tau,
            "use_2dpe": bool(use_2dpe),
            "pe_pairs": int(pe_pairs),
            "pe_alpha": float(pe_alpha),
            "epochs_per_cycle": args.epochs,
            "cycles": args.cycles,
        },
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
