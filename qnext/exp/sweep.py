# qnext/exp/sweep.py
import argparse, os, json, random, subprocess
from pathlib import Path
from statistics import mean

# =========================
# I/O & CSV parsing utils
# =========================
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_train_log(log_path: Path):
    """
    train_cls가 쓰는 train_log.csv (최신 포맷 가정):
    epoch,train_loss,early_acc,val_mine_acc,test_acc,margin_p10,margin_p50,margin_p90,slot_offdiag_mean

    과거 버전 호환: 일부 컬럼 없으면 None으로 채움.
    """
    if not log_path.exists():
        return []
    with open(log_path, "r") as f:
        header = f.readline().strip().split(",")
        cols = {k: i for i, k in enumerate(header)}

        def get(parts, name, cast=float, default=None):
            try:
                idx = cols[name]
                v = parts[idx].strip()
                if v == "" or v is None:
                    return default
                return cast(v)
            except Exception:
                return default

        rows = []
        for line in f:
            parts = [x.strip() for x in line.rstrip().split(",")]
            if len(parts) < len(header):
                # 패딩
                parts = parts + [""] * (len(header) - len(parts))
            rows.append({
                "epoch": get(parts, "epoch", cast=lambda x: int(float(x)), default=None),
                "train_loss": get(parts, "train_loss", default=None),
                "early_acc": get(parts, "early_acc", default=None),
                "val_acc": get(parts, "val_mine_acc", default=get(parts, "val_acc", default=None)),
                "test_acc": get(parts, "test_acc", default=None),
                "m10": get(parts, "margin_p10", default=None),
                "m50": get(parts, "margin_p50", default=None),
                "m90": get(parts, "margin_p90", default=None),
                "slot_offdiag_mean": get(parts, "slot_offdiag_mean", default=None),
            })
    return rows

def last_metrics(rows):
    if not rows:
        return {"val_acc": 0.0, "test_acc": 0.0, "m50": 0.0, "slot_offdiag_mean": 1.0}
    r = rows[-1]
    return {
        "val_acc": r.get("val_acc") or 0.0,
        "test_acc": r.get("test_acc") or 0.0,
        "m50": r.get("m50") or 0.0,
        "slot_offdiag_mean": (r.get("slot_offdiag_mean") if r.get("slot_offdiag_mean") is not None else 1.0),
    }

def obj_score(metrics, penalty_offdiag=0.5, penalty_margin=0.5, offdiag_thr=0.70, m50_thr=3.30):
    """
    목적함수: val_acc - penalty
      - 슬롯 간 유사도(낮을수록 좋음) 기준 페널티
      - 마진 중앙값 m50(높을수록 좋음) 기준 페널티
    """
    offdiag_pen = penalty_offdiag * max(0.0, (metrics["slot_offdiag_mean"] or 1.0) - offdiag_thr)
    margin_pen  = penalty_margin  * max(0.0, m50_thr - (metrics["m50"] or 0.0))
    return (metrics["val_acc"] or 0.0) - offdiag_pen - margin_pen

def run_train_cls(base_cmd: str, out_dir: Path, epochs: int, seed: int, extra_args: dict, env=None, quiet=False):
    mkdir(out_dir)
    # 베이스 + 오버라이드 인자 구성
    arg_line = base_cmd.strip()
    arg_line += f" --epochs {epochs} --seed {seed}"
    for k, v in extra_args.items():
        if isinstance(v, bool):
            arg_line += f" --{k} {int(v)}"
        else:
            arg_line += f" --{k} {v}"
    arg_line += f" --out_dir {str(out_dir)}"

    if not quiet:
        print("[RUN]", arg_line)

    proc = subprocess.run(["/bin/bash", "-lc", arg_line], env=env)
    return proc.returncode == 0

# =========================
# Preset samplers
# =========================
def sampler_pack_A(rng, fixed):  # LR/동결 빅레인지
    U = rng.uniform; C = rng.choice
    d = {
        "freeze_encoder_epochs": int(C([0, 4, 6, 10, 16, 20])),
        "lr": round(U(1e-4, 2e-3), 6),
        "wta_tau_start": 0.9,
        "wta_tau_end": round(U(0.70, 0.82), 3),
        # slotdrop: 중간 프로파일 고정 + 약간의 드리프트
        "slotdrop_start": round(U(0.06, 0.12), 3),
        "slotdrop_mid":   round(U(0.36, 0.44), 3),
        "slotdrop_end":   round(U(0.26, 0.30), 3),
        "slotdrop_t_warm": 0.30,
        "slotdrop_t_peak": round(U(0.88, 0.93), 2),
        "diversity_enable": 1,
        "diversity_lambda": 0.006,
        "feature_dropout_p": 0.25,
        "head_dropout_p": 0.38,
    }
    d.update(fixed); return d

def sampler_pack_B(rng, fixed):  # SlotDrop 모양 실험
    C = rng.choice; U = rng.uniform
    # 네 가지 대표 형태 + 소폭 지터
    shapes = [
        {"name":"U", "start":0.30, "mid":0.10, "end":0.30, "t_peak":0.50},
        {"name":"up", "start":0.05, "mid":0.20, "end":0.40, "t_peak":0.95},
        {"name":"plateau","start":0.28, "mid":0.35, "end":0.32, "t_peak":0.85},
        {"name":"flat", "start":0.30, "mid":0.30, "end":0.30, "t_peak":0.80},
    ]
    s = dict(C(shapes))
    d = {
        "freeze_encoder_epochs": 6,
        "lr": 4e-4,
        "wta_tau_start": 0.9,
        "wta_tau_end": 0.75,
        "slotdrop_start": round(max(0.0, min(0.5, s["start"] + U(-0.02, 0.02))), 3),
        "slotdrop_mid":   round(max(0.0, min(0.6, s["mid"]   + U(-0.03, 0.03))), 3),
        "slotdrop_end":   round(max(0.0, min(0.6, s["end"]   + U(-0.02, 0.02))), 3),
        "slotdrop_t_warm": 0.30,
        "slotdrop_t_peak": round(max(0.5, min(0.98, s["t_peak"] + U(-0.03, 0.03))), 2),
        "diversity_enable": 1,
        "diversity_lambda": 0.006,
        "feature_dropout_p": 0.25,
        "head_dropout_p": 0.38,
    }
    d.update(fixed); return d

def sampler_pack_C(rng, fixed):  # 규제 강도 대조
    U = rng.uniform
    d = {
        "freeze_encoder_epochs": 6,
        "lr": 4e-4,
        "wta_tau_start": 0.9,
        "wta_tau_end": 0.75,
        "slotdrop_start": 0.10, "slotdrop_mid": 0.40, "slotdrop_end": 0.28,
        "slotdrop_t_warm": 0.30, "slotdrop_t_peak": 0.90,
        "diversity_enable": 1,
        "diversity_lambda": round(U(0.0, 0.02), 4),
        "feature_dropout_p": round(U(0.05, 0.40), 2),
        "head_dropout_p": round(U(0.10, 0.60), 2),
    }
    d.update(fixed); return d

def sampler_pack_D(rng, fixed):  # WTA 강성/느슨 + k
    C = rng.choice; U = rng.uniform
    d = {
        "freeze_encoder_epochs": 6,
        "lr": 4e-4,
        "wta_tau_start": 0.9,
        "wta_tau_end": float(C([0.68, 0.70, 0.74, 0.78, 0.82])),
        "wta_k": int(C([1, 3, 5])),
        "slotdrop_start": 0.10, "slotdrop_mid": 0.40, "slotdrop_end": 0.28,
        "slotdrop_t_warm": 0.30, "slotdrop_t_peak": 0.90,
        "diversity_enable": 1,
        "diversity_lambda": 0.006,
        "feature_dropout_p": 0.25,
        "head_dropout_p": 0.38,
    }
    d.update(fixed); return d

def sampler_pack_E(rng, fixed):  # 2DPE/어텐션 모드/슬롯수 영향 (슬롯수 옵션화 가능 시)
    # 주: 현 train_cls가 S(슬롯 개수) 옵션을 안 받으면, 여기선 2DPE on/off & attn_mode만 변주
    C = rng.choice
    use_2dpe = int(C([0,1]))
    d = {
        "freeze_encoder_epochs": 6,
        "lr": 4e-4,
        "wta_tau_start": 0.9,
        "wta_tau_end": 0.75,
        "slotdrop_start": 0.10, "slotdrop_mid": 0.40, "slotdrop_end": 0.28,
        "slotdrop_t_warm": 0.30, "slotdrop_t_peak": 0.90,
        "diversity_enable": 1, "diversity_lambda": 0.006,
        "feature_dropout_p": 0.25, "head_dropout_p": 0.38,
        # 변주
        "attn_mode": str(C(["slot","local"])),
        "use_2dpe": use_2dpe,
        "pe_alpha": (0.5 if use_2dpe==1 else 0.5),
        "pe_pairs": 16,
    }
    d.update(fixed); return d

def sampler_pack_F(rng, fixed):  # 동결 & (본 학습) encoder 학습 강도 대비
    C = rng.choice; U = rng.uniform
    d = {
        "freeze_encoder_epochs": int(C([0, 6, 12])),
        "lr": float(C([2e-4, 4e-4, 8e-4, 1.2e-3])),
        "wta_tau_start": 0.9,
        "wta_tau_end": round(U(0.72, 0.78), 3),
        "slotdrop_start": 0.10, "slotdrop_mid": 0.40, "slotdrop_end": 0.28,
        "slotdrop_t_warm": 0.30, "slotdrop_t_peak": 0.90,
        "diversity_enable": 1, "diversity_lambda": 0.006,
        "feature_dropout_p": 0.25, "head_dropout_p": 0.38,
        # encoder_lr_scale은 post-hoc 튠에서 쓰이지만,
        # 본 학습에서의 간접적 강도는 lr / freeze로 유도.
    }
    d.update(fixed); return d

PRESETS = {
    "A": sampler_pack_A,
    "B": sampler_pack_B,
    "C": sampler_pack_C,
    "D": sampler_pack_D,
    "E": sampler_pack_E,
    "F": sampler_pack_F,
}

def default_sampler(rng, fixed):  # 이전(좁은) 기본 랜덤
    U = rng.uniform
    d = {
        "freeze_encoder_epochs": int(random.choice([6,8,10,12])),
        "lr": round(U(2e-4, 8e-4), 6),
        "wta_tau_start": 0.9,
        "wta_tau_end": round(U(0.72, 0.78), 3),
        "slotdrop_start": round(U(0.05, 0.15), 3),
        "slotdrop_mid":   round(U(0.30, 0.45), 3),
        "slotdrop_end":   round(U(0.26, 0.34), 3),
        "slotdrop_t_warm": 0.30,
        "slotdrop_t_peak": round(U(0.85, 0.95), 2),
        "diversity_enable": 1,
        "diversity_lambda": round(U(0.002, 0.008), 4),
        "feature_dropout_p": round(U(0.20, 0.30), 2),
        "head_dropout_p": round(U(0.30, 0.40), 2),
    }
    d.update(fixed); return d

def canonical_name(cand: dict):
    keys = [
        "lr","freeze_encoder_epochs","wta_tau_end",
        "slotdrop_start","slotdrop_mid","slotdrop_end","slotdrop_t_peak",
        "diversity_lambda","feature_dropout_p","head_dropout_p",
        "attn_mode","use_2dpe","wta_k"
    ]
    name_parts = []
    for k in keys:
        if k in cand:
            name_parts.append(f"{k}={cand[k]}")
    return "_".join(name_parts)

# =========================
# ASHA-lite
# =========================
def successive_halving(
    work_dir: Path,
    init_from_ae: str,
    seeds=(41,42,43),
    stage_epochs=(10,30,50),
    top_frac=(1/3, 1/3),
    n_init=24,
    sampler_fn=None,
    fixed_args=None,
    env=None,
):
    rng = random.Random(20250922)
    fixed_args = fixed_args or {}
    sampler_fn = sampler_fn or default_sampler

    mkdir(work_dir)

    def base_cmd():
        # 공통 고정: 모델/데이터/변경 적은 것
        return (
            "PYTHONPATH=. python -m qnext.train.train_cls "
            f"--data_root ./data "
            f"--init_from_ae {init_from_ae} "
            "--enc_act gelu "
            "--wta_mode soft_topk --wta_k 3 "
            "--attn_mode slot "
            "--use_2dpe 1 --pe_pairs 16 --pe_alpha 0.5 "
            "--jitter_px 2 --jitter_alpha 0.2 "
            "--diversity_enable 1 "
        )

    def eval_pool(pool, stage_idx:int):
        ep = stage_epochs[stage_idx]
        stage_tag = f"stage{stage_idx+1}_{ep}ep"
        scored = []

        for i, cand in enumerate(pool):
            name = canonical_name(cand)
            trial_dir = work_dir / stage_tag / f"{i:03d}_{name}"
            mkdir(trial_dir)

            val_scores = []; test_scores = []; m50s = []; offdiags = []
            for sd in seeds:
                out_dir = trial_dir / f"seed{sd}"
                ok = run_train_cls(
                    base_cmd=base_cmd(),
                    out_dir=out_dir,
                    epochs=ep,
                    seed=sd,
                    extra_args=cand,
                    env=env,
                    quiet=False,
                )
                rows = read_train_log(out_dir / "train_log.csv")
                mets = last_metrics(rows)
                val_scores.append(mets["val_acc"])
                test_scores.append(mets["test_acc"])
                m50s.append(mets["m50"])
                offdiags.append(mets["slot_offdiag_mean"])

            avg_val = mean(val_scores) if val_scores else 0.0
            avg_test = mean(test_scores) if test_scores else 0.0
            avg_m50 = mean(m50s) if m50s else 0.0
            avg_off = mean(offdiags) if offdiags else 1.0
            score = obj_score({"val_acc":avg_val,"m50":avg_m50,"slot_offdiag_mean":avg_off})

            scored.append((score, avg_val, avg_test, avg_m50, avg_off, cand, trial_dir))
            with open(trial_dir / "summary.json", "w") as f:
                json.dump({
                    "stage": stage_tag, "candidate": cand,
                    "val_acc_mean": avg_val,
                    "test_acc_mean": avg_test,
                    "m50_mean": avg_m50,
                    "offdiag_mean": avg_off,
                    "obj_score": score,
                    "seeds": list(seeds),
                }, f, indent=2)

        scored.sort(key=lambda t: t[0], reverse=True)
        with open(work_dir / f"{stage_tag}_leaderboard.json", "w") as f:
            lb = []
            for rank, (score, val, test, m50, off, cand, tdir) in enumerate(scored, start=1):
                lb.append({
                    "rank": rank, "obj_score": score,
                    "val_acc": val, "test_acc": test, "m50": m50, "offdiag": off,
                    "candidate": cand, "trial_dir": str(tdir)
                })
            json.dump(lb, f, indent=2)
        return scored

    # 0) 초기 풀
    pool = [sampler_fn(rng, fixed_args) for _ in range(n_init)]

    # Stage 1
    S1 = eval_pool(pool, 0)
    k1 = max(1, int(len(S1) * top_frac[0]))
    pool2 = [t[5] for t in S1[:k1]]

    # Stage 2
    S2 = eval_pool(pool2, 1)
    k2 = max(1, int(len(S2) * top_frac[1]))
    pool3 = [t[5] for t in S2[:k2]]

    # Stage 3
    S3 = eval_pool(pool3, 2)
    final = sorted(S3, key=lambda t: t[0], reverse=True)

    with open(work_dir / "FINAL_leaderboard.json", "w") as f:
        lb = []
        for rank, (score, val, test, m50, off, cand, tdir) in enumerate(final, start=1):
            lb.append({
                "rank": rank, "obj_score": score,
                "val_acc": val, "test_acc": test, "m50": m50, "offdiag": off,
                "candidate": cand, "trial_dir": str(tdir)
            })
        json.dump(lb, f, indent=2)

    print("\n=== FINAL TOP-5 ===")
    for i, item in enumerate(final[:5], start=1):
        print(f"[{i}] score={item[0]:.4f}, val={item[1]*100:.2f}, test={item[2]*100:.2f}, m50={item[3]:.3f}, off={item[4]:.3f}")
    print(f"\nResults saved under: {str(work_dir)}")

# =========================
# CLI
# =========================
def parse_fixed_kv(s: str):
    if not s or not s.strip():
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        v = v.strip()
        # 숫자 or bool 캐스팅 시도
        if v in ["0","1"]:
            out[k] = int(v)
        else:
            try:
                if "." in v or "e" in v or "E" in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=str, default="sweeps/slot_asha")
    ap.add_argument("--init_from_ae", type=str, required=True)
    ap.add_argument("--n_init", type=int, default=24)
    ap.add_argument("--seeds", type=str, default="41,42,43")
    ap.add_argument("--stage_epochs", type=str, default="10,30,50")
    ap.add_argument("--top_frac", type=str, default="0.33,0.33")
    ap.add_argument("--preset", type=str, default="", choices=["","A","B","C","D","E","F"],
                    help="미리 정의된 넓은 탐색 세계선 프리셋")
    ap.add_argument("--fixed", type=str, default="",
                    help="추가 고정 인자: key=value,key=value (예: enc_act=gelu,attn_mode=slot)")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    mkdir(work_dir)

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    stage_epochs = tuple(int(e) for e in args.stage_epochs.split(",") if e.strip())
    tf = tuple(float(x) for x in args.top_frac.split(","))
    fixed = parse_fixed_kv(args.fixed)

    sampler_fn = PRESETS.get(args.preset, default_sampler)

    successive_halving(
        work_dir=work_dir,
        init_from_ae=args.init_from_ae,
        seeds=seeds,
        stage_epochs=stage_epochs,
        top_frac=tf,
        n_init=args.n_init,
        sampler_fn=sampler_fn,
        fixed_args=fixed,
    )

if __name__ == "__main__":
    main()
