# qnext/train/compactify_frontend.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from qnext.core.frontend import Frontend
from qnext.core.sharpen import sharpen_and_prune
from qnext.core.compact import indices_from_mask, hard_compact_frontend, export_frontend
from qnext.core.vis import save_filters_grid


def _save_hist(values, out_png, bins=50, title=None, xlabel=None):
    values = np.asarray(values).reshape(-1)
    plt.figure(figsize=(5,3.2))
    plt.hist(values, bins=bins, edgecolor="black", linewidth=0.3)
    if title:  plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def _percentiles(x, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x).reshape(-1)
    return {int(p): float(np.percentile(x, p)) for p in ps}


@torch.no_grad()
def _weight_energy(fe: Frontend):
    # conv1 weight energy proxy: L2 norm per filter
    W = fe.conv1.weight.detach().cpu().view(fe.conv1.out_channels, -1)
    e = torch.norm(W, p=2, dim=1).numpy()  # [K]
    return e


def _load_or_compute_energy(fe: Frontend, sharp_dir: Path):
    # 1) activation-based energy if available
    act_energy_npy = sharp_dir / "energy.npy"
    if act_energy_npy.exists():
        try:
            e = np.load(act_energy_npy)
            if e.ndim == 1 and e.shape[0] == fe.conv1.out_channels:
                return e, "activation"
        except Exception:
            pass
    # 2) fallback: weight-based proxy
    return _weight_energy(fe), "weight_proxy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--ae_dir", type=str, required=True,
                    help="Directory containing conv1.npy/proj_down.npz from AE")
    ap.add_argument("--out_dir", type=str, default="outputs/ae_compact")

    # 기존 기준들
    ap.add_argument("--winner_thresh", type=float, default=0.005,
                    help="winner-rate(사용도) 하한. 하한 미만이면 프루닝 후보.")
    ap.add_argument("--sim_thresh", type=float, default=0.98,
                    help="코사인 유사도 중복 컷 임계값.")

    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)

    # === 신규 기준들 ===
    ap.add_argument("--energy_percentile", type=float, default=5,
                    help="에너지 하위 퍼센타일 컷. e < p% 이고 winner<thresh 이면 1차 프루닝.")
    ap.add_argument("--prune_cap", type=float, default=0.25,
                    help="전체 K 대비 최대 프루닝 비율 상한(과컷 방지).")
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sharp_dir = out_dir / "sharp"; sharp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = Frontend(K=150, D=64, enc_act="relu", wta_mode="none").to(device)
    loaded = fe.load_from_ae_dir(args.ae_dir)
    K_init = fe.conv1.out_channels
    print(f"[LOAD] from '{args.ae_dir}' -> {loaded}; K={K_init}, D={fe.D}", flush=True)

    # --- Step 1: 통계 수집 (winner-rate, redundancy sim, (opt) activation energy) ---
    print("[STEP] sharpen & prune stats pass...", flush=True)
    wr, rr, sim = sharpen_and_prune(
        fe, data_root=args.data_root,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        winner_thresh=args.winner_thresh,
        sim_thresh=args.sim_thresh,
        device=device,
        out_dir=str(sharp_dir),
        log_every=args.log_every, verbose=True
    )
    # wr: winner rate per filter  [K]
    # rr: runner-up rate per filter [K] (옵션)
    # sim: similarity matrix (K,K) or top similarities (구현에 따라 상이)

    # 저장 및 히스토그램
    np.save(sharp_dir / "winner_rate.npy", wr.detach().cpu().numpy())
    if rr is not None:
        np.save(sharp_dir / "runner_rate.npy", rr.detach().cpu().numpy())

    _save_hist(wr.detach().cpu().numpy(), out_dir / "hist_winner_rate.png",
               bins=50, title="Winner-rate histogram", xlabel="winner-rate")
    if rr is not None:
        _save_hist(rr.detach().cpu().numpy(), out_dir / "hist_runner_rate.png",
                   bins=50, title="Runner-rate histogram", xlabel="runner-up-rate")

    # sim 저장/요약
    if sim is not None:
        if torch.is_tensor(sim): sim_np = sim.detach().cpu().numpy()
        else: sim_np = np.asarray(sim)
        np.save(sharp_dir / "similarity.npy", sim_np)
        # 상위 분위수 로깅용 (대각 제외)
        if sim_np.ndim == 2 and sim_np.shape[0] == sim_np.shape[1]:
            S = sim_np.copy()
            np.fill_diagonal(S, -1.0)
            sim_vals = S[S >= -0.5]  # 유효값만
            sim_p = _percentiles(sim_vals, ps=(50, 90, 95, 99, 99.5, 99.9))
        else:
            sim_p = _percentiles(sim_np, ps=(50, 90, 95, 99, 99.5, 99.9))
    else:
        sim_p = {}

    # --- Step 2: 에너지 산출 (activation 기반 있으면 사용, 없으면 weight-proxy) ---
    energy, e_mode = _load_or_compute_energy(fe, sharp_dir)
    np.save(sharp_dir / "energy_used.npy", energy)
    _save_hist(energy, out_dir / "hist_energy.png", bins=50,
               title=f"Energy histogram ({e_mode})", xlabel="energy")

    # --- Step 3: 프루닝 후보 결정 ---
    wr_np = wr.detach().cpu().numpy()
    # 1) 사용도 + 에너지 AND 규칙
    energy_cut = np.percentile(energy, args.energy_percentile)
    mask_use = wr_np < args.winner_thresh
    mask_energy = energy < energy_cut
    base_prune = mask_use & mask_energy

    # 2) 중복 컷: sim > sim_thresh 군집에서 추가 컷 (여기선 간단히: 높은 유사도 짝 중에서 winner-rate 낮은 쪽 우선)
    dup_prune = np.zeros_like(base_prune, dtype=bool)
    if sim is not None:
        if torch.is_tensor(sim): sim_np = sim.detach().cpu().numpy()
        else: sim_np = np.asarray(sim)
        if sim_np.ndim == 2 and sim_np.shape[0] == sim_np.shape[1]:
            S = sim_np
            np.fill_diagonal(S, 0.0)
            # 간단한 휴리스틱: 행별 최대 유사 대상만 보고, 임계 초과면 둘 중 winner-rate 낮은 쪽 후보 지정
            max_idx = np.argmax(S, axis=1)      # [K]
            max_val = S[np.arange(S.shape[0]), max_idx]
            for i, j, v in zip(range(S.shape[0]), max_idx, max_val):
                if v > args.sim_thresh:
                    if wr_np[i] <= wr_np[j]:
                        dup_prune[i] = True
                    else:
                        dup_prune[j] = True

    prune_candidates = base_prune | dup_prune

    # 3) 과컷 방지: cap 적용
    max_remove = int(np.floor(K_init * args.prune_cap))
    cand_idx = np.nonzero(prune_candidates)[0].tolist()
    if len(cand_idx) > max_remove:
        # 우선순위: winner-rate 낮은 순, 그다음 energy 낮은 순
        order = sorted(cand_idx, key=lambda k: (wr_np[k], energy[k]))
        keep_subset = set(order[:max_remove])
        prune_mask = np.zeros(K_init, dtype=bool)
        prune_mask[list(keep_subset)] = True
    else:
        prune_mask = prune_candidates

    # 프루닝 마스크에서 '남길 것'의 인덱스로 변환
    keep_idx = np.nonzero(~prune_mask)[0]
    # channel_mask를 해당 keep_idx로 설정
    # indices_from_mask는 채널 마스크 기반이므로 여기선 직접 keep_idx를 사용
    # 하드 컴팩션 실행
    K_eff = hard_compact_frontend(fe, torch.tensor(keep_idx, dtype=torch.long))

    print(f"[COMPACT] K_eff={K_eff} (from K_init={K_init})", flush=True)

    # --- Step 4: 결과 아웃풋/시각화 ---
    print("[STEP] export compact frontend...", flush=True)
    export_frontend(fe, args.out_dir)

    # 필터 그리드 (컴팩션 후)
    try:
        save_filters_grid(fe.conv1.weight.detach().cpu(), str(out_dir / "filters_grid_compact.png"))
    except Exception as e:
        print(f"[WARN] save_filters_grid failed: {e}")

    # 요약 저장
    summary = {
        "K_init": int(K_init),
        "K_eff": int(K_eff),
        "winner_thresh": float(args.winner_thresh),
        "sim_thresh": float(args.sim_thresh),
        "energy_percentile": float(args.energy_percentile),
        "energy_cut": float(energy_cut),
        "prune_cap": float(args.prune_cap),
        "energy_mode": e_mode,
        "winner_rate_percentiles": _percentiles(wr_np),
        "energy_percentiles": _percentiles(energy),
        "sim_percentiles": sim_p,
        "removed_count": int(K_init - len(keep_idx)),
        "removed_ratio": float((K_init - len(keep_idx)) / max(1, K_init)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] exported to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
