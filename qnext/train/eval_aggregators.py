# qnext/train/eval_aggregators.py

import argparse, torch, json
from pathlib import Path
from qnext.models.trunk import Trunk
from qnext.core.data import get_dataloaders
from qnext.core.utils import set_seed

def eval_on_valmine(model, dl, device, aggregator):
    model.aggregator = aggregator
    model.eval(); total=0; correct=0
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x); pred = logits.argmax(dim=1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct / max(1,total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="outputs/agg_sweep")
    ap.add_argument("--aggregators", type=str, nargs="+", default=["logit_mean","prob_mean","prob_max"])
    ap.add_argument("--attn_mode", type=str, default=None, choices=[None,"slot","local"])
    # (옵션) 인자에서 덮어쓰고 싶을 때 사용할 수 있게 추가
    ap.add_argument("--enc_act", type=str, default=None)
    ap.add_argument("--wta_mode", type=str, default=None)
    ap.add_argument("--wta_k", type=int, default=None)
    ap.add_argument("--wta_tau", type=float, default=None)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, dl_val_mine, _ = get_dataloaders(data_root=args.data_root, split_mode="remedial")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}

    # 1) ckpt 설정과 CLI 인자 병합 (CLI가 우선, 미지정이면 ckpt 사용, 그래도 없으면 기본)
    attn_mode = args.attn_mode or cfg.get("attn_mode", "slot")
    enc_act   = args.enc_act  or cfg.get("enc_act",  "gelu")
    wta_mode  = args.wta_mode or cfg.get("wta_mode", "soft_topk")
    wta_k     = args.wta_k    if args.wta_k is not None else cfg.get("wta_k", 3)
    wta_tau   = args.wta_tau  if args.wta_tau is not None else cfg.get("wta_tau", 0.7)

    # 2) 모델 생성
    model = Trunk(enc_act=enc_act, wta_mode=wta_mode, wta_tau=wta_tau,
                  wta_k=wta_k, attn_mode=attn_mode)

    # 3) CKPT의 K_eff에 맞춰 프런트엔드 리사이즈 후 state_dict 로드
    sd = ckpt["model"]
    if "enc.conv1.weight" in sd and hasattr(model, "enc") and hasattr(model.enc, "_resize_channels"):
        K_eff = sd["enc.conv1.weight"].shape[0]
        model.enc._resize_channels(K_new=K_eff)
        # channel_mask는 (1,K,1,1) 형태 보장
        if getattr(model.enc, "channel_mask", None) is not None and model.enc.channel_mask.shape[1] != K_eff:
            model.enc.register_buffer("channel_mask", torch.ones(1, K_eff, 1, 1, device=model.enc.proj_down.weight.device))

    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()

    # 4) 스윕
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for agg in args.aggregators:
        acc = eval_on_valmine(model, dl_val_mine, device, agg)
        print(f"{agg}: {acc*100:.2f}%"); results.append((agg, acc))

    with open(Path(args.out_dir)/"agg_results.csv","w") as f:
        for k,v in results: f.write(f"{k},{v}\n")
    best = max(results, key=lambda t:t[1])
    print(f"[BEST] {best[0]} ({best[1]*100:.2f}%)")
    with open(Path(args.out_dir)/"summary.json","w") as f:
        json.dump({"best": {"aggregator": best[0], "acc": float(best[1])},
                   "results": {k: float(v) for k,v in results},
                   "attn_mode": attn_mode}, f, indent=2)

if __name__=="__main__":
    main()
