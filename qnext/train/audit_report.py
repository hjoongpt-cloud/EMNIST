import argparse, json
from pathlib import Path
import torch

from qnext.core.data import get_dataloaders
from qnext.models.trunk import Trunk
from qnext.core.viz import (
    save_filters_grid, save_winner_runner_hist, save_similarity_heatmap,
    save_slot_overlay, save_logits_bar, save_margin_hist
)
from qnext.core.utils import confusion_matrix

@torch.no_grad()
def compute_filter_usage(frontend, dl, device, max_batches=300):
    K = frontend.K
    winner = torch.zeros(K, dtype=torch.long)
    runner = torch.zeros(K, dtype=torch.long)
    for i,(x,_) in enumerate(dl):
        if i>=max_batches: break
        x = x.to(device)
        a = frontend.conv1(x) * frontend.channel_mask
        score = torch.abs(a)
        top2 = torch.topk(score, k=2, dim=1)
        win_idx = top2.indices[:,0]
        run_idx = top2.indices[:,1]
        for b in range(x.size(0)):
            wi = win_idx[b].view(-1).cpu()
            ri = run_idx[b].view(-1).cpu()
            for k in wi: winner[int(k)] += 1
            for k in ri: runner[int(k)] += 1
    total_sites = winner.sum().item() if winner.sum().item()>0 else 1
    winner_rate = winner.float()/float(total_sites)
    runner_rate = runner.float()/float(total_sites)
    return winner_rate, runner_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="outputs/audit")
    ap.add_argument("--n_random", type=int, default=32)
    ap.add_argument("--n_errors", type=int, default=32)
    ap.add_argument("--attn_mode", type=str, default=None, choices=[None,"slot","local"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, dl_val_mine, dl_test = get_dataloaders(data_root=args.data_root, split_mode="remedial")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}

    # 1) ckpt설정 + CLI 병합 (CLI가 우선, 없으면 ckpt, 그래도 없으면 기본)
    attn_mode = args.attn_mode or cfg.get("attn_mode", "slot")
    enc_act   = getattr(args, "enc_act", None)  or cfg.get("enc_act",  "gelu")
    wta_mode  = getattr(args, "wta_mode", None) or cfg.get("wta_mode", "soft_topk")
    wta_k     = getattr(args, "wta_k", None)
    wta_k     = wta_k if wta_k is not None else cfg.get("wta_k", 3)
    wta_tau   = getattr(args, "wta_tau", None)
    wta_tau   = wta_tau if wta_tau is not None else cfg.get("wta_tau", 0.7)

    # 2) 모델 생성
    model = Trunk(enc_act=enc_act, wta_mode=wta_mode, wta_tau=wta_tau,
                wta_k=wta_k, attn_mode=attn_mode)

    # 3) === 중요: ckpt의 K_eff에 맞춰 프런트엔드 리사이즈 ===
    sd = ckpt["model"]
    if "enc.conv1.weight" in sd and hasattr(model, "enc") and hasattr(model.enc, "_resize_channels"):
        K_eff = sd["enc.conv1.weight"].shape[0]
        model.enc._resize_channels(K_new=K_eff)
        # channel_mask 브로드캐스트 형태 보장 (1,K,1,1)
        if getattr(model.enc, "channel_mask", None) is not None and model.enc.channel_mask.shape[1] != K_eff:
            model.enc.register_buffer("channel_mask", torch.ones(1, K_eff, 1, 1, device=model.enc.proj_down.weight.device))

    # 4) 로드 및 디바이스 이동
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()


    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    save_filters_grid(model.enc.conv1.weight, str(out_dir/"filters_grid.png"))
    wr, rr = compute_filter_usage(model.enc, dl_test, device, max_batches=200)
    save_winner_runner_hist(wr, rr, str(out_dir/"filter_usage.png"))

    W = model.enc.conv1.weight.detach().cpu().view(model.enc.K, -1)
    Wn = torch.nn.functional.normalize(W, dim=1)
    S = torch.matmul(Wn, Wn.t())
    save_similarity_heatmap(S, str(out_dir/"filter_similarity.png"), title="conv1 cos sim")

    xs = []
    for i,(x,_) in enumerate(dl_test):
        xs.append(x); 
        if len(xs)*x.size(0) >= max(1,args.n_random): break
    Xr = torch.cat(xs, dim=0)
    save_slot_overlay(Xr[0], model.slots.masks, str(out_dir/"slot_overlay_random.png"))

    model.eval(); logits_all=[]; y_all=[]
    with torch.no_grad():
        for x,y in dl_test:
            x = x.to(device); y=y.to(device)
            logits, aux = model(x, return_aux=True)
            logits_all.append(logits.cpu()); y_all.append(y.cpu())
    logits = torch.cat(logits_all, dim=0); y = torch.cat(y_all, dim=0)
    cm = confusion_matrix(logits, y, num_classes=logits.size(1))
    torch.save({"confusion": cm}, str(out_dir/"confusion.pt"))

    pred = logits.argmax(dim=1)
    err_idx = torch.nonzero(pred!=y, as_tuple=False).squeeze(1)
    if err_idx.numel() > 0:
        sel = err_idx[:args.n_errors]
        logit_y = logits.gather(1, y.view(-1,1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.view(-1,1), False)
        rival = logits.masked_fill(~mask, float('-inf')).amax(dim=1)
        margins = (logit_y - rival)
        save_margin_hist(margins[sel], str(out_dir/"error_margins.png"))

    feats = []
    with torch.no_grad():
        for i,(x,_) in enumerate(dl_test):
            x = x.to(device)
            _, aux = model(x, return_aux=True)
            feats.append(aux["slot_logits"].cpu())
            if i>=10: break
    if feats:
        F = torch.cat(feats, dim=0)
        N,S,C = F.shape
        Fp = torch.softmax(F, dim=-1)
        Fp = Fp / (Fp.norm(dim=2, keepdim=True).clamp_min(1e-6))
        sims = []
        for n in range(min(N,512)):
            G = torch.matmul(Fp[n], Fp[n].t())
            sims.append(G)
        Sm = torch.stack(sims).mean(dim=0)
        save_similarity_heatmap(Sm, str(out_dir/"slot_similarity.png"), title="slot cos sim (avg)")
        torch.save({"slot_similarity": Sm}, str(out_dir/"slot_similarity.pt"))

    with open(out_dir/"summary.json", "w") as f:
        json.dump({"attn_mode": attn_mode,
                   "winner_rate_mean": float(wr.mean()),
                   "runner_rate_mean": float(rr.mean())}, f, indent=2)
    print(f"[AUDIT] saved to {args.out_dir}")

if __name__=="__main__":
    main()
