import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, accuracy_score

from src.models.expert_model_j import ExpertModelJ

@torch.no_grad()
def forward_full(model, x):
    logits_all, p, _ = model.forward_all_expert_logits(x)
    mix = (p.unsqueeze(-1) * logits_all).sum(dim=1)
    return logits_all, p, mix


def ce_from_logits(logits, y):
    return F.cross_entropy(logits, y, reduction='none')


def top2_margin(logits):
    # logits: [B,C]
    top2 = torch.topk(logits, 2, dim=1).values
    return (top2[:,0] - top2[:,1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c','--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--hard_ratio', type=float, default=0.1, help='상위 p% CE로 hard set 규정')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = OmegaConf.load(args.config)
    model = ExpertModelJ(cfg).to(device)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(sd, strict=False)
    model.eval()

    nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(nm['mean'], nm['std'])])
    test_ds = EMNIST('data', split='balanced', train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    E = cfg.model.num_experts
    C = cfg.model.num_classes

    all_y, ce_mix_list, ent_p_list, maxp_list = [], [], [], []
    pred_soft, pred_hard, pred_orac, pred_rand = [], [], [], []
    hard_idx_list = []

    # hard 전용 통계 버퍼
    per_exp_true_hard = [ [] for _ in range(E) ]
    per_exp_pred_hard = [ [] for _ in range(E) ]
    class_exp_soft_hard = torch.zeros(C, E, dtype=torch.double)
    class_exp_hard_hard = torch.zeros(C, E, dtype=torch.long)

    rng = np.random.default_rng(0)

    # 1) 전체 테스트셋 forward 수집
    logits_all_all, p_all, mix_all = [], [], []
    for x, y in test_loader:
        x = x.to(device); y = y.to(device)
        logits_all, p, mix = forward_full(model, x)
        logits_all_all.append(logits_all.cpu())
        p_all.append(p.cpu())
        mix_all.append(mix.cpu())
        all_y.append(y.cpu())

    logits_all_all = torch.cat(logits_all_all)
    p_all = torch.cat(p_all)
    mix_all = torch.cat(mix_all)
    y_all = torch.cat(all_y)

    # 2) 난이도 측정(CE/margin/entropy)
    ce_mix = ce_from_logits(mix_all, y_all)               # [N]
    margin_mix = top2_margin(mix_all)                     # [N]
    p_dist = p_all / (p_all.sum(1, keepdim=True) + 1e-9)
    ent_p = -(p_dist * (p_dist+1e-9).log()).sum(1)        # [N]
    maxp = p_dist.max(1).values

    N = y_all.size(0)
    k = int(max(1, round(args.hard_ratio * N)))
    hard_sel = torch.topk(ce_mix, k).indices              # CE 상위 k개 → hard set
    easy_sel = torch.topk(-ce_mix, k).indices             # CE 하위 k개 → easy set(동일 크기 비교용)

    # 3) hard/easy 각각에서 soft/hard/oracle/random 정확도
    def eval_subset(idx):
        idx = idx.cpu()
        # soft
        y_soft = mix_all[idx].argmax(1)
        # hard via chosen expert
        hard_e = p_all[idx].argmax(1)                # (k,)
        chosen_logits = logits_all_all[idx, hard_e]

        y_hard = chosen_logits.argmax(1)
        # oracle(best expert per sample)
        best_logits, _ = logits_all_all[idx].max(1)     # [k,C]
        y_orac = best_logits.argmax(1)
        # random expert
        rand_e = torch.from_numpy(rng.integers(0, E, size=idx.size(0))).long()
        y_rand = logits_all_all[idx, rand_e].argmax(1)
        acc_soft = (y_soft == y_all[idx]).float().mean().item()
        acc_hard = (y_hard == y_all[idx]).float().mean().item()
        acc_orac = (y_orac == y_all[idx]).float().mean().item()
        acc_rand = (y_rand == y_all[idx]).float().mean().item()
        # p mean / entropy
        p_mean = p_all[idx].mean(0).numpy().tolist()
        ent_mean = float(ent_p[idx].mean().item())
        return acc_soft, acc_hard, acc_orac, acc_rand, p_mean, ent_mean

    acc_soft_h, acc_hard_h, acc_orac_h, acc_rand_h, pmean_h, ent_h = eval_subset(hard_sel)
    acc_soft_e, acc_hard_e, acc_orac_e, acc_rand_e, pmean_e, ent_e = eval_subset(easy_sel)

    # 4) hard subset 세부: per-expert confusion & class→expert 라우팅
    hard_e_all = p_all.argmax(1)
    for e in range(E):
        m = (hard_e_all[hard_sel] == e)
        if m.any():
            t = y_all[hard_sel][m]
            pred = logits_all_all[hard_sel][m, e].argmax(1)
            per_exp_true_hard[e].append(t)
            per_exp_pred_hard[e].append(pred)

    # class routing on hard
    for c in y_all.unique().tolist():
        mask = (y_all == c)
        m = torch.zeros_like(mask); m[hard_sel] = True
        sel = mask & m
        if sel.any():
            class_exp_soft_hard[c] += p_all[sel].sum(0).double()
            class_exp_hard_hard[c] += torch.bincount(hard_e_all[sel], minlength=E)

    # 저장물
    metrics = {
        'hard_ratio': args.hard_ratio,
        'hard': {
            'acc_soft': acc_soft_h, 'acc_hard': acc_hard_h, 'acc_oracle': acc_orac_h, 'acc_random': acc_rand_h,
            'p_mean': pmean_h, 'gate_entropy_mean': ent_h,
        },
        'easy': {
            'acc_soft': acc_soft_e, 'acc_hard': acc_hard_e, 'acc_oracle': acc_orac_e, 'acc_random': acc_rand_e,
            'p_mean': pmean_e, 'gate_entropy_mean': ent_e,
        },
        'corr': {
            'pearson(entropy_p, CE)': float(np.corrcoef(ent_p.numpy(), ce_mix.numpy())[0,1]),
            'pearson(max_p, CE)': float(np.corrcoef(maxp.numpy(), ce_mix.numpy())[0,1])
        }
    }
    with open(os.path.join(args.out_dir,'hardcase_metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)

    # per-expert confusion on hard
    for e in range(E):
        if per_exp_true_hard[e]:
            t = torch.cat(per_exp_true_hard[e]).numpy()
            p_ = torch.cat(per_exp_pred_hard[e]).numpy()
            cm = confusion_matrix(t, p_, labels=list(range(C)))
            np.savetxt(os.path.join(args.out_dir, f'expert{e}_hard_confusion.csv'), cm, fmt='%d', delimiter=',')

    # class→expert(soft/hard) on hard subset (정규화 없이 raw counts/sum)
    np.savetxt(os.path.join(args.out_dir,'class_to_expert_soft_hard.csv'), class_exp_soft_hard.numpy(), delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.out_dir,'class_to_expert_hard_hard.csv'), class_exp_hard_hard.numpy(), delimiter=',', fmt='%d')

    print('[K2-hardcase] saved: hardcase_metrics.json, expert{e}_hard_confusion.csv, class_to_expert_*_hard.csv')

if __name__ == '__main__':
    main()