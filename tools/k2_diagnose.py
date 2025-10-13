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

DEFAULT_CLASS_TO_EXPERT = {
    0:0, 24:0,
    1:1, 18:1, 21:1,
    9:2, 44:2,
    15:3, 40:3
}

@torch.no_grad()
def run_batch(model, x):
    logits_all, p, gate_logits = model.forward_all_expert_logits(x)
    # soft mixture
    mix = (p.unsqueeze(-1) * logits_all).sum(dim=1)
    # hard
    hard_idx = p.argmax(dim=1)
    hard = logits_all[torch.arange(x.size(0), device=x.device), hard_idx]
    return logits_all, p, mix, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c','--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--k0_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = OmegaConf.load(args.config)
    model = ExpertModelJ(cfg).to(device)
    # 로드 (weights_only=False 권장; 신뢰된 ckpt)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 데이터
    nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(nm['mean'], nm['std'])])
    test_ds = EMNIST('data', split='balanced', train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    E = cfg.model.num_experts
    C = cfg.model.num_classes
    cls2exp = torch.full((C,), -1, dtype=torch.long)
    for c,e in DEFAULT_CLASS_TO_EXPERT.items(): cls2exp[c] = e

    # 통계 누적
    all_y, pred_soft, pred_hard = [], [], []
    expert_use_counts = torch.zeros(E, dtype=torch.long)
    class_exp_soft = torch.zeros(C, E, dtype=torch.double)
    class_exp_hard = torch.zeros(C, E, dtype=torch.long)
    entropy_sum, n_samples = 0.0, 0

    # per-expert confusion용 버퍼
    per_exp_true = [ [] for _ in range(E) ]
    per_exp_pred = [ [] for _ in range(E) ]

    # 어블레이션: oracle & random
    pred_oracle, pred_random = [], []
    rng = np.random.default_rng(0)

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        logits_all, p, mix, hard_logits = run_batch(model, x)

        # 기본 예측
        pred_soft.append(mix.argmax(1).cpu())
        pred_hard.append(hard_logits.argmax(1).cpu())
        all_y.append(y.cpu())

        # expert usage (hard argmax)
        hard_idx = p.argmax(1).cpu()
        for e in range(E):
            expert_use_counts[e] += int((hard_idx == e).sum().item())

        # class→expert routing
        for c in y.unique().tolist():
            mask_dev = (y == c)
            mask_cpu = mask_dev.cpu()
            class_exp_soft[c] += p[mask_dev].sum(dim=0).cpu().double()
            class_exp_hard[c] += torch.bincount(hard_idx[mask_cpu], minlength=E)

        # entropy
        ent = -(p * (p+1e-9).log()).sum(dim=1).mean().item()
        entropy_sum += ent * x.size(0); n_samples += x.size(0)

        # per-expert confusion (hard routed subset, expert logits만 사용)
        hard_e = p.argmax(1)
        chosen_logits = logits_all[torch.arange(x.size(0), device=x.device), hard_e]
        chosen_pred = chosen_logits.argmax(1)
        for e in range(E):
            m = (hard_e == e)
            if m.any():
                per_exp_true[e].append(y[m].cpu())
                per_exp_pred[e].append(chosen_pred[m].cpu())

        # oracle gating (mapped만 one-hot expert, 나머지는 soft mix)
        oracle_logits = mix.clone()
        mapped = cls2exp[y.cpu()].to(device).ge(0)
        if mapped.any():
            tgt_e = cls2exp[y[mapped].cpu()].to(device)
            rows = torch.arange(mapped.sum().item(), device=device)
            oracle_logits[mapped] = logits_all[mapped][rows, tgt_e]
        pred_oracle.append(oracle_logits.argmax(1).cpu())

        # random gating (uniform random expert 선택)
        rand_e = torch.from_numpy(rng.integers(0, E, size=x.size(0))).to(device)
        rand_logits = logits_all[torch.arange(x.size(0), device=x.device), rand_e]
        pred_random.append(rand_logits.argmax(1).cpu())

    y_all = torch.cat(all_y)
    y_soft = torch.cat(pred_soft)
    y_hard = torch.cat(pred_hard)
    y_orac = torch.cat(pred_oracle)
    y_rand = torch.cat(pred_random)

    # 정확도
    acc_soft = accuracy_score(y_all, y_soft)
    acc_hard = accuracy_score(y_all, y_hard)
    acc_orac = accuracy_score(y_all, y_orac)
    acc_rand = accuracy_score(y_all, y_rand)

    # per-expert confusion matrix
    per_exp_cm = {}
    for e in range(E):
        if per_exp_true[e]:
            t = torch.cat(per_exp_true[e]).numpy()
            p_ = torch.cat(per_exp_pred[e]).numpy()
            cm = confusion_matrix(t, p_, labels=list(range(C)))
            per_exp_cm[e] = cm.tolist()
            np.savetxt(os.path.join(args.out_dir, f'expert{e}_confusion.csv'), cm, fmt='%d', delimiter=',')

    # class→expert soft/hard 분포
    soft_mat = class_exp_soft.numpy()
    hard_mat = class_exp_hard.numpy()
    # 정규화(soft는 평균 p, hard는 빈도 비율)
    class_counts = np.bincount(y_all.numpy(), minlength=C).reshape(-1,1)
    soft_avg = np.divide(soft_mat, np.maximum(class_counts,1), where=class_counts>0)
    hard_avg = np.divide(hard_mat, np.maximum(class_counts,1), where=class_counts>0)
    np.savetxt(os.path.join(args.out_dir,'class_to_expert_soft.csv'), soft_avg, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.out_dir,'class_to_expert_hard.csv'), hard_avg, delimiter=',', fmt='%.6f')

    # purity: 매핑 클래스가 타겟 expert로 hard 라우팅된 비율
    purity = {}
    for c,e in DEFAULT_CLASS_TO_EXPERT.items():
        idx = (y_all.numpy() == c)
        if idx.sum() == 0:
            purity[c] = None
        else:
            purity[c] = float((hard_avg[c, e]))

    # 게이트 사용량/엔트로피
    entropy_mean = float(entropy_sum / n_samples)
    expert_usage = expert_use_counts.numpy().tolist()

    # 전체 소프트/하드/오라클/랜덤 정확도 저장
    metrics = {
        'acc_soft': float(acc_soft),
        'acc_hard': float(acc_hard),
        'acc_oracle': float(acc_orac),
        'acc_random': float(acc_rand),
        'entropy_mean': entropy_mean,
        'expert_usage_counts': expert_usage,
        'purity_by_class': purity,
    }
    with open(os.path.join(args.out_dir,'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)

    print('[K2-diagnose] saved:')
    print(' - metrics.json (acc_soft/hard/oracle/random, entropy, usage, purity)')
    print(' - class_to_expert_soft.csv / class_to_expert_hard.csv')
    print(' - expert{e}_confusion.csv (per expert)')

if __name__ == '__main__':
    main()