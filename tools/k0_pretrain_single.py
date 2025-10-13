import os, json, argparse
import numpy as np
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_k import ExpertModelJ
from src.utils.moe_analysis import MoeAnalyzer  # 존재하지만 여기선 사용하지 않음
from src.training.train_expert_analyze import train_epoch, evaluate


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-o', '--out_dir', required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(nm['mean'], nm['std'])
    ])
    train_ds = EMNIST('data', split='balanced', train=True,  download=True, transform=transform)
    test_ds  = EMNIST('data', split='balanced', train=False, download=True, transform=transform)

    summary = []

    for seed in cfg.get('seeds', [42]):
        set_seed(seed)
        run_dir = os.path.join(args.out_dir, f'seed_{seed}')
        os.makedirs(run_dir, exist_ok=True)
        ckpt_dir = os.path.join(run_dir, 'ckpts'); os.makedirs(ckpt_dir, exist_ok=True)

        train_loader = DataLoader(train_ds, batch_size=cfg.opt.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = ExpertModelJ(cfg).to(device)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.opt.lr)

        for ep in range(1, cfg.opt.epochs+1):
            ce, lb, ent, acc = train_epoch(model, train_loader, optimizer, device, cfg, analyzer=None, epoch=ep)
            print(f"[K0][seed {seed}] Ep{ep:02d} CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} ACC {acc:.4f}")

        test_acc = evaluate(model, test_loader, device)
        print(f"[K0][seed {seed}] TestAcc {test_acc:.4f}")

        torch.save({
            'model': model.state_dict(),
            'cfg': OmegaConf.to_container(cfg, resolve=True),
            'seed': seed,
        }, os.path.join(ckpt_dir, 'k0_single.pt'))
        summary.append({'seed': seed, 'test_acc': float(test_acc)})

    with open(os.path.join(args.out_dir, 'summary_k0.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()