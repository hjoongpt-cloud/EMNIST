# tools/l1_supcon_train.py
import os, json, argparse, random
import numpy as np, torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_j import ExpertModelJ
from src.training.train_expert_analyze import extract_features
from tools.weighted_supcon import WeightedSupCon

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def aug_light():
    # 아주 가벼운 증강 (EMNIST)
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.07,0.07), fill=0),
        transforms.ToTensor()
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k0_config', required=True)
    ap.add_argument('--k0_ckpt', required=True)
    ap.add_argument('--gate_ckpt', default=None)             # K1 ckpt (clusters 추출용)
    ap.add_argument('--clusters_json', required=True)         # l0_make_clusters_from_k1 출력
    ap.add_argument('--out_ckpt', required=True)              # L ckpt 경로
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lambda_supcon', type=float, default=0.2)
    ap.add_argument('--lambda_ce', type=float, default=0.1)
    ap.add_argument('--anchor_hard_weight', type=float, default=1.15)
    ap.add_argument('--hard_anchor_cap', type=float, default=0.4)
    ap.add_argument('--semi_hard_margin', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cfg/model
    cfg = OmegaConf.load(args.k0_config)
    nm = cfg.get('normalize', {'mean':[0.1307],'std':[0.3081]})
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(nm['mean'], nm['std'])])

    model = ExpertModelJ(cfg).to(device)
    sd0 = torch.load(args.k0_ckpt, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(sd0, strict=False)

    # 데이터
    train_ds = EMNIST('data', split='balanced', train=True,  download=True, transform=tf)
    test_ds  = EMNIST('data', split='balanced', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 클러스터 로드: {gid: [class,...]}
    cc = json.load(open(args.clusters_json))
    raw = cc.get('clusters', cc)
    if isinstance(raw, dict):
        gid_to_classes = {int(g): [int(c) for c in cls_list] for g, cls_list in raw.items()}
    elif isinstance(raw, list):
        gid_to_classes = {g: [int(c) for c in cls_list] for g, cls_list in enumerate(raw)}
    else:
        raise ValueError("clusters_json must be dict{gid:[...]} or list[[...],...]")

    C = cfg.model.num_classes
    cls2gid = torch.full((C,), -1, dtype=torch.long)
    for g, cls_list in gid_to_classes.items():
        for c in cls_list:
            if 0 <= c < C:
                cls2gid[c] = g
            else:
                print(f"[L] WARNING: class id {c} out of range 0..{C-1}")
    cls2gid_t = cls2gid  # already torch

    # 어려운 클래스 집합: 제공되면 사용, 없으면 클러스터 포함 전부
    hard_classes = set(cc.get('hard_classes', sum(gid_to_classes.values(), [])))

    # 옵티마이저: trunk만 업데이트(헤드/게이트는 X)
    for p in model.classifier.parameters(): p.requires_grad = False
    if hasattr(model, "moe"):
        for p in model.moe.parameters(): p.requires_grad = False
    opt = optim.AdamW([p for p in model.trunk.parameters() if p.requires_grad], lr=1e-3)

    supcon = WeightedSupCon(temperature=0.07, semi_hard_margin=args.semi_hard_margin)

    # 증강
    aug1, aug2 = aug_light(), aug_light()

    def build_anchor_w(y):
        w = torch.ones_like(y, dtype=torch.float)
        mask_hard = torch.zeros_like(y, dtype=torch.bool)
        for hc in hard_classes:
            mask_hard |= (y == hc)
        # cap: 배치 내 hard 앵커 비율 제한
        if mask_hard.float().mean().item() > args.hard_anchor_cap:
            # 랜덤하게 일부만 유지
            idx = torch.nonzero(mask_hard, as_tuple=False).squeeze(1)
            keep = int(args.hard_anchor_cap * y.size(0))
            sel = idx[torch.randperm(idx.numel())[:keep]]
            new_mask = torch.zeros_like(mask_hard); new_mask[sel] = True
            mask_hard = new_mask
        w = torch.where(mask_hard, torch.full_like(w, args.anchor_hard_weight), w)
        return w

    for ep in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            x1 = aug1.transforms[0](x.cpu()).to(device) if isinstance(aug1, transforms.Compose) else x
            x2 = aug2.transforms[0](x.cpu()).to(device) if isinstance(aug2, transforms.Compose) else x

            z1 = model.forward_features(x1)
            z2 = model.forward_features(x2)
            feats = torch.stack([z1, z2], dim=1)  # [B,2,32]

            # CE는 원 네트워크 경로로 아주 작게
            logits, _ = model(x)
            ce = F.cross_entropy(logits, y)

            # anchor weight + cluster id
            aw = build_anchor_w(y)
            loss_sup = supcon(feats, y, anchor_w=aw, cluster_id=cls2gid_t)

            loss = args.lambda_supcon * loss_sup + args.lambda_ce * ce
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)

        print(f"[L] Ep{ep:02d} Loss {tot/len(train_ds):.4f}")

    # 저장(트렁크만 바뀜)
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': OmegaConf.to_container(cfg, resolve=True)}, args.out_ckpt)
    print(f"[L] saved → {args.out_ckpt}")

if __name__ == '__main__':
    main()
