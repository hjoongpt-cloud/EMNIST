import os, json, argparse
import numpy as np
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.expert_model_k import ExpertModelJ
from src.training.train_expert_analyze import train_epoch, evaluate
import torch.nn as nn

NUM_EXPERTS = 4

class GatingNet(nn.Module):
    """게이트를 K1과 동일 구조로 다시 정의해 MoE의 gate를 교체."""
    def __init__(self, input_dim, num_experts=NUM_EXPERTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, num_experts)
        )
    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def set_requires_grad(m, flag: bool):
    if m is None: return
    for p in m.parameters(): p.requires_grad = flag


def init_experts_from_single(model_multi: ExpertModelJ, k0_ckpt_path: str, noise_std: float = 0.0):
    """K0에서 학습된 단일 expert를 모든 expert에 복제. 선택적으로 가우시안 노이즈 추가."""
    ck = torch.load(k0_ckpt_path, map_location='cpu', weights_only=False)['model']
    # 단일 expert 키만 추출
    single = {k.replace('moe.experts.0.', ''): v for k, v in ck.items() if k.startswith('moe.experts.0.')}

    with torch.no_grad():
        for e, expert in enumerate(model_multi.moe.experts):
            sd = expert.state_dict()
            for k in sd.keys():
                if k in single:
                    sd[k].copy_(single[k])
            expert.load_state_dict(sd)
            if noise_std > 0:
                for p in expert.parameters():
                    p.add_(noise_std * torch.randn_like(p))

    print(f"[K2] experts initialized from K0 (noise_std={noise_std})")
def build_frozen_gate_teacher(D, num_experts, ckpt_path, device):
    teacher = GatingNet(D, num_experts)
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    teacher.load_state_dict(sd, strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher.to(device)

def load_gate_into_model(model: ExpertModelJ, gate_ckpt: str):
    ck = torch.load(gate_ckpt, map_location='cpu', weights_only=False)
    D = model.D
    num_experts = model.moe.num_experts

    # 기존 MoE gate를 K1 구조로 교체 후 로드
    new_gate = GatingNet(D, num_experts=num_experts)
    missing, unexpected = new_gate.load_state_dict(ck['state_dict'], strict=False)
    print(f"[K2] load gate missing={missing}, unexpected={unexpected}")
    model.moe.gate = new_gate.to(next(model.parameters()).device)
    
def load_k0_into_model_excluding_gate(model, k0_ckpt):
    ck0 = torch.load(k0_ckpt, map_location='cpu', weights_only=False)['model']
    # moe.gate.* 키들 제거해서 shape mismatch 회피
    ck0_no_gate = {k: v for k, v in ck0.items() if not k.startswith('moe.gate.')}
    missing, unexpected = model.load_state_dict(ck0_no_gate, strict=False)
    print(f"[K2] load K0 into model(excl gate): missing={missing}, unexpected={unexpected}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-o', '--out_dir', required=True)
    ap.add_argument('--k0_ckpt', required=True)
    ap.add_argument('--gate_ckpt', required=True)
    ap.add_argument('--freeze_trunk_epochs', type=int, default=None)
    ap.add_argument('--freeze_gate_epochs', type=int, default=None)
    ap.add_argument('--expert_noise_std', type=float, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 스케줄 기본값: config→cli override
    freeze_trunk_E = cfg.get('schedule', {}).get('freeze_trunk_epochs', 0) if args.freeze_trunk_epochs is None else args.freeze_trunk_epochs
    freeze_gate_E  = cfg.get('schedule', {}).get('freeze_gate_epochs',  0) if args.freeze_gate_epochs  is None else args.freeze_gate_epochs
    noise_std = cfg.get('init', {}).get('add_expert_noise_std', 0.0) if args.expert_noise_std is None else args.expert_noise_std

    # 데이터
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

        # 모델 생성 & 초기화
        model = ExpertModelJ(cfg).to(device)
        # 전문가 복제 초기화
        load_k0_into_model_excluding_gate(model, args.k0_ckpt)

        # 2) 그 다음, expert0 → 나머지 전문가로 복제 (필요 시 노이즈)
        init_experts_from_single(model, args.k0_ckpt, noise_std=noise_std)

        # 3) 게이트 교체/로드
        load_gate_into_model(model, args.gate_ckpt)
        model.moe.gate_temperature = 3.0      # 초반 납작
        model.moe.gate_mix_alpha   = 0.2      # teacher:student = 0.8:0.2
        model.moe.stopgrad_gate    = True     # 초반 보호막 on
        # attach teacher
        model.moe.gate_teacher = build_frozen_gate_teacher(model.D, model.moe.num_experts, args.gate_ckpt, device)
        # Freeze 스케줄 초기 설정(초기 epoch 동안 고정)
        set_requires_grad(model.trunk, False if freeze_trunk_E > 0 else True)
        set_requires_grad(model.moe.gate, False if freeze_gate_E > 0 else True)

        gate_params    = list(model.moe.gate.parameters())
        expert_params  = [p for e in model.moe.experts for p in e.parameters()]
        trunk_params   = list(model.trunk.parameters())
        head_params    = list(model.classifier.parameters())

        optimizer = optim.AdamW([
            {'params': expert_params, 'lr': cfg.opt.lr},
            {'params': head_params,   'lr': cfg.opt.lr},
            {'params': gate_params,   'lr': cfg.opt.lr * 0.1},    # gate LR 10배 낮게
            {'params': trunk_params,  'lr': cfg.opt.lr * 0.3},    # trunk 낮게(선택)
        ])
        STOPGRAD_E = 8         # stopgrad 유지 epoch
        TEMP_START, TEMP_END = 3.0, 1.0
        ALPHA_START, ALPHA_END = 0.2, 1.0
        
        model.eval()
        with torch.no_grad():
            x,_ = next(iter(train_loader)); x = x.to(device)
            feat = model.forward_features(x)

            # 학생 게이트
            p_student = torch.softmax(model.moe.gate(feat), dim=1)
            print("[K2] student p.mean:", p_student.mean(0).cpu().numpy())
            print("[K2] student hard:", p_student.argmax(1).cpu().bincount(minlength=NUM_EXPERTS).numpy())

            # teacher 게이트 (K1 ckpt)
            teacher = build_frozen_gate_teacher(model.D, model.moe.num_experts, args.gate_ckpt, device)
            p_teacher = torch.softmax(teacher(feat), dim=1)
            print("[K2] teacher p.mean:", p_teacher.mean(0).cpu().numpy())
            print("[K2] teacher hard:", p_teacher.argmax(1).cpu().bincount(minlength=NUM_EXPERTS).numpy())
        model.train()
        
        for ep in range(1, cfg.opt.epochs+1):
            t = min(1.0, max(0.0, (ep-1)/(cfg.opt.epochs-1)))
            model.moe.gate_temperature = TEMP_START + (TEMP_END - TEMP_START)*t
            model.moe.gate_mix_alpha   = ALPHA_START + (ALPHA_END - ALPHA_START)*t
            model.moe.stopgrad_gate    = (ep <= STOPGRAD_E)            

            # epoch 경과에 따른 unfreeze
            if ep == freeze_trunk_E + 1:
                set_requires_grad(model.trunk, True)
                # 옵티마이저 재생성 (파라미터 그룹 유지)
                gate_params    = list(model.moe.gate.parameters())
                expert_params  = [p for e in model.moe.experts for p in e.parameters()]
                trunk_params   = list(model.trunk.parameters())
                head_params    = list(model.classifier.parameters())
                optimizer = optim.AdamW([
                    {'params': expert_params, 'lr': cfg.opt.lr},
                    {'params': head_params,   'lr': cfg.opt.lr},
                    {'params': gate_params,   'lr': cfg.opt.lr * 0.1},
                    {'params': trunk_params,  'lr': cfg.opt.lr * 0.3},
                ])

            ce, lb, ent, acc = train_epoch(model, train_loader, optimizer, device, cfg, analyzer=None, epoch=ep)
            print(f"[K2][seed {seed}] Ep{ep:02d} CE {ce:.4f} LB {lb:.4f} ENT {ent:.4f} ACC {acc:.4f}")

        test_acc = evaluate(model, test_loader, device)
        print(f"[K2][seed {seed}] TestAcc {test_acc:.4f}")

        torch.save({
            'model': model.state_dict(),
            'cfg': OmegaConf.to_container(cfg, resolve=True),
            'seed': seed,
        }, os.path.join(ckpt_dir, 'k2_moe.pt'))
        summary.append({'seed': seed, 'test_acc': float(test_acc)})

    with open(os.path.join(args.out_dir, 'summary_k2.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()