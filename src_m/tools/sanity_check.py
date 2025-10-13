
import torch, argparse, numpy as np
from src_m.models.trunk_m import TrunkM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot_M", type=int, default=12)
    ap.add_argument("--slot_aggregate", type=str, default="proj32", choices=["proj32","concat","mean"])
    args = ap.parse_args()

    trunk = TrunkM(slot_M=args.slot_M, slot_aggregate=args.slot_aggregate)
    x = torch.randn(8,1,28,28)
    Z, aux = trunk(x)
    A_maps = aux["A_maps"]; S = aux["S_slots"]; head_energy = aux["head_energy"]
    print("Z:", tuple(Z.shape))
    print("A_maps:", tuple(A_maps.shape))
    print("S_slots:", tuple(S.shape))
    print("head_energy:", tuple(head_energy.shape))
    # quick metrics
    B,M,H,W = A_maps.shape
    P = A_maps.view(B,M,-1); P = P / (P.sum(-1, keepdim=True)+1e-8)
    H_cur = -(P * (P+1e-8).log()).sum(-1)  # (B,M)
    dead_ratio = float((P.sum(-1)<1e-6).float().mean())
    print(f"entropy mean={H_cur.mean().item():.3f}  dead_ratio={dead_ratio:.3f}")

if __name__ == "__main__":
    main()
