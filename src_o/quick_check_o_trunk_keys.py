# -*- coding: utf-8 -*-
# src_o/quick_check_o_trunk_keys.py
import argparse, sys
import torch
from src_o.o_trunk import OTrunk

def _get_sd_from_ckpt(ckpt):
    """
    다양한 포맷 지원:
      - {'trunk': state_dict, 'head': ...}
      - {'state_dict': {...}}  (여기 안에 'trunk.' prefix일 수도)
      - {'model': {...}}       (여기 안에 'trunk.' / 'module.trunk.' prefix일 수도)
      - 평평한 state_dict 그 자체
    반환: (name, sd)  name은 어떤 키에서 뽑았는지 힌트용
    """
    if isinstance(ckpt, dict):
        # 우선순위 후보
        for k in ("trunk", "state_dict", "model", "net"):
            if k in ckpt and isinstance(ckpt[k], dict) and all(isinstance(v, torch.Tensor) for v in ckpt[k].values()):
                return k, ckpt[k]
        # 평평한 state_dict로 저장된 경우
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return "flat", ckpt
    raise RuntimeError(f"Unsupported ckpt format: type={type(ckpt)}, top-level keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}")

def _carve_trunk_subdict(sd, trunk):
    """
    trunk.state_dict()의 키들과 모양을 기준으로,
    sd에서 가능한 가장 많은 키를 매칭해서 추출.
    prefix 시도 순서: (1) exact (2) 'trunk.' (3) 'module.trunk.'
    그래도 부족하면 suffix 기반 모模匹配(형상 일치) 시도.
    """
    cur = trunk.state_dict()
    def try_prefix(pref):
        inter = {}
        for k in cur.keys():
            src_k = k if pref == "" else f"{pref}{k}"
            if src_k in sd and sd[src_k].shape == cur[k].shape:
                inter[k] = sd[src_k]
        return inter

    # 1) exact
    inter = try_prefix("")
    # 2) trunk.
    if len(inter) < len(cur)//2:
        inter = try_prefix("trunk.")
    # 3) module.trunk.
    if len(inter) < len(cur)//2:
        inter = try_prefix("module.trunk.")

    # 4) 마지막 수단: suffix 매칭(형상 동일한 것만)
    if len(inter) < len(cur)//2:
        suffix_inter = {}
        # 역인덱스(모양 -> 후보키) 만들어서 빠르게 비교
        shape_buckets = {}
        for k_sd, v in sd.items():
            shape_buckets.setdefault(tuple(v.shape), []).append(k_sd)
        for k in cur.keys():
            shp = tuple(cur[k].shape)
            cands = shape_buckets.get(shp, [])
            # 끝이 일치하는 애를 우선
            hit = None
            for cand in cands:
                if cand.endswith(k):
                    hit = cand; break
            if hit is None and cands:
                # 그 외 아무거나 한 개라도
                hit = cands[0]
            if hit is not None:
                suffix_inter[k] = sd[hit]
        if len(suffix_inter) > len(inter):
            inter = suffix_inter

    # missing / mismatched / extra 계산
    loaded = list(inter.keys())
    missing = [k for k in cur.keys() if k not in inter]
    # 여분키: sd에는 있는데 trunk에 없는 키
    extra = [k for k in sd.keys()
             if not (k in cur or k.startswith("trunk.") or k.startswith("module.trunk."))]  # 대충의 힌트

    return inter, missing, extra, len(cur)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conv1_filters", required=True, help="np.load 가능한 9x9x150 필터 npy 경로")
    args = ap.parse_args()

    # 현재 코드의 OTrunk 인스턴스 생성(키 기준)
    trunk = OTrunk(args.conv1_filters)
    cur = trunk.state_dict()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    src_name, sd = _get_sd_from_ckpt(ckpt)
    inter, missing, extra, total = _carve_trunk_subdict(sd, trunk)

    print(f"[ckpt] source key = '{src_name}'  (num_keys_in_that_block={len(sd)})")
    print(f"[trunk] expected keys: {total}")
    print(f"[match] loaded={len(inter)}  missing={len(missing)}  (coverage={100.0*len(inter)/max(1,total):.1f}%)")

    if missing:
        print("\n--- Missing (in ckpt or shape mismatch) ---")
        for k in missing:
            shp_ckpt = sd.get(k, sd.get(f'trunk.{k}', sd.get(f'module.trunk.{k}', None)))
            shp_ckpt = (tuple(shp_ckpt.shape) if isinstance(shp_ckpt, torch.Tensor) else None)
            print(f"  - {k:40s}  cur={tuple(cur[k].shape)}  ckpt={shp_ckpt}")

    # 여분키는 너무 많을 수 있으니 앞 몇 개만
    if extra:
        print("\n--- Extra keys in ckpt (not used by current OTrunk) [showing up to 20] ---")
        for k in extra[:20]:
            print("  -", k)

    # 간단 로드 시뮬
    new_state = cur.copy()
    new_state.update(inter)
    trunk.load_state_dict(new_state, strict=False)
    print("\n[status] Simulated load into OTrunk completed (strict=False).")

if __name__ == "__main__":
    main()
