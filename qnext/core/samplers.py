# qnext/core/samplers.py
import torch
import numpy as np
import random

class ReplayBatcher:
    def __init__(self, anchors, replay, batch_size=512, ratio=0.5, drop_last=True, seed=42):
        # <-- 여기만 추가/수정
        a = torch.as_tensor(anchors, dtype=torch.long).view(-1)  # list/np/tensor 모두 OK
        r = torch.as_tensor(replay,  dtype=torch.long).view(-1)
        self.anchors = a.clone()
        self.replay  = r.clone()
        # -------------- 여기까지 --------------

        self.batch_size = int(batch_size)
        self.ratio = float(ratio)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(seed)

    def __iter__(self):
        # 필요 앵커/리플레이 샘플 수
        nA = max(1, int(round(self.batch_size * self.ratio))) if len(self.anchors) > 0 else 0
        nR = self.batch_size - nA

        # 인덱스 셔플
        a_idx = self.anchors[torch.randperm(len(self.anchors))] if len(self.anchors) > 0 else torch.empty(0, dtype=torch.long)
        r_idx = self.replay[torch.randperm(len(self.replay))]   if len(self.replay)  > 0 else torch.empty(0, dtype=torch.long)

        pa = 0; pr = 0
        while True:
            if nA > 0 and pa + nA > len(a_idx):
                # 앵커가 바닥나면 끝(혹은 다시 셔플해서 계속하고 싶으면 여기서 재셔플)
                break
            if pr + nR > len(r_idx):
                # 리플레이 바닥
                break

            if nA > 0:
                ba = a_idx[pa:pa+nA]; pa += nA
            else:
                ba = torch.empty(0, dtype=torch.long)

            br = r_idx[pr:pr+nR]; pr += nR
            batch = torch.cat([ba, br], dim=0)
            if batch.numel() < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self):
        # 대략치: 두 풀에서 몇 배치가 나오는지의 min
        nA = max(1, int(round(self.batch_size * self.ratio))) if len(self.anchors) > 0 else 0
        nR = self.batch_size - nA
        if nA == 0 and nR == 0:
            return 0
        a_batches = (len(self.anchors) // nA) if nA > 0 else float("inf")
        r_batches = (len(self.replay)  // nR) if nR > 0 else float("inf")
        return int(min(a_batches, r_batches))
