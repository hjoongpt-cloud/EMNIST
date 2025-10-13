# qnext/core/confusion.py
import torch

@torch.no_grad()
def build_confusion_pairs(all_true, all_pred, all_margin, top_ratio=0.3):
    """
    all_*: (N,) Tensor(cpu)
    return: list of (c, r) tuples ranked by (freq desc, median_margin asc)
    """
    y = all_true
    p = all_pred
    m = all_margin
    C = {}
    for c, r, mm in zip(y.tolist(), p.tolist(), m.tolist()):
        if c == r: continue
        key = (c, r)
        if key not in C: C[key] = []
        C[key].append(mm)
    stats = []
    for (c, r), margins in C.items():
        arr = torch.tensor(margins)
        stats.append(((c, r), len(margins), float(arr.median())))
    # 빈도 desc, 중앙마진 asc
    stats.sort(key=lambda t: (-t[1], t[2]))
    K = len(set(y.tolist()))
    P = max(1, int(round(K * top_ratio)))
    pairs = [t[0] for t in stats[:P]]
    return pairs, stats
