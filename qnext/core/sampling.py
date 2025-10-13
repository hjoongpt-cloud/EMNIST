import torch
from torch.utils.data import TensorDataset

@torch.no_grad()
def mine_low_margin(model, dl, device, margin_thresh=0.5, per_class_cap=64, hard_cap=4096, num_classes=47):
    model.eval()
    per_class = {c: [] for c in range(num_classes)}
    for x,y in dl:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        logit_y = logits.gather(1, y.view(-1,1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.view(-1,1), False)
        rival = logits.masked_fill(~mask, float('-inf')).amax(dim=1)
        m_true = logit_y - rival
        pred = logits.argmax(dim=1)
        pick = (pred != y) | (m_true < margin_thresh)
        idx = torch.nonzero(pick, as_tuple=False).squeeze(1)
        for i in idx.tolist():
            c = int(y[i].item())
            per_class[c].append((float(m_true[i].item()), x[i].detach().cpu(), c))
    xs, ys = [], []
    for c in range(num_classes):
        buf = per_class[c]
        if not buf: continue
        buf.sort(key=lambda t: t[0])
        for (_, xi, yi) in buf[:per_class_cap]:
            xs.append(xi); ys.append(yi)
    if not xs: return None
    if len(xs) > hard_cap:
        xs = xs[:hard_cap]; ys = ys[:hard_cap]
    X = torch.stack(xs); Y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(X, Y)
