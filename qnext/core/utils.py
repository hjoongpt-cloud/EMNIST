import torch, random, numpy as np

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def renorm_conv1(model):
    if hasattr(model, "enc") and hasattr(model.enc, "renorm_conv1"):
        model.enc.renorm_conv1()

@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(dim=1)==y).float().mean().item()

@torch.no_grad()
def confusion_matrix(logits, y, num_classes=47):
    pred = logits.argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=logits.device)
    for t, p in zip(y.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm.cpu()

def percentile(t, q):
    t = t.detach().float().view(-1)
    if t.numel()==0: return 0.0
    k = int(round((q/100.0) * (t.numel()-1)))
    v, _ = torch.kthvalue(t, max(1,k))
    return float(v.item())
