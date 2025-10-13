import torch

def prob_mean(logits_s):
    probs = torch.softmax(logits_s, dim=-1)
    p = probs.mean(dim=1)
    return torch.log(p.clamp_min(1e-12))

def prob_max(logits_s):
    probs = torch.softmax(logits_s, dim=-1)
    p = probs.max(dim=1).values
    return torch.log(p.clamp_min(1e-12))

def logit_mean(logits_s):
    return logits_s.mean(dim=1)

def margin_switch(logits_s, base="logit_mean", alt="prob_max", tau=0.2):
    if base=="logit_mean": base_out = logit_mean(logits_s)
    elif base=="prob_mean": base_out = prob_mean(logits_s)
    elif base=="prob_max": base_out = prob_max(logits_s)
    else: raise ValueError("unknown base")
    if alt=="logit_mean": alt_out = logit_mean(logits_s)
    elif alt=="prob_mean": alt_out = prob_mean(logits_s)
    elif alt=="prob_max": alt_out = prob_max(logits_s)
    else: raise ValueError("unknown alt")
    top2 = torch.topk(base_out, k=2, dim=1).values
    margin = top2[:,0] - top2[:,1]
    use_alt = (margin < tau).float().view(-1,1)
    return alt_out * use_alt + base_out * (1.0 - use_alt)

def aggregate_logits(logits_s, mode="logit_mean", **kwargs):
    if mode=="logit_mean": return logit_mean(logits_s)
    if mode=="prob_mean": return prob_mean(logits_s)
    if mode=="prob_max": return prob_max(logits_s)
    if mode=="margin_switch": return margin_switch(logits_s, **kwargs)
    raise ValueError(f"unknown aggregator {mode}")
