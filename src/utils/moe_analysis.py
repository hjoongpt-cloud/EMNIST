import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F


def _entropy(p, eps: float = 1e-9):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)

def _pairwise_cosine(mat: torch.Tensor):  # [E, D]
    m = mat / (mat.norm(dim=1, keepdim=True) + 1e-9)
    return (m @ m.t()).cpu().numpy()


class MoeAnalyzer:
    """
    Collect and summarize MoE routing/oracle statistics per epoch.
    Call order each epoch:
        begin_epoch()
        (for every batch in train loop) accumulate_batch(...), log_grad_norms(...)
        stats = end_epoch()
        finalize_epoch(...)
    """
    def __init__(self, num_experts: int, num_classes: int):
        self.E = num_experts
        self.C = num_classes
        # buffers reset every epoch
        self.p_list = []          # [B,E]
        self.y_list = []          # [B]
        self.z_list = []          # optional gate logits [B,E]
        # oracle / gating metrics
        self.ce_chosen_list = []  # [B]
        self.ce_best_list   = []  # [B]
        self.best_e_list    = []  # [B]
        self.chosen_e_list  = []  # [B]
        # grad norms (router vs experts)
        self.grad_norm_router = []
        self.grad_norm_expert = []

    # -------- epoch hooks ---------
    def begin_epoch(self):
        self.p_list.clear(); self.y_list.clear(); self.z_list.clear()
        self.ce_chosen_list.clear(); self.ce_best_list.clear()
        self.best_e_list.clear(); self.chosen_e_list.clear()
        self.grad_norm_router.clear(); self.grad_norm_expert.clear()

    def accumulate_batch(self, *, p: torch.Tensor, y: torch.Tensor,
                         gate_logits: torch.Tensor = None,
                         ce_chosen: torch.Tensor = None,
                         ce_best: torch.Tensor = None,
                         best_e: torch.Tensor = None,
                         chosen_e: torch.Tensor = None):
        self.p_list.append(p.detach().cpu())
        self.y_list.append(y.detach().cpu())
        if gate_logits is not None:
            self.z_list.append(gate_logits.detach().cpu())
        if ce_chosen is not None:
            self.ce_chosen_list.append(ce_chosen.detach().cpu())
        if ce_best is not None:
            self.ce_best_list.append(ce_best.detach().cpu())
        if best_e is not None:
            self.best_e_list.append(best_e.detach().cpu())
        if chosen_e is not None:
            self.chosen_e_list.append(chosen_e.detach().cpu())

    def log_grad_norms(self, model):
        g_r2 = 0.0; g_e2 = 0.0
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.data.norm().item()
            if 'moe.gate' in n:
                g_r2 += g * g
            elif 'moe.experts' in n:
                g_e2 += g * g
        self.grad_norm_router.append(math.sqrt(g_r2))
        self.grad_norm_expert.append(math.sqrt(g_e2))

    def end_epoch(self):
        # concat
        P = torch.cat(self.p_list, dim=0) if self.p_list else torch.empty(0, self.E)
        Y = torch.cat(self.y_list, dim=0) if self.y_list else torch.empty(0, dtype=torch.long)
        Z = torch.cat(self.z_list, dim=0) if self.z_list else None
        N = P.size(0)

        # usage / entropy / top1 share
        usage = P.mean(dim=0).cpu().numpy() if N>0 else np.zeros(self.E)
        ent = _entropy(P).cpu().numpy() if N>0 else np.zeros(0)
        top1 = P.argmax(dim=1) if N>0 else torch.empty(0, dtype=torch.long)
        top1_counts = torch.bincount(top1, minlength=self.E).float() / (N if N>0 else 1)
        top1_counts = top1_counts.cpu().numpy()

        # class-conditional routing
        pe_c = np.zeros((self.C, self.E), dtype=np.float32)
        pc_e = np.zeros((self.E, self.C), dtype=np.float32)
        if N>0:
            y_np = Y.cpu().numpy()
            p_np = P.cpu().numpy()
            for c in range(self.C):
                mask = (y_np == c)
                if mask.sum()>0:
                    pe_c[c] = p_np[mask].mean(axis=0)
            exp_assign = p_np.argmax(axis=1)
            for i in range(N):
                pc_e[exp_assign[i], y_np[i]] += 1
            pc_e_den = pc_e.sum(axis=1, keepdims=True) + 1e-9
            pc_e = pc_e / pc_e_den

        # oracle stats
        if self.ce_best_list:
            ce_chosen = torch.cat(self.ce_chosen_list).numpy()
            ce_best   = torch.cat(self.ce_best_list).numpy()
            best_e    = torch.cat(self.best_e_list).numpy()
            chosen_e  = torch.cat(self.chosen_e_list).numpy()
            misroute_mask = (best_e != chosen_e)
            gate_regret = (ce_chosen - ce_best)  # per sample
            misroute_rate = float(misroute_mask.mean())
            avg_regret = float(gate_regret.mean())
        else:
            misroute_rate = 0.0; avg_regret = 0.0

        stats = {
            'N': int(N),
            'usage': usage.tolist(),
            'entropy_mean': float(ent.mean()) if ent.size else 0.0,
            'entropy_std': float(ent.std()) if ent.size else 0.0,
            'top1_share': top1_counts.tolist(),
            'P_expert_given_class': pe_c.tolist(),
            'P_class_given_expert': pc_e.tolist(),
            'grad_router_mean': float(np.mean(self.grad_norm_router)) if self.grad_norm_router else 0.0,
            'grad_expert_mean': float(np.mean(self.grad_norm_expert)) if self.grad_norm_expert else 0.0,
            # oracle/gap
            'misroute_rate': misroute_rate,
            'gate_regret_ce': avg_regret
        }
        return stats

    def finalize_epoch(self, out_dir: str, epoch: int, model):
        os.makedirs(out_dir, exist_ok=True)
        # expert weight cosine (last Linear in each expert)
        Ws = []
        for e in range(self.E):
            block = model.moe.experts[e]
            w = None
            for m in reversed(list(block.modules())):
                if isinstance(m, torch.nn.Linear):
                    w = m.weight.detach().cpu().view(-1)
                    break
            if w is None:
                # fallback
                w = torch.randn(8)
            Ws.append(w)
        Ws = torch.stack(Ws, 0)
        cos = _pairwise_cosine(Ws)
        np.save(os.path.join(out_dir, f"cosine_ep{epoch:03d}.npy"), cos)