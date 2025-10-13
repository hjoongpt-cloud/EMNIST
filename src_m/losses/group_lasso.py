
import torch

def group_lasso_penalty_from_head(head, lambda_gl: float = 1e-3):
    return head.group_lasso_penalty(lambda_gl=lambda_gl)

@torch.no_grad()
def prune_head_by_threshold(head, threshold: float):
    return head.prune_by_threshold(threshold)
