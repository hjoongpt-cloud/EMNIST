import torch

@torch.no_grad()
def indices_from_mask(mask_tensor, thresh=0.5):
    # mask_tensor: [K,1,1,1] or [K]; return indices to keep
    m = mask_tensor.view(-1).float()
    keep = torch.nonzero(m > thresh, as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        keep = torch.topk(m, k=1).indices
    return keep

@torch.no_grad()
def hard_compact_frontend(frontend, keep_idx):
    """Slice conv1/proj_down by keep_idx and rebuild layers with K_eff channels.
    Returns K_eff."""
    device = frontend.conv1.weight.device
    K_eff = int(keep_idx.numel())
    import torch.nn as nn
    conv1_new = nn.Conv2d(1, K_eff, kernel_size=9, padding=4, bias=False).to(device)
    proj_new  = nn.Conv2d(K_eff, frontend.D, kernel_size=1, bias=True).to(device)
    with torch.no_grad():
        conv1_new.weight.copy_(frontend.conv1.weight[keep_idx])
        proj_new.weight.copy_(frontend.proj_down.weight[:, keep_idx])
        if frontend.proj_down.bias is not None:
            proj_new.bias.copy_(frontend.proj_down.bias)
    frontend.conv1 = conv1_new
    frontend.proj_down = proj_new
    frontend.register_buffer("channel_mask", torch.ones(K_eff,1,1,1, device=device))
    frontend.K = K_eff
    return K_eff

def export_frontend(frontend, out_dir):
    from pathlib import Path
    import numpy as np
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(str(Path(out_dir)/"conv1.npy"), frontend.conv1.weight.detach().cpu().numpy())
    np.savez(str(Path(out_dir)/"proj_down.npz"),
             weight=frontend.proj_down.weight.detach().cpu().numpy(),
             bias=(frontend.proj_down.bias.detach().cpu().numpy() if frontend.proj_down.bias is not None else None))
