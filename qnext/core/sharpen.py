from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

@torch.no_grad()
def _collect_stats(frontend, dl, device, max_batches=400, log_every=20):
    K = frontend.K
    winner = torch.zeros(K, dtype=torch.long)
    runner = torch.zeros(K, dtype=torch.long)
    energy = torch.zeros(K, dtype=torch.float32)
    for i, (x,_) in enumerate(dl):
        if i>=max_batches: break
        x = x.to(device)
        a = frontend.conv1(x) * frontend.channel_mask
        score = torch.abs(a)
        top2 = torch.topk(score, k=2, dim=1)
        win_idx = top2.indices[:,0]
        run_idx = top2.indices[:,1]
        for b in range(x.size(0)):
            wi = win_idx[b].view(-1).cpu()
            ri = run_idx[b].view(-1).cpu()
            for k in wi: winner[int(k)] += 1
            for k in ri: runner[int(k)] += 1
        e = (a*a).sum(dim=(0,2,3)).detach().cpu()
        energy += e
        if (i+1) % max(1,log_every) == 0:
            done = i+1
            print(f"[SHARP] stats {done}/{max_batches} batches", flush=True)
    return winner, runner, energy

@torch.no_grad()
def _kmeans_centers(patches, K, iters=20):
    N,D = patches.shape
    device = patches.device
    idx = torch.randperm(N, device=device)[:K]
    C = patches[idx].clone()
    for _ in range(iters):
        dist = torch.cdist(patches, C, p=2)
        assign = dist.argmin(dim=1)
        for k in range(K):
            mask = (assign==k)
            if mask.any():
                C[k] = patches[mask].mean(dim=0)
    C = F.normalize(C, dim=1)
    return C

@torch.no_grad()
def sharpen_and_prune(frontend, data_root="./data", sample_images=20000,
                      batch_size=512, max_batches=200, winner_thresh=0.005,
                      sim_thresh=0.98, device=None, out_dir="outputs/ae/sharp",
                      log_every=20, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend.eval(); frontend = frontend.to(device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.ToTensor()
    ds = datasets.EMNIST(data_root, split='balanced', train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if verbose: print(f"[SHARP] collecting stats up to {max_batches} batches...", flush=True)
    winner, runner, energy = _collect_stats(frontend, dl, device, max_batches=max_batches, log_every=log_every)
    total_sites = winner.sum().item() if winner.sum().item()>0 else 1
    winner_rate = winner.float() / float(total_sites)
    runner_rate = runner.float() / float(total_sites)

    W = frontend.conv1.weight.detach().cpu()
    patches = W.view(W.size(0), -1)
    patches = torch.cat([patches, patches*0.0+1e-6], dim=0)
    C = _kmeans_centers(patches.to(device), frontend.K, iters=10).cpu()

    Wn = torch.nn.functional.normalize(W.view(W.size(0), -1), dim=1)
    Sim = torch.matmul(Wn, C.t())

    low_use = (winner_rate < winner_thresh)
    low_energy = (energy < energy.mean()*0.2)
    prune = (low_use & low_energy)
    mask = torch.ones_like(frontend.channel_mask)              # (1,K,1,1)
    prune_idx = prune.view(-1).nonzero(as_tuple=False).squeeze(1)
    if prune_idx.numel() > 0:
        mask[:, prune_idx, :, :] = 0.0
    frontend.channel_mask.copy_(mask.to(frontend.channel_mask.device))

    torch.save({
        "winner": winner, "runner": runner,
        "winner_rate": winner_rate, "runner_rate": runner_rate,
        "energy": energy, "sim": Sim,
        "prune_mask": mask.squeeze().cpu(),
    }, str(Path(out_dir)/"sharp_stats.pt"))
    if verbose:
        kept = int((mask.view(-1)>0.5).sum().item())
        print(f"[SHARP] done. keep~{kept}/{frontend.K} (by mask>0.5)", flush=True)
    return winner_rate, runner_rate, Sim
