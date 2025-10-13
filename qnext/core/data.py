import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def _stratified_split_indices(labels, ratios, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    classes = np.unique(labels)
    buckets = {i: [] for i in range(len(ratios))}
    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx); start=0
        for i, r in enumerate(ratios):
            cnt = int(round(r * n)) if i < len(ratios) - 1 else (n - start)
            buckets[i].extend(idx[start:start+cnt]); start += cnt
    return [np.array(sorted(v)).tolist() for v in buckets.values()]

def get_dataloaders(data_root="./data", split_mode="remedial",
                    val_ratio=0.1, early_ratio=0.1,
                    batch_size=512, num_workers=4, seed=42):
    tfm = transforms.ToTensor()
    ds_tr_full = datasets.EMNIST(data_root, split='balanced', train=True, download=True, transform=tfm)
    ds_te      = datasets.EMNIST(data_root, split='balanced', train=False, download=True, transform=tfm)
    labels = np.array([y for _, y in ds_tr_full])

    if split_mode == "remedial":
        train_idx, valmine_idx = _stratified_split_indices(labels, [1 - val_ratio, val_ratio], seed=seed)
        tr_labels = labels[train_idx]
        train_main_idx, train_early_idx = _stratified_split_indices(tr_labels, [1 - early_ratio, early_ratio], seed=seed+1)
        train_main_idx = [train_idx[i] for i in train_main_idx]
        train_early_idx = [train_idx[i] for i in train_early_idx]

        ds_train_main = Subset(ds_tr_full, train_main_idx)
        ds_train_early = Subset(ds_tr_full, train_early_idx)
        ds_val_mine = Subset(ds_tr_full, valmine_idx)

        dl_train = DataLoader(ds_train_main, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_train_early = DataLoader(ds_train_early, batch_size=1024, shuffle=False, num_workers=max(1,num_workers//2), pin_memory=True)
        dl_val_mine = DataLoader(ds_val_mine, batch_size=1024, shuffle=False, num_workers=max(1,num_workers//2), pin_memory=True)
        dl_test  = DataLoader(ds_te, batch_size=1024, shuffle=False, num_workers=max(1,num_workers//2), pin_memory=True)

        splits = {"train_main_idx": train_main_idx, "train_early_idx": train_early_idx, "val_mine_idx": valmine_idx}
    else:
        raise NotImplementedError("Only 'remedial' split is implemented")

    Path("outputs/splits").mkdir(parents=True, exist_ok=True)
    with open("outputs/splits/split_remedial.json", "w") as f:
        json.dump(splits, f)

    return dl_train, dl_train_early, dl_val_mine, dl_test
