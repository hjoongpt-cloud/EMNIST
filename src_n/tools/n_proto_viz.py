# src_m/tools/proto_viz.py
import os, json, argparse, glob
import numpy as np
import matplotlib.pyplot as plt
from src_common.labels import emnist_char, emnist_tag

try:
    from sklearn.decomposition import PCA
    HAS_PCA = True
except Exception:
    HAS_PCA = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# ===================== 공통 유틸 =====================

def build_feature_match_dim(S_slots, XY, target_dim, xy_weight=1.0):
    Dslot = None if S_slots is None else S_slots.shape[-1]
    if target_dim == 2:
        xy = XY / (np.linalg.norm(XY, axis=-1, keepdims=True) + 1e-8)
        return xy
    if S_slots is None:
        raise ValueError("Prototypes expect S or S+XY, but S_slots missing.")
    if target_dim == Dslot:
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        return S
    if target_dim == Dslot + 2:
        S = S_slots / (np.linalg.norm(S_slots, axis=-1, keepdims=True) + 1e-8)
        X = XY       / (np.linalg.norm(XY,       axis=-1, keepdims=True) + 1e-8)
        return np.concatenate([S, X * float(xy_weight)], axis=-1)
    raise ValueError(f"Unsupported target_dim={target_dim} (Dslot={Dslot})")

def load_prototypes(path):
    with open(path, "r") as f: data = json.load(f)
    per_class = data.get("per_class", {})

    C_list, C_cls = [], []
    for c_str, block in per_class.items():
        cid = int(c_str)
        vecs = []
        if isinstance(block, dict):
            mus = block.get("mu") or block.get("center") or block.get("centers")
            if mus is not None:
                arr = np.asarray(mus, np.float32)
                if arr.ndim==1: vecs = [arr]
                else: vecs = [np.asarray(v,np.float32) for v in arr]
        elif isinstance(block, list):
            for p in block:
                mu = p.get("mu") if isinstance(p, dict) else p
                if mu is None: continue
                vecs.append(np.asarray(mu, np.float32).reshape(-1))
        else:
            vecs = [np.asarray(block, np.float32).reshape(-1)]
        for v in vecs:
            C_list.append(v); C_cls.append(cid)
    C = np.stack(C_list, 0) if len(C_list) else np.zeros((0,2), np.float32)
    return C, np.asarray(C_cls, np.int64)

def load_dump_dir(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if not paths: raise FileNotFoundError(dump_dir)

    imgs, S_slots, XY, E, M, Y, PRED, Aup = [], [], [], [], [], [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        imgs.append(z["image"].astype(np.uint8))
        XY.append(z["XY"].astype(np.float32))
        E.append((z["slot_prob"] if "slot_prob" in z else z["energy_norm"]).astype(np.float32))
        M.append(int(z["slot_mask"].shape[0]))
        Y.append(int(z["clazz"]))
        PRED.append(int(z["pred"]) if "pred" in z else int(z["clazz"]))
        Aup.append(z["A_upsampled"].astype(np.float32) if "A_upsampled" in z else None)
        S_slots.append(z["S_slots"].astype(np.float32) if "S_slots" in z else None)

    Mmax = max(M)
    D = 0
    for s in S_slots:
        if s is not None:
            D = max(D, s.shape[1])
    N = len(paths)
    S_np = None
    if D>0: S_np = np.zeros((N,Mmax,D), np.float32)
    XY_np= np.zeros((N,Mmax,2), np.float32)
    E_np = np.zeros((N,Mmax), np.float32)
    A_np = None
    if any(a is not None for a in Aup):
        A_np = np.zeros((N,Mmax,28,28), np.float32)

    for i in range(N):
        Mi = XY[i].shape[0]
        XY_np[i,:Mi]= XY[i]
        E_np[i,:Mi] = E[i]
        if S_np is not None and S_slots[i] is not None:
            S_np[i,:Mi] = S_slots[i]
        if A_np is not None and Aup[i] is not None:
            A_np[i,:Mi] = Aup[i]

    return {
        "paths": paths,
        "image": np.stack(imgs,0),
        "S_slots": S_np,
        "XY": XY_np,
        "E": E_np,
        "Mmax": Mmax,
        "Y": np.asarray(Y, np.int64),
        "pred": np.asarray(PRED, np.int64),
        "Aup": A_np
    }

def pick_top_slot(E_row):
    if E_row.sum() <= 0:
        return int(np.argmax(E_row))
    return int(np.argmax(E_row))

def assign_all(features_top, C):
    if C.shape[0] == 0:
        return np.zeros((features_top.shape[0],0)), np.zeros((features_top.shape[0],), np.int64), np.zeros((features_top.shape[0],), np.float32)
    d2 = ((features_top[:,None,:] - C[None,:,:])**2).sum(-1)  # [N,K]
    nn = np.argmin(d2, axis=1)
    dist = np.sqrt(np.min(d2, axis=1))
    return d2, nn, dist

# ===================== 시각화 =====================

def montage_one(proto_id, data, C, C_cls, nn, dist, top_idx, out_dir, title_prefix, N=16):
    os.makedirs(out_dir, exist_ok=True)
    y = data["Y"]; img = data["image"]; XY = data["XY"]; Aup = data["Aup"]; E = data["E"]
    cls = int(C_cls[proto_id])
    sel_in  = np.where((nn==proto_id) & (y==cls))[0]
    sel_out = np.where((nn==proto_id) & (y!=cls))[0]
    sel_in  = sel_in[np.argsort(dist[sel_in])][:N]
    sel_out = sel_out[np.argsort(dist[sel_out])][:N]

    for tag, S in [("IN", sel_in), ("OUT", sel_out)]:
        if S.size == 0: continue
        cols = min(N, 12); rows = int(np.ceil(S.size/cols))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 3.1*rows))
        axes = np.array(axes).reshape(rows, cols)
        for k, ax in enumerate(axes.ravel()):
            ax.axis("off")
            if k >= S.size: continue
            i = S[k]; m = top_idx[i]
            im = img[i]
            ax.imshow(im, cmap="magma", vmin=0, vmax=255)
            cx = int(np.clip(XY[i,m,0]*28, 0, 27)); cy = int(np.clip(XY[i,m,1]*28, 0, 27))
            ax.scatter([cx],[cy], s=20, marker='o', edgecolor='c', facecolor='none', linewidths=1.5)
            if Aup is not None and Aup[i].sum() > 0:
                ax.imshow(Aup[i,m], cmap="magma", alpha=0.35)
            ax.set_title(f"y={y[i]} d={dist[i]:.2f} e={E[i,m]:.2f}", fontsize=8)
        cls_tag = emnist_tag(cls)  # 예: '26_Q'
        fig.suptitle(f"class {cls_tag} / proto {proto_id} ({tag})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"c{cls_tag}_k{proto_id:04d}_{tag}.png"), dpi=160)
        plt.close(fig)

def plot_energy_heatmap(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    y, E = data["Y"], data["E"]
    Epos = E.copy(); Epos[Epos<0]=0
    s = Epos.sum(axis=1, keepdims=True) + 1e-8
    En = Epos / s
    gmean = En.mean(axis=0)
    order = np.argsort(-gmean)
    En_ord = En[:,order]
    Cn = int(y.max()+1)
    H = np.zeros((Cn, En.shape[1]), np.float32)
    for c in range(Cn):
        idx = np.where(y==c)[0]
        if idx.size: H[c] = En_ord[idx].mean(0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,10))
    plt.imshow(H, aspect='auto', cmap='viridis')
    plt.xlabel("slot (sorted by global usage)"); plt.ylabel("class")
    plt.title("Avg normalized slot_prob per class × slot"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "class_slot_energy_heatmap.png"), dpi=160); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(En[En>0].ravel(), bins=50)
    plt.title("slot_prob (normalized)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "energy_hist.png"), dpi=160); plt.close()

def plot_projection(features_top, C, nn, dist, out_path, use_umap=False):
    if use_umap and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15, metric="euclidean", random_state=0)
        XY = reducer.fit_transform(np.concatenate([features_top, C], 0))
        Zf, Zc = XY[:features_top.shape[0]], XY[features_top.shape[0]:]
    else:
        if not HAS_PCA: return
        pca = PCA(n_components=2, random_state=0)
        XY = pca.fit_transform(np.concatenate([features_top, C], 0))
        Zf, Zc = XY[:features_top.shape[0]], XY[features_top.shape[0]:]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    plt.scatter(Zf[:,0], Zf[:,1], s=5, c=np.clip(dist,0,np.percentile(dist,95)), cmap="viridis", alpha=0.55, label="slots")
    plt.scatter(Zc[:,0], Zc[:,1], s=50, marker="*", c="k", edgecolors="w", linewidths=0.5, label="prototypes")
    plt.legend(); plt.title("Top-slot features vs prototypes (color = NN distance)")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=1.0)
    ap.add_argument("--only_correct", type=int, default=1)
    ap.add_argument("--montage_N", type=int, default=12)
    ap.add_argument("--topk_protos", type=int, default=20)
    ap.add_argument("--use_umap", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_dump_dir(args.dump_dir)
    C, C_cls = load_prototypes(args.proto_json)
    if C.shape[0] == 0:
        print("[viz] No prototypes found.")
        return
    K, d = C.shape

    N, M = data["XY"].shape[:2]
    # top-slot index per sample
    top_idx = np.array([pick_top_slot(data["E"][i]) for i in range(N)], np.int64)

    # filter only-correct
    if args.only_correct:
        keep = (data["Y"] == data["pred"])
        for k in ["image","S_slots","XY","E","Aup","Y","pred"]:
            if data[k] is not None: data[k] = data[k][keep]
        top_idx = top_idx[keep]
        N = int(top_idx.shape[0])

    # feature build to match prototype dim
    feats_all = build_feature_match_dim(
        data["S_slots"], data["XY"], target_dim=d, xy_weight=args.xy_weight
    )
    feats_top = feats_all[np.arange(N), top_idx]  # [N,d]

    # assign
    _, nn, dist = assign_all(feats_top, C)

    # top prototypes by support
    sup = np.bincount(nn, minlength=K)
    sel_proto = np.argsort(-sup)[:args.topk_protos]

    # montages
    mon_dir = os.path.join(args.out_dir, "montage")
    os.makedirs(mon_dir, exist_ok=True)
    for pid in sel_proto:
        if sup[pid] == 0: continue
        montage_one(pid, data, C, C_cls, nn, dist, top_idx, mon_dir,
                    title_prefix=f"class {int(C_cls[pid])}", N=args.montage_N)

    # energy plots
    plot_energy_heatmap(data, os.path.join(args.out_dir, "energy"))

    # projection
    plot_projection(feats_top, C, nn, dist, out_path=os.path.join(args.out_dir, "features_vs_protos.png"), use_umap=bool(args.use_umap))

    print(f"[viz] saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
