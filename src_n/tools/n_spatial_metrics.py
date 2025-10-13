# src_n/tools/n_spatial_metrics.py
import os, json, glob, argparse, numpy as np
import matplotlib.pyplot as plt

def build_feature(S, XY, mode="s+xy", xy_weight=0.2):
    if mode=="s":
        return S / (np.linalg.norm(S, axis=-1, keepdims=True)+1e-8)
    if mode=="xy":
        return XY / (np.linalg.norm(XY, axis=-1, keepdims=True)+1e-8)
    S1 = S /(np.linalg.norm(S, axis=-1, keepdims=True)+1e-8)
    X1 = XY/(np.linalg.norm(XY, axis=-1, keepdims=True)+1e-8)
    return np.concatenate([S1, X1*xy_weight], axis=-1)

def load_protos(path):
    with open(path,"r") as f: data=json.load(f)
    per = data.get("per_class",{})
    C_list, C_cls = [], []
    for c_str, block in per.items():
        c=int(c_str)
        if isinstance(block, dict):
            mus=block.get("mu") or block.get("center") or block.get("centers")
            if mus is None: continue
            arr=np.asarray(mus,np.float32)
            if arr.ndim==1: C_list.append(arr); C_cls.append(c)
            else:
                for v in arr: C_list.append(np.asarray(v,np.float32)); C_cls.append(c)
        elif isinstance(block,list):
            for p in block:
                mu=p.get("mu") if isinstance(p,dict) else p
                if mu is None: continue
                C_list.append(np.asarray(mu,np.float32).reshape(-1)); C_cls.append(c)
        else:
            C_list.append(np.asarray(block,np.float32).reshape(-1)); C_cls.append(c)
    C=np.stack(C_list,0) if len(C_list)>0 else np.zeros((0,2),np.float32)
    return C, np.asarray(C_cls,np.int64)

def entropy(p):
    p = p[p>0]
    if p.size==0: return 0.0
    return float(-(p*np.log(p)).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_mode", default="s+xy", choices=["s","xy","s+xy"])
    ap.add_argument("--xy_weight", type=float, default=0.2)
    ap.add_argument("--slots_topk", type=int, default=3)
    ap.add_argument("--q_iou", type=float, default=0.1)
    ap.add_argument("--grid", type=int, default=4)  # coverage grid
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    C, C_cls = load_protos(args.proto_json)
    K = C.shape[0]
    if K==0:
        print("[metrics] no prototypes."); return

    paths=sorted(glob.glob(os.path.join(args.dump_dir,"*.npz")))
    # per-proto accumulators
    Hgrid=args.grid
    heat = [np.zeros((Hgrid,Hgrid),np.float32) for _ in range(K)]
    xy_list = [[] for _ in range(K)]

    # per-sample diversity
    iou_list=[]; xyvar_list=[]

    for p in paths:
        z=np.load(p, allow_pickle=True)
        XY=z["XY"].astype(np.float32)           # (M,2) in [0,1]
        P =(z["slot_prob"] if "slot_prob" in z else z["energy_norm"]).astype(np.float32)  # (M,)
        S = z["S_slots"].astype(np.float32) if "S_slots" in z else None
        A = z["A_maps"].astype(np.float32)      # (M,H,W)

        if args.feature_mode!="s" and S is None:
            # fall back to xy-only
            feats = XY / (np.linalg.norm(XY,axis=-1,keepdims=True)+1e-8)
        else:
            feats = build_feature(S, XY, args.feature_mode, args.xy_weight)  # (M,d)

        # topk per sample for diversity stats
        idx = np.argsort(-P)[:max(1,args.slots_topk)]
        # IoU (top-q masks)
        H,W=A.shape[1],A.shape[2]
        def topq_mask(a,q):
            flat=a.reshape(-1)
            k=max(1,int(round(q*flat.size)))
            th=np.partition(flat,-k)[-k]
            return (a>=th)
        ms=[topq_mask(A[i], args.q_iou) for i in idx]
        # pairwise IoU
        if len(ms)>=2:
            vals=[]
            for i in range(len(ms)):
                for j in range(i+1,len(ms)):
                    inter=np.logical_and(ms[i],ms[j]).sum()
                    union=np.logical_or(ms[i],ms[j]).sum()
                    vals.append( float(inter)/max(1, int(union)) )
            iou_list.append( np.mean(vals) )
        else:
            iou_list.append(0.0)
        # XY variance
        xyvar_list.append( float(np.var(XY[idx], axis=0).mean()) )

        # assign each slot to NN proto
        d2=((feats[:,None,:]-C[None,:,:])**2).sum(-1)  # (M,K)
        nn=np.argmin(d2,axis=1)
        for m,k in enumerate(nn):
            # accumulate XY into grid
            gx=min(Hgrid-1, max(0,int(XY[m,0]*Hgrid)))
            gy=min(Hgrid-1, max(0,int(XY[m,1]*Hgrid)))
            heat[k][gy,gx]+=1.0
            xy_list[k].append(XY[m])

    # per-proto entropy & coverage
    import csv
    with open(os.path.join(args.out_dir,"proto_spatial_summary.csv"),"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["proto_id","class","count","entropy","entropy_norm","coverage4x4"])
        for k in range(K):
            h=heat[k]
            cnt=h.sum()
            if cnt>0:
                p=(h/cnt).reshape(-1)
                H=entropy(p)
                Hmax=np.log(h.size)
                cov=(h>0).mean()
            else:
                H=0.0; Hmax=np.log(h.size); cov=0.0
            w.writerow([k, int(C_cls[k]), int(cnt), f"{H:.6f}", f"{(H/Hmax if Hmax>0 else 0):.6f}", f"{cov:.6f}"])
            # save heatmap
            plt.figure(figsize=(2.8,2.8))
            plt.imshow(h, cmap="magma")
            plt.title(f"proto {k} (c={int(C_cls[k])})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"proto_{k:04d}_grid.png"), dpi=140)
            plt.close()

    # diversity histograms
    for name, arr in [("iou_mean", np.array(iou_list)), ("xyvar", np.array(xyvar_list))]:
        plt.figure(figsize=(4,3))
        plt.hist(arr, bins=50)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"hist_{name}.png"), dpi=140)
        plt.close()

    print(f"[metrics] saved → {args.out_dir}")
if __name__=="__main__":
    main()
