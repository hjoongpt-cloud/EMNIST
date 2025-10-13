# src_m/tools/proto_sweep.py
import argparse, subprocess, json, os, sys
from datetime import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--proto_json", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=3.0)
    ap.add_argument("--gate_delta", type=float, default=0.4)
    ap.add_argument("--grid", type=str, default="0.6,0.75,0.9,0.95,1.0")
    ap.add_argument("--learn_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_top_p", type=float, default=1.0)
    ap.add_argument("--work_dir", type=str, default="sweep_runs")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    grid = [float(t) for t in args.grid.split(",")]

    rows=[]
    for p in grid:
        cmd = [
            sys.executable, "src_m/tools/proto_fuse.py",
            "--dump_dir", args.dump_dir,
            "--proto_json", args.proto_json,
            "--alpha", str(args.alpha),
            "--beta",  str(args.beta),
            "--gate_delta", str(args.gate_delta),
            "--top_p", str(p),
            "--learn_epochs", str(args.learn_epochs),
            "--lr", str(args.lr),
            "--train_top_p", str(args.train_top_p),
        ]
        print("[sweep] running:", " ".join(cmd))
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(out.stdout)
        # 간단히 stdout에서 acc 라인 파싱
        fused = None; base=None
        for line in out.stdout.splitlines():
            if "base_acc=" in line and "fused_acc=" in line:
                # n=18800 | base_acc=86.38% | fused_acc=86.32% | Δ=-0.05
                try:
                    items = line.split("|")
                    base = float(items[1].split("=")[1].strip().rstrip("%"))
                    fused= float(items[2].split("=")[1].strip().rstrip("%"))
                except Exception: pass
        rows.append((p, base, fused))

    # 요약
    print("\n=== Sweep summary ===")
    for p, b, f in rows:
        print(f"top_p={p:.2f}  base={b:.2f}%  fused={f:.2f}%  Δ={(f-b) if (b is not None and f is not None) else float('nan'):+.2f}")

    with open(os.path.join(args.work_dir, f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump([dict(top_p=p, base=b, fused=f) for (p,b,f) in rows], f, indent=2)

if __name__ == "__main__":
    main()
