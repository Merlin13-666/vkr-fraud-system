# scripts/19_ablation_gnn.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _run(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (code={p.returncode}): {' '.join(cmd)}")


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="A5.2: GNN ablation grid (lite)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    # grid toggles
    parser.add_argument("--neighbors-a", type=str, default="15,10")
    parser.add_argument("--neighbors-b", type=str, default="10,5", help="Optional alt fanout for 2 runs")
    parser.add_argument("--use-neighbors-b", action="store_true", help="If set, include neighbors-b for layers=2 only")

    parser.add_argument("--embed-dims", type=str, default="32,64", help="e.g. '32,64' or '64,128'")
    parser.add_argument("--layers", type=str, default="1,2,3", help="e.g. '1,2,3'")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Keep fixed for compact grid")

    # training budget
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    embed_dims = [int(x.strip()) for x in args.embed_dims.split(",") if x.strip()]
    layers_list = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    if not embed_dims or not layers_list:
        raise ValueError("Empty grid: check --embed-dims and --layers")

    runs: List[Tuple[int, int, str]] = []
    for L in layers_list:
        for E in embed_dims:
            runs.append((L, E, args.neighbors_a))

    # optional 2 extra runs to show neighbor influence (compact)
    if args.use_neighbors_b:
        # add 2 runs for L=2 (or if 2 not in grid - take first layer)
        L_ref = 2 if 2 in layers_list else layers_list[0]
        for E in embed_dims[:2]:
            runs.append((L_ref, E, args.neighbors_b))

    results = []
    for (L, E, nei) in runs:
        tag = "ablation"
        suffix = f"{tag}__L{L}_H{args.hidden_dim}_E{E}_N{nei.replace(',', '-')}"
        cmd = [
            sys.executable, "-m", "scripts.04_train_gnn",
            "--device", args.device,
            "--seed", str(args.seed),
            "--num-layers", str(L),
            "--hidden-dim", str(args.hidden_dim),
            "--embed-dim", str(E),
            "--neighbors", nei,
            "--max-epochs", str(args.max_epochs),
            "--patience", str(args.patience),
            "--tag", tag,
            "--out-suffix", f"L{L}_H{args.hidden_dim}_E{E}_N{nei.replace(',', '-')}",
        ]
        _run(cmd)

        metrics_path = eval_dir / f"gnn_metrics__L{L}_H{args.hidden_dim}_E{E}_N{nei.replace(',', '-')}.json"
        if not metrics_path.exists():
            # fallback: in case you changed naming
            raise FileNotFoundError(f"Expected metrics not found: {metrics_path}")

        j = _read_json(metrics_path)
        results.append({
            "run": j.get("run_suffix"),
            "num_layers": j["config"]["num_layers"],
            "hidden_dim": j["config"]["hidden_dim"],
            "embed_dim": j["config"]["embed_dim"],
            "neighbors": ",".join(str(x) for x in (j["sampler"].get("neighbors_list") or [])),
            "val_pr_auc": j["val"]["pr_auc"],
            "val_logloss": j["val"]["logloss"],
            "val_roc_auc": j["val"]["roc_auc"],
            "test_pr_auc": j["test"]["pr_auc"],
            "test_logloss": j["test"]["logloss"],
            "test_roc_auc": j["test"]["roc_auc"],
            "metrics_path": str(metrics_path).replace("\\", "/"),
        })

    df = pd.DataFrame(results)
    out_csv = eval_dir / "gnn_ablation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[A5.2] Saved: {out_csv}")

    # plot: PR-AUC vs layers (use best embed per layer)
    # for clarity: pick max val_pr_auc for each layer
    best = df.sort_values(["num_layers", "val_pr_auc"], ascending=[True, False]).groupby("num_layers", as_index=False).head(1)
    best = best.sort_values("num_layers")

    plt.figure()
    plt.plot(best["num_layers"], best["val_pr_auc"], marker="o")
    plt.xlabel("num_layers")
    plt.ylabel("VAL PR-AUC")
    plt.title("GNN ablation: PR-AUC vs layers (best embed per layer)")
    out_png = eval_dir / "gnn_ablation_pr_auc_vs_layers.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[A5.2] Saved: {out_png}")

    # also print compact table to console
    print("\n[A5.2] Top runs (by VAL PR-AUC):")
    top = df.sort_values("val_pr_auc", ascending=False).head(10)[
        ["num_layers", "embed_dim", "neighbors", "val_pr_auc", "val_logloss", "test_pr_auc", "test_logloss"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

