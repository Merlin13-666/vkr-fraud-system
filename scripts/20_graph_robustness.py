from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_gnn_internal_metrics(metrics_path: Path) -> dict:
    """
    Expected (from 04_train_gnn): {"val": {...}, "test": {...}, ...}
    """
    obj = _read_json(metrics_path)
    return {
        "val_pr_auc": float(obj["val"]["pr_auc"]),
        "val_logloss": float(obj["val"]["logloss"]),
        "test_pr_auc": float(obj["test"]["pr_auc"]),
        "test_logloss": float(obj["test"]["logloss"]),
    }


def _read_gnn_external_metrics(val_path: Path, test_path: Path) -> dict:
    """
    Expected (from 08_predict_gnn_external): json with keys: logloss, pr_auc, roc_auc, etc.
    """
    jv = _read_json(val_path)
    jt = _read_json(test_path)
    return {
        "val_pr_auc": float(jv["pr_auc"]),
        "val_logloss": float(jv["logloss"]),
        "test_pr_auc": float(jt["pr_auc"]),
        "test_logloss": float(jt["logloss"]),
    }


def _read_fusion_external_metrics(path: Path) -> dict:
    """
    Robust reader for fusion external metrics.

    Supported formats:
      - {"val_external": {...}, "test_external": {...}, ...}
      - {"VAL_EXTERNAL": {...}, "TEST_EXTERNAL": {...}, ...}
      - {"val": {...}, "test": {...}, ...}   (fallback)
    """
    obj = _read_json(path)

    def pick(*keys: str):
        for k in keys:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return None

    val_block = pick("val_external", "VAL_EXTERNAL", "val", "VAL")
    test_block = pick("test_external", "TEST_EXTERNAL", "test", "TEST")

    if val_block is None or test_block is None:
        raise KeyError(
            f"Unexpected fusion metrics format in {path}. "
            f"Top-level keys: {list(obj.keys())}"
        )

    return {
        "val_pr_auc": float(val_block.get("pr_auc")),
        "val_logloss": float(val_block.get("logloss")),
        "test_pr_auc": float(test_block.get("pr_auc")),
        "test_logloss": float(test_block.get("logloss")),
    }


def _drop_edges_random(edges: pd.DataFrame, drop_pct: float, seed: int) -> pd.DataFrame:
    """
    Randomly drop drop_pct fraction of rows from edges dataframe.
    Keeps schema as-is.
    """
    if not (0.0 < drop_pct < 1.0):
        raise ValueError(f"drop_pct must be in (0,1), got {drop_pct}")

    n = len(edges)
    keep_n = int(round(n * (1.0 - drop_pct)))
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(n, size=keep_n, replace=False)
    keep_idx.sort()
    return edges.iloc[keep_idx].reset_index(drop=True)


def _agg_name(agg: str) -> str:
    return "mean" if agg == "mean" else "median"


def _agg(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str], agg: str) -> pd.DataFrame:
    if agg == "mean":
        g = df.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
        # flatten columns
        g.columns = [
            f"{a}_{b}" if b else a
            for (a, b) in [(c[0], c[1] if len(c) > 1 else "") for c in g.columns.to_flat_index()]
        ]
        # fix group cols names (they become like drop_pct_)
        for col in group_cols:
            if f"{col}_" in g.columns:
                g = g.rename(columns={f"{col}_": col})
        return g
    else:
        # median without std
        g = df.groupby(group_cols, as_index=False)[metric_cols].median()
        return g


def _errorbar(x, y_mean, y_std, label: str) -> None:
    # If std is all NaN (single seed), draw as normal line
    if y_std is None or (np.isnan(y_std).all()):
        plt.plot(x, y_mean, marker="o", label=label)
    else:
        plt.errorbar(x, y_mean, yerr=y_std, marker="o", capsize=3, label=label)


# -------------------------
# Data
# -------------------------

@dataclass
class RawRow:
    drop_pct: float
    seed: int
    edges_total: int
    edges_kept: int

    # Internal (from 04_train_gnn)
    gnn_int_val_pr_auc: float
    gnn_int_val_logloss: float
    gnn_int_test_pr_auc: float
    gnn_int_test_logloss: float

    # External (from 08_predict_gnn_external)
    gnn_ext_val_pr_auc: float
    gnn_ext_val_logloss: float
    gnn_ext_test_pr_auc: float
    gnn_ext_test_logloss: float

    # Fusion external (from 10_train_fusion_external)
    fusion_ext_val_pr_auc: float
    fusion_ext_val_logloss: float
    fusion_ext_test_pr_auc: float
    fusion_ext_test_logloss: float


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("A4.4 Graph robustness: edge dropout experiments (external + seed-avg)")
    ap.add_argument("--drops", type=str, default="0.1,0.3,0.5,0.75", help="Comma-separated drop pcts, e.g. 0.1,0.3,0.5")
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 1,2,3,4,5")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--neighbors", type=str, default="15,10", help="Neighbors list like 15,10")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=64)

    ap.add_argument("--skip-fusion", action="store_true", help="If set, do not run A10 fusion_external")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    drops = [float(x.strip()) for x in args.drops.split(",") if x.strip()]
    if not drops:
        raise ValueError("No drops provided")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    base_graph_dir = Path("artifacts/graph")
    base_eval_dir = Path("artifacts/evaluation")
    base_fusion_dir = Path("artifacts/fusion")

    # required base artifacts
    edges_src = base_graph_dir / "edges.parquet"
    node_map_src = base_graph_dir / "node_map.parquet"
    graph_info_src = base_graph_dir / "graph_info.json"

    # required tabular preds for fusion
    tab_val_src = base_eval_dir / "val_pred_tabular.parquet"
    tab_test_src = base_eval_dir / "test_pred_tabular.parquet"

    for p in [edges_src, node_map_src, graph_info_src]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p} (run A4 first)")
    if not tab_val_src.exists() or not tab_test_src.exists():
        raise FileNotFoundError("Missing tabular preds. Run A3 first (01_train_tabular).")

    edges = pd.read_parquet(edges_src)
    total_edges = int(len(edges))

    out_eval_root = base_eval_dir / "robustness"
    _ensure_dir(out_eval_root)

    raw_rows: List[RawRow] = []

    for drop_pct in drops:
        tag = f"r{int(round(drop_pct * 100)):02d}"

        for seed in seeds:
            run_tag = f"{tag}_s{seed}"

            graph_dir = base_graph_dir / "robustness" / run_tag
            eval_dir = base_eval_dir / "robustness" / run_tag
            fusion_dir = base_fusion_dir / "robustness" / run_tag

            _ensure_dir(graph_dir)
            _ensure_dir(eval_dir)
            _ensure_dir(fusion_dir)

            print(f"\n[A4.4] === DROP {drop_pct:.2f} ({tag}) | seed={seed} ===")

            # Copy base artifacts except edges (tx_index is produced by 03_make_graph_data)
            _copy_if_exists(node_map_src, graph_dir / "node_map.parquet")
            _copy_if_exists(graph_info_src, graph_dir / "graph_info.json")

            # Drop edges and save
            dropped = _drop_edges_random(edges, drop_pct=drop_pct, seed=seed)
            kept_edges = int(len(dropped))
            dropped.to_parquet(graph_dir / "edges.parquet", index=False)
            print(f"[A4.4] edges: total={total_edges}, kept={kept_edges}")

            # Copy tabular preds for fusion stage
            _copy_if_exists(tab_val_src, eval_dir / "val_pred_tabular.parquet")
            _copy_if_exists(tab_test_src, eval_dir / "test_pred_tabular.parquet")

            # Env overrides for ALL next scripts
            env = os.environ.copy()
            env["FRAUD_GRAPH_DIR"] = str(graph_dir)
            env["FRAUD_EVAL_DIR"] = str(eval_dir)
            env["FRAUD_FUSION_DIR"] = str(fusion_dir)

            py = sys.executable  # portable

            # 1) A4.2 make_graph_data
            _run([py, "-m", "scripts.03_make_graph_data"], env=env)

            # 2) A5 train_gnn
            neighbors_tag = args.neighbors.replace(",", "-")
            out_suffix = f"ROB_{tag}_S{seed}_L{args.num_layers}_H{args.hidden_dim}_E{args.embed_dim}_N{neighbors_tag}"
            _run(
                [
                    py, "-m", "scripts.04_train_gnn",
                    "--device", args.device,
                    "--seed", str(seed),
                    "--num-layers", str(args.num_layers),
                    "--hidden-dim", str(args.hidden_dim),
                    "--embed-dim", str(args.embed_dim),
                    "--neighbors", args.neighbors,
                    "--max-epochs", str(args.max_epochs),
                    "--patience", str(args.patience),
                    "--tag", "robustness",
                    "--out-suffix", out_suffix,
                ],
                env=env,
            )

            # Internal metrics file
            gnn_metrics_path = eval_dir / f"gnn_metrics__{out_suffix}.json"
            if not gnn_metrics_path.exists():
                candidates = sorted(eval_dir.glob("gnn_metrics__*.json"))
                if not candidates:
                    raise FileNotFoundError(f"No gnn_metrics__*.json found in {eval_dir}")
                gnn_metrics_path = candidates[-1]

            gnn_int = _read_gnn_internal_metrics(gnn_metrics_path)

            # 3) A9 external predict (VAL+TEST)
            _run([py, "-m", "scripts.08_predict_gnn_external", "--split", "val"], env=env)
            _run([py, "-m", "scripts.08_predict_gnn_external", "--split", "test"], env=env)

            ext_val_path = eval_dir / "gnn_external_metrics_val.json"
            ext_test_path = eval_dir / "gnn_external_metrics_test.json"
            if not ext_val_path.exists() or not ext_test_path.exists():
                raise FileNotFoundError(f"Missing external metrics in {eval_dir} (run A9 failed?)")

            gnn_ext = _read_gnn_external_metrics(ext_val_path, ext_test_path)

            # 4) A9.3 calibration (kept for pipeline consistency)
            _run([py, "-m", "scripts.09_calibrate_gnn"], env=env)

            # 5) A10 fusion external (optional)
            fusion_ext_val_pr = fusion_ext_val_ll = fusion_ext_test_pr = fusion_ext_test_ll = float("nan")
            if not args.skip_fusion:
                _run([py, "-m", "scripts.10_train_fusion_external"], env=env)

                fusion_metrics_path = eval_dir / "fusion_metrics_external.json"
                if not fusion_metrics_path.exists():
                    raise FileNotFoundError(f"Missing: {fusion_metrics_path}")

                fm = _read_fusion_external_metrics(fusion_metrics_path)
                fusion_ext_val_pr = fm["val_pr_auc"]
                fusion_ext_val_ll = fm["val_logloss"]
                fusion_ext_test_pr = fm["test_pr_auc"]
                fusion_ext_test_ll = fm["test_logloss"]

            raw_rows.append(
                RawRow(
                    drop_pct=drop_pct,
                    seed=seed,
                    edges_total=total_edges,
                    edges_kept=kept_edges,

                    gnn_int_val_pr_auc=gnn_int["val_pr_auc"],
                    gnn_int_val_logloss=gnn_int["val_logloss"],
                    gnn_int_test_pr_auc=gnn_int["test_pr_auc"],
                    gnn_int_test_logloss=gnn_int["test_logloss"],

                    gnn_ext_val_pr_auc=gnn_ext["val_pr_auc"],
                    gnn_ext_val_logloss=gnn_ext["val_logloss"],
                    gnn_ext_test_pr_auc=gnn_ext["test_pr_auc"],
                    gnn_ext_test_logloss=gnn_ext["test_logloss"],

                    fusion_ext_val_pr_auc=fusion_ext_val_pr,
                    fusion_ext_val_logloss=fusion_ext_val_ll,
                    fusion_ext_test_pr_auc=fusion_ext_test_pr,
                    fusion_ext_test_logloss=fusion_ext_test_ll,
                )
            )

    # -------------------------
    # Save RAW + AGG
    # -------------------------
    df_raw = pd.DataFrame([r.__dict__ for r in raw_rows]).sort_values(["drop_pct", "seed"])
    out_raw = out_eval_root / "graph_robustness_raw.csv"
    df_raw.to_csv(out_raw, index=False)
    print(f"\n[A4.4] Saved RAW: {out_raw}")

    metric_cols = [
        "edges_kept",
        "gnn_int_val_pr_auc", "gnn_int_val_logloss", "gnn_int_test_pr_auc", "gnn_int_test_logloss",
        "gnn_ext_val_pr_auc", "gnn_ext_val_logloss", "gnn_ext_test_pr_auc", "gnn_ext_test_logloss",
        "fusion_ext_val_pr_auc", "fusion_ext_val_logloss", "fusion_ext_test_pr_auc", "fusion_ext_test_logloss",
    ]

    # For aggregation, edges_kept is deterministic for a given seed drop? Actually depends on rounding but same per seed.
    # We'll aggregate it too (mean/std) to be safe.
    df_agg = _agg(df_raw, group_cols=["drop_pct"], metric_cols=metric_cols, agg=args.agg).sort_values("drop_pct")

    out_csv = out_eval_root / "graph_robustness.csv"
    df_agg.to_csv(out_csv, index=False)
    print(f"[A4.4] Saved AGG: {out_csv}")

    # -------------------------
    # PLOTS (VKR-main: EXTERNAL)
    # -------------------------
    x = df_agg["drop_pct"] * 100.0

    # External plot (main)
    fig_path_ext = out_eval_root / "graph_robustness_external_pr_auc_vs_drop.png"
    plt.figure()
    _errorbar(
        x,
        df_agg.get("gnn_ext_val_pr_auc_mean", df_agg.get("gnn_ext_val_pr_auc")),
        df_agg.get("gnn_ext_val_pr_auc_std"),
        "GNN external VAL PR-AUC",
    )
    _errorbar(
        x,
        df_agg.get("gnn_ext_test_pr_auc_mean", df_agg.get("gnn_ext_test_pr_auc")),
        df_agg.get("gnn_ext_test_pr_auc_std"),
        "GNN external TEST PR-AUC",
    )

    if not args.skip_fusion:
        _errorbar(
            x,
            df_agg.get("fusion_ext_val_pr_auc_mean", df_agg.get("fusion_ext_val_pr_auc")),
            df_agg.get("fusion_ext_val_pr_auc_std"),
            "FUSION external VAL PR-AUC",
        )
        _errorbar(
            x,
            df_agg.get("fusion_ext_test_pr_auc_mean", df_agg.get("fusion_ext_test_pr_auc")),
            df_agg.get("fusion_ext_test_pr_auc_std"),
            "FUSION external TEST PR-AUC",
        )

    plt.xlabel("Edge drop, %")
    plt.ylabel("PR-AUC")
    plt.title(f"Graph robustness (external): PR-AUC vs edge dropout (agg={_agg_name(args.agg)}, seeds={len(seeds)})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_ext, dpi=150)
    plt.close()
    print(f"[A4.4] Saved: {fig_path_ext}")

    # Internal plot (optional but useful)
    fig_path_int = out_eval_root / "graph_robustness_internal_pr_auc_vs_drop.png"
    plt.figure()
    _errorbar(
        x,
        df_agg.get("gnn_int_val_pr_auc_mean", df_agg.get("gnn_int_val_pr_auc")),
        df_agg.get("gnn_int_val_pr_auc_std"),
        "GNN internal VAL PR-AUC",
    )
    _errorbar(
        x,
        df_agg.get("gnn_int_test_pr_auc_mean", df_agg.get("gnn_int_test_pr_auc")),
        df_agg.get("gnn_int_test_pr_auc_std"),
        "GNN internal TEST PR-AUC",
    )
    plt.xlabel("Edge drop, %")
    plt.ylabel("PR-AUC")
    plt.title(f"Graph robustness (internal): PR-AUC vs edge dropout (agg={_agg_name(args.agg)}, seeds={len(seeds)})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_int, dpi=150)
    plt.close()
    print(f"[A4.4] Saved: {fig_path_int}")

    # -------------------------
    # Console summary (external = main)
    # -------------------------
    print("\n[A4.4] Summary (external, aggregated):")
    show_cols = [
        "drop_pct",
        "gnn_ext_val_pr_auc_mean" if "gnn_ext_val_pr_auc_mean" in df_agg.columns else "gnn_ext_val_pr_auc",
        "gnn_ext_test_pr_auc_mean" if "gnn_ext_test_pr_auc_mean" in df_agg.columns else "gnn_ext_test_pr_auc",
    ]
    if not args.skip_fusion:
        show_cols += [
            "fusion_ext_val_pr_auc_mean" if "fusion_ext_val_pr_auc_mean" in df_agg.columns else "fusion_ext_val_pr_auc",
            "fusion_ext_test_pr_auc_mean" if "fusion_ext_test_pr_auc_mean" in df_agg.columns else "fusion_ext_test_pr_auc",
        ]
    print(df_agg[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()