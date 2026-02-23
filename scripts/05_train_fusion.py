from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from fraud_system.models.fusion import FusionModel, FusionConfig
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def _merge_preds(tab: pd.DataFrame, gnn: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Merge predictions by transaction_id. We expect:
    tab:  transaction_id, p_tabular, target
    gnn:  transaction_id, p_gnn, target (+ tx_node_id optionally)
    """
    # Keep only required cols (safe)
    tab = tab[["transaction_id", "p_tabular", "target"]].copy()
    gnn_cols = [c for c in ["transaction_id", "p_gnn", "target"] if c in gnn.columns]
    gnn = gnn[gnn_cols].copy()

    m = tab.merge(gnn, on=["transaction_id", "target"], how="inner")
    if len(m) == 0:
        raise ValueError(
            f"[A6] Empty merge for {split_name}. "
            f"Reason: different transaction_id sets. "
            f"Use tabular(train) with gnn(internal splits) for train-only graph mode."
        )
    return m


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "logloss": binary_logloss(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
    }


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    fusion_dir = Path("artifacts/fusion")
    _ensure_dir(fusion_dir)

    # -------------------------
    # Load predictions
    # -------------------------
    tab_train = _read_parquet(eval_dir / "train_pred_tabular.parquet")
    gnn_train = _read_parquet(eval_dir / "train_pred_gnn.parquet")
    gnn_val = _read_parquet(eval_dir / "val_pred_gnn.parquet")
    gnn_test = _read_parquet(eval_dir / "test_pred_gnn.parquet")

    # IMPORTANT:
    # GNN val/test are internal splits inside train-graph,
    # so we must use tabular(train) to align transaction_id.
    train_m = _merge_preds(tab_train, gnn_train, "TRAIN (tabular=train, gnn=train)")
    val_m = _merge_preds(tab_train, gnn_val, "VAL_INTERNAL (tabular=train, gnn=val_gnn)")
    test_m = _merge_preds(tab_train, gnn_test, "TEST_INTERNAL (tabular=train, gnn=test_gnn)")

    # -------------------------
    # Train fusion (stacking)
    # -------------------------
    y_train = train_m["target"].to_numpy()
    p_tab_train = train_m["p_tabular"].to_numpy()
    p_gnn_train = train_m["p_gnn"].to_numpy()

    cfg = FusionConfig(C=1.0, max_iter=200, random_state=42)
    fusion = FusionModel(cfg).fit(p_tab_train, p_gnn_train, y_train)

    # Predict
    p_f_train = fusion.predict_proba(p_tab_train, p_gnn_train)

    y_val = val_m["target"].to_numpy()
    p_f_val = fusion.predict_proba(val_m["p_tabular"].to_numpy(), val_m["p_gnn"].to_numpy())

    y_test = test_m["target"].to_numpy()
    p_f_test = fusion.predict_proba(test_m["p_tabular"].to_numpy(), test_m["p_gnn"].to_numpy())

    # Metrics
    train_metrics = _metrics(y_train, p_f_train)
    val_metrics = _metrics(y_val, p_f_val)
    test_metrics = _metrics(y_test, p_f_test)

    print(f"[A6] TRAIN: logloss={train_metrics['logloss']:.6f}, pr_auc={train_metrics['pr_auc']:.6f}, roc_auc={train_metrics['roc_auc']:.6f}")
    print(f"[A6] VAL_INTERNAL: logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A6] TEST_INTERNAL: logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # -------------------------
    # Save model
    # -------------------------
    model_path = fusion_dir / "fusion.pkl"
    fusion.save(model_path)
    print(f"[A6] Saved model: {model_path}")

    # -------------------------
    # Save predictions
    # -------------------------
    def _save_df(m: pd.DataFrame, p: np.ndarray, out_path: Path) -> None:
        out = m[["transaction_id", "target"]].copy()
        out["p_fusion"] = p.astype("float64")
        out.to_parquet(out_path, index=False)
        print(f"[A6] Saved preds: {out_path}")

    _save_df(train_m, p_f_train, eval_dir / "train_pred_fusion.parquet")
    _save_df(val_m, p_f_val, eval_dir / "val_pred_fusion_internal.parquet")
    _save_df(test_m, p_f_test, eval_dir / "test_pred_fusion_internal.parquet")

    # -------------------------
    # Save metrics json
    # -------------------------
    out = {
        "mode": "train-only graph; internal gnn val/test inside train period",
        "train": train_metrics,
        "val_internal": val_metrics,
        "test_internal": test_metrics,
        "config": cfg.__dict__,
        "n_train": int(len(train_m)),
        "n_val_internal": int(len(val_m)),
        "n_test_internal": int(len(test_m)),
    }
    metrics_path = eval_dir / "fusion_metrics_internal.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[A6] Saved metrics: {metrics_path}")

    # -------------------------
    # PR curve on VAL_INTERNAL
    # -------------------------
    precision, recall, _ = pr_curve_points(y_val, p_f_val)
    pr_path = eval_dir / "pr_curve_fusion_internal.png"
    plot_pr_curve(precision, recall, str(pr_path), title="PR Curve (Fusion, VAL_INTERNAL)")
    print(f"[A6] Saved PR curve: {pr_path}")

    # Print weights for interpretation
    w = fusion.model.coef_[0]
    b = fusion.model.intercept_[0]
    print(f"[A6] Fusion weights: w_tabular={w[0]:.4f}, w_gnn={w[1]:.4f}, bias={b:.4f}")

    print("[A6] Done.")


if __name__ == "__main__":
    main()