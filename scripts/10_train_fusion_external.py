from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)


def _merge_preds_external(tab: pd.DataFrame, gnn: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Merge external predictions by transaction_id + target.

    Expected:
      tab: transaction_id, p_tabular, target
      gnn: transaction_id, p_gnn_external_cal (preferred) OR p_gnn_external, target
    """
    tab = tab[["transaction_id", "p_tabular", "target"]].copy()

    gnn_col = None
    if "p_gnn_external_cal" in gnn.columns:
        gnn_col = "p_gnn_external_cal"
    elif "p_gnn_external" in gnn.columns:
        gnn_col = "p_gnn_external"
    else:
        raise ValueError(
            f"[A10] GNN external file for {split_name} must contain p_gnn_external_cal or p_gnn_external"
        )

    gnn = gnn[["transaction_id", gnn_col, "target"]].copy()
    gnn = gnn.rename(columns={gnn_col: "p_gnn"})

    # strong sanity: drop dup transaction_id (shouldn't happen, but protects pipeline)
    tab = tab.drop_duplicates("transaction_id")
    gnn = gnn.drop_duplicates("transaction_id")

    m = tab.merge(gnn, on=["transaction_id", "target"], how="inner")
    if len(m) == 0:
        raise ValueError(
            f"[A10] Empty merge for {split_name}. "
            f"Check that tabular and gnn external predictions were generated on the same split."
        )
    return m


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_score = _clip(y_score)
    return {
        "logloss": float(binary_logloss(y_true, y_score)),
        "pr_auc": float(pr_auc(y_true, y_score)),
        "roc_auc": float(roc_auc(y_true, y_score)),
    }


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    fusion_dir = Path("artifacts/fusion")
    _ensure_dir(fusion_dir)

    # -------------------------
    # Load predictions (external)
    # -------------------------
    tab_val = _read_parquet(eval_dir / "val_pred_tabular.parquet")
    tab_test = _read_parquet(eval_dir / "test_pred_tabular.parquet")

    # prefer calibrated external gnn preds if available
    gnn_val_path_cal = eval_dir / "val_pred_gnn_external_calibrated.parquet"
    gnn_test_path_cal = eval_dir / "test_pred_gnn_external_calibrated.parquet"
    gnn_val_path = eval_dir / "val_pred_gnn_external.parquet"
    gnn_test_path = eval_dir / "test_pred_gnn_external.parquet"

    gnn_val = _read_parquet(gnn_val_path_cal if gnn_val_path_cal.exists() else gnn_val_path)
    gnn_test = _read_parquet(gnn_test_path_cal if gnn_test_path_cal.exists() else gnn_test_path)

    # -------------------------
    # Merge (external val/test)
    # -------------------------
    val_m = _merge_preds_external(tab_val, gnn_val, "VAL_EXTERNAL")
    test_m = _merge_preds_external(tab_test, gnn_test, "TEST_EXTERNAL")

    # -------------------------
    # Train fusion on external VAL
    # -------------------------
    y_val = val_m["target"].to_numpy(dtype=np.int8)
    p_tab_val = _clip(val_m["p_tabular"].to_numpy())
    p_gnn_val = _clip(val_m["p_gnn"].to_numpy())

    cfg = FusionConfig(C=1.0, max_iter=400, random_state=42)
    fusion = FusionModel(cfg).fit(p_tab_val, p_gnn_val, y_val)

    # Predict on VAL and TEST
    p_f_val = _clip(fusion.predict_proba(p_tab_val, p_gnn_val))

    y_test = test_m["target"].to_numpy(dtype=np.int8)
    p_f_test = _clip(
        fusion.predict_proba(
            _clip(test_m["p_tabular"].to_numpy()),
            _clip(test_m["p_gnn"].to_numpy()),
        )
    )

    # Metrics
    val_metrics = _metrics(y_val, p_f_val)
    test_metrics = _metrics(y_test, p_f_test)

    print(
        f"[A10] VAL_EXTERNAL:  logloss={val_metrics['logloss']:.6f}, "
        f"pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}"
    )
    print(
        f"[A10] TEST_EXTERNAL: logloss={test_metrics['logloss']:.6f}, "
        f"pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}"
    )

    # -------------------------
    # Save model
    # -------------------------
    model_path = fusion_dir / "fusion_external.pkl"
    fusion.save(model_path)
    print(f"[A10] Saved model: {model_path}")

    # -------------------------
    # Save metrics json
    # -------------------------
    # weights for interpretation
    w = fusion.model.coef_[0]
    b = fusion.model.intercept_[0]

    out: Dict[str, Any] = {
        "mode": "external fusion; trained on external VAL, evaluated on external TEST",

        # --- Добавлено для аудита ВКР ---
        "trained_on": "external_val",
        "used_gnn": "calibrated" if "calibrated" in str(
            (gnn_val_path_cal if gnn_val_path_cal.exists() else gnn_val_path)
        ) else "raw",
        # ---------------------------------

        "val_external": val_metrics,
        "test_external": test_metrics,

        "config": cfg.__dict__,
        "n_val_external": int(len(val_m)),
        "n_test_external": int(len(test_m)),

        "fusion_weights": {
            "w_tabular": float(w[0]),
            "w_gnn": float(w[1]),
            "bias": float(b),
        },

        "inputs": {
            "tab_val": "artifacts/evaluation/val_pred_tabular.parquet",
            "tab_test": "artifacts/evaluation/test_pred_tabular.parquet",
            "gnn_val": str((gnn_val_path_cal if gnn_val_path_cal.exists() else gnn_val_path)).replace("\\", "/"),
            "gnn_test": str((gnn_test_path_cal if gnn_test_path_cal.exists() else gnn_test_path)).replace("\\", "/"),
        },
    }

    metrics_path = eval_dir / "fusion_metrics_external.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[A10] Saved metrics: {metrics_path}")

    # -------------------------
    # PR curve on VAL_EXTERNAL
    # -------------------------
    precision, recall, _ = pr_curve_points(y_val, p_f_val)
    pr_path = eval_dir / "pr_curve_fusion_external.png"
    plot_pr_curve(precision, recall, str(pr_path), title="PR Curve (Fusion, VAL_EXTERNAL)")
    print(f"[A10] Saved PR curve: {pr_path}")

    # Optional: save preds (useful for audit/future steps)
    def _save_df(m: pd.DataFrame, p: np.ndarray, out_path: Path) -> None:
        df_out = m[["transaction_id", "target"]].copy()
        df_out["p_fusion_external"] = p.astype("float64")
        df_out.to_parquet(out_path, index=False)
        print(f"[A10] Saved preds: {out_path}")

    _save_df(val_m, p_f_val, eval_dir / "val_pred_fusion_external.parquet")
    _save_df(test_m, p_f_test, eval_dir / "test_pred_fusion_external.parquet")

    print(f"[A10] Fusion weights: w_tabular={w[0]:.4f}, w_gnn={w[1]:.4f}, bias={b:.4f}")
    print("[A10] Done.")


if __name__ == "__main__":
    main()