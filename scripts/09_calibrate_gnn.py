from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_system.evaluation.metrics import pr_auc, roc_auc, binary_logloss
from fraud_system.evaluation.calibration import fit_temperature_on_logits, apply_temperature


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    out_dir = Path("artifacts/graph")
    _ensure_dir(eval_dir)
    _ensure_dir(out_dir)

    val_path = eval_dir / "val_pred_gnn_external.parquet"
    test_path = eval_dir / "test_pred_gnn_external.parquet"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run: python -m scripts.08_predict_gnn_external --split val")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run: python -m scripts.08_predict_gnn_external --split test")

    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    for df, name in [(df_val, "val"), (df_test, "test")]:
        need = {"target", "logit_gnn_external", "p_gnn_external"}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"{name} file missing columns {miss}. Ensure 08_predict_gnn_external saves logits.")

    y_val = df_val["target"].to_numpy(dtype=np.int8)
    z_val = df_val["logit_gnn_external"].to_numpy(dtype=np.float64)

    res = fit_temperature_on_logits(y_val, z_val, T_min=0.5, T_max=10.0, n_grid=250)

    # Apply to val/test
    df_val["p_gnn_external_cal"] = np.clip(apply_temperature(z_val, res.T), 1e-6, 1 - 1e-6)

    y_test = df_test["target"].to_numpy(dtype=np.int8)
    z_test = df_test["logit_gnn_external"].to_numpy(dtype=np.float64)
    df_test["p_gnn_external_cal"] = np.clip(apply_temperature(z_test, res.T), 1e-6, 1 - 1e-6)

    # Metrics
    val_metrics = {
        "logloss": float(binary_logloss(y_val, df_val["p_gnn_external_cal"].to_numpy())),
        "pr_auc": float(pr_auc(y_val, df_val["p_gnn_external_cal"].to_numpy())),
        "roc_auc": float(roc_auc(y_val, df_val["p_gnn_external_cal"].to_numpy())),
    }
    test_metrics = {
        "logloss": float(binary_logloss(y_test, df_test["p_gnn_external_cal"].to_numpy())),
        "pr_auc": float(pr_auc(y_test, df_test["p_gnn_external_cal"].to_numpy())),
        "roc_auc": float(roc_auc(y_test, df_test["p_gnn_external_cal"].to_numpy())),
    }

    cal_art = {
        "method": "temperature_scaling",
        "T": float(res.T),
        "val_logloss_before": float(res.val_logloss_before),
        "val_logloss_after": float(res.val_logloss_after),
        "val_metrics_calibrated": val_metrics,
        "test_metrics_calibrated": test_metrics,
        "source_val": str(val_path).replace("\\", "/"),
        "source_test": str(test_path).replace("\\", "/"),
    }

    out_json = out_dir / "gnn_temperature.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cal_art, f, ensure_ascii=False, indent=2)

    # Save updated parquet
    val_out = eval_dir / "val_pred_gnn_external_calibrated.parquet"
    test_out = eval_dir / "test_pred_gnn_external_calibrated.parquet"
    df_val.to_parquet(val_out, index=False)
    df_test.to_parquet(test_out, index=False)

    print(f"[A9.3] Temperature scaling fitted on external VAL: T={res.T:.4f}")
    print(f"[A9.3] VAL logloss before={res.val_logloss_before:.6f} after={val_metrics['logloss']:.6f}")
    print(f"[A9.3] TEST calibrated: logloss={test_metrics['logloss']:.6f} pr_auc={test_metrics['pr_auc']:.6f} roc_auc={test_metrics['roc_auc']:.6f}")
    print(f"[A9.3] Saved: {out_json}")
    print(f"[A9.3] Saved: {val_out}")
    print(f"[A9.3] Saved: {test_out}")


if __name__ == "__main__":
    main()