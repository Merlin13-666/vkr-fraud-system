from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from lightgbm import early_stopping, log_evaluation

from fraud_system.features.tabular import TabularSpec, get_feature_cols, detect_feature_types, build_xy, sanity_check_types, refine_numeric_columns
from fraud_system.models.tabular_lgbm import LGBMConfig, fit_with_early_stopping, save_model, predict_proba, build_tabular_pipeline
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = _read_yaml("configs/base.yaml")
    seed = int(cfg.get("seed", 42))

    processed_path = Path(cfg["data"]["processed_path"])
    train_path = processed_path / "train.parquet"
    val_path = processed_path / "val.parquet"
    test_path = processed_path / "test.parquet"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    spec = TabularSpec()

    feature_cols = get_feature_cols(df_train, spec)
    num_cols, cat_cols = detect_feature_types(df_train, feature_cols)

    print("[A3] Example cat cols:", cat_cols[:20])
    print(f"[A3] Train rows={len(df_train)}, Val rows={len(df_val)}, Test rows={len(df_test)}")
    print(f"[A3] Features={len(feature_cols)}, Num cols={len(num_cols)}, Cat cols={len(cat_cols)}")

    X_train, y_train = build_xy(df_train, spec, feature_cols)
    X_val, y_val = build_xy(df_val, spec, feature_cols)
    X_test, y_test = build_xy(df_test, spec, feature_cols)

    # второй проход: переносим "грязные" числовые в категориальные
    num_cols, cat_cols = refine_numeric_columns(X_train, num_cols, cat_cols)

    print(f"[A3] Refined: Num cols={len(num_cols)}, Cat cols={len(cat_cols)}")

    sanity_check_types(X_train, num_cols, cat_cols)

    model_cfg = LGBMConfig(seed=seed)
    model_params = {
        "n_estimators": model_cfg.n_estimators,
        "learning_rate": model_cfg.learning_rate,
        "num_leaves": model_cfg.num_leaves,
        "max_depth": model_cfg.max_depth,
        "subsample": model_cfg.subsample,
        "colsample_bytree": model_cfg.colsample_bytree,
        "reg_alpha": model_cfg.reg_alpha,
        "reg_lambda": model_cfg.reg_lambda,
        "n_jobs": model_cfg.n_jobs,
        "seed": model_cfg.seed,
        "early_stopping_rounds": 50,
    }

    model = fit_with_early_stopping(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cfg=model_cfg,
        stopping_rounds=50,
        log_period=50,
    )

    best_iter = getattr(model.named_steps["model"], "best_iteration_", None)
    print(f"[A3] best_iteration={best_iter}")

    best_iter = int(best_iter) if best_iter is not None else None

    p_val = predict_proba(model, X_val)
    p_test = predict_proba(model, X_test)

    metrics = {
        "best_iteration": best_iter,
        "model_params": model_params,
        "val": {
            "logloss": binary_logloss(y_val.to_numpy(), p_val),
            "pr_auc": pr_auc(y_val.to_numpy(), p_val),
            "roc_auc": roc_auc(y_val.to_numpy(), p_val),
        },
        "test": {
            "logloss": binary_logloss(y_test.to_numpy(), p_test),
            "pr_auc": pr_auc(y_test.to_numpy(), p_test),
            "roc_auc": roc_auc(y_test.to_numpy(), p_test),
        },
        "meta": {
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_test": int(len(df_test)),
            "n_features": int(len(feature_cols)),
            "n_num": int(len(num_cols)),
            "n_cat": int(len(cat_cols)),
            "seed": int(seed),
        },
    }

    print(f"[A3] VAL:  logloss={metrics['val']['logloss']:.6f}, pr_auc={metrics['val']['pr_auc']:.6f}, roc_auc={metrics['val']['roc_auc']:.6f}")
    print(f"[A3] TEST: logloss={metrics['test']['logloss']:.6f}, pr_auc={metrics['test']['pr_auc']:.6f}, roc_auc={metrics['test']['roc_auc']:.6f}")

    metrics["model_params"]["best_iteration"] = best_iter
    # --- Save artifacts ---
    model_path = Path("artifacts/tabular/model.pkl")
    _ensure_dir(model_path.parent)
    save_model(model, str(model_path))

    eval_dir = Path("artifacts/evaluation")
    _ensure_dir(eval_dir)

    val_pred = pd.DataFrame({
        "transaction_id": df_val[spec.id_col].astype("int64"),
        "time": df_val[spec.time_col].astype("int64"),
        "target": y_val.astype("int8"),
        "p_tabular": p_val.astype("float64"),
    })
    test_pred = pd.DataFrame({
        "transaction_id": df_test[spec.id_col].astype("int64"),
        "time": df_test[spec.time_col].astype("int64"),
        "target": y_test.astype("int8"),
        "p_tabular": p_test.astype("float64"),
    })

    val_pred.to_parquet(eval_dir / "val_pred_tabular.parquet", index=False)
    test_pred.to_parquet(eval_dir / "test_pred_tabular.parquet", index=False)

    print(f"[A3] Saved preds: {eval_dir / 'val_pred_tabular.parquet'}")
    print(f"[A3] Saved preds: {eval_dir / 'test_pred_tabular.parquet'}")

    precision, recall, _ = pr_curve_points(y_val.to_numpy(), p_val)
    plot_pr_curve(precision, recall, str(eval_dir / "pr_curve_tabular.png"), "PR Curve (Tabular, VAL)")

    with open(eval_dir / "tabular_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # фиксируем используемые колонки (для воспроизводимости)
    with open(eval_dir / "tabular_feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(
            {"feature_cols": feature_cols, "num_cols": num_cols, "cat_cols": cat_cols},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[A3] Saved model: {model_path}")
    print(f"[A3] Saved metrics: {eval_dir / 'tabular_metrics.json'}")
    print(f"[A3] Saved PR curve: {eval_dir / 'pr_curve_tabular.png'}")
    print("[A3] Done.")


if __name__ == "__main__":
    main()