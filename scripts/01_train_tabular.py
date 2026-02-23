from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
import yaml

from fraud_system.features.tabular import (
    TabularSpec,
    get_feature_cols,
    detect_feature_types,
    build_xy,
    sanity_check_types,
    refine_numeric_columns,
)
from fraud_system.models.tabular_lgbm import (
    LGBMConfig,
    fit_with_early_stopping,
    save_model,
    predict_proba,
)
from fraud_system.evaluation.metrics import pr_auc, binary_logloss, pr_curve_points, roc_auc
from fraud_system.evaluation.plots import plot_pr_curve


# -----------------------
# Utils
# -----------------------

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _calc_metrics(y_true, y_score) -> Dict[str, float]:
    y_true = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true
    return {
        "logloss": binary_logloss(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
    }


def _make_pred_df(df: pd.DataFrame, spec: TabularSpec, y, p, score_col: str = "p_tabular") -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": df[spec.id_col].astype("int64"),
        "time": df[spec.time_col].astype("int64"),
        "target": y.astype("int8"),
        score_col: p.astype("float64"),
    })


# -----------------------
# Main
# -----------------------

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

    # -------- Feature selection / typing
    feature_cols = get_feature_cols(df_train, spec)
    num_cols, cat_cols = detect_feature_types(df_train, feature_cols)

    print("[A3] Example cat cols:", cat_cols[:20])
    print(f"[A3] Train rows={len(df_train)}, Val rows={len(df_val)}, Test rows={len(df_test)}")
    print(f"[A3] Features={len(feature_cols)}, Num cols={len(num_cols)}, Cat cols={len(cat_cols)}")

    # -------- Build X/y
    X_train, y_train = build_xy(df_train, spec, feature_cols)
    X_val, y_val = build_xy(df_val, spec, feature_cols)
    X_test, y_test = build_xy(df_test, spec, feature_cols)

    # Second pass: move "dirty numeric" to categorical
    num_cols, cat_cols = refine_numeric_columns(X_train, num_cols, cat_cols)
    print(f"[A3] Refined: Num cols={len(num_cols)}, Cat cols={len(cat_cols)}")

    sanity_check_types(X_train, num_cols, cat_cols)

    # -------- Train model (logloss + early stopping)
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
    best_iter = int(best_iter) if best_iter is not None else None
    print(f"[A3] best_iteration={best_iter}")

    # -------- Predict: TRAIN/VAL/TEST
    p_train = predict_proba(model, X_train)
    p_val = predict_proba(model, X_val)
    p_test = predict_proba(model, X_test)

    train_metrics = _calc_metrics(y_train, p_train)
    val_metrics = _calc_metrics(y_val, p_val)
    test_metrics = _calc_metrics(y_test, p_test)

    print(f"[A3] TRAIN: logloss={train_metrics['logloss']:.6f}, pr_auc={train_metrics['pr_auc']:.6f}, roc_auc={train_metrics['roc_auc']:.6f}")
    print(f"[A3] VAL:   logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A3] TEST:  logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # -------- Save artifacts
    model_path = Path("artifacts/tabular/model.pkl")
    _ensure_dir(model_path.parent)
    save_model(model, str(model_path))
    print(f"[A3] Saved model: {model_path}")

    eval_dir = Path("artifacts/evaluation")
    _ensure_dir(eval_dir)

    # Save predictions (now includes TRAIN for fusion)
    train_pred = _make_pred_df(df_train, spec, y_train, p_train, score_col="p_tabular")
    val_pred = _make_pred_df(df_val, spec, y_val, p_val, score_col="p_tabular")
    test_pred = _make_pred_df(df_test, spec, y_test, p_test, score_col="p_tabular")

    train_pred_path = eval_dir / "train_pred_tabular.parquet"
    val_pred_path = eval_dir / "val_pred_tabular.parquet"
    test_pred_path = eval_dir / "test_pred_tabular.parquet"

    train_pred.to_parquet(train_pred_path, index=False)
    val_pred.to_parquet(val_pred_path, index=False)
    test_pred.to_parquet(test_pred_path, index=False)

    print(f"[A3] Saved preds: {train_pred_path}")
    print(f"[A3] Saved preds: {val_pred_path}")
    print(f"[A3] Saved preds: {test_pred_path}")

    # Save PR curve (VAL)
    precision, recall, _ = pr_curve_points(y_val.to_numpy(), p_val)
    pr_path = eval_dir / "pr_curve_tabular.png"
    plot_pr_curve(precision, recall, str(pr_path), "PR Curve (Tabular, VAL)")
    print(f"[A3] Saved PR curve: {pr_path}")

    # Save metrics (now includes train)
    metrics = {
        "best_iteration": best_iter,
        "model_params": {**model_params, "best_iteration": best_iter},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
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

    metrics_path = eval_dir / "tabular_metrics.json"
    _save_json(metrics_path, metrics)
    print(f"[A3] Saved metrics: {metrics_path}")

    # Save used columns (reproducibility)
    spec_path = eval_dir / "tabular_feature_spec.json"
    _save_json(spec_path, {"feature_cols": feature_cols, "num_cols": num_cols, "cat_cols": cat_cols})
    print(f"[A3] Saved feature spec: {spec_path}")

    print("[A3] Done.")


if __name__ == "__main__":
    main()