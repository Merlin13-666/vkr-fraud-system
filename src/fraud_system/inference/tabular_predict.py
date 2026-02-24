from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import joblib
import numpy as np
import pandas as pd

from fraud_system.evaluation.thresholding import assign_zone


# =========================
# IO helpers
# =========================

def load_model(path: Path):
    return joblib.load(path)


def load_thresholds(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_spec(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Feature alignment
# =========================

def prepare_features(df: pd.DataFrame, feature_spec: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align raw df to feature list used in training.
    feature_spec must contain:
      - feature_cols
    """
    feature_cols = feature_spec["feature_cols"]

    missing = set(feature_cols) - set(df.columns)
    for col in missing:
        df[col] = np.nan

    X = df[feature_cols].copy()
    return X, df["transaction_id"]


# =========================
# SHAP helpers (pipeline-safe)
# =========================

def _unwrap_pipeline(model) -> Tuple[Optional[Any], Any]:
    """
    If model is sklearn Pipeline -> return (preprocessor, estimator).
    Else -> (None, model).
    """
    try:
        from sklearn.pipeline import Pipeline
    except Exception:
        Pipeline = None

    if Pipeline is not None and isinstance(model, Pipeline):
        if len(model.steps) == 0:
            raise ValueError("Empty sklearn Pipeline")
        if len(model.steps) == 1:
            return None, model.steps[-1][1]
        return model[:-1], model.steps[-1][1]

    return None, model


def _get_feature_names_after_preprocess(pre: Any, X_raw: pd.DataFrame) -> List[str]:
    if pre is None:
        return list(X_raw.columns)

    if hasattr(pre, "get_feature_names_out"):
        try:
            names = list(pre.get_feature_names_out())
            return [str(n) for n in names]
        except Exception:
            pass

    # fallback
    return []


def _transform(pre: Any, X_raw: pd.DataFrame) -> Tuple[Any, List[str]]:
    if pre is None:
        return X_raw, list(X_raw.columns)

    Xt = pre.transform(X_raw)
    names = _get_feature_names_after_preprocess(pre, X_raw)
    if names:
        return Xt, names

    # worst-case fallback
    m = Xt.shape[1]
    return Xt, [f"f_{i}" for i in range(m)]


def _force_dense(Xt: Any, max_cells: int) -> np.ndarray:
    """
    Force dense for SHAP stability.
    Convert sparse->dense only if n*m <= max_cells.
    """
    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(Xt)
    except Exception:
        is_sparse = False

    n, m = Xt.shape
    if n * m > max_cells:
        raise RuntimeError(
            f"Too large to densify for SHAP reasons: n*m={n*m} > max_cells={max_cells}. "
            f"Reduce --reasons-max-rows."
        )

    if is_sparse:
        return Xt.toarray().astype(np.float32, copy=False)
    return np.asarray(Xt, dtype=np.float32)


def _normalize_shap_to_pos_class(values: Any) -> np.ndarray:
    """
    Normalize SHAP outputs to shape (n, m) for positive class.

    Handles:
    - list/tuple [class0, class1] -> class1
    - shap.Explanation -> .values
    - ndarray (n, m)
    - ndarray (n, m, 2) -> [:, :, 1]
    - ndarray (n, 2, m) -> [:, 1, :]
    """
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, (list, tuple)):
        if len(values) >= 2:
            values = values[1]
        else:
            values = values[0]

    arr = np.asarray(values)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[1] == 2:
            return arr[:, 1, :]
        raise RuntimeError(f"Unexpected 3D SHAP shape: {arr.shape}")

    raise RuntimeError(f"Unexpected SHAP shape: {arr.shape}")


def _shap_top_reasons(
    model,
    X_raw: pd.DataFrame,
    topk: int = 5,
    max_rows: int = 5000,
    max_cells: int = 15_000_000,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Compute per-row topK SHAP reasons as JSON string.

    Important:
    - model may be sklearn Pipeline (preprocess + estimator).
    - For SHAP stability we transform and force dense on a limited subset of rows.
    - Reasons are returned for transformed feature space (after preprocessing).
    """
    import shap  # optional dependency

    n_total = len(X_raw)
    use_n = min(int(max_rows), n_total)
    X_use = X_raw.iloc[:use_n].copy()

    pre, est = _unwrap_pipeline(model)
    Xt, feat_names = _transform(pre, X_use)

    Xt_dense = _force_dense(Xt, max_cells=max_cells)

    explainer = shap.TreeExplainer(est)
    sv_raw = explainer.shap_values(Xt_dense)
    shap_values = _normalize_shap_to_pos_class(sv_raw)  # (use_n, m)

    if shap_values.ndim != 2:
        raise RuntimeError(f"Unexpected normalized SHAP shape: {shap_values.shape}")

    if shap_values.shape[1] != len(feat_names):
        raise RuntimeError(
            f"Mismatch: shap dim={shap_values.shape[1]} vs feature names={len(feat_names)}"
        )

    # feature values to include in reasons (on transformed features)
    vals = Xt_dense
    cols = feat_names

    out = []
    for i in range(use_n):
        row_sv = shap_values[i]
        idx = np.argsort(np.abs(row_sv))[::-1][:topk]
        reasons = []
        for j in idx:
            v = vals[i, j]
            # JSON-friendly
            if np.isnan(v):
                v_out = None
            else:
                v_out = float(v)
            reasons.append(
                {
                    "feature": cols[j],
                    "value": v_out,
                    "shap": float(row_sv[j]),
                }
            )
        out.append(json.dumps(reasons, ensure_ascii=False))

    reasons_series = pd.Series(out, index=X_use.index, dtype="string")

    info = {
        "n_total": int(n_total),
        "n_used": int(use_n),
        "truncated": bool(use_n < n_total),
        "topk": int(topk),
        "max_rows": int(max_rows),
        "transformed_dim": int(len(cols)),
        "note": "Reasons are computed in transformed feature space (after preprocessing).",
    }
    return reasons_series, info


# =========================
# Main predict with policy
# =========================

def predict_with_policy(
    df: pd.DataFrame,
    model,
    thresholds: Dict[str, Any],
    feature_spec: Dict[str, Any],
    with_reasons: bool = False,
    reasons_topk: int = 5,
    reasons_max_rows: int = 5000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    X_raw, tx_ids = prepare_features(df, feature_spec)
    X_raw = X_raw.copy()

    # Pipeline predict_proba works on raw features
    p = model.predict_proba(X_raw)[:, 1].astype("float64")

    t_review = float(thresholds["t_review"])
    t_deny = float(thresholds["t_deny"])

    decision = assign_zone(p, t_review=t_review, t_deny=t_deny)

    out = pd.DataFrame(
        {
            "transaction_id": tx_ids.astype("int64"),
            "p_tabular": p,
            "decision": decision,
        }
    )

    summary: Dict[str, Any] = {
        "n_transactions": int(len(out)),
        "t_review": float(t_review),
        "t_deny": float(t_deny),
        "share_allow": float((decision == "allow").mean()),
        "share_review": float((decision == "review").mean()),
        "share_deny": float((decision == "deny").mean()),
        "with_reasons": bool(with_reasons),
    }

    if with_reasons:
        reasons_json, info = _shap_top_reasons(
            model=model,
            X_raw=X_raw,
            topk=int(reasons_topk),
            max_rows=int(reasons_max_rows),
            max_cells=15_000_000,
        )
        out["top_reasons"] = None
        out.loc[reasons_json.index, "top_reasons"] = reasons_json.values

        summary["reasons_info"] = info
        if info.get("truncated"):
            summary["warning"] = (
                "Reasons computed only for the first rows due to --reasons-max-rows limit "
                "(see reasons_info)."
            )

    return out, summary