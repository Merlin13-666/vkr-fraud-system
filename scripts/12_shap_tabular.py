from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed")

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/")


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path}")


def _prepare_raw_features(df: pd.DataFrame, feature_spec: Dict[str, Any]) -> pd.DataFrame:
    feature_cols = feature_spec["feature_cols"]
    missing = set(feature_cols) - set(df.columns)
    for col in missing:
        df[col] = np.nan
    return df[feature_cols].copy()


def _unwrap_pipeline(model: Any) -> Tuple[Any, Any]:
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

    return []


def _transform(pre: Any, X_raw: pd.DataFrame) -> Tuple[Any, List[str]]:
    if pre is None:
        return X_raw, list(X_raw.columns)

    Xt = pre.transform(X_raw)
    names = _get_feature_names_after_preprocess(pre, X_raw)
    if names:
        return Xt, names

    m = Xt.shape[1]
    return Xt, [f"f_{i}" for i in range(m)]


def _force_dense(Xt: Any, max_cells: int, debug: Dict[str, Any]) -> np.ndarray:
    """
    Force dense for SHAP stability.
    Convert sparse->dense only if n*m <= max_cells.
    """
    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(Xt)
    except Exception:
        is_sparse = False

    if hasattr(Xt, "shape"):
        n, m = Xt.shape
    else:
        arr = np.asarray(Xt)
        n, m = arr.shape

    debug["xt_is_sparse"] = bool(is_sparse)
    debug["xt_shape"] = [int(n), int(m)]
    debug["max_cells"] = int(max_cells)
    debug["n_cells"] = int(n * m)

    if n * m > max_cells:
        raise RuntimeError(
            f"Too large to densify safely for SHAP: n*m={n*m} > max_cells={max_cells}. "
            f"Reduce --sample or increase max_cells."
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


def main() -> None:
    parser = argparse.ArgumentParser(description="B1: SHAP global interpretation for tabular LightGBM pipeline")
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--model", default="artifacts/tabular/model.pkl")
    parser.add_argument("--feature-spec", default="artifacts/evaluation/tabular_feature_spec.json")
    parser.add_argument("--sample", type=int, default=4000, help="SHAP sample size (dense for stability)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-display", type=int, default=30)
    parser.add_argument("--no-beeswarm", action="store_true")
    parser.add_argument("--max-cells", type=int, default=10_000_000, help="Max n*m allowed for densify")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    spec_path = Path(args.feature_spec)

    out_assets = Path("reports") / "assets"
    out_tables = Path("reports") / "tables"
    _ensure_dir(out_assets)
    _ensure_dir(out_tables)

    print(f"[B1.1] Loading data: {data_path}")
    df = _load_df(data_path)

    print(f"[B1.1] Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"[B1.1] Loading feature spec: {spec_path}")
    feature_spec = _read_json(spec_path)

    X_raw = _prepare_raw_features(df, feature_spec)

    n_total = len(X_raw)
    n_sample = min(int(args.sample), n_total)
    if n_sample < n_total:
        X_raw = X_raw.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    else:
        X_raw = X_raw.reset_index(drop=True)

    pre, est = _unwrap_pipeline(model)
    Xt, feat_names = _transform(pre, X_raw)

    print(f"[B1.1] SHAP sample: {len(X_raw)} rows")
    print(f"[B1.1] Transformed features: {len(feat_names)}")

    debug: Dict[str, Any] = {
        "x_type_before_dense": str(type(Xt)),
    }

    # Force dense (this is the key fix)
    Xt_dense = _force_dense(Xt, max_cells=int(args.max_cells), debug=debug)
    debug["x_type_after_dense"] = str(type(Xt_dense))
    debug["x_dtype_after_dense"] = str(Xt_dense.dtype)

    # TreeExplainer on final estimator
    explainer = shap.TreeExplainer(est)

    # Use shap_values (stable for TreeExplainer)
    sv_raw = explainer.shap_values(Xt_dense)
    shap_values = _normalize_shap_to_pos_class(sv_raw)

    debug["shap_values_shape"] = list(np.asarray(shap_values).shape)
    debug_path = out_tables / "shap_debug.json"
    _write_json(debug_path, debug)

    if shap_values.ndim != 2:
        raise RuntimeError(f"Unexpected normalized SHAP shape: {shap_values.shape} (see {debug_path})")

    if shap_values.shape[1] != len(feat_names):
        raise RuntimeError(
            f"Feature dim mismatch: shap={shap_values.shape[1]} vs names={len(feat_names)} (see {debug_path})"
        )

    mean_abs = np.mean(np.abs(shap_values), axis=0).astype(float)

    imp = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False)

    top_csv = out_tables / "shap_top_features.csv"
    imp.to_csv(top_csv, index=False)
    print(f"[B1.1] Saved: {_norm(top_csv)}")

    bar_png = out_assets / "shap_summary_bar.png"
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=None,
        feature_names=feat_names,
        plot_type="bar",
        show=False,
        max_display=int(args.max_display),
    )
    plt.tight_layout()
    plt.savefig(bar_png, dpi=150)
    plt.close()
    print(f"[B1.1] Saved: {_norm(bar_png)}")

    bees_png: Optional[Path] = None
    if not args.no_beeswarm:
        bees_png = out_assets / "shap_summary_beeswarm.png"
        plt.figure()
        shap.summary_plot(
            shap_values,
            Xt_dense,
            feature_names=feat_names,
            show=False,
            max_display=int(args.max_display),
        )
        plt.tight_layout()
        plt.savefig(bees_png, dpi=150)
        plt.close()
        print(f"[B1.1] Saved: {_norm(bees_png)}")
    else:
        print("[B1.1] Beeswarm skipped (--no-beeswarm).")

    meta = {
        "data": _norm(data_path),
        "model": _norm(model_path),
        "feature_spec": _norm(spec_path),
        "n_total": int(n_total),
        "n_used": int(len(X_raw)),
        "transformed_dim": int(len(feat_names)),
        "saved": {
            "top_features_csv": _norm(top_csv),
            "bar_png": _norm(bar_png),
            "beeswarm_png": _norm(bees_png) if bees_png else None,
            "debug_json": _norm(debug_path),
        },
    }
    meta_path = out_tables / "shap_meta.json"
    _write_json(meta_path, meta)
    print(f"[B1.1] Saved: {_norm(meta_path)}")

    print("[B1.1] Done.")


if __name__ == "__main__":
    main()