from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score


# =========================
# Utils
# =========================

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _fmt(x: Any, nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _safe_series_str(s: pd.Series) -> pd.Series:
    # string dtype + safe NaN handling
    return s.astype("object").where(~s.isna(), other=None).astype("string")


def _plot_pr_curve(y_true: np.ndarray, p: np.ndarray, out_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, p)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(booster: lgb.Booster, feat_names: List[str], out_path: Path, topk: int = 15) -> None:
    imp = booster.feature_importance(importance_type="gain")
    df = pd.DataFrame({"feature": feat_names, "gain": imp})
    df = df.sort_values("gain", ascending=False).head(topk)

    plt.figure()
    plt.barh(df["feature"][::-1], df["gain"][::-1])
    plt.xlabel("Total gain")
    plt.title(f"LightGBM feature importance (top-{topk}, gain)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _calc_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "logloss": float(log_loss(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
    }


# =========================
# Graph artifacts
# =========================

@dataclass
class GraphArtifacts:
    graph_info: Dict[str, Any]
    node_map: pd.DataFrame
    edges: pd.DataFrame


def _load_graph_artifacts() -> GraphArtifacts:
    graph_dir = Path("artifacts/graph")
    gi = _read_json(graph_dir / "graph_info.json") or {}

    node_map = pd.read_parquet(graph_dir / "node_map.parquet")
    edges = pd.read_parquet(graph_dir / "edges.parquet")

    return GraphArtifacts(graph_info=gi, node_map=node_map, edges=edges)


def _entity_cols_from_graph_info(graph_info: Dict[str, Any]) -> Dict[str, List[str]]:
    entity_cols = graph_info.get("entity_cols")
    if not isinstance(entity_cols, dict) or not entity_cols:
        raise RuntimeError("[A6.1] graph_info.json has no 'entity_cols'. Run A4 first.")
    out: Dict[str, List[str]] = {}
    for et, cols in entity_cols.items():
        if isinstance(cols, list) and cols:
            out[str(et)] = [str(c) for c in cols]
    return out


def _build_entity_value(col: str, value: Any) -> Optional[str]:
    if value is None or value is pd.NA:
        return None
    return f"{col}::{value}"


def _compute_entity_degree(edges: pd.DataFrame) -> pd.DataFrame:
    # degree(entity)=count(tx->entity edges)
    deg = (
        edges.groupby(["dst_type", "dst_id"], as_index=False)
        .size()
        .rename(columns={"size": "entity_degree"})
    )
    return deg


# =========================
# Feature engineering (Iteration 2)
# =========================

def _prepare_entity_lookup(ga: GraphArtifacts) -> pd.DataFrame:
    node_map = ga.node_map.copy()
    node_map["entity_type"] = node_map["entity_type"].astype("string")
    node_map["entity_value"] = node_map["entity_value"].astype("string")
    node_map["node_id"] = node_map["node_id"].astype(np.int64)

    edges = ga.edges.copy()
    edges["dst_type"] = edges["dst_type"].astype("string")
    edges["dst_id"] = edges["dst_id"].astype(np.int64)

    deg = _compute_entity_degree(edges)
    deg = deg.rename(columns={"dst_type": "entity_type", "dst_id": "node_id"})

    lookup = node_map.merge(deg, how="left", on=["entity_type", "node_id"], validate="1:1")
    lookup["entity_degree"] = lookup["entity_degree"].fillna(0).astype(np.int64)

    # “rarity” — полезная метрика для антифрода: редкие сущности (маленькая степень) часто важнее
    lookup["entity_rarity"] = 1.0 / (1.0 + lookup["entity_degree"].astype(np.float32))

    return lookup


def _build_graph_metrics_features(
    df: pd.DataFrame,
    entity_cols: Dict[str, List[str]],
    entity_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iteration 2 features:
      - global: known/unknown counts, shares
      - per-type: known_cnt, unknown_cnt, degree stats, rarity stats
      - log1p transforms for stability
    """
    tx_id_col = "transaction_id"
    if tx_id_col not in df.columns:
        raise RuntimeError(f"[A6.1] Missing column '{tx_id_col}'")

    parts: List[pd.DataFrame] = []
    for et, cols in entity_cols.items():
        for c in cols:
            if c not in df.columns:
                continue
            tmp = df[[tx_id_col, c]].copy()
            tmp[c] = _safe_series_str(tmp[c])
            tmp["entity_type"] = et
            tmp["entity_value"] = tmp[c].map(lambda v: _build_entity_value(c, v))
            tmp = tmp.drop(columns=[c])
            tmp = tmp.dropna(subset=["entity_value"])
            parts.append(tmp)

    if not parts:
        raise RuntimeError("[A6.1] No entity columns present in df. Check graph_info.entity_cols and your dataset.")

    neigh = pd.concat(parts, ignore_index=True)

    neigh = neigh.merge(
        entity_lookup[["entity_type", "entity_value", "node_id", "entity_degree", "entity_rarity"]],
        how="left",
        on=["entity_type", "entity_value"],
        validate="m:1",
    )

    neigh["known"] = (~neigh["node_id"].isna()).astype(np.int8)
    neigh["unknown"] = (neigh["node_id"].isna()).astype(np.int8)

    # --- global aggregates
    g = neigh.groupby(tx_id_col, as_index=False).agg(
        neighbors_total=("entity_value", "size"),
        neighbors_known=("known", "sum"),
        neighbors_unknown=("unknown", "sum"),
        uniq_known=("node_id", pd.Series.nunique),
        deg_sum=("entity_degree", "sum"),
        deg_mean=("entity_degree", "mean"),
        deg_max=("entity_degree", "max"),
        rarity_mean=("entity_rarity", "mean"),
        rarity_min=("entity_rarity", "min"),
    )

    # shares
    g["unknown_share"] = (g["neighbors_unknown"] / g["neighbors_total"]).astype(np.float32)

    # --- per-type aggregates
    type_aggs: List[pd.DataFrame] = []
    for et in sorted(entity_cols.keys()):
        sub = neigh[neigh["entity_type"] == et]
        if sub.empty:
            continue
        a = sub.groupby(tx_id_col, as_index=False).agg(
            **{
                f"{et}__neighbors_total": ("entity_value", "size"),
                f"{et}__known": ("known", "sum"),
                f"{et}__unknown": ("unknown", "sum"),
                f"{et}__uniq_known": ("node_id", pd.Series.nunique),
                f"{et}__deg_sum": ("entity_degree", "sum"),
                f"{et}__deg_mean": ("entity_degree", "mean"),
                f"{et}__deg_max": ("entity_degree", "max"),
                f"{et}__rarity_mean": ("entity_rarity", "mean"),
                f"{et}__rarity_min": ("entity_rarity", "min"),
            }
        )
        # per-type unknown share
        a[f"{et}__unknown_share"] = (a[f"{et}__unknown"] / np.maximum(a[f"{et}__neighbors_total"], 1)).astype(np.float32)
        type_aggs.append(a)

    out = g
    for a in type_aggs:
        out = out.merge(a, on=tx_id_col, how="left")

    # fill NaNs in numeric features
    for c in out.columns:
        if c == tx_id_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # stabilize heavy-tail
    for c in out.columns:
        if c == tx_id_col:
            continue
        if c.endswith("__deg_sum") or c.endswith("__deg_max") or c in ["deg_sum", "deg_max", "neighbors_total", "neighbors_known", "neighbors_unknown"]:
            out[f"log1p__{c}"] = np.log1p(out[c].astype(np.float32))

    return out


def _align_features(df_base: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = df_base[["transaction_id"]].merge(feats, on="transaction_id", how="left")
    for c in out.columns:
        if c == "transaction_id":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def main() -> None:
    print("[A6.1][ITER2] Graph-metrics baseline: stronger features + LightGBM")

    data_dir = Path("data/processed")
    eval_dir = Path("artifacts/evaluation")
    graph_dir = Path("artifacts/graph")
    _ensure_dir(eval_dir)

    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "transaction_id" not in df.columns or "target" not in df.columns:
            raise RuntimeError(f"[A6.1] {name} missing required columns: transaction_id/target")

    ga = _load_graph_artifacts()
    entity_cols = _entity_cols_from_graph_info(ga.graph_info)
    print(f"[A6.1] entity_cols: {entity_cols}")

    entity_lookup = _prepare_entity_lookup(ga)

    X_train_raw = _build_graph_metrics_features(train, entity_cols, entity_lookup)
    X_val_raw = _build_graph_metrics_features(val, entity_cols, entity_lookup)
    X_test_raw = _build_graph_metrics_features(test, entity_cols, entity_lookup)

    Xtr = _align_features(train, X_train_raw)
    Xva = _align_features(val, X_val_raw)
    Xte = _align_features(test, X_test_raw)

    feat_cols = [c for c in Xtr.columns if c != "transaction_id"]
    print(f"[A6.1] Features: {len(feat_cols)}")

    y_train = train["target"].to_numpy()
    y_val = val["target"].to_numpy()
    y_test = test["target"].to_numpy()

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = (neg / max(pos, 1.0))

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "scale_pos_weight": float(pos_weight),
    }

    dtrain = lgb.Dataset(Xtr[feat_cols], label=y_train)
    dval = lgb.Dataset(Xva[feat_cols], label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=80),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=3000,
        callbacks=callbacks,
    )

    best_it = int(booster.best_iteration or 0)
    print(f"[A6.1] best_iteration={best_it}")

    p_train = booster.predict(Xtr[feat_cols], num_iteration=best_it)
    p_val = booster.predict(Xva[feat_cols], num_iteration=best_it)
    p_test = booster.predict(Xte[feat_cols], num_iteration=best_it)

    m_train = _calc_metrics(y_train, p_train)
    m_val = _calc_metrics(y_val, p_val)
    m_test = _calc_metrics(y_test, p_test)

    print(f"[A6.1] TRAIN: logloss={_fmt(m_train['logloss'])}, pr_auc={_fmt(m_train['pr_auc'])}, roc_auc={_fmt(m_train['roc_auc'])}")
    print(f"[A6.1] VAL:   logloss={_fmt(m_val['logloss'])}, pr_auc={_fmt(m_val['pr_auc'])}, roc_auc={_fmt(m_val['roc_auc'])}")
    print(f"[A6.1] TEST:  logloss={_fmt(m_test['logloss'])}, pr_auc={_fmt(m_test['pr_auc'])}, roc_auc={_fmt(m_test['roc_auc'])}")

    # Save preds
    def _save_pred(df_base: pd.DataFrame, p: np.ndarray, out_path: Path) -> None:
        out = df_base[["transaction_id", "target"]].copy()
        out["p_graph_metrics"] = p.astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)

    _save_pred(train, p_train, eval_dir / "train_pred_graph_metrics.parquet")
    _save_pred(val, p_val, eval_dir / "val_pred_graph_metrics.parquet")
    _save_pred(test, p_test, eval_dir / "test_pred_graph_metrics.parquet")

    pr_path = eval_dir / "pr_curve_graph_metrics.png"
    _plot_pr_curve(y_val, p_val, pr_path, "PR curve: Graph-metrics baseline (VAL)")

    imp_path = eval_dir / "graph_metrics_feature_importance.png"
    _plot_feature_importance(booster, feat_cols, imp_path, topk=15)

    out_json = {
        "train": m_train,
        "val": m_val,
        "test": m_test,
        "best_iteration": best_it,
        "class_balance": {"pos": int(pos), "neg": int(neg), "pos_weight": float(pos_weight)},
        "features": feat_cols,
        "sources": {
            "graph_info": str((graph_dir / "graph_info.json").as_posix()),
            "node_map": str((graph_dir / "node_map.parquet").as_posix()),
            "edges": str((graph_dir / "edges.parquet").as_posix()),
        },
        "notes": {
            "mode": "inductive_baseline_iter2",
            "val_test_use_train_graph_only": True,
            "meaning": "Control model: uses only graph topology statistics (no GNN).",
        },
    }
    _save_json(eval_dir / "graph_metrics_baseline_metrics.json", out_json)

    # Save feature table for debugging / appendix
    Xtr_out = Xtr.copy()
    Xtr_out["target"] = y_train.astype(np.int8)
    Xtr_out.to_parquet(eval_dir / "graph_metrics_features_train.parquet", index=False)

    print(f"[A6.1] Saved: {eval_dir/'graph_metrics_baseline_metrics.json'}")
    print(f"[A6.1] Saved: {imp_path}")
    print(f"[A6.1] Done.")


if __name__ == "__main__":
    main()