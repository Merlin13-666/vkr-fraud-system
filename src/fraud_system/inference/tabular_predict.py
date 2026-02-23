from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from fraud_system.evaluation.thresholding import assign_zone


def load_model(path: Path):
    return joblib.load(path)


def load_thresholds(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_spec(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_features(df: pd.DataFrame, feature_spec: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align raw df to feature list used in training.
    feature_spec must contain:
      - feature_cols
      - cat_cols
      - num_cols
    """
    feature_cols = feature_spec["feature_cols"]

    # Ensure all required columns exist
    missing = set(feature_cols) - set(df.columns)
    for col in missing:
        df[col] = np.nan

    X = df[feature_cols].copy()
    return X, df["transaction_id"]


def predict_with_policy(
    df: pd.DataFrame,
    model,
    thresholds: Dict,
    feature_spec: Dict,
) -> Tuple[pd.DataFrame, Dict]:

    X, tx_ids = prepare_features(df, feature_spec)
    X = X.copy()  # гарантируем DataFrame
    p = model.predict_proba(X)[:, 1]

    t_review = thresholds["t_review"]
    t_deny = thresholds["t_deny"]

    decision = assign_zone(p, t_review=t_review, t_deny=t_deny)

    out = pd.DataFrame({
        "transaction_id": tx_ids.astype("int64"),
        "p_tabular": p.astype("float64"),
        "decision": decision,
    })

    summary = {
        "n_transactions": int(len(out)),
        "t_review": float(t_review),
        "t_deny": float(t_deny),
        "share_allow": float((decision == "allow").mean()),
        "share_review": float((decision == "review").mean()),
        "share_deny": float((decision == "deny").mean()),
    }

    return out, summary