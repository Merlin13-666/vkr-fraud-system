from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ThresholdPolicy:
    t_review: float
    t_deny: float
    max_fpr_deny: float
    max_review_share: float


def _safe_clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, t: float) -> Dict[str, int]:
    y_true = y_true.astype(int)
    y_hat = (y_score >= t).astype(int)

    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rates_from_confusion(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "fpr": float(fpr)}


def find_threshold_by_max_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    """
    Find the highest threshold t such that FPR(t) <= max_fpr.
    We scan unique scores (descending). CPU-safe for 100k-500k rows.
    """
    y_true = y_true.astype(int)
    y_score = _safe_clip(y_score)

    # Sort by score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    n_neg = int((y_sorted == 0).sum())
    if n_neg == 0:
        return 1.0

    fp = 0
    tn = n_neg

    # We move threshold from +inf downwards: include more predicted positives
    # Track at score boundaries
    best_t = 1.0

    i = 0
    while i < len(s_sorted):
        t = s_sorted[i]

        # include all with score == t as positive
        j = i
        while j < len(s_sorted) and s_sorted[j] == t:
            if y_sorted[j] == 0:
                fp += 1
                tn -= 1
            j += 1

        cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if cur_fpr <= max_fpr:
            best_t = float(t)
            # since we're descending, lower thresholds will only increase FPR,
            # so keep the last valid (lowest) OR we want the highest threshold?
            # For deny we want strict: choose highest threshold meeting constraint -> update only once.
            # But scanning descending, first time FPR <= max_fpr is always true (fp=0). We need highest threshold
            # that still meets constraint, i.e. keep updating while condition holds.
        else:
            # once exceeded, further lowering threshold will not reduce FPR
            break

        i = j

    return float(best_t)


def find_threshold_for_review_share(y_score: np.ndarray, t_deny: float, max_review_share: float) -> float:
    """
    Choose t_review such that share(REVIEW) <= max_review_share, given fixed t_deny.
    REVIEW zone is [t_review, t_deny). So we pick t_review as a quantile of scores below t_deny.
    """
    y_score = _safe_clip(y_score)
    if not (0.0 < max_review_share < 1.0):
        raise ValueError("max_review_share must be in (0,1)")

    below = y_score[y_score < t_deny]
    if len(below) == 0:
        return float(max(0.0, t_deny - 1e-6))

    # We want P(t_review <= score < t_deny) <= max_review_share
    # Equivalent: t_review is (1 - max_review_share) quantile of all scores,
    # but restricted below deny makes it more stable.
    q = 1.0 - max_review_share
    t_review = float(np.quantile(below, q))
    t_review = min(t_review, float(t_deny) - 1e-6)
    t_review = max(t_review, 0.0)
    return float(t_review)


def assign_zone(y_score: np.ndarray, t_review: float, t_deny: float) -> np.ndarray:
    """
    Returns array of strings: 'allow' / 'review' / 'deny'
    """
    y_score = _safe_clip(y_score)
    zones = np.full(shape=len(y_score), fill_value="allow", dtype=object)
    zones[y_score >= t_review] = "review"
    zones[y_score >= t_deny] = "deny"
    return zones


def build_decision_table(y_true: np.ndarray, y_score: np.ndarray, t_review: float, t_deny: float) -> pd.DataFrame:
    """
    Build table with per-zone stats and also binary metrics for deny threshold.
    """
    y_true = y_true.astype(int)
    y_score = _safe_clip(y_score)

    zones = assign_zone(y_score, t_review=t_review, t_deny=t_deny)

    rows = []
    total = len(y_true)

    for zone in ["allow", "review", "deny"]:
        m = zones == zone
        n = int(m.sum())
        if n == 0:
            rows.append({
                "zone": zone,
                "n": 0,
                "share": 0.0,
                "fraud_rate": 0.0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            })
            continue

        y_z = y_true[m]
        s_z = y_score[m]

        fraud_rate = float(y_z.mean())
        # For zone-specific confusion, we interpret "positive" as being in that zone or higher?
        # Here we keep zone-specific fraud rate and counts only.
        rows.append({
            "zone": zone,
            "n": n,
            "share": n / total,
            "fraud_rate": fraud_rate,
            "score_min": float(s_z.min()),
            "score_max": float(s_z.max()),
        })

    df = pd.DataFrame(rows)

    # Deny threshold metrics (binary decision: deny vs not deny)
    cm = confusion_at_threshold(y_true, y_score, t=t_deny)
    rates = rates_from_confusion(cm)
    df_metrics = pd.DataFrame([{
        "t_review": float(t_review),
        "t_deny": float(t_deny),
        **cm,
        **rates,
    }])

    return df, df_metrics


def fit_policy_on_val(y_val: np.ndarray, p_val: np.ndarray, max_fpr_deny: float, max_review_share: float) -> ThresholdPolicy:
    t_deny = find_threshold_by_max_fpr(y_val, p_val, max_fpr=max_fpr_deny)
    t_review = find_threshold_for_review_share(p_val, t_deny=t_deny, max_review_share=max_review_share)
    return ThresholdPolicy(
        t_review=float(t_review),
        t_deny=float(t_deny),
        max_fpr_deny=float(max_fpr_deny),
        max_review_share=float(max_review_share),
    )

def zone_shares_df(zones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: decision_zones_*.csv df with columns zone,n,share,fraud_rate,...
    Output: simplified df zone,share (for plotting)
    """
    out = zones_df[["zone", "share"]].copy()
    out["share"] = out["share"].astype(float)
    return out


def estimate_cost_by_zones(
    df_pred: pd.DataFrame,
    proba_col: str,
    t_review: float,
    t_deny: float,
    c_fp_deny: float = 50.0,
    c_fn_allow: float = 200.0,
    c_review: float = 2.0,
) -> dict:
    """
    df_pred must contain: target, proba_col
    Computes total and per-transaction expected realized cost under zone decisions.
    """
    y = df_pred["target"].to_numpy().astype(int)
    p = df_pred[proba_col].to_numpy().astype(float)

    zones = assign_zone(p, t_review=t_review, t_deny=t_deny)

    cost = np.zeros(len(y), dtype=float)

    # deny: cost only when legit (false positive deny)
    deny = zones == "deny"
    cost[deny & (y == 0)] = c_fp_deny

    # review: always costs manual review
    review = zones == "review"
    cost[review] = c_review

    # allow: cost only when fraud (missed fraud)
    allow = zones == "allow"
    cost[allow & (y == 1)] = c_fn_allow

    return {
        "c_fp_deny": float(c_fp_deny),
        "c_fn_allow": float(c_fn_allow),
        "c_review": float(c_review),
        "total_cost": float(cost.sum()),
        "avg_cost_per_tx": float(cost.mean()),
        "cost_by_zone": {
            "deny": float(cost[deny].sum()),
            "review": float(cost[review].sum()),
            "allow": float(cost[allow].sum()),
        },
        "share_by_zone": {
            "deny": float(deny.mean()),
            "review": float(review.mean()),
            "allow": float(allow.mean()),
        }
    }