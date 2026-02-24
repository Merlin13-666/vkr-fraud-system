from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from fraud_system.evaluation.metrics import pr_auc, roc_auc, binary_logloss
from fraud_system.evaluation.thresholding import (
    fit_policy_on_val,
    build_decision_table,
    estimate_cost_by_zones,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_pred(path: Path, proba_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing preds file: {path}")

    df = pd.read_parquet(path)
    need = {"transaction_id", "target", proba_col}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}")

    df = df[["transaction_id", "target", proba_col]].copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["target"] = df["target"].astype("int8")
    df[proba_col] = df[proba_col].astype(float)
    return df


def _basic_metrics(df: pd.DataFrame, proba_col: str) -> Dict[str, float]:
    y = df["target"].to_numpy()
    p = df[proba_col].to_numpy()
    return {
        "logloss": float(binary_logloss(y, p)),
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
    }


def _plot_zone_shares(zdf: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["allow", "review", "deny"]
    zdf = zdf.set_index("zone").reindex(order).reset_index()
    shares = zdf["share"].to_numpy()

    plt.figure()
    plt.bar(zdf["zone"], shares)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Share of transactions")
    plt.xlabel("Zone")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _evaluate_one(
    name: str,
    val_path: Path,
    test_path: Path,
    proba_col: str,
    thr_dir: Path,
    eval_dir: Path,
    max_fpr_deny: float,
    max_review_share: float,
    costs: Dict[str, float],
) -> Dict[str, Any]:
    """
    Evaluate one scorer:
      - metrics on VAL/TEST
      - fit thresholds on VAL
      - decision tables on VAL/TEST
      - zone share plots
      - realized cost on TEST
    """
    val_df = _load_pred(val_path, proba_col=proba_col)
    test_df = _load_pred(test_path, proba_col=proba_col)

    val_metrics = _basic_metrics(val_df, proba_col)
    test_metrics = _basic_metrics(test_df, proba_col)

    print(
        f"[A7][{name}] VAL:  logloss={val_metrics['logloss']:.6f}, "
        f"pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}"
    )
    print(
        f"[A7][{name}] TEST: logloss={test_metrics['logloss']:.6f}, "
        f"pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}"
    )

    # Fit thresholds on VAL
    policy = fit_policy_on_val(
        y_val=val_df["target"].to_numpy(),
        p_val=val_df[proba_col].to_numpy(),
        max_fpr_deny=max_fpr_deny,
        max_review_share=max_review_share,
    )

    # Build decision tables
    val_zones, val_bin = build_decision_table(
        y_true=val_df["target"].to_numpy(),
        y_score=val_df[proba_col].to_numpy(),
        t_review=policy.t_review,
        t_deny=policy.t_deny,
    )
    test_zones, test_bin = build_decision_table(
        y_true=test_df["target"].to_numpy(),
        y_score=test_df[proba_col].to_numpy(),
        t_review=policy.t_review,
        t_deny=policy.t_deny,
    )

    # Save threshold policy
    thresholds_path = thr_dir / f"thresholds_{name.lower()}.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(policy.__dict__, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {thresholds_path}")

    # Save decision tables
    val_zones_path = eval_dir / f"decision_zones_{name.lower()}_val.csv"
    test_zones_path = eval_dir / f"decision_zones_{name.lower()}_test.csv"
    val_bin_path = eval_dir / f"decision_binary_{name.lower()}_val.csv"
    test_bin_path = eval_dir / f"decision_binary_{name.lower()}_test.csv"

    val_zones.to_csv(val_zones_path, index=False)
    test_zones.to_csv(test_zones_path, index=False)
    val_bin.to_csv(val_bin_path, index=False)
    test_bin.to_csv(test_bin_path, index=False)

    print(f"[A7] Saved: {val_zones_path}")
    print(f"[A7] Saved: {test_zones_path}")
    print(f"[A7] Saved: {val_bin_path}")
    print(f"[A7] Saved: {test_bin_path}")

    # Zone share plots
    zone_val_png = eval_dir / f"zone_share_{name.lower()}_val.png"
    zone_test_png = eval_dir / f"zone_share_{name.lower()}_test.png"

    _plot_zone_shares(pd.read_csv(val_zones_path), zone_val_png, f"Zone Shares ({name}, VAL)")
    _plot_zone_shares(pd.read_csv(test_zones_path), zone_test_png, f"Zone Shares ({name}, TEST)")

    print(f"[A7] Saved: {zone_val_png}")
    print(f"[A7] Saved: {zone_test_png}")

    # Realized cost on TEST
    cost = estimate_cost_by_zones(
        df_pred=test_df,
        proba_col=proba_col,
        t_review=policy.t_review,
        t_deny=policy.t_deny,
        c_fp_deny=float(costs.get("c_fp_deny", 50.0)),
        c_fn_allow=float(costs.get("c_fn_allow", 200.0)),
        c_review=float(costs.get("c_review", 2.0)),
    )
    cost_path = eval_dir / f"cost_{name.lower()}_test.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {cost_path}")

    return {
        "val": val_metrics,
        "test": test_metrics,
        "threshold_policy": policy.__dict__,
        "paths": {
            "val_pred": str(val_path).replace("\\", "/"),
            "test_pred": str(test_path).replace("\\", "/"),
            "thresholds": str(thresholds_path).replace("\\", "/"),
            "decision_zones_val": str(val_zones_path).replace("\\", "/"),
            "decision_zones_test": str(test_zones_path).replace("\\", "/"),
            "decision_binary_val": str(val_bin_path).replace("\\", "/"),
            "decision_binary_test": str(test_bin_path).replace("\\", "/"),
            "zone_share_val_png": str(zone_val_png).replace("\\", "/"),
            "zone_share_test_png": str(zone_test_png).replace("\\", "/"),
            "cost_test": str(cost_path).replace("\\", "/"),
        },
        "proba_col": proba_col,
        "costs": costs,
        "constraints": {
            "max_fpr_deny": float(max_fpr_deny),
            "max_review_share": float(max_review_share),
        },
    }


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    thr_dir = Path("artifacts/thresholds")
    _ensure_dir(thr_dir)

    # Policy constraints (can tweak later)
    max_fpr_deny = 0.01
    max_review_share = 0.10

    # Cost model (keep same as before)
    costs = {"c_fp_deny": 50.0, "c_fn_allow": 200.0, "c_review": 2.0}

    summary: Dict[str, Any] = {}

    # ==============
    # TABULAR (EXTERNAL)
    # ==============
    summary["tabular"] = _evaluate_one(
        name="TABULAR",
        val_path=eval_dir / "val_pred_tabular.parquet",
        test_path=eval_dir / "test_pred_tabular.parquet",
        proba_col="p_tabular",
        thr_dir=thr_dir,
        eval_dir=eval_dir,
        max_fpr_deny=max_fpr_deny,
        max_review_share=max_review_share,
        costs=costs,
    )

    # ==============
    # FUSION_EXTERNAL (HONEST)
    # ==============
    # created by scripts.10_train_fusion_external
    summary["fusion_external"] = _evaluate_one(
        name="FUSION_EXTERNAL",
        val_path=eval_dir / "val_pred_fusion_external.parquet",
        test_path=eval_dir / "test_pred_fusion_external.parquet",
        proba_col="p_fusion_external",
        thr_dir=thr_dir,
        eval_dir=eval_dir,
        max_fpr_deny=max_fpr_deny,
        max_review_share=max_review_share,
        costs=costs,
    )

    summary_path = eval_dir / "evaluate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {summary_path}")

    print("[A7] Done.")


if __name__ == "__main__":
    main()