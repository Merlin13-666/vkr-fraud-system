from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from fraud_system.evaluation.thresholding import estimate_cost_by_zones
from fraud_system.evaluation.metrics import pr_auc, roc_auc, binary_logloss
from fraud_system.evaluation.thresholding import fit_policy_on_val, build_decision_table


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_pred(path: Path, proba_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"transaction_id", "target", proba_col}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}")
    df = df[["transaction_id", "target", proba_col]].copy()
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df["target"] = df["target"].astype("int8")
    return df


def _basic_metrics(df: pd.DataFrame, proba_col: str) -> Dict[str, float]:
    y = df["target"].to_numpy()
    p = df[proba_col].to_numpy()
    return {
        "logloss": float(binary_logloss(y, p)),
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
    }


def main() -> None:
    eval_dir = Path("artifacts/evaluation")
    thr_dir = Path("artifacts/thresholds")
    _ensure_dir(thr_dir)

    # Policy constraints (can tweak later)
    max_fpr_deny = 0.01
    max_review_share = 0.10

    # ==============
    # TABULAR (EXTERNAL)
    # ==============
    val_path = eval_dir / "val_pred_tabular.parquet"
    test_path = eval_dir / "test_pred_tabular.parquet"

    val_df = _load_pred(val_path, proba_col="p_tabular")
    test_df = _load_pred(test_path, proba_col="p_tabular")

    val_metrics = _basic_metrics(val_df, "p_tabular")
    test_metrics = _basic_metrics(test_df, "p_tabular")

    print(f"[A7][TABULAR] VAL:  logloss={val_metrics['logloss']:.6f}, pr_auc={val_metrics['pr_auc']:.6f}, roc_auc={val_metrics['roc_auc']:.6f}")
    print(f"[A7][TABULAR] TEST: logloss={test_metrics['logloss']:.6f}, pr_auc={test_metrics['pr_auc']:.6f}, roc_auc={test_metrics['roc_auc']:.6f}")

    # Fit thresholds on VAL, apply to TEST
    policy = fit_policy_on_val(
        y_val=val_df["target"].to_numpy(),
        p_val=val_df["p_tabular"].to_numpy(),
        max_fpr_deny=max_fpr_deny,
        max_review_share=max_review_share,
    )

    # Build decision tables
    val_zones, val_bin = build_decision_table(
        y_true=val_df["target"].to_numpy(),
        y_score=val_df["p_tabular"].to_numpy(),
        t_review=policy.t_review,
        t_deny=policy.t_deny,
    )
    test_zones, test_bin = build_decision_table(
        y_true=test_df["target"].to_numpy(),
        y_score=test_df["p_tabular"].to_numpy(),
        t_review=policy.t_review,
        t_deny=policy.t_deny,
    )

    # Save outputs
    thresholds_path = thr_dir / "thresholds_tabular.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(policy.__dict__, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {thresholds_path}")

    val_zones_path = eval_dir / "decision_zones_tabular_val.csv"
    test_zones_path = eval_dir / "decision_zones_tabular_test.csv"
    val_bin_path = eval_dir / "decision_binary_tabular_val.csv"
    test_bin_path = eval_dir / "decision_binary_tabular_test.csv"

    val_zones.to_csv(val_zones_path, index=False)
    test_zones.to_csv(test_zones_path, index=False)
    val_bin.to_csv(val_bin_path, index=False)
    test_bin.to_csv(test_bin_path, index=False)

    print(f"[A7] Saved: {val_zones_path}")
    print(f"[A7] Saved: {test_zones_path}")
    print(f"[A7] Saved: {val_bin_path}")
    print(f"[A7] Saved: {test_bin_path}")

    # Save summary metrics
    summary = {
        "tabular": {
            "val": val_metrics,
            "test": test_metrics,
            "threshold_policy": policy.__dict__,
        }
    }
    summary_path = eval_dir / "evaluate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {summary_path}")

    # Load saved zone tables for plotting
    vz = pd.read_csv(val_zones_path)
    tz = pd.read_csv(test_zones_path)

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

    _plot_zone_shares(vz, eval_dir / "zone_share_tabular_val.png", "Zone Shares (Tabular, VAL)")
    _plot_zone_shares(tz, eval_dir / "zone_share_tabular_test.png", "Zone Shares (Tabular, TEST)")
    print(f"[A7] Saved: {eval_dir / 'zone_share_tabular_val.png'}")
    print(f"[A7] Saved: {eval_dir / 'zone_share_tabular_test.png'}")

    # Cost on TEST (realized, based on true labels)
    cost = estimate_cost_by_zones(
        df_pred=test_df,
        proba_col="p_tabular",
        t_review=policy.t_review,
        t_deny=policy.t_deny,
        c_fp_deny=50.0,
        c_fn_allow=200.0,
        c_review=2.0,
    )
    cost_path = eval_dir / "cost_tabular_test.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost, f, ensure_ascii=False, indent=2)
    print(f"[A7] Saved: {cost_path}")

    print("[A7] Done.")


if __name__ == "__main__":
    main()