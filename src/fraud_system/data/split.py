from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

import pandas as pd


@dataclass
class SplitInfo:
    time_col: str
    ratios: Tuple[float, float, float]
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    time_min: int
    time_max: int
    time_train_end: int
    time_val_end: int
    fraud_rate_total: float
    fraud_rate_train: float
    fraud_rate_val: float
    fraud_rate_test: float


def _fraud_rate(df: pd.DataFrame, target_col: str = "target") -> float:
    if len(df) == 0:
        return 0.0
    return float(df[target_col].mean())


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "time",
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    target_col: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Делит датасет по времени (по отсортированному time_col) на train/val/test.
    ratios должны суммироваться в 1.0.
    """
    r_sum = sum(ratios)
    if abs(r_sum - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got sum={r_sum} and ratios={ratios}")

    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in dataframe")

    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # остаток — в test
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Split sizes are invalid: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}. "
            "Adjust ratios or ensure dataset is large enough."
        )

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    time_min = int(df[time_col].min())
    time_max = int(df[time_col].max())
    time_train_end = int(train_df[time_col].max())
    time_val_end = int(val_df[time_col].max())

    info = SplitInfo(
        time_col=time_col,
        ratios=ratios,
        n_total=n,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        time_min=time_min,
        time_max=time_max,
        time_train_end=time_train_end,
        time_val_end=time_val_end,
        fraud_rate_total=_fraud_rate(df, target_col),
        fraud_rate_train=_fraud_rate(train_df, target_col),
        fraud_rate_val=_fraud_rate(val_df, target_col),
        fraud_rate_test=_fraud_rate(test_df, target_col),
    )

    return train_df, val_df, test_df, asdict(info)