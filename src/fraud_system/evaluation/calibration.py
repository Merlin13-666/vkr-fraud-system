from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TemperatureScalingResult:
    T: float
    val_logloss_before: float
    val_logloss_after: float


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = y.astype(np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-6)
    # sigmoid
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_on_logits(
    y: np.ndarray,
    logits: np.ndarray,
    T_min: float = 0.5,
    T_max: float = 10.0,
    n_grid: int = 200,
) -> TemperatureScalingResult:
    """
    Simple, stable grid-search for Temperature Scaling:
    minimize logloss(y, sigmoid(logit/T)) over T in [T_min, T_max].
    """
    y = y.astype(np.int8)
    logits = logits.astype(np.float64)

    p0 = 1.0 / (1.0 + np.exp(-logits))
    ll0 = _binary_logloss(y, p0)

    Ts = np.linspace(T_min, T_max, n_grid)
    best_T = float(Ts[0])
    best_ll = 1e18

    for T in Ts:
        p = apply_temperature(logits, float(T))
        ll = _binary_logloss(y, p)
        if ll < best_ll:
            best_ll = ll
            best_T = float(T)

    return TemperatureScalingResult(
        T=best_T,
        val_logloss_before=ll0,
        val_logloss_after=float(best_ll),
    )