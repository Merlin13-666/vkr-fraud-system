from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def binary_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    eps = 1e-15
    y_score = np.clip(y_score, eps, 1 - eps)
    return float(log_loss(y_true, y_score))


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return precision, recall, thresholds

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))