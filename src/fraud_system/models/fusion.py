from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class FusionConfig:
    C: float = 1.0
    max_iter: int = 200
    random_state: int = 42


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = _clip_prob(p, eps=eps)
    return np.log(p / (1.0 - p))


class FusionModel:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.model: Optional[LogisticRegression] = None

    def fit(self, p_tabular: np.ndarray, p_gnn: np.ndarray, y: np.ndarray) -> "FusionModel":
        X = np.vstack([logit(p_tabular), logit(p_gnn)]).T
        self.model = LogisticRegression(
            C=self.cfg.C,
            max_iter=self.cfg.max_iter,
            random_state=self.cfg.random_state,
            solver="lbfgs",
        )
        self.model.fit(X, y.astype(int))
        return self

    def predict_proba(self, p_tabular: np.ndarray, p_gnn: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FusionModel is not fitted")
        X = np.vstack([logit(p_tabular), logit(p_gnn)]).T
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg, "model": self.model}, f)

    @staticmethod
    def load(path: Path) -> "FusionModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        fm = FusionModel(obj["cfg"])
        fm.model = obj["model"]
        return fm