from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class LGBMConfig:
    seed: int = 42
    n_estimators: int = 10000
    learning_rate: float = 0.05
    num_leaves: int = 64
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    n_jobs: int = -1


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Preprocess:
      - numeric: median impute
      - categorical: constant impute + OHE
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    return pre


def build_lgbm(cfg: LGBMConfig) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
    )


def build_tabular_pipeline(num_cols: List[str], cat_cols: List[str], cfg: LGBMConfig) -> Pipeline:
    """
    Обычный pipeline без early stopping (на всякий случай оставляем).
    """
    pre = build_preprocessor(num_cols, cat_cols)
    model = build_lgbm(cfg)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model),
    ])
    return pipe


def fit_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    cfg: LGBMConfig,
    stopping_rounds: int = 50,
    log_period: int = 50,
) -> Pipeline:
    """
    ВАЖНО:
    LightGBM eval_set должен быть числовым (после preprocess).
    Поэтому:
      1) fit preprocess на train
      2) transform train/val
      3) fit LGBM с early stopping на transformed val
      4) собираем Pipeline обратно (fitted pre + fitted model)
    """
    pre = build_preprocessor(num_cols, cat_cols)
    model = build_lgbm(cfg)

    Xt_train = pre.fit_transform(X_train, y_train)
    Xt_val = pre.transform(X_val)

    model.fit(
        Xt_train,
        y_train,
        eval_set=[(Xt_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=stopping_rounds),
            log_evaluation(period=log_period),
        ],
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", model),
    ])
    return pipe


def save_model(model: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]