from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from fraud_system.api.errors import PredictError
from fraud_system.api.settings import ApiSettings


def _read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_any(path: str) -> Any:
    """
    Надёжная загрузка модели:
    - сначала joblib.load (под sklearn/lightgbm это обычно правильно)
    - если вдруг файл не joblib — пробуем pickle
    """
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        # fallback: pickle
        import pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise PredictError(
                "Не удалось загрузить модель.\n"
                f"joblib: {type(e_joblib).__name__}: {e_joblib}\n"
                f"pickle: {type(e_pickle).__name__}: {e_pickle}\n"
                "Проверь, что artifacts/tabular/model.pkl создан в этой же версии Python/библиотек."
            )


def _extract_columns_from_feature_spec(spec: Dict[str, Any]) -> Optional[List[str]]:
    """
    Достаём expected columns из tabular_feature_spec.json.
    Обычно там есть feature_cols.
    """
    if not isinstance(spec, dict):
        return None

    cols = spec.get("feature_cols")
    if isinstance(cols, list) and cols and all(isinstance(x, str) for x in cols):
        # remove duplicates preserving order
        return list(dict.fromkeys(cols))

    # fallback: берём самый длинный список строк внутри json
    candidates: List[List[str]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if x and all(isinstance(t, str) for t in x) and len(x) >= 10:
                candidates.append(x)
            else:
                for v in x:
                    walk(v)

    walk(spec)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return list(dict.fromkeys(candidates[0]))


def _load_thresholds(path: str) -> Dict[str, float]:
    """
    Поддерживаем несколько форматов:
    - {"review": 0.7, "deny": 0.9}
    - {"allow": 0.3, "review": 0.7, "deny": 0.9}
    - {"thresholds": {"review":..., "deny":...}}
    """
    data = _read_json(path) or {}
    if "thresholds" in data and isinstance(data["thresholds"], dict):
        data = data["thresholds"]

    out: Dict[str, float] = {}
    for k in ("allow", "review", "deny"):
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)

    out.setdefault("review", 0.7)
    out.setdefault("deny", 0.9)
    return out


def _make_decision(p: float, thr: Dict[str, float]) -> str:
    deny = float(thr.get("deny", 0.9))
    review = float(thr.get("review", 0.7))
    if p >= deny:
        return "deny"
    if p >= review:
        return "review"
    return "allow"


def _prepare_dataframe(rows: List[Dict[str, Any]], expected_columns: List[str]) -> pd.DataFrame:
    """
    Ключевая штука: создаём df строго с expected_columns.
    Недостающее -> NaN, лишнее -> игнор.
    Поэтому можно присылать TransactionAmt+ProductCD и не падать на 431 фиче.
    """
    data = []
    base_template = {c: np.nan for c in expected_columns}

    for r in rows:
        base = dict(base_template)
        for k, v in (r or {}).items():
            if k in base:
                base[k] = v
        data.append(base)

    return pd.DataFrame(data, columns=expected_columns)


def _split_pipeline(model: Any) -> Tuple[Any, Any]:
    """
    Если model = sklearn.Pipeline(steps=[('pre', ...), ('model', LGBMClassifier)])
    то вернём (pre, estimator). Иначе (None, model).
    """
    pre = None
    est = model
    if hasattr(model, "named_steps"):
        pre = model.named_steps.get("pre")
        est = model.named_steps.get("model", model)
    return pre, est


def _try_contrib(estimator: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    LightGBM sklearn API: predict(pred_contrib=True)
    Возвращает shape (n, n_features+1), где последний — bias.
    """
    try:
        if hasattr(estimator, "predict"):
            contrib = estimator.predict(X, pred_contrib=True)
            arr = np.asarray(contrib)
            if arr.ndim == 2 and arr.shape[0] == len(X):
                return arr
    except Exception:
        return None
    return None


@dataclass
class LoadedArtifacts:
    tabular_model: Any
    thresholds: Dict[str, float]
    expected_columns: List[str]
    feature_spec: Dict[str, Any]
    preprocessor: Any = None
    estimator: Any = None
    transformed_feature_names: Optional[List[str]] = None


class FraudService:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self._artifacts: Optional[LoadedArtifacts] = None
        self._last_error: Optional[str] = None

    def is_ready(self) -> bool:
        return self._artifacts is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def artifacts(self) -> LoadedArtifacts:
        if self._artifacts is None:
            raise PredictError("Service not ready")
        return self._artifacts

    def reload(self) -> None:
        """
        Важно: не роняем процесс при проблемах — сохраняем ошибку в last_error.
        """
        try:
            model = _load_model_any(self.settings.tabular_model_path)
            feature_spec = _read_json(self.settings.feature_spec_path)
            thresholds = _load_thresholds(self.settings.thresholds_path)

            cols = _extract_columns_from_feature_spec(feature_spec)
            if not cols:
                cols = list(getattr(model, "feature_names_in_", []) or [])
            if not cols:
                raise PredictError(
                    "Не удалось определить список ожидаемых колонок.\n"
                    "Проверь artifacts/evaluation/tabular_feature_spec.json (feature_cols)."
                )

            pre, est = _split_pipeline(model)

            feat_names = None
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                try:
                    feat_names = list(pre.get_feature_names_out())
                except Exception:
                    feat_names = None

            self._artifacts = LoadedArtifacts(
                tabular_model=model,
                thresholds=thresholds,
                expected_columns=cols,
                feature_spec=feature_spec,
                preprocessor=pre,
                estimator=est,
                transformed_feature_names=feat_names,
            )
            self._last_error = None
        except Exception as e:
            self._artifacts = None
            self._last_error = f"{type(e).__name__}: {e}"

    def predict_tabular(
        self,
        rows: List[Dict[str, Any]],
        with_reasons: bool,
        reasons_topk: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        art = self.artifacts

        X = _prepare_dataframe(rows, art.expected_columns)

        # predict_proba
        try:
            p = art.tabular_model.predict_proba(X)[:, 1]
        except Exception:
            # иногда sklearn Pipeline ожидает ndarray — но у тебя сейчас df работает;
            # оставим fallback на values
            p = art.tabular_model.predict_proba(X.values)[:, 1]

        items: List[Dict[str, Any]] = []

        # reasons: быстро через LightGBM contrib (если доступно и не слишком много строк)
        reasons_allowed = (
            self.settings.enable_reasons
            and bool(with_reasons)
            and len(rows) <= self.settings.reasons_max_rows
        )

        contrib = None
        if reasons_allowed and art.estimator is not None and art.preprocessor is not None:
            # трансформируем фичи как в пайплайне
            try:
                Xt = art.preprocessor.transform(X)
                # для LGBM sklearn обычно принимает numpy/scipy
                contrib = _try_contrib(art.estimator, Xt)
            except Exception:
                contrib = None

        for i in range(len(rows)):
            score = float(p[i])
            decision = _make_decision(score, art.thresholds)

            top_reasons = None
            if reasons_allowed and contrib is not None:
                row_contrib = np.asarray(contrib[i])
                # last is bias
                if row_contrib.ndim == 1 and row_contrib.shape[0] >= 2:
                    contrib_wo_bias = row_contrib[:-1]

                    names = art.transformed_feature_names
                    if names and len(names) == len(contrib_wo_bias):
                        pairs = list(zip(names, contrib_wo_bias))
                    else:
                        # fallback: если не знаем transformed names — покажем raw columns (хуже, но стабильно)
                        raw_names = art.expected_columns[: len(contrib_wo_bias)]
                        pairs = list(zip(raw_names, contrib_wo_bias))

                    pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    pairs = pairs[: max(1, int(reasons_topk))]

                    # value: попробуем взять из исходного rows (только для raw cols)
                    raw_row = rows[i] or {}
                    out_reasons = []
                    for feat, c in pairs:
                        out_reasons.append(
                            {
                                "feature": str(feat),
                                "contribution": float(c),
                                "value": raw_row.get(feat) if feat in raw_row else None,
                            }
                        )
                    top_reasons = out_reasons

            items.append(
                {
                    "risk_score": score,
                    "decision": decision,
                    "top_reasons": top_reasons,
                }
            )

        return items, "tabular"

    def features_info(self) -> Dict[str, Any]:
        """
        Для /features: список ожидаемых колонок + группы из feature_spec.
        """
        art = self.artifacts
        spec = art.feature_spec or {}
        return {
            "n_features": len(art.expected_columns),
            "expected_columns": art.expected_columns,
            "groups": {
                "num_cols": spec.get("num_cols", []),
                "cat_cols": spec.get("cat_cols", []),
            },
        }