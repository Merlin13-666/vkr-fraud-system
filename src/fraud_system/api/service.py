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

    # нормализуем к deny/review
    if "deny" not in out and "t_deny" in out:
        out["deny"] = out["t_deny"]
    if "review" not in out and "t_review" in out:
        out["review"] = out["t_review"]

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

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

@dataclass
class FusionWeights:
    w_tabular: float
    w_gnn: float
    bias: float

@dataclass
class LoadedArtifacts:
    tabular_model: Any
    thresholds_tabular: Dict[str, float]
    expected_columns: List[str]
    feature_spec: Dict[str, Any]
    preprocessor: Any = None
    estimator: Any = None
    transformed_feature_names: Optional[List[str]] = None
    # fusion external
    fusion_external_model: Any = None  # may be sklearn LogisticRegression-wrapped or custom
    fusion_external_thresholds: Optional[Dict[str, float]] = None
    fusion_external_weights: Optional[FusionWeights] = None
    fusion_external_meta: Optional[Dict[str, Any]] = None


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
            # --- tabular ---
            model = _load_model_any(self.settings.tabular_model_path)
            feature_spec = _read_json(self.settings.feature_spec_path)
            thresholds_tab = _load_thresholds(self.settings.thresholds_tabular_path)

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

            # --- fusion external (optional) ---
            fusion_model = None
            if Path(self.settings.fusion_external_model_path).exists():
                try:
                    fusion_model = _load_model_any(self.settings.fusion_external_model_path)
                except Exception:
                    fusion_model = None

            fusion_thr = _load_thresholds(self.settings.thresholds_fusion_external_path)

            fusion_meta = _read_json(self.settings.fusion_external_metrics_path) or None
            fusion_w = None
            if fusion_meta and isinstance(fusion_meta.get("fusion_weights"), dict):
                fw = fusion_meta["fusion_weights"]
                if all(k in fw for k in ("w_tabular", "w_gnn", "bias")):
                    fusion_w = FusionWeights(
                        w_tabular=float(fw["w_tabular"]),
                        w_gnn=float(fw["w_gnn"]),
                        bias=float(fw["bias"]),
                    )

            self._artifacts = LoadedArtifacts(
                tabular_model=model,
                thresholds_tabular=thresholds_tab,
                expected_columns=cols,
                feature_spec=feature_spec,
                preprocessor=pre,
                estimator=est,
                transformed_feature_names=feat_names,
                fusion_external_model=fusion_model,
                fusion_external_thresholds=fusion_thr,
                fusion_external_weights=fusion_w,
                fusion_external_meta=fusion_meta,
            )
            self._last_error = None
        except Exception as e:
            self._artifacts = None
            self._last_error = f"{type(e).__name__}: {e}"

    def _predict_tabular_scores(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        art = self.artifacts
        X = _prepare_dataframe(rows, art.expected_columns)
        # ВАЖНО: оставляем DataFrame, чтобы не было sklearn warning про feature names
        p = art.tabular_model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype="float64")

    def predict_tabular(
        self,
        rows: List[Dict[str, Any]],
        with_reasons: bool,
        reasons_topk: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        art = self.artifacts
        p = self._predict_tabular_scores(rows)

        reasons_allowed = (
            self.settings.enable_reasons
            and bool(with_reasons)
            and len(rows) <= self.settings.reasons_max_rows
        )

        contrib = None
        if reasons_allowed and art.estimator is not None and art.preprocessor is not None:
            try:
                X = _prepare_dataframe(rows, art.expected_columns)
                Xt = art.preprocessor.transform(X)
                contrib = _try_contrib(art.estimator, Xt)
            except Exception:
                contrib = None

        items: List[Dict[str, Any]] = []
        for i in range(len(rows)):
            score = float(p[i])
            decision = _make_decision(score, art.thresholds_tabular)

            top_reasons = None
            if reasons_allowed and contrib is not None:
                row_contrib = np.asarray(contrib[i])
                if row_contrib.ndim == 1 and row_contrib.shape[0] >= 2:
                    contrib_wo_bias = row_contrib[:-1]
                    names = art.transformed_feature_names
                    if names and len(names) == len(contrib_wo_bias):
                        pairs = list(zip(names, contrib_wo_bias))
                    else:
                        raw_names = art.expected_columns[: len(contrib_wo_bias)]
                        pairs = list(zip(raw_names, contrib_wo_bias))

                    pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)
                    pairs = pairs[: max(1, int(reasons_topk))]

                    raw_row = rows[i] or {}
                    top_reasons = [
                        {
                            "feature": str(feat),
                            "contribution": float(c),
                            "value": raw_row.get(feat) if feat in raw_row else None,
                        }
                        for feat, c in pairs
                    ]

            items.append({"risk_score": score, "decision": decision, "top_reasons": top_reasons})

        return items, "tabular"

    def predict_fusion_external(
        self,
        rows: List[Dict[str, Any]],
        gnn_scores: List[float],
        with_reasons: bool,
        reasons_topk: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        art = self.artifacts
        if art.fusion_external_thresholds is None:
            raise PredictError("Fusion external thresholds are not loaded.")

        if len(gnn_scores) != len(rows):
            raise PredictError("gnn_scores length must match number of rows.")

        p_tab = self._predict_tabular_scores(rows)
        p_gnn = np.asarray(gnn_scores, dtype="float64")

        if np.any(~np.isfinite(p_gnn)):
            raise PredictError("gnn_score contains non-finite values.")
        if np.any(p_gnn < 0.0) or np.any(p_gnn > 1.0):
            raise PredictError("gnn_score must be in [0, 1].")

        p_f = None

        # 1) Try saved model (if supports predict_proba(p_tab, p_gnn))
        if art.fusion_external_model is not None:
            try:
                if hasattr(art.fusion_external_model, "predict_proba"):
                    p_f = art.fusion_external_model.predict_proba(p_tab, p_gnn)
            except Exception:
                p_f = None

        # 2) Fallback: weights from metrics (logit scheme)
        if p_f is None:
            if art.fusion_external_weights is None:
                raise PredictError(
                    "Fusion model is not available and fusion_weights are missing "
                    "(check artifacts/evaluation/fusion_metrics_external.json)."
                )
            w = art.fusion_external_weights
            z = w.bias + w.w_tabular * _logit(p_tab) + w.w_gnn * _logit(p_gnn)
            p_f = _sigmoid(z)

        p_f = np.asarray(p_f, dtype="float64")
        p_f = np.clip(p_f, 1e-6, 1.0 - 1e-6)

        # reasons: табличные причины (объяснимость интерпретируемой части системы)
        tab_items, _ = self.predict_tabular(rows=rows, with_reasons=with_reasons, reasons_topk=reasons_topk)

        items: List[Dict[str, Any]] = []
        for i in range(len(rows)):
            score = float(p_f[i])
            decision = _make_decision(score, art.fusion_external_thresholds)
            items.append(
                {
                    "risk_score": score,
                    "decision": decision,
                    "top_reasons": tab_items[i].get("top_reasons"),
                }
            )

        return items, "fusion_external"

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

    def models_info(self) -> Dict[str, Any]:
        art = self.artifacts
        return {
            "available": ["tabular", "fusion_external"],
            "default_model": self.settings.default_model,
            "tabular": {
                "ready": True,
                "model_path": self.settings.tabular_model_path,
                "thresholds_path": self.settings.thresholds_tabular_path,
                "feature_spec_path": self.settings.feature_spec_path,
            },
            "fusion_external": {
                "ready": bool(art.fusion_external_thresholds is not None),
                "model_path": self.settings.fusion_external_model_path,
                "thresholds_path": self.settings.thresholds_fusion_external_path,
                "metrics_path": self.settings.fusion_external_metrics_path,
                "loaded_pkl": bool(art.fusion_external_model is not None),
                "loaded_weights_fallback": bool(art.fusion_external_weights is not None),
                "meta": art.fusion_external_meta or {},
            },
        }