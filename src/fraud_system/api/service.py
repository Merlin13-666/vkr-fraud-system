from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from fraud_system.api.errors import PredictError
from fraud_system.api.settings import ApiSettings


def _read_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_any(path: str) -> Any:
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        import pickle
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise PredictError(
                "Не удалось загрузить модель.\n"
                f"joblib: {type(e_joblib).__name__}: {e_joblib}\n"
                f"pickle: {type(e_pickle).__name__}: {e_pickle}\n"
                "Проверь, что artifacts создан в той же версии Python/библиотек."
            )


def _extract_columns_from_feature_spec(spec: Dict[str, Any]) -> Optional[List[str]]:
    if not isinstance(spec, dict):
        return None
    cols = spec.get("feature_cols")
    if isinstance(cols, list) and cols and all(isinstance(x, str) for x in cols):
        return list(dict.fromkeys(cols))

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
    data = _read_json(path) or {}

    # поддержка формата {"thresholds": {...}}
    if isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]

    out: Dict[str, float] = {}

    # 1) основной формат allow/review/deny
    for k in ("allow", "review", "deny"):
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)

    # 2) твой формат t_review/t_deny
    if "review" not in out:
        v = data.get("t_review")
        if isinstance(v, (int, float)):
            out["review"] = float(v)

    if "deny" not in out:
        v = data.get("t_deny")
        if isinstance(v, (int, float)):
            out["deny"] = float(v)

    # дефолты на крайний случай (если файл не тот/пустой)
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
    base_template = {c: np.nan for c in expected_columns}
    data = []
    for r in rows:
        base = dict(base_template)
        for k, v in (r or {}).items():
            if k in base:
                base[k] = v
        data.append(base)
    return pd.DataFrame(data, columns=expected_columns)


def _split_pipeline_smart(model: Any) -> Tuple[Any, Any]:
    """
    Лучший вариант для твоего кейса:
    - если есть named_steps: берём ('pre', 'model')
    - иначе если Pipeline.steps: last = estimator, prev = preprocessor-wrapper
    - иначе: (None, model)
    """
    # 1) BEST: pipeline with explicit names (your case)
    if hasattr(model, "named_steps"):
        ns = model.named_steps
        pre = ns.get("pre") or ns.get("preprocess") or ns.get("prep") or ns.get("transform")
        est = ns.get("model") or ns.get("clf") or ns.get("classifier") or ns.get("lgbm")
        if pre is not None and est is not None:
            return pre, est

    # 2) fallback: generic Pipeline.steps
    steps = getattr(model, "steps", None)
    if isinstance(steps, list) and len(steps) >= 2:
        est = steps[-1][1]

        class _PreWrap:
            def __init__(self, pipe):
                self.pipe = pipe

            def transform(self, X):
                out = X
                for _, step in self.pipe:
                    if hasattr(step, "transform"):
                        out = step.transform(out)
                return out

            def get_feature_names_out(self):
                for _, step in reversed(self.pipe):
                    if hasattr(step, "get_feature_names_out"):
                        return step.get_feature_names_out()
                raise AttributeError("no get_feature_names_out")

        pre = _PreWrap(steps[:-1])
        return pre, est

    return None, model


def _try_contrib(estimator: Any, Xt: Any) -> Optional[np.ndarray]:
    """
    Надёжно для LightGBM в sklearn.Pipeline:
    - сначала пробуем estimator.predict(..., pred_contrib=True) (если поддерживается),
    - затем самый стабильный вариант: estimator.booster_.predict(..., pred_contrib=True).
    """
    # нормализуем вход (часто это scipy sparse)
    try:
        if hasattr(Xt, "tocsr"):
            Xt_use = Xt.tocsr()
        else:
            Xt_use = Xt
    except Exception:
        Xt_use = Xt

    # 1) sklearn wrapper (может не поддерживаться)
    try:
        if hasattr(estimator, "predict"):
            contrib = estimator.predict(Xt_use, pred_contrib=True)
            arr = np.asarray(contrib)
            if arr.ndim == 2:
                return arr
    except Exception:
        pass

    # 2) booster_.predict (самый стабильный путь)
    try:
        booster = getattr(estimator, "booster_", None)
        if booster is not None and hasattr(booster, "predict"):
            contrib = booster.predict(Xt_use, pred_contrib=True)
            arr = np.asarray(contrib)
            if arr.ndim == 2:
                return arr
    except Exception:
        pass

    return None

def _shap_values_binary(explainer: Any, X: Any) -> np.ndarray:
    """
    Возвращает SHAP values для positive class (class=1) как (n_rows, n_features).
    Поддерживает:
      - explainer(X) -> shap.Explanation
      - explainer.shap_values(X) -> list[np.ndarray] или np.ndarray
    """
    sv: Any = None

    # 1) Новый стиль: explainer(X) -> Explanation
    try:
        exp = explainer(X)
        if hasattr(exp, "values"):
            sv = exp.values
    except Exception:
        sv = None

    # 2) Старый стиль: shap_values()
    if sv is None:
        sv = explainer.shap_values(X)

    # 2a) binary: list[class0, class1] -> class1
    if isinstance(sv, list):
        if len(sv) >= 2:
            sv = sv[1]
        elif len(sv) == 1:
            sv = sv[0]

    sv = np.asarray(sv)

    # Иногда прилетает (n, 2, m) или (2, n, m) — аккуратно режем
    # (редко, но лучше подстраховаться)
    if sv.ndim == 3:
        # попробуем интерпретировать как (classes, n, m) или (n, classes, m)
        if sv.shape[0] == 2:
            sv = sv[1]
        elif sv.shape[1] == 2:
            sv = sv[:, 1, :]
        else:
            raise PredictError(f"Unexpected SHAP 3D shape: {sv.shape}")

    # Гарантируем 2D
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    if sv.ndim != 2:
        raise PredictError(f"Unexpected SHAP values shape: {sv.shape}")

    return sv

def _reason_value_from_raw(raw_row: Dict[str, Any], feat_name: str) -> Any:
    # 1) прямое совпадение (редко, но бывает)
    if feat_name in raw_row:
        return raw_row.get(feat_name)

    # 2) one-hot вида "col_value" или "col___MISSING__"
    # берем часть до первого "_"
    if "_" in feat_name:
        base = feat_name.split("_", 1)[0]
        if base in raw_row:
            return raw_row.get(base)

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

    fusion_external_model: Any = None
    fusion_external_thresholds: Optional[Dict[str, float]] = None
    fusion_external_weights: Optional[FusionWeights] = None
    fusion_external_meta: Optional[Dict[str, Any]] = None

    # --- SHAP (optional, for reasons)
    shap_explainer: Any = None
    shap_uses_transformed: bool = True
    shap_feature_names: Optional[List[str]] = None
    shap_model_output: str = "raw"  # "raw" (log-odds) - fastest/stable


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
                    "Не удалось определить expected columns. Проверь tabular_feature_spec.json (feature_cols)."
                )

            pre, est = _split_pipeline_smart(model)

            # --- SHAP explainer (optional) ---
            shap_explainer = None
            shap_uses_transformed = True

            # Создаём explainer только если reasons потенциально нужны
            if self.settings.enable_reasons:
                try:
                    # Важно: объяснять лучше estimator/booster, а не весь pipeline
                    booster = getattr(est, "booster_", None)
                    if booster is not None:
                        shap_explainer = shap.TreeExplainer(booster)
                    else:
                        shap_explainer = shap.TreeExplainer(est)
                except Exception:
                    shap_explainer = None
                    shap_uses_transformed = True

            feat_names = None
            if pre is not None:
                try:
                    if hasattr(pre, "get_feature_names_out"):
                        feat_names = list(pre.get_feature_names_out())
                except Exception:
                    feat_names = None

            if not feat_names:
                feat_names = cols

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

                shap_explainer=shap_explainer,
                shap_uses_transformed=shap_uses_transformed,

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

        shap_vals = None
        shap_names = art.transformed_feature_names or art.expected_columns

        if reasons_allowed and art.shap_explainer is not None:
            try:
                X = _prepare_dataframe(rows, art.expected_columns)

                # если есть препроцессор — объясняем уже transformed space
                if art.preprocessor is not None and hasattr(art.preprocessor, "transform"):
                    Xt = art.preprocessor.transform(X)
                    # SHAP иногда не дружит со sparse — конвертим безопасно
                    try:
                        if hasattr(Xt, "toarray"):
                            Xt_shap = Xt.toarray()
                        else:
                            Xt_shap = Xt
                    except Exception:
                        Xt_shap = Xt

                    shap_vals = _shap_values_binary(art.shap_explainer, Xt_shap)
                else:
                    shap_vals = _shap_values_binary(art.shap_explainer, X)
            except Exception as e:
                self._last_error = f"SHAP error: {type(e).__name__}: {e}"
                shap_vals = None
                print(self._last_error)

        items: List[Dict[str, Any]] = []
        for i in range(len(rows)):
            score = float(p[i])
            decision = _make_decision(score, art.thresholds_tabular)

            top_reasons = None
            if reasons_allowed and shap_vals is not None:
                row_sv = np.asarray(shap_vals[i])  # теперь это точно 1D (n_features,)
                nn = min(len(shap_names), len(row_sv))
                pairs = list(zip(shap_names[:nn], row_sv[:nn]))
                pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)
                pairs = pairs[: max(1, int(reasons_topk))]

                raw_row = rows[i] or {}
                top_reasons = [
                    {
                        "feature": str(feat),
                        "contribution": float(c),
                        "value": _reason_value_from_raw(raw_row, str(feat)),
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
        if art.fusion_external_model is not None:
            try:
                if hasattr(art.fusion_external_model, "predict_proba"):
                    p_f = art.fusion_external_model.predict_proba(p_tab, p_gnn)
            except Exception:
                p_f = None

        if p_f is None:
            if art.fusion_external_weights is None:
                raise PredictError(
                    "Fusion model not available and fusion_weights missing "
                    "(check artifacts/evaluation/fusion_metrics_external.json)."
                )
            w = art.fusion_external_weights
            z = w.bias + w.w_tabular * _logit(p_tab) + w.w_gnn * _logit(p_gnn)
            p_f = _sigmoid(z)

        p_f = np.asarray(p_f, dtype="float64")
        p_f = np.clip(p_f, 1e-6, 1.0 - 1e-6)

        tab_items, _ = self.predict_tabular(rows=rows, with_reasons=with_reasons, reasons_topk=reasons_topk)

        items: List[Dict[str, Any]] = []
        for i in range(len(rows)):
            score = float(p_f[i])
            decision = _make_decision(score, art.fusion_external_thresholds)
            items.append(
                {"risk_score": score, "decision": decision, "top_reasons": tab_items[i].get("top_reasons")}
            )

        return items, "fusion_external"

    def features_info(self) -> Dict[str, Any]:
        art = self.artifacts
        spec = art.feature_spec or {}
        return {
            "n_features": len(art.expected_columns),
            "expected_columns": art.expected_columns,
            "groups": {"num_cols": spec.get("num_cols", []), "cat_cols": spec.get("cat_cols", [])},
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