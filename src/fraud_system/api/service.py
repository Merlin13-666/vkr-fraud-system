from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


def _load_json(path: str) -> Dict[str, Any]:
    if path is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model(path: str):
    """
    Надёжная загрузка модели:
      - joblib.load() для sklearn/lightgbm пайплайнов
      - pickle здесь НЕ используем специально (часто ломается при версиях)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model with joblib.load('{path}'). "
            f"Most common причина: несовместимость версий sklearn/lightgbm. Error: {e}"
        ) from e


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass
class LoadedArtifacts:
    tabular_model: Any
    thresholds_tabular: Dict[str, Any]
    feature_spec: Dict[str, Any]


class FraudService:
    """
    Сервис скоринга.

    Главное улучшение:
      - даже если клиент прислал 2 фичи, мы достроим DataFrame до полного списка фич,
        который ожидает ColumnTransformer (из feature_spec).
    """

    def __init__(
        self,
        tabular_model_path: str,
        thresholds_tabular_path: str,
        feature_spec_path: str,
        fusion_model_path: Optional[str] = None,
        thresholds_fusion_external_path: Optional[str] = None,
        enable_reasons: bool = True,
    ):
        self.tabular_model_path = tabular_model_path
        self.thresholds_tabular_path = thresholds_tabular_path
        self.feature_spec_path = feature_spec_path

        self.fusion_model_path = fusion_model_path
        self.thresholds_fusion_external_path = thresholds_fusion_external_path

        self.enable_reasons = enable_reasons

        self.art: Optional[LoadedArtifacts] = None
        self.reload()

    @property
    def paths(self) -> Dict[str, Any]:
        return {
            "tabular_model_path": self.tabular_model_path,
            "thresholds_tabular_path": self.thresholds_tabular_path,
            "feature_spec_path": self.feature_spec_path,
            "fusion_model_path": self.fusion_model_path,
            "thresholds_fusion_external_path": self.thresholds_fusion_external_path,
        }

    def reload(self) -> Dict[str, Any]:
        tabular_model = _load_model(self.tabular_model_path)
        thresholds_tabular = _load_json(self.thresholds_tabular_path)
        feature_spec = _load_json(self.feature_spec_path)

        self.art = LoadedArtifacts(
            tabular_model=tabular_model,
            thresholds_tabular=thresholds_tabular,
            feature_spec=feature_spec,
        )
        return self.paths

    # --------------------------
    # Feature alignment
    # --------------------------

    def _expected_feature_names(self) -> List[str]:
        """
        Берём список ожидаемых фич.
        1) Из feature_spec.json (предпочтительно)
        2) Если нет — пытаемся достать из sklearn pipeline / preprocessor
        """
        assert self.art is not None
        spec = self.art.feature_spec or {}

        # самый частый вариант: feature_spec["features"] = [...]
        if isinstance(spec.get("features"), list) and len(spec["features"]) > 0:
            return list(spec["features"])

        # fallback: пытаемся из модели
        m = self.art.tabular_model
        try:
            # sklearn Pipeline: steps['pre'] = ColumnTransformer
            pre = m.named_steps.get("pre")  # type: ignore[attr-defined]
            if pre is not None and hasattr(pre, "feature_names_in_"):
                return list(pre.feature_names_in_)  # type: ignore[attr-defined]
        except Exception:
            pass

        # если совсем ничего — оставим то, что пришло (но тогда модель может упасть)
        return []

    def _to_aligned_df(self, features_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Делает DataFrame, где есть ВСЕ ожидаемые колонки (в нужном порядке).
        Недостающие -> NaN.
        """
        expected = self._expected_feature_names()

        df = pd.DataFrame(features_list)

        if expected:
            # добавим недостающие
            for c in expected:
                if c not in df.columns:
                    df[c] = np.nan
            # лишние можно оставить — ColumnTransformer обычно игнорит лишние, но на всякий случай:
            df = df[expected]
        return df

    # --------------------------
    # Thresholds / decision
    # --------------------------

    def _decision_from_thresholds(self, p: float, thresholds: Dict[str, Any]) -> str:
        """
        thresholds_tabular.json у тебя может быть разный.
        Поддержим 2 варианта:
          A) {"allow": 0.2, "deny": 0.8} => review между ними
          B) {"thresholds": {"allow":..., "deny":...}}
        """
        t = thresholds.get("thresholds", thresholds)

        allow_t = t.get("allow", 0.2)
        deny_t = t.get("deny", 0.8)

        p = float(p)
        if p >= deny_t:
            return "deny"
        if p >= allow_t:
            return "review"
        return "allow"

    # --------------------------
    # Explainability (простая заглушка)
    # --------------------------

    def _top_reasons_stub(self, topk: int) -> List[Dict[str, Any]]:
        # Пока без SHAP (чтобы не тормозить и не падать).
        # В /docs явно пишем, что это заглушка/опционально.
        return []

    # --------------------------
    # Main predict
    # --------------------------

    def predict(
        self,
        model: str,
        records: List[Dict[str, Any]],
        with_reasons: bool,
        reasons_topk: int,
        reasons_max_rows: int,
    ) -> Dict[str, Any]:
        assert self.art is not None

        if model != "tabular":
            raise ValueError(f"Unsupported model='{model}'. Сейчас реализован 'tabular'.")

        # records: [{"transaction_id": "...", "features": {...}}, ...]
        tx_ids = [r["transaction_id"] for r in records]
        feats = [r["features"] for r in records]

        X = self._to_aligned_df(feats)

        # predict_proba
        p = self.art.tabular_model.predict_proba(X)[:, 1]

        items = []
        for i, tx_id in enumerate(tx_ids):
            score = float(p[i])
            decision = self._decision_from_thresholds(score, self.art.thresholds_tabular)

            reasons = None
            if self.enable_reasons and with_reasons:
                reasons = self._top_reasons_stub(reasons_topk)

            items.append(
                {
                    "transaction_id": tx_id,
                    "risk_score": score,
                    "decision": decision,
                    "top_reasons": reasons if reasons is not None else None,
                }
            )

        return {
            "model": "tabular",
            "input_format": "rows",  # app.py подставит корректно сверху
            "items": items,
            "request_id": "n/a",  # app.py заменит на request_id из middleware
        }