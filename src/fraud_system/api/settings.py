from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    # service
    service_name: str = "vkr-fraud-api"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # auth
    api_key: str = "change-me"
    api_key_header: str = "X-API-Key"

    # artifacts: tabular
    tabular_model_path: str = "artifacts/tabular/model.pkl"
    thresholds_tabular_path: str = "artifacts/thresholds/thresholds_tabular.json"
    feature_spec_path: str = "artifacts/evaluation/tabular_feature_spec.json"

    # artifacts: fusion external (main “system mode”)
    fusion_external_model_path: str = "artifacts/fusion/fusion_external.pkl"
    thresholds_fusion_external_path: str = "artifacts/thresholds/thresholds_fusion_external.json"
    fusion_external_metrics_path: str = "artifacts/evaluation/fusion_metrics_external.json"

    # defaults
    default_model: str = "tabular"  # or "fusion_external"

    # reasons
    enable_reasons: bool = True
    reasons_topk_default: int = 5
    reasons_max_rows: int = 500  # safety cap for heavy batches

    # ops
    enable_metrics: bool = True

    @classmethod
    def from_env(cls) -> "ApiSettings":
        def _get_bool(name: str, default: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

        def _get_int(name: str, default: int) -> int:
            v = os.getenv(name)
            if v is None or str(v).strip() == "":
                return default
            try:
                return int(v)
            except Exception:
                return default

        return cls(
            service_name=os.getenv("FRAUD_API_SERVICE_NAME", cls.service_name),
            host=os.getenv("FRAUD_API_HOST", cls.host),
            port=_get_int("FRAUD_API_PORT", cls.port),
            log_level=os.getenv("FRAUD_API_LOG_LEVEL", cls.log_level),
            api_key=os.getenv("FRAUD_API_API_KEY", cls.api_key),
            api_key_header=os.getenv("FRAUD_API_API_KEY_HEADER", cls.api_key_header),

            tabular_model_path=os.getenv("FRAUD_API_TABULAR_MODEL_PATH", cls.tabular_model_path),
            thresholds_tabular_path=os.getenv("FRAUD_API_THRESHOLDS_TABULAR_PATH", cls.thresholds_tabular_path),
            feature_spec_path=os.getenv("FRAUD_API_FEATURE_SPEC_PATH", cls.feature_spec_path),

            fusion_external_model_path=os.getenv("FRAUD_API_FUSION_EXTERNAL_MODEL_PATH", cls.fusion_external_model_path),
            thresholds_fusion_external_path=os.getenv("FRAUD_API_THRESHOLDS_FUSION_EXTERNAL_PATH", cls.thresholds_fusion_external_path),
            fusion_external_metrics_path=os.getenv("FRAUD_API_FUSION_EXTERNAL_METRICS_PATH", cls.fusion_external_metrics_path),

            default_model=os.getenv("FRAUD_API_DEFAULT_MODEL", cls.default_model),

            enable_reasons=_get_bool("FRAUD_API_ENABLE_REASONS", cls.enable_reasons),
            reasons_topk_default=_get_int("FRAUD_API_REASONS_TOPK_DEFAULT", cls.reasons_topk_default),
            reasons_max_rows=_get_int("FRAUD_API_REASONS_MAX_ROWS", cls.reasons_max_rows),
            enable_metrics=_get_bool("FRAUD_API_ENABLE_METRICS", cls.enable_metrics),
        )

    def auth_enabled(self) -> bool:
        k = (self.api_key or "").strip()
        return k != "" and k.lower() != "change-me"