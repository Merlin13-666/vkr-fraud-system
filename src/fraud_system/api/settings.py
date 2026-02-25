from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return v


def _env_bool(key: str, default: bool) -> bool:
    v = _env(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ApiSettings:
    """
    Настройки API.

    Важно:
      - API key берётся из переменной окружения FRAUD_API_API_KEY
      - По умолчанию host=127.0.0.1, port=8000
    """

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # Security
    api_key: str = "change-me"  # перезапишется из env при создании через from_env()
    api_key_header: str = "X-API-Key"
    disable_auth: bool = False  # удобно для локальных тестов (НЕ для прод)

    # Artifacts
    tabular_model_path: str = "artifacts/tabular/model.pkl"
    fusion_model_path: Optional[str] = None

    thresholds_tabular_path: str = "artifacts/thresholds/thresholds_tabular.json"
    thresholds_fusion_external_path: Optional[str] = None

    feature_spec_path: str = "artifacts/evaluation/tabular_feature_spec.json"

    # Explainability
    enable_reasons: bool = True

    @staticmethod
    def from_env() -> "ApiSettings":
        return ApiSettings(
            host=_env("FRAUD_API_HOST", "127.0.0.1") or "127.0.0.1",
            port=int(_env("FRAUD_API_PORT", "8000") or "8000"),
            log_level=_env("FRAUD_API_LOG_LEVEL", "info") or "info",
            api_key=_env("FRAUD_API_API_KEY", "change-me") or "change-me",
            api_key_header=_env("FRAUD_API_API_KEY_HEADER", "X-API-Key") or "X-API-Key",
            disable_auth=_env_bool("FRAUD_API_DISABLE_AUTH", False),
            tabular_model_path=_env("FRAUD_API_TABULAR_MODEL_PATH", "artifacts/tabular/model.pkl")
            or "artifacts/tabular/model.pkl",
            fusion_model_path=_env("FRAUD_API_FUSION_MODEL_PATH", None),
            thresholds_tabular_path=_env(
                "FRAUD_API_THRESHOLDS_TABULAR_PATH", "artifacts/thresholds/thresholds_tabular.json"
            )
            or "artifacts/thresholds/thresholds_tabular.json",
            thresholds_fusion_external_path=_env("FRAUD_API_THRESHOLDS_FUSION_EXTERNAL_PATH", None),
            feature_spec_path=_env(
                "FRAUD_API_FEATURE_SPEC_PATH", "artifacts/evaluation/tabular_feature_spec.json"
            )
            or "artifacts/evaluation/tabular_feature_spec.json",
            enable_reasons=_env_bool("FRAUD_API_ENABLE_REASONS", True),
        )


@dataclass(frozen=True)
class BuildInfo:
    service: str = "vkr-fraud-api"
    version: str = _env("FRAUD_API_VERSION", "0.1.0") or "0.1.0"