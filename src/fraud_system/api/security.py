from __future__ import annotations

from fastapi import HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from fraud_system.api.settings import ApiSettings


def build_api_key_scheme(settings: ApiSettings) -> APIKeyHeader:
    """
    Схема для Swagger (/docs): появится кнопка Authorize и заголовок X-API-Key.
    auto_error=False — чтобы мы сами делали понятные ошибки и могли отключать auth.
    """
    return APIKeyHeader(name=settings.api_key_header, auto_error=False)


def require_api_key(settings: ApiSettings, api_key: str | None) -> None:
    """
    Проверка API-ключа.
    Если auth отключен (ключ пустой/change-me) — пропускаем.
    Иначе требуем заголовок X-API-Key и сравниваем с settings.api_key.
    """
    if not settings.auth_enabled():
        return

    if api_key is None or str(api_key).strip() == "":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key (send header {settings.api_key_header}).",
        )

    if str(api_key).strip() != str(settings.api_key).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )