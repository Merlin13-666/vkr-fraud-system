from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from .settings import ApiSettings


def require_api_key(
    settings: ApiSettings = Depends(ApiSettings.from_env),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Простая авторизация по заголовку X-API-Key.

    Как использовать:
      - выставь env:  $env:FRAUD_API_API_KEY="super-secret"
      - отправь заголовок:  X-API-Key: super-secret
    """
    if settings.disable_auth:
        return

    if x_api_key is None or str(x_api_key).strip() == "":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key (send header X-API-Key).",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )