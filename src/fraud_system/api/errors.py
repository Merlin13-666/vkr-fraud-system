from __future__ import annotations

import traceback
from fastapi import Request
from fastapi.responses import JSONResponse


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Чтобы в клиенте не было просто 'Internal Server Error' без смысла,
    возвращаем JSON с request_id (если проставлен middleware) и кратким сообщением.

    ВАЖНО: stack trace выдаём только как строку (без секретов), и то можно потом выключить.
    """
    request_id = request.headers.get("X-Request-ID") or request.state.__dict__.get("request_id")
    payload = {
        "detail": "Internal Server Error",
        "request_id": request_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    # Небольшой трейс полезен для разработки
    payload["trace"] = traceback.format_exc(limit=25)
    return JSONResponse(status_code=500, content=payload)