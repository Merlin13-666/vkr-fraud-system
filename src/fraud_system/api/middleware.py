from __future__ import annotations

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Проставляет X-Request-ID на каждый ответ.
    Если клиент прислал свой X-Request-ID — используем его.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Добавляет X-Response-Time-ms (полезно для дебага)."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        response: Response = await call_next(request)
        dt_ms = int((time.time() - t0) * 1000)
        response.headers["X-Response-Time-ms"] = str(dt_ms)
        return response