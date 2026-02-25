from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fraud_system.api.metrics import Metrics


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, metrics: Metrics):
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        seconds = time.perf_counter() - start

        # группируем по “роуту” аккуратно: берём path без query-string
        path = request.url.path
        self.metrics.observe(request.method, path, response.status_code, seconds)
        return response