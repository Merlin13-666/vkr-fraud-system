from __future__ import annotations

import json
import time
import uuid
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fraud_system.api.metrics import Metrics


def _log_json(event: dict) -> None:
    # stdout/stderr соберёт uvicorn; формат JSON удобен для ELK/Graylog
    try:
        print(json.dumps(event, ensure_ascii=False))
    except Exception:
        # fallback
        print(str(event))


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
        status_code: Optional[int] = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            seconds = time.perf_counter() - start
            path = request.url.path
            self.metrics.observe(request.method, path, int(status_code or 500), seconds)

            _log_json(
                {
                    "ts": time.time(),
                    "type": "http_request",
                    "request_id": getattr(request.state, "request_id", None),
                    "method": request.method,
                    "path": path,
                    "status": int(status_code or 500),
                    "latency_ms": round(seconds * 1000.0, 3),
                    "client_ip": request.client.host if request.client else None,
                }
            )