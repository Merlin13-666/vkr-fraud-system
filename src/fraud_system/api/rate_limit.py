from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


@dataclass
class _Bucket:
    tokens: float
    last_ts: float


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.

    Key:
      - if auth enabled and X-API-Key is present -> key = "key:<api_key>"
      - else key = "ip:<client_ip>"

    Settings:
      - rps: refill rate (tokens per second)
      - burst: max bucket size
    """

    def __init__(
        self,
        app,
        *,
        header_name: str,
        rps: float,
        burst: float,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.header_name = header_name
        self.rps = max(0.1, float(rps))
        self.burst = max(1.0, float(burst))
        self.enabled = bool(enabled)

        self._buckets: Dict[str, _Bucket] = {}
        self._last_cleanup = time.time()

    def _key(self, request: Request) -> str:
        api_key = request.headers.get(self.header_name)
        if api_key:
            return f"key:{api_key}"
        ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"

    def _allow(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        now = time.time()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self.burst, last_ts=now)
            self._buckets[key] = b

        # refill
        dt = max(0.0, now - b.last_ts)
        b.tokens = min(self.burst, b.tokens + dt * self.rps)
        b.last_ts = now

        if b.tokens >= cost:
            b.tokens -= cost
            return True, b.tokens
        return False, b.tokens

    def _cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < 60.0:
            return
        self._last_cleanup = now
        # удаляем очень старые ключи
        dead = [k for k, b in self._buckets.items() if now - b.last_ts > 15 * 60]
        for k in dead:
            self._buckets.pop(k, None)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)

        self._cleanup()

        key = self._key(request)
        ok, left = self._allow(key, cost=1.0)
        if not ok:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try later. (bucket_left={left:.2f})",
            )

        return await call_next(request)