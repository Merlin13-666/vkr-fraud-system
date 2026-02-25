from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fastapi.responses import PlainTextResponse

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except Exception:  # pragma: no cover
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    generate_latest = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"


@dataclass
class Metrics:
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled or Counter is None:
            self.requests_total = None
            self.request_latency = None
            return

        self.requests_total = Counter(
            "fraud_api_requests_total",
            "Total requests",
            ["method", "path", "status"],
        )
        self.request_latency = Histogram(
            "fraud_api_request_latency_seconds",
            "Request latency",
            ["method", "path"],
        )

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        if not self.enabled or self.requests_total is None or self.request_latency is None:
            return
        self.requests_total.labels(method=method, path=path, status=str(status)).inc()
        self.request_latency.labels(method=method, path=path).observe(seconds)

    def endpoint(self):
        if not self.enabled or generate_latest is None:
            return PlainTextResponse("metrics disabled", status_code=404)

        data = generate_latest()
        return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


class _Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def seconds(self) -> float:
        return time.perf_counter() - self.t0