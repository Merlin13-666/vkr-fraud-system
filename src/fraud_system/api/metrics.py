from __future__ import annotations

# Оставляем файл для расширения.
# Сейчас метрики отдаём напрямую через prometheus_client.generate_latest() в app.py

#
# from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# from starlette.responses import Response
#
#
# REQUESTS_TOTAL = Counter(
#     "fraud_api_requests_total",
#     "Total requests",
#     ["method", "path", "status"],
# )
#
# REQUEST_LATENCY = Histogram(
#     "fraud_api_request_latency_seconds",
#     "Request latency",
#     ["method", "path"],
# )
#
#
# def metrics_response() -> Response:
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)