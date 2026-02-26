from __future__ import annotations

import os
import platform
import sys
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List
import concurrent.futures
from collections import deque

from fraud_system.api.rate_limit import RateLimitMiddleware
from fastapi import Body, Depends, FastAPI, Request
from fraud_system.api.errors import install_error_handlers, http_400, http_503
from fraud_system.api.metrics import Metrics
from fraud_system.api.middleware import MetricsMiddleware, RequestIdMiddleware
from fraud_system.api.schemas import (
    PredictRequestCanonical,
    PredictRequestRows,
    PredictResponse,
    PredictItem,
    ReasonItem,
    ReadyResponse,
    VersionResponse,
    ConfigResponse,
)
from fraud_system.api.security import build_api_key_scheme, require_api_key
from fraud_system.api.service import FraudService
from fraud_system.api.settings import ApiSettings


_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat()

def _docs_ru(settings: ApiSettings) -> str:
    hdr = settings.api_key_header
    auth_state = "ВКЛЮЧЕНА ✅" if settings.auth_enabled() else "ОТКЛЮЧЕНА ✅ (ключ пустой или change-me)"

    return f"""
## VKR Fraud API — демо API для ВКР (антифрод скоринг)

### Что делает сервис
Сервис принимает транзакции (набор признаков) и возвращает:
- **risk_score** — вероятность мошенничества (0..1)
- **decision** — решение по порогам: `allow / review / deny`
- **top_reasons** — (опционально) причины / вклады признаков для top-k

---

## Авторизация
Текущее состояние: **{auth_state}**

Если включена авторизация — **каждый запрос** к `POST /predict` требует заголовок:

- `{hdr}: <ваш_ключ>`

Пример в PowerShell:
```powershell
-Headers @{{ "{hdr}" = "super-secret" }}
```
---
## Режимы модели (важно для ВКР)
### 1) model = `tabular`
Используется LightGBM (табличная модель), причины (`top_reasons`) доступны.

### 2) model = `fusion_external` (основной “системный” режим)
Используется честный fusion (external): объединяет tabular + gnn_score по формуле A10 (через logit).
**В запросе нужно передать `gnn_score` (0..1) на каждую транзакцию**.

---

## Endpoints (рекомендованный способ)
- `POST /predict/rows` — быстрый формат rows
- `POST /predict/canonical` — полный формат canonical (options + reasons)
- `POST /predict` — совместимость (auto-detect)
---
## Форматы входа: rows vs canonical

### 1) rows (короткий формат)

Когда нужно “быстро закинуть фичи”.
transaction_id создаётся автоматически: row_1, row_2, ...
```json
{{
  "rows": [
    {{"TransactionAmt": 100.0, "ProductCD": "W"}}
  ]
}}
```
### 2) canonical (расширенный формат)

Когда нужен свой transaction_id и нужно управлять options.
```json
{{
  "options": {{
    "model": "tabular",
    "with_reasons": true,
    "reasons_topk": 5
  }},
  "transactions": [
    {{
      "transaction_id": "tx_1",
      "features": {{"TransactionAmt": 100.0, "ProductCD": "W"}}
    }}
  ]
}}
```
---
### Что означает risk_score и decision
- risk_score ∈ [0..1] — скор/вероятность мошенничества (чем больше, тем “хуже”).
- decision вычисляется по порогам из `thresholds_tabular.json`:
    - `risk_score >= deny` → `deny`
    - `risk_score >= review` → `review`
    - иначе → `allow`
---
### Что такое top_reasons

Это список top-k причин (вклады признаков), которые сильнее всего повлияли на скор.
Возвращается только если `with_reasons=true` (canonical).

Ограничение безопасности: если строк слишком много — reasons могут быть отключены.
---
## Типичные ошибки и как их чинить
### 401 Invalid API key

Причина: не передан заголовок `{hdr}` или неверный ключ.

Решение: задай ключ и перезапусти сервис:
```powershell
$env:FRAUD_API_API_KEY="super-secret"
```
И передавай заголовок:
```powershell
-Headers @{{ "{hdr}" = "super-secret" }}
```
### 400 invalid payload

Причина: тело запроса не соответствует rows или canonical.

Решение: см. примеры выше или открой `GET /examples`.

### 503 not ready

Причина: сервис не загрузил модель/пороги/spec.

Решение: проверь `GET /ready`.

### 405 Method Not Allowed на /predict

Норма: `/predict` принимает только POST.
___
## Служебные endpoints
- `GET /` — краткая информация
- `GET /docs` — документация (эта страница)
- `GET /health` — жив ли сервис
- `GET /ready` — готовность + пути к артефактам
- `GET /features` — список признаков, группы и легенда
- `GET /metrics` — метрики (если включены)
- `GET /examples` — готовые примеры запросов
""".strip()

def _legend_for_feature(name: str) -> str:
# Короткая “расшифровка”, чтобы /docs выглядел человечно
    if name == "TransactionAmt":
        return "Сумма транзакции"
    if name == "ProductCD":
        return "Код продукта/типа операции"
    if name.startswith("card"):
        return "Атрибуты карты/банка/категории"
    if name.startswith("addr"):
        return "Адресные признаки"
    if name.startswith("dist"):
        return "Дистанционные признаки"
    if name.startswith("C"):
        return "Счётчики/агрегаты поведения (C*)"
    if name.startswith("D"):
        return "Временные дельты/интервалы (D*)"
    if name.startswith("M"):
        return "Match/флаги совпадений (M*)"
    if name.startswith("V"):
        return "Сгенерированные Vesta признаки (V*)"
    if name.startswith("id"):
        return "Identity признаки устройства/браузера (id_*)"
    if name in {"DeviceType", "DeviceInfo"}:
        return "Тип/информация об устройстве"
    if name in {"P_emaildomain", "R_emaildomain"}:
        return "Домены email покупателя/получателя"
    return "Признак (см. датасет/feature_spec)"


def create_app(settings: ApiSettings) -> FastAPI:
    tags_metadata = [
        {"name": "Service", "description": "Технические эндпоинты (health/ready/version/config)."},
        {"name": "Prediction", "description": "Скоринг транзакций и решение allow/review/deny."},
        {"name": "Ops", "description": "Метрики и сервисные операции."},
    ]

    app = FastAPI(
        title=settings.service_name,
        version="1.0",
        description=_docs_ru(settings),
        openapi_tags=tags_metadata,
    )

    install_error_handlers(app)

    metrics = Metrics(enabled=settings.enable_metrics)
    svc = FraudService(settings=settings)
    svc.reload()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    audit = deque(maxlen=int(settings.audit_max_events))

    app.add_middleware(RequestIdMiddleware)

    app.add_middleware(
        RateLimitMiddleware,
        header_name=settings.api_key_header,
        rps=settings.rate_limit_rps,
        burst=settings.rate_limit_burst,
        enabled=True,
    )

    if settings.enable_metrics:
        app.add_middleware(MetricsMiddleware, metrics=metrics)

    api_key_scheme = build_api_key_scheme(settings)

    def _auth_dep(api_key: str | None = Depends(api_key_scheme)) -> None:
        require_api_key(settings, api_key)

    # ---------------- Service

    @app.get("/version", response_model=VersionResponse, tags=["Service"])
    def version():
        return {
            "service": settings.service_name,
            "version": "1.0",
            "git_commit": (os.getenv("GIT_COMMIT") or os.getenv("FRAUD_API_GIT_COMMIT") or "unknown"),
            "started_at_utc": _STARTED_AT_UTC,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }

    @app.get("/config", response_model=ConfigResponse, tags=["Service"], summary="Безопасный срез настроек (без ключей)")
    def config():
        return {
            "service_name": settings.service_name,
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "auth_enabled": settings.auth_enabled(),
            "api_key_header": settings.api_key_header,
            "default_model": settings.default_model,
            "enable_reasons": settings.enable_reasons,
            "reasons_topk_default": settings.reasons_topk_default,
            "reasons_max_rows": settings.reasons_max_rows,
            "enable_metrics": settings.enable_metrics,
            "tabular_model_path": settings.tabular_model_path,
            "thresholds_tabular_path": settings.thresholds_tabular_path,
            "feature_spec_path": settings.feature_spec_path,
            "fusion_external_model_path": settings.fusion_external_model_path,
            "thresholds_fusion_external_path": settings.thresholds_fusion_external_path,
            "fusion_external_metrics_path": settings.fusion_external_metrics_path,
        }

    @app.get("/", tags=["Service"])
    def root():
        return {"service": settings.service_name, "docs": "/docs", "health": "/health", "version": "/version"}

    @app.get("/health", tags=["Service"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, tags=["Service"])
    def ready():
        return {
            "ready": svc.is_ready(),
            "model_path": settings.tabular_model_path,
            "thresholds_tabular_path": settings.thresholds_tabular_path,
            "feature_spec_path": settings.feature_spec_path,
            "fusion_external_model_path": settings.fusion_external_model_path,
            "thresholds_fusion_external_path": settings.thresholds_fusion_external_path,
            "fusion_external_metrics_path": settings.fusion_external_metrics_path,
            "default_model": settings.default_model,
            "error": svc.last_error,
        }

    # ---------------- Ops

    @app.get("/models", tags=["Ops"], summary="Состояние моделей (для аудита/ВКР)")
    def models():
        if not svc.is_ready():
            raise http_503("Service not ready. Call /ready")
        return svc.models_info()

    @app.get("/metrics", tags=["Ops"])
    def metrics_endpoint():
        return metrics.endpoint()

    @app.get("/audit/recent", tags=["Ops"], summary="Последние запросы (ring buffer) — для ВКР/аудита")
    def audit_recent(limit: int = 50):
        lim = max(1, min(int(limit), int(settings.audit_max_events)))
        data = list(audit)[-lim:]
        return {"limit": lim, "items": data}

    @app.post(
        "/reload",
        tags=["Ops"],
        dependencies=[Depends(_auth_dep)],
        summary="Перезагрузить артефакты (модель/пороги/spec) без рестарта",
    )
    def reload_endpoint():
        svc.reload()
        if not svc.is_ready():
            raise http_503(f"Reload failed: {svc.last_error}")
        return {"ok": True}

    @app.get("/features", tags=["Ops"], summary="Список признаков и их группы (num/cat) + легенда")
    def features():
        if not svc.is_ready():
            raise http_503("Service not ready. Call /ready")
        info = svc.features_info()
        expected = info.get("expected_columns", [])
        info["legend"] = {c: _legend_for_feature(c) for c in expected}
        info["fusion_external_hint"] = {"requires": "gnn_score per transaction", "gnn_score_range": "[0,1]"}
        return info

    @app.get("/examples", tags=["Ops"], summary="Готовые примеры запросов и PowerShell-команды")
    def examples():
        hdr = settings.api_key_header
        base = f"http://{settings.host}:{settings.port}"
        return {
            "powershell_rows_tabular": [
                '$payload = @{ rows = @(@{ TransactionAmt = 100.0; ProductCD = "W" }) } | ConvertTo-Json -Depth 10',
                f'Invoke-RestMethod -Uri "{base}/predict/rows" -Method Post -ContentType "application/json" -Headers @{{ "{hdr}" = "super-secret" }} -Body $payload',
            ],
            "powershell_canonical_tabular_reasons": [
                '$payload = @{ options = @{ model="tabular"; with_reasons=$true; reasons_topk=5 }; transactions = @(@{ transaction_id="tx_1"; features=@{ TransactionAmt=100.0; ProductCD="W" } }) } | ConvertTo-Json -Depth 20',
                f'Invoke-RestMethod -Uri "{base}/predict/canonical" -Method Post -ContentType "application/json" -Headers @{{ "{hdr}" = "super-secret" }} -Body $payload',
            ],
            "powershell_canonical_fusion_external": [
                '$payload = @{ options = @{ model="fusion_external"; with_reasons=$true; reasons_topk=5 }; transactions = @(@{ transaction_id="tx_1"; gnn_score=0.12; features=@{ TransactionAmt=100.0; ProductCD="W" } }) } | ConvertTo-Json -Depth 20',
                f'Invoke-RestMethod -Uri "{base}/predict/canonical" -Method Post -ContentType "application/json" -Headers @{{ "{hdr}" = "super-secret" }} -Body $payload',
            ],
        }

    # ---------------- Prediction helpers

    def _build_response(
        request: Request,
        input_format: str,
        tx_ids: List[str],
        items_raw: List[Dict[str, Any]],
        used_model: str,
    ) -> PredictResponse:
        rid = getattr(request.state, "request_id", None) or str(uuid.uuid4())
        items: List[PredictItem] = []
        for i, it in enumerate(items_raw):
            reasons = None
            if it.get("top_reasons") is not None:
                reasons = [ReasonItem(**r) for r in it["top_reasons"]]
            items.append(
                PredictItem(
                    transaction_id=tx_ids[i],
                    risk_score=float(it["risk_score"]),
                    decision=it["decision"],
                    top_reasons=reasons,
                )
            )
        return PredictResponse(model=used_model, input_format=input_format, items=items, request_id=rid)

    def _run_model(
            model_name: str,
            rows: List[Dict[str, Any]],
            gnn_scores: List[float],
            with_reasons: bool,
            reasons_topk: int,
    ) -> tuple[List[Dict[str, Any]], str]:
        m = str(model_name).strip().lower()
        if m not in {"tabular", "fusion_external"}:
            raise http_400("options.model must be one of: tabular, fusion_external")

        # validate for fusion early (чтобы не гонять executor впустую)
        if m == "fusion_external":
            if any(not np.isfinite(x) for x in gnn_scores):
                raise http_400("model=fusion_external requires gnn_score for each transaction (0..1).")
            if any((x < 0.0) or (x > 1.0) for x in gnn_scores):
                raise http_400("gnn_score must be in [0,1].")

        timeout_s = max(0.1, float(settings.predict_timeout_ms) / 1000.0)

        def _call() -> tuple[List[Dict[str, Any]], str]:
            if m == "tabular":
                return svc.predict_tabular(
                    rows=rows,
                    with_reasons=with_reasons,
                    reasons_topk=reasons_topk,
                )
            return svc.predict_fusion_external(
                rows=rows,
                gnn_scores=[float(x) for x in gnn_scores],
                with_reasons=with_reasons,
                reasons_topk=reasons_topk,
            )

        fut = executor.submit(_call)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise http_503(
                f"Prediction timeout ({settings.predict_timeout_ms} ms). "
                "Reduce batch size or disable reasons."
            )

    # ---------------- New explicit endpoints

    @app.post(
        "/predict/rows",
        response_model=PredictResponse,
        tags=["Prediction"],
        dependencies=[Depends(_auth_dep)],
        summary="Скоринг rows (явный эндпоинт). По умолчанию reasons выключены.",
    )
    def predict_rows(request: Request, req: PredictRequestRows = Body(...)):
        if not svc.is_ready():
            raise http_503("Service not ready. Call /ready")

        rows_raw = req.rows
        tx_ids = [f"row_{i + 1}" for i in range(len(rows_raw))]

        options = req.options
        model_name = (options.model if options else settings.default_model)

        # rows: быстрый режим, но если options.with_reasons=true — разрешим
        with_reasons = bool(options.with_reasons) if options else False
        reasons_topk = int(
            (options.reasons_topk if options else settings.reasons_topk_default)
            or settings.reasons_topk_default
        )

        gnn_scores: List[float] = []
        rows: List[Dict[str, Any]] = []
        for r in rows_raw:
            rr = dict(r or {})
            g = rr.pop("gnn_score", None)
            if g is None:
                gnn_scores.append(float("nan"))
            else:
                try:
                    gnn_scores.append(float(g))
                except Exception:
                    gnn_scores.append(float("nan"))
            rows.append(rr)

        items_raw, used_model = _run_model(
            model_name=model_name,
            rows=rows,
            gnn_scores=gnn_scores,
            with_reasons=with_reasons,
            reasons_topk=reasons_topk,
        )

        # -------- audit (4.6)
        counts = {"allow": 0, "review": 0, "deny": 0}
        for it in items_raw:
            d = it.get("decision")
            if d in counts:
                counts[d] += 1

        audit.append(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, "request_id", None),
                "endpoint": "predict/rows",
                "model_requested": str(model_name),
                "model_used": used_model,
                "n_items": len(items_raw),
                "decisions": counts,
                "with_reasons": bool(with_reasons),
                "reasons_topk": int(reasons_topk),
            }
        )
        # --------

        return _build_response(request, "rows", tx_ids, items_raw, used_model)

    @app.post(
        "/predict/canonical",
        response_model=PredictResponse,
        tags=["Prediction"],
        dependencies=[Depends(_auth_dep)],
        summary="Скоринг canonical (явный эндпоинт). Полные options + reasons.",
    )
    def predict_canonical(request: Request, req: PredictRequestCanonical = Body(...)):
        if not svc.is_ready():
            raise http_503("Service not ready. Call /ready")

        model_name = req.options.model or settings.default_model
        with_reasons = bool(req.options.with_reasons)
        reasons_topk = int(req.options.reasons_topk or settings.reasons_topk_default)

        tx_ids = [t.transaction_id for t in req.transactions]
        rows = [t.features for t in req.transactions]
        gnn_scores = [float(t.gnn_score) if t.gnn_score is not None else float("nan") for t in req.transactions]

        items_raw, used_model = _run_model(
            model_name=model_name,
            rows=rows,
            gnn_scores=gnn_scores,
            with_reasons=with_reasons,
            reasons_topk=reasons_topk,
        )

        # -------- audit (4.6)
        counts = {"allow": 0, "review": 0, "deny": 0}
        for it in items_raw:
            d = it.get("decision")
            if d in counts:
                counts[d] += 1

        audit.append(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, "request_id", None),
                "endpoint": "predict/canonical",
                "model_requested": str(model_name),
                "model_used": used_model,
                "n_items": len(items_raw),
                "decisions": counts,
                "with_reasons": bool(with_reasons),
                "reasons_topk": int(reasons_topk),
            }
        )
        # --------

        return _build_response(request, "canonical", tx_ids, items_raw, used_model)

    # ---------------- Backward compatible /predict (auto-detect)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Prediction"],
        dependencies=[Depends(_auth_dep)],
        summary="Скоринг (совместимость): auto-detect rows/canonical. Лучше использовать /predict/rows или /predict/canonical.",
    )
    def predict_compat(
        request: Request,
        payload: Dict[str, Any] = Body(..., description="Тело запроса (rows или canonical)."),
    ):
        if "rows" in payload:
            req = PredictRequestRows(**payload)
            return predict_rows(request, req)
        if "transactions" in payload:
            req = PredictRequestCanonical(**payload)
            return predict_canonical(request, req)
        raise http_400("Неверный формат тела. Нужно либо {'rows':[...]} либо {'options':..., 'transactions':[...]}")

    return app