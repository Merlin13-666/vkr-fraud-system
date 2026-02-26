from __future__ import annotations

import platform
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

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
)
from fraud_system.api.security import build_api_key_scheme, require_api_key
from fraud_system.api.service import FraudService
from fraud_system.api.settings import ApiSettings


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
    "input_format": "canonical",
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

def _get_git_commit_short() -> str:
    """
    Пытаемся получить короткий hash текущего коммита.
    Работает, если запускается из git-репозитория и git доступен в PATH.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        s = (out or "").strip()
        return s if s else "unknown"
    except Exception:
        return "unknown"

def create_app(settings: ApiSettings) -> FastAPI:
    tags_metadata = [
        {"name": "Service", "description": "Технические эндпоинты (health/ready)."},
        {"name": "Prediction", "description": "Скоринг транзакций и решение allow/review/deny."},
        {"name": "Ops", "description": "Метрики и сервисные операции."},
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    git_commit = _get_git_commit_short()

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

    app.add_middleware(RequestIdMiddleware)
    if settings.enable_metrics:
        app.add_middleware(MetricsMiddleware, metrics=metrics)

    api_key_scheme = build_api_key_scheme(settings)

    def _auth_dep(api_key: str | None = Depends(api_key_scheme)) -> None:
        require_api_key(settings, api_key)

    # -----------------
    # Service
    # -----------------

    @app.get("/", tags=["Service"])
    def root():
        return {"service": settings.service_name, "docs": "/docs", "health": "/health", "version": "/version"}

    @app.get("/version", response_model=VersionResponse, tags=["Service"], summary="Build/runtime info for audit")
    def version():
        return VersionResponse(
            service=settings.service_name,
            version="1.0",
            git_commit=git_commit,
            started_at_utc=started_at,
            python=platform.python_version(),
            platform=f"{platform.system()} {platform.release()} ({platform.platform()})",
        )

    @app.get("/health", tags=["Service"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, tags=["Service"])
    def ready():
        return {
            "ready": svc.is_ready(),
            "model_path": settings.tabular_model_path,
            "thresholds_path": settings.thresholds_path,
            "feature_spec_path": settings.feature_spec_path,
            "default_model": settings.default_model,
            "error": svc.last_error,
        }

    # -----------------
    # Ops
    # -----------------

    @app.get("/metrics", tags=["Ops"])
    def metrics_endpoint():
        return metrics.endpoint()

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
        return info

    @app.get("/examples", tags=["Ops"], summary="Готовые примеры запросов и PowerShell-команды")
    def examples():
        hdr = settings.api_key_header
        return {
            "powershell_rows": [
                '$payload = @{ rows = @(@{ TransactionAmt = 100.0; ProductCD = "W" }) } | ConvertTo-Json -Depth 10',
                f'Invoke-RestMethod -Uri "http://{settings.host}:{settings.port}/predict" -Method Post -ContentType "application/json" -Headers @{{ "{hdr}" = "super-secret" }} -Body $payload',
            ],
            "powershell_canonical": [
                '$payload = @{ options = @{ model="tabular"; input_format="canonical"; with_reasons=$true; reasons_topk=5 }; transactions = @(@{ transaction_id="tx_1"; features=@{ TransactionAmt=100.0; ProductCD="W" } }) } | ConvertTo-Json -Depth 20',
                f'Invoke-RestMethod -Uri "http://{settings.host}:{settings.port}/predict" -Method Post -ContentType "application/json" -Headers @{{ "{hdr}" = "super-secret" }} -Body $payload',
            ],
        }

    # -----------------
    # Prediction
    # -----------------

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Prediction"],
        dependencies=[Depends(_auth_dep)],
        summary="Скоринг транзакций (tabular). Поддерживает rows и canonical.",
    )
    def predict(
            request: Request,
            payload: Dict[str, Any] = Body(
                ...,
                description="Тело запроса (rows или canonical). См. описание выше и /examples.",
            ),
    ):
        if not svc.is_ready():
            raise http_503("Service not ready. Call /ready")

        rid = getattr(request.state, "request_id", None) or str(uuid.uuid4())

        # Определяем формат по ключам
        if "rows" in payload:
            req = PredictRequestRows(**payload)
            input_format = "rows"
            rows = req.rows
            tx_ids = [f"row_{i + 1}" for i in range(len(rows))]
            with_reasons = False
            reasons_topk = settings.reasons_topk_default
        elif "transactions" in payload:
            req = PredictRequestCanonical(**payload)
            input_format = "canonical"
            with_reasons = req.options.with_reasons
            reasons_topk = req.options.reasons_topk or settings.reasons_topk_default
            tx_ids = [t.transaction_id for t in req.transactions]
            rows = [t.features for t in req.transactions]
        else:
            raise http_400("Неверный формат тела. Нужно либо {'rows':[...]} либо {'options':..., 'transactions':[...]}")

        items_raw, model_name = svc.predict_tabular(
            rows=rows,
            with_reasons=with_reasons,
            reasons_topk=reasons_topk,
        )

        items: List[PredictItem] = []
        for i, it in enumerate(items_raw):
            reasons = None
            if it.get("top_reasons") is not None:
                reasons = [ReasonItem(**r) for r in it["top_reasons"]]
            items.append(
                PredictItem(
                    transaction_id=tx_ids[i],
                    risk_score=it["risk_score"],
                    decision=it["decision"],
                    top_reasons=reasons,
                )
            )

        return PredictResponse(
            model=model_name,
            input_format=input_format,
            items=items,
            request_id=rid,
        )

    return app