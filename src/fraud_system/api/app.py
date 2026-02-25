from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .errors import unhandled_exception_handler
from .middleware import RequestIdMiddleware, TimingMiddleware
from .schemas import PredictRequest, PredictResponse, ReloadResponse
from .security import require_api_key
from .service import FraudService
from .settings import ApiSettings, BuildInfo


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    build = BuildInfo()

    tags = [
        {
            "name": "Сервис",
            "description": (
                "Проверка работоспособности сервиса, readiness и метрики.\n\n"
                "**/docs** — это интерактивная документация Swagger: можно прямо там отправлять запросы."
            ),
        },
        {
            "name": "Скоринг",
            "description": (
                "Скоринг транзакций.\n\n"
                "Поддерживаются два формата входа:\n"
                "1) **rows** (короткий): `{ \"rows\": [ { ...features... } ] }`\n"
                "2) **canonical** (полный): `{ \"options\": {...}, \"transactions\": [ {\"transaction_id\":..., \"features\": {...}} ] }`\n\n"
                "⚠️ Можно присылать не все фичи — недостающие будут добавлены как пустые (NaN), "
                "чтобы модель (ColumnTransformer) не падала."
            ),
        },
        {
            "name": "Администрирование",
            "description": "Перезагрузка артефактов (модель/пороги/feature_spec) без рестарта.",
        },
    ]

    app = FastAPI(
        title="VKR Fraud System API",
        version=build.version,
        description=(
            "API для скоринга банковских транзакций (табличная модель).\n\n"
            "## Авторизация\n"
            "Используется заголовок **X-API-Key**.\n\n"
            "### Пример (PowerShell)\n"
            "```powershell\n"
            "$env:FRAUD_API_API_KEY=\"super-secret\"\n"
            "python -m scripts.13_serve_api\n"
            "\n"
            "$payload = @{ rows = @(@{ TransactionAmt = 100.0; ProductCD = \"W\" }) } | ConvertTo-Json -Depth 10\n"
            "Invoke-RestMethod -Uri \"http://127.0.0.1:8000/predict\" -Method Post -ContentType \"application/json\" "
            "-Headers @{ \"X-API-Key\" = \"super-secret\" } -Body $payload\n"
            "```\n\n"
            "## rows vs canonical\n"
            "- **rows** — быстрее и проще (только фичи, transaction_id генерируется автоматически)\n"
            "- **canonical** — полный контроль над transaction_id + options\n"
        ),
        openapi_tags=tags,
    )

    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    svc = FraudService(
        tabular_model_path=settings.tabular_model_path,
        fusion_model_path=settings.fusion_model_path,
        thresholds_tabular_path=settings.thresholds_tabular_path,
        thresholds_fusion_external_path=settings.thresholds_fusion_external_path,
        feature_spec_path=settings.feature_spec_path,
        enable_reasons=settings.enable_reasons,
    )

    @app.get("/", tags=["Сервис"])
    def index():
        return {
            "service": build.service,
            "version": build.version,
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
        }

    @app.get("/health", tags=["Сервис"])
    def health():
        return {"status": "ok", "service": build.service, "version": build.version}

    @app.get("/ready", tags=["Сервис"])
    def ready():
        return {"ready": True, "artifacts": svc.paths}

    @app.get("/metrics", tags=["Сервис"])
    def metrics() -> PlainTextResponse:
        data = generate_latest()
        return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Скоринг"],
        dependencies=[Depends(require_api_key)],
        summary="Скоринг транзакций (rows или canonical)",
        description=(
            "### Вариант 1: rows\n"
            "```json\n"
            "{\n"
            "  \"rows\": [\n"
            "    {\"TransactionAmt\": 100.0, \"ProductCD\": \"W\"}\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "### Вариант 2: canonical\n"
            "```json\n"
            "{\n"
            "  \"options\": {\"model\":\"tabular\",\"input_format\":\"canonical\",\"with_reasons\":false,\"reasons_topk\":5},\n"
            "  \"transactions\": [\n"
            "    {\"transaction_id\":\"tx_1\",\"features\":{\"TransactionAmt\":100.0,\"ProductCD\":\"W\"}}\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "⚠️ Если прислали мало фич — сервис сам добавит недостающие как пустые значения, "
            "чтобы модель не падала."
        ),
    )
    def predict(req: PredictRequest, request: Request):
        rid = request.state.request_id

        # Нормализуем rows/canonical в единый records
        if req.rows is not None:
            records = [
                {"transaction_id": f"row_{i+1}", "features": row}
                for i, row in enumerate(req.rows)
            ]
            input_format = "rows"
        else:
            records = [{"transaction_id": t.transaction_id, "features": t.features} for t in (req.transactions or [])]
            input_format = "canonical"

        out = svc.predict(
            model=req.options.model,
            records=records,
            with_reasons=req.options.with_reasons,
            reasons_topk=req.options.reasons_topk,
            reasons_max_rows=req.options.reasons_max_rows,
        )

        # Проставим корректные поля ответа
        out["request_id"] = rid
        out["input_format"] = input_format
        return out

    @app.post(
        "/reload",
        response_model=ReloadResponse,
        tags=["Администрирование"],
        dependencies=[Depends(require_api_key)],
        summary="Перезагрузить артефакты (модель/пороги/spec) без рестарта",
    )
    def reload_artifacts(request: Request):
        rid = request.state.request_id
        svc.reload()
        return {"status": "ok", "reloaded": True, "artifacts": {**svc.paths, "request_id": rid}}

    return app