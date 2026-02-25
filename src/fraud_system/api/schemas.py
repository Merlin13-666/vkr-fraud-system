from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


ModelName = Literal["tabular", "fusion_external"]
InputFormat = Literal["rows", "canonical"]


class PredictOptions(BaseModel):
    model: ModelName = Field(default="tabular", description="Какую модель использовать.")
    input_format: InputFormat = Field(
        default="rows",
        description=(
            "Формат входа.\n"
            "- rows: короткая форма (просто список объектов с фичами)\n"
            "- canonical: полная форма (transactions[{transaction_id, features}])"
        ),
    )
    with_reasons: bool = Field(
        default=False,
        description="Пытаться вернуть объяснения (top_reasons). Если выключено — top_reasons будет null.",
    )
    reasons_topk: int = Field(default=5, ge=1, le=50, description="Сколько top причин вернуть (если включено).")
    reasons_max_rows: int = Field(default=30000, ge=1, le=200000, description="Лимит строк для explainability.")


class CanonicalTransaction(BaseModel):
    transaction_id: str = Field(..., description="ID транзакции в ответе (любая строка).", examples=["tx_1"])
    features: Dict[str, Any] = Field(
        ...,
        description="Словарь фич. Можно присылать не все — недостающие будут добавлены автоматически как пустые.",
        examples=[{"TransactionAmt": 100.0, "ProductCD": "W"}],
    )


class PredictRequest(BaseModel):
    """
    Единый запрос, который умеет принимать ДВА варианта:

    1) rows (короткий):
        { "rows": [ { ...features... }, { ... } ] }

    2) canonical (полный):
        {
          "options": {...},
          "transactions": [
            { "transaction_id": "...", "features": {...} }
          ]
        }
    """

    # short form
    rows: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Короткий формат. Список объектов-фич.",
        examples=[[{"TransactionAmt": 100.0, "ProductCD": "W"}]],
    )

    # canonical form
    options: PredictOptions = Field(default_factory=PredictOptions)
    transactions: Optional[List[CanonicalTransaction]] = Field(
        default=None,
        description="Полный формат. Список транзакций (transaction_id + features).",
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> "PredictRequest":
        # Разрешаем:
        # - либо rows
        # - либо transactions
        if self.rows is None and (self.transactions is None or len(self.transactions) == 0):
            raise ValueError('Send either "rows" or "transactions".')
        if self.rows is not None and self.transactions is not None:
            raise ValueError('Send only one of: "rows" OR "transactions", not both.')
        return self


class TopReason(BaseModel):
    feature: str
    contribution: float = Field(description="Относительный вклад (пока эвристика/заглушка).")


class PredictItem(BaseModel):
    transaction_id: str
    risk_score: float = Field(description="Вероятность мошенничества (0..1).")
    decision: Literal["allow", "review", "deny"] = Field(description="Решение по порогам.")
    top_reasons: Optional[List[TopReason]] = Field(default=None, description="Объяснения (если включено).")


class PredictResponse(BaseModel):
    model: ModelName
    input_format: InputFormat
    items: List[PredictItem]
    request_id: str


class ReloadResponse(BaseModel):
    status: str
    reloaded: bool
    artifacts: Dict[str, Any]