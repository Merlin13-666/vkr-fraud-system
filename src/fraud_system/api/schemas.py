from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- INPUT ---

class PredictOptions(BaseModel):
    model: str = Field(default="tabular", description="Какую модель использовать (пока доступна tabular).")
    input_format: Literal["canonical"] = Field(default="canonical", description="Формат input (для канонического запроса).")
    with_reasons: bool = Field(default=False, description="Вернуть top_reasons (вклады признаков).")
    reasons_topk: int = Field(default=5, ge=1, le=50, description="Сколько причин вернуть (top-k).")


class CanonicalTransaction(BaseModel):
    transaction_id: str = Field(..., description="Идентификатор транзакции (ваш).")
    features: Dict[str, Any] = Field(..., description="Словарь фич (можно присылать частично).")


class PredictRequestCanonical(BaseModel):
    options: PredictOptions = Field(default_factory=PredictOptions)
    transactions: List[CanonicalTransaction]


class PredictRequestRows(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="Список строк-фич. transaction_id будет row_1, row_2, ...")


# --- OUTPUT ---

class ReasonItem(BaseModel):
    feature: str = Field(..., description="Название признака.")
    contribution: float = Field(..., description="Вклад признака в скор (приближенно, по contrib).")
    value: Optional[Any] = Field(default=None, description="Значение признака в запросе (если было).")


class PredictItem(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Вероятность fraud (от 0 до 1).")
    decision: Literal["allow", "review", "deny"] = Field(..., description="Решение по порогам.")
    top_reasons: Optional[List[ReasonItem]] = Field(default=None, description="Причины (если запрошены).")


class PredictResponse(BaseModel):
    model: str
    input_format: str
    items: List[PredictItem]
    request_id: str


class ReadyResponse(BaseModel):
    ready: bool
    model_path: str
    thresholds_path: str
    feature_spec_path: str
    default_model: str
    error: Optional[str] = None