from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import pandas as pd


@dataclass
class TabularSpec:
    target_col: str = "target"
    id_col: str = "transaction_id"
    time_col: str = "time"


def get_feature_cols(df: pd.DataFrame, spec: TabularSpec) -> List[str]:
    """
    Возвращает список фичей (всё, кроме id/time/target).
    """
    exclude = {spec.target_col, spec.id_col, spec.time_col}
    return [c for c in df.columns if c not in exclude]


def detect_feature_types(df: pd.DataFrame, feature_cols: List[str], sample_size: int = 2000,
                         non_numeric_threshold: float = 0.01) -> Tuple[List[str], List[str]]:
    """
    Устойчивое определение типов:
    - если колонка содержит строковые/нечисловые значения (даже при странном dtype),
      мы относим её к categorical.
    - иначе numeric.

    non_numeric_threshold: доля значений в сэмпле, которые не удалось привести к числу.
    """
    sample = df[feature_cols].head(sample_size)

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in feature_cols:
        s = sample[c]

        # Если dtype уже object/category -> считаем категориальной сразу
        if s.dtype == "object" or str(s.dtype) == "category":
            cat_cols.append(c)
            continue

        # Пробуем привести к числу по фактическим значениям
        s_num = pd.to_numeric(s, errors="coerce")
        non_num_rate = float(s_num.isna().mean() - s.isna().mean())  # "добавочные" NaN из-за coercion

        # Если есть существенная доля нечисловых значений -> categorical
        if non_num_rate > non_numeric_threshold:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return num_cols, cat_cols


def build_xy(df: pd.DataFrame, spec: TabularSpec, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Строим X и y.
    """
    X = df[feature_cols].copy()
    y = df[spec.target_col].astype("int8")
    return X, y


def sanity_check_types(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str], sample_size: int = 50) -> None:
    """
    Проверяем, что num_cols реально числовые (конвертируются в float).
    """
    bad = []
    for c in num_cols:
        s = X[c].head(sample_size)
        # если появляются "новые NaN" при to_numeric => там есть мусор типа 'W'
        s_num = pd.to_numeric(s, errors="coerce")
        extra_nan = s_num.isna().sum() - s.isna().sum()
        if extra_nan > 0:
            bad.append(c)

    if bad:
        example_col = bad[0]
        ex = X[example_col].dropna().astype(str).head(10).tolist()
        raise ValueError(
            f"Non-numeric values found in num_cols. Example bad column='{example_col}'. "
            f"Sample values={ex}. Total bad columns={len(bad)} (showing first 10): {bad[:10]}"
        )

def refine_numeric_columns(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    sample_size: int = 5000,
    non_numeric_threshold: float = 0.001,
) -> Tuple[List[str], List[str]]:
    """
    Второй проход: проверяем num_cols на наличие нечисловых значений.
    Если доля "нечислового мусора" выше порога — переносим колонку в cat_cols.

    non_numeric_threshold:
      0.001 = 0.1% "мусора" в сэмпле уже считаем признак неустойчиво числовым.
    """
    sample = X[num_cols].head(sample_size)

    move_to_cat: List[str] = []
    for c in num_cols:
        s = sample[c]
        s_num = pd.to_numeric(s, errors="coerce")
        extra_nan = float(s_num.isna().mean() - s.isna().mean())  # NaN из-за coercion
        if extra_nan > non_numeric_threshold:
            move_to_cat.append(c)

    if move_to_cat:
        num_cols = [c for c in num_cols if c not in move_to_cat]
        cat_cols = cat_cols + move_to_cat

    return num_cols, cat_cols