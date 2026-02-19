"""
Загрузка, чистка и валидация данных о пожарах.

Функции:
- load_fact_sheet: загрузка листа Excel с нормализацией
- sheet_period: определение периода по имени листа
- clean_fire_data: основная функция чистки и валидации
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from .schema import (
    CODE_COLS,
    HEADER_ROWS,
    MAX_DISTANCE,
    MAX_FLOORS,
    MAX_YEAR,
    MIN_YEAR,
    OUTPUT_COLS,
    PERIOD_MAPPING,
    RU_COLS,
    RU_TO_EN,
    TIME_COLS,
)
from .utils import compute_rank_ref_v2, first_int, normalize_text, parse_time


def load_fact_sheet(
    sheet_name: str, xl: pd.ExcelFile, header_row: Optional[int] = None
) -> pd.DataFrame:
    """
    Загрузка факта из листа Excel с нормализацией колонок.

    Args:
        sheet_name: имя листа
        xl: pd.ExcelFile объект
        header_row: номер строки заголовка (опционально)

    Returns:
        DataFrame с нормализованными колонками
    """
    # Определение номера строки заголовка
    header = HEADER_ROWS.get("default")
    for prefix, row in HEADER_ROWS.items():
        if sheet_name.startswith(prefix):
            header = row
            break
    if header_row is not None:
        header = header_row

    df = xl.parse(sheet_name, header=header)

    # Нормализация имён колонок
    from .utils import map_col, norm_col

    df.columns = [norm_col(c) for c in df.columns]

    # Удаление колонок Unnamed
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Маппинг колонок
    col_map = {c: map_col(c) for c in df.columns}
    col_map = {k: v for k, v in col_map.items() if v}
    df = df.rename(columns=col_map)

    # Удаление дубликатов колонок
    df = df.loc[:, ~df.columns.duplicated()]

    # Добавление отсутствующих колонок
    for col in RU_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[RU_COLS].copy()
    df["source_sheet"] = sheet_name

    return df


def sheet_period(name: str) -> str:
    """
    Определение периода по имени листа.
    """
    for prefix, period in PERIOD_MAPPING.items():
        if name.startswith(prefix):
            return period
    return "other"


def clean_fire_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Очистка и валидация данных о пожарах.

    Выполняет:
    1. Переименование колонок RU → EN
    2. Валидация дат (2000–2020)
    3. Извлечение кодов из текстовых полей
    4. Парсинг временных колонок
    5. Создание severity_score и rank_ref (исследовательский)
    6. Флаги качества (выбросы, пропуски, некорректные значения)

    Args:
        df: DataFrame с исходными данными (колонки на русском)

    Returns:
        (clean_df, quality_report)
        - clean_df: очищенный DataFrame с флагами
        - quality_report: dict со статистикой проблем
    """
    df = df.copy()

    # 1. Переименование колонок
    df = df.rename(columns=RU_TO_EN)
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Валидация дат
    df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
    df["year"] = df["fire_date"].dt.year
    df["month"] = df["fire_date"].dt.month
    df["flag_date_outlier"] = (df["year"] < MIN_YEAR) | (df["year"] > MAX_YEAR)
    df.loc[df["flag_date_outlier"], ["fire_date", "year", "month"]] = [
        pd.NaT,
        np.nan,
        np.nan,
    ]

    # 3. Извлечение кодов из текстовых полей
    for col in CODE_COLS:
        df[f"{col}_code"] = df[col].apply(first_int)
        df[f"{col}_text"] = normalize_text(df[col])

    # Парсинг числовых колонок
    for col in ["equipment_count", "nozzle_count"]:
        df[col] = df[col].apply(first_int)

    # 4. Преобразование числовых колонок
    num_cols = [
        "building_floors",
        "fire_floor",
        "distance_to_station",
        "fatalities",
        "injuries",
        "direct_damage",
        "people_saved",
        "people_evacuated",
        "assets_saved",
        "equipment_count",
        "nozzle_count",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Парсинг временных колонок
    for col in TIME_COLS:
        parsed = df[col].apply(parse_time)
        df[col + "_min"] = parsed.apply(lambda x: x[0])
        df[col + "_invalid"] = parsed.apply(lambda x: x[1])

    # 6. Логарифм ущерба для severity
    df["direct_damage_log"] = np.log1p(df["direct_damage"])

    # 7. Severity score (исследовательский признак)
    severity = (
        df["fatalities"].fillna(0) * 5
        + df["injuries"].fillna(0) * 2
        + df["direct_damage_log"]
    )
    all_missing = df[["fatalities", "injuries", "direct_damage"]].isna().all(axis=1)
    df["severity_score"] = severity.mask(all_missing, np.nan)

    # 8. Rank ref (исследовательский, не по ТЗ)
    df["rank_ref"], rank_info = compute_rank_ref_v2(df["severity_score"])

    # 9. Флаги качества

    # Выбросы этажности
    df["flag_floor_outlier"] = df["building_floors"] > MAX_FLOORS

    # Выбросы расстояния
    df["flag_distance_outlier"] = df["distance_to_station"] > MAX_DISTANCE

    # Несогласованность этажей (fire_floor > building_floors)
    df["flag_floor_inconsistent"] = (
        df["fire_floor"].notna()
        & df["building_floors"].notna()
        & (df["fire_floor"] > df["building_floors"])
    )

    # Отрицательные значения
    df["flag_negative_values"] = df[num_cols].lt(0).any(axis=1)

    # Выбросы ущерба (99.5-й перцентиль)
    damage_limit = (
        df["direct_damage"].dropna().quantile(0.995)
        if df["direct_damage"].notna().any()
        else np.nan
    )
    df["flag_damage_outlier"] = df["direct_damage"] > damage_limit

    # Некорректные времена
    time_invalid_cols = [c for c in df.columns if c.endswith("_invalid")]
    df["flag_time_invalid"] = df[time_invalid_cols].any(axis=1)

    # Пропуски выходов (нет ни fatalities, ни injuries, ни direct_damage)
    df["flag_missing_outputs"] = df[OUTPUT_COLS].isna().all(axis=1)

    # 10. Отчёт о качестве
    issues = {
        "invalid_time_counts": {
            c: int(df[c].sum()) for c in df.columns if c.endswith("_invalid")
        },
        "floor_outliers": int(df["flag_floor_outlier"].sum()),
        "distance_outliers": int(df["flag_distance_outlier"].sum()),
        "floor_inconsistent": int(df["flag_floor_inconsistent"].sum()),
        "negative_values": int(df["flag_negative_values"].sum()),
        "damage_outliers": int(df["flag_damage_outlier"].sum()),
        "missing_outputs": int(df["flag_missing_outputs"].sum()),
        "rank_quantiles": rank_info,
    }

    return df, issues
