"""
Загрузка, чистка и валидация данных о пожарах.

Функции:
- load_fact_sheet: загрузка листа Excel с нормализацией
- sheet_period: определение периода по имени листа
- clean_fire_data: основная функция чистки и валидации
"""

import hashlib
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


SOURCE_PRIORITY_MAP = {
    "lpr_manual_input": 110,
    "бд-1": 100,
    "1...": 80,
    "2...": 90,
}


def _infer_source_priority(source_sheet: Any) -> int:
    value = str(source_sheet or "").strip().lower()
    for prefix, priority in SOURCE_PRIORITY_MAP.items():
        if prefix in value:
            return priority
    return 10


def _stable_hash(parts: list[str]) -> str:
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _normalize_free_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def build_event_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Derive event identity and duplicate handling metadata."""
    enriched = df.copy()
    fire_dates = pd.to_datetime(enriched.get("fire_date"), errors="coerce")
    years = pd.to_numeric(enriched.get("year"), errors="coerce")

    row_ids = pd.to_numeric(enriched.get("row_id"), errors="coerce")
    region_codes = pd.to_numeric(enriched.get("region_code"), errors="coerce")
    settlement = pd.to_numeric(enriched.get("settlement_type_code"), errors="coerce")
    enterprise = pd.to_numeric(enriched.get("enterprise_type_code"), errors="coerce")
    floors = pd.to_numeric(enriched.get("building_floors"), errors="coerce")
    fire_floor = pd.to_numeric(enriched.get("fire_floor"), errors="coerce")
    source_item = pd.to_numeric(enriched.get("source_item_code"), errors="coerce")
    object_name = enriched.get("object_name", pd.Series(index=enriched.index, dtype=object))
    address = enriched.get("address", pd.Series(index=enriched.index, dtype=object))

    key_parts = pd.DataFrame(
        {
            "row_id": row_ids.map(lambda value: "" if pd.isna(value) else str(int(value))),
            "fire_date": fire_dates.dt.strftime("%Y-%m-%d").fillna(""),
            "year": years.map(lambda value: "" if pd.isna(value) else str(int(value))),
            "region_code": region_codes.map(lambda value: "" if pd.isna(value) else str(int(value))),
            "settlement_type_code": settlement.map(
                lambda value: "" if pd.isna(value) else str(int(value))
            ),
            "enterprise_type_code": enterprise.map(
                lambda value: "" if pd.isna(value) else str(int(value))
            ),
            "building_floors": floors.map(lambda value: "" if pd.isna(value) else str(int(value))),
            "fire_floor": fire_floor.map(lambda value: "" if pd.isna(value) else str(int(value))),
            "source_item_code": source_item.map(
                lambda value: "" if pd.isna(value) else str(int(value))
            ),
            "object_name": object_name.map(_normalize_free_text),
            "address": address.map(_normalize_free_text),
        }
    )

    quality_counts = key_parts.replace("", np.nan).count(axis=1)
    low_confidence = quality_counts < 5
    fingerprints = key_parts.apply(lambda row: "|".join(row.astype(str).tolist()), axis=1)
    fallback_fingerprints = key_parts.drop(columns=["row_id"]).apply(
        lambda row: "|".join(row.astype(str).tolist()),
        axis=1,
    )
    event_fingerprint = np.where(low_confidence, fallback_fingerprints, fingerprints)
    event_id = pd.Series(event_fingerprint).map(
        lambda value: f"evt_{_stable_hash([str(value)])}" if str(value).strip("|") else None
    )
    duplicate_group_id = pd.Series(fallback_fingerprints).map(
        lambda value: f"dup_{_stable_hash([str(value)])}" if str(value).strip("|") else None
    )

    enriched["event_fingerprint"] = event_fingerprint
    enriched["event_id"] = event_id
    enriched["duplicate_group_id"] = duplicate_group_id
    enriched["event_id_low_confidence"] = low_confidence.astype(bool)
    enriched["source_priority"] = enriched.get("source_sheet", pd.Series(index=enriched.index, dtype=object)).map(
        _infer_source_priority
    )

    completeness_columns = [
        "row_id",
        "fire_date",
        "year",
        "region_code",
        "settlement_type_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "source_item_code",
        "object_name",
        "address",
        "equipment",
        "equipment_count",
        "nozzle_count",
    ]
    available_columns = [column for column in completeness_columns if column in enriched.columns]
    enriched["_completeness_score"] = enriched[available_columns].notna().sum(axis=1)
    enriched["_group_size"] = enriched.groupby("duplicate_group_id")["duplicate_group_id"].transform("size")
    sort_columns = ["duplicate_group_id", "_completeness_score", "source_priority"]
    ascending = [True, False, False]
    if "created_at" in enriched.columns:
        sort_columns.append("created_at")
        ascending.append(True)
    enriched = enriched.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="stable",
    )
    enriched["is_canonical_event_record"] = (
        enriched.groupby("duplicate_group_id").cumcount() == 0
    )

    duplicate_policy = pd.Series("unique", index=enriched.index, dtype=object)
    duplicate_mask = enriched["_group_size"] > 1
    duplicate_policy.loc[duplicate_mask] = "canonical_event_only"
    conflict_fields = [column for column in ["equipment", "equipment_count", "nozzle_count"] if column in enriched.columns]
    if conflict_fields:
        conflict_groups = (
            enriched.groupby("duplicate_group_id")[conflict_fields]
            .nunique(dropna=True)
            .max(axis=1)
        )
        conflict_ids = conflict_groups[conflict_groups > 1].index
        duplicate_policy.loc[enriched["duplicate_group_id"].isin(conflict_ids)] = "conflict_kept"
    enriched["duplicate_policy"] = duplicate_policy
    enriched = enriched.sort_index()
    enriched = enriched.drop(columns=["_completeness_score", "_group_size"], errors="ignore")
    return enriched


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
    df = build_event_identity(df)
    df["rank_label_source"] = np.where(df["rank_tz"].notna() if "rank_tz" in df.columns else False, "historical_vector", None)
    df["human_verified"] = False
    df["usable_for_training"] = df["is_canonical_event_record"].fillna(False)
    df["rank_normative_version"] = None
    df["rank_quality_flags"] = None
    df["predicted_rank_at_decision"] = np.nan
    issues["canonical_rows"] = int(df["is_canonical_event_record"].sum())
    issues["duplicate_rows"] = int((~df["is_canonical_event_record"]).sum())
    issues["event_id_coverage_pct"] = float(df["event_id"].notna().mean() * 100)
    return df, issues
