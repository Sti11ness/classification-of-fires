"""
Shared rank_tz ML contract for training, deployment, and UI.

This module centralizes:
- feature set contracts for rank_tz
- deploy roles (production vs offline benchmark)
- preprocessor artifact generation and application
- rank/class mappings
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd

RANK_TZ_SCHEMA_VERSION = "rank_tz_schema_v2"
PRODUCTION_DEPLOYMENT_ROLE = "rank_tz_lpr_production"
OFFLINE_DEPLOYMENT_ROLE = "rank_tz_offline_benchmark"

RANK_TO_CLASS_MAP = {
    1.0: 1,
    1.5: 2,
    2.0: 3,
    3.0: 4,
    4.0: 5,
    5.0: 6,
}
CLASS_TO_RANK_MAP = {value: key for key, value in RANK_TO_CLASS_MAP.items()}

FIELD_SPECS: dict[str, dict[str, Any]] = {
    "region_code": {
        "name": "region_code",
        "label": "Код региона",
        "type": "int",
        "min": 0,
        "max": 99999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "settlement_type_code": {
        "name": "settlement_type_code",
        "label": "Тип населенного пункта",
        "type": "int",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "fire_protection_code": {
        "name": "fire_protection_code",
        "label": "Вид пожарной охраны",
        "type": "int",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "enterprise_type_code": {
        "name": "enterprise_type_code",
        "label": "Тип предприятия",
        "type": "int",
        "min": 0,
        "max": 999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "building_floors": {
        "name": "building_floors",
        "label": "Этажность здания",
        "type": "int",
        "min": 0,
        "max": 200,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "fire_floor": {
        "name": "fire_floor",
        "label": "Этаж пожара",
        "type": "int",
        "min": -1,
        "max": 200,
        "null_sentinel": -1,
        "allow_negative_input": True,
    },
    "fire_resistance_code": {
        "name": "fire_resistance_code",
        "label": "Степень огнестойкости",
        "type": "int",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "source_item_code": {
        "name": "source_item_code",
        "label": "Источник зажигания",
        "type": "int",
        "min": 0,
        "max": 999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "distance_to_station": {
        "name": "distance_to_station",
        "label": "Расстояние до части",
        "type": "float",
        "min": 0.0,
        "max": 1000.0,
        "null_sentinel": -1.0,
        "suffix": " км",
        "decimals": 2,
        "allow_negative_input": False,
    },
    "fatalities": {
        "name": "fatalities",
        "label": "Погибшие",
        "type": "int",
        "min": 0,
        "max": 1000,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "injuries": {
        "name": "injuries",
        "label": "Пострадавшие",
        "type": "int",
        "min": 0,
        "max": 5000,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "direct_damage": {
        "name": "direct_damage",
        "label": "Прямой ущерб",
        "type": "float",
        "min": 0.0,
        "max": 1000000000.0,
        "null_sentinel": -1.0,
        "suffix": " руб.",
        "decimals": 2,
        "allow_negative_input": False,
    },
    "t_detect_min": {
        "name": "t_detect_min",
        "label": "Время обнаружения",
        "type": "int",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
        "suffix": " мин",
        "allow_negative_input": False,
    },
    "t_report_min": {
        "name": "t_report_min",
        "label": "Время сообщения",
        "type": "int",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
        "suffix": " мин",
        "allow_negative_input": False,
    },
    "t_arrival_min": {
        "name": "t_arrival_min",
        "label": "Время прибытия",
        "type": "int",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
        "suffix": " мин",
        "allow_negative_input": False,
    },
    "t_first_hose_min": {
        "name": "t_first_hose_min",
        "label": "Время подачи первого ствола",
        "type": "int",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
        "suffix": " мин",
        "allow_negative_input": False,
    },
}

FEATURE_SET_SPECS: dict[str, dict[str, Any]] = {
    "basic": {
        "feature_set": "basic",
        "feature_order": [
            "building_floors",
            "fire_floor",
            "distance_to_station",
            "fatalities",
            "injuries",
            "direct_damage",
        ],
        "deployment_role": OFFLINE_DEPLOYMENT_ROLE,
        "offline_only": True,
        "default_fill_strategy": "constant",
        "default_fill_value": 0.0,
        "label": "Basic (legacy baseline)",
    },
    "extended": {
        "feature_set": "extended",
        "feature_order": [
            "region_code",
            "settlement_type_code",
            "fire_protection_code",
            "enterprise_type_code",
            "building_floors",
            "fire_floor",
            "fire_resistance_code",
            "source_item_code",
            "distance_to_station",
            "fatalities",
            "injuries",
            "direct_damage",
            "t_detect_min",
            "t_report_min",
            "t_arrival_min",
            "t_first_hose_min",
            "t_contained_min",
            "t_extinguished_min",
        ],
        "deployment_role": OFFLINE_DEPLOYMENT_ROLE,
        "offline_only": True,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Extended (offline benchmark)",
    },
    "online_tactical": {
        "feature_set": "online_tactical",
        "feature_order": [
            "region_code",
            "settlement_type_code",
            "fire_protection_code",
            "enterprise_type_code",
            "building_floors",
            "fire_floor",
            "fire_resistance_code",
            "source_item_code",
            "distance_to_station",
            "t_detect_min",
            "t_report_min",
            "t_arrival_min",
            "t_first_hose_min",
        ],
        "deployment_role": PRODUCTION_DEPLOYMENT_ROLE,
        "offline_only": False,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Online tactical (production)",
    },
    "enhanced_tactical": {
        "feature_set": "enhanced_tactical",
        "feature_order": [
            "region_code",
            "settlement_type_code",
            "fire_protection_code",
            "enterprise_type_code",
            "building_floors",
            "fire_floor",
            "fire_resistance_code",
            "source_item_code",
            "distance_to_station",
            "day_of_week",
            "month",
            "season",
            "is_weekend",
            "is_high_rise",
            "is_super_high_rise",
            "basement_fire",
            "top_floor_fire",
            "fire_floor_ratio",
            "is_far",
            "is_very_far",
            "distance_category",
            "building_floors_bin",
            "risk_category_missing",
            "fpo_class_missing",
            "t_detect_min",
            "t_report_min",
            "t_arrival_min",
            "t_first_hose_min",
            "delta_detect_to_report",
            "delta_report_to_arrival",
            "delta_arrival_to_hose",
        ],
        "deployment_role": OFFLINE_DEPLOYMENT_ROLE,
        "offline_only": True,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Enhanced tactical (offline experiment)",
    },
}


def get_feature_set_spec(feature_set: str) -> dict[str, Any]:
    """Return a feature set spec with a defensive copy."""
    if feature_set not in FEATURE_SET_SPECS:
        raise ValueError(f"Unknown rank_tz feature set: {feature_set}")
    return deepcopy(FEATURE_SET_SPECS[feature_set])


def get_input_schema(feature_set: str) -> list[dict[str, Any]]:
    """Return UI/input schema for a feature set."""
    spec = get_feature_set_spec(feature_set)
    schema: list[dict[str, Any]] = []
    for feature_name in spec["feature_order"]:
        field = deepcopy(FIELD_SPECS.get(feature_name, {"name": feature_name, "label": feature_name}))
        field.setdefault("type", "float")
        field.setdefault("null_sentinel", -1.0)
        schema.append(field)
    return schema


def map_rank_series_to_classes(y: pd.Series) -> pd.Series:
    """Map rank_tz float labels to classifier classes."""
    return y.map(lambda value: RANK_TO_CLASS_MAP.get(float(value)) if pd.notna(value) else np.nan)


def class_list_to_rank_values(classes: list[int]) -> list[float]:
    """Map classifier integer classes back to rank_tz float values."""
    return [CLASS_TO_RANK_MAP.get(int(value), float(value)) for value in classes]


def build_preprocessor_artifact(
    df: pd.DataFrame,
    *,
    feature_order: list[str],
    feature_set: str,
    fill_strategy: str,
    fill_value: Optional[float],
    training_rows: int,
    test_size: float,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build a serializable preprocessor artifact and transformed features."""
    frame = ensure_feature_frame(df, feature_order)
    fill_values: dict[str, float] = {}

    for column in feature_order:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if fill_strategy == "median":
            fill = numeric.median()
            if pd.isna(fill):
                fill = 0.0
        elif fill_strategy == "constant":
            fill = 0.0 if fill_value is None else float(fill_value)
        else:
            raise ValueError(f"Unsupported fill_strategy for preprocessor artifact: {fill_strategy}")

        fill_values[column] = float(fill)
        frame[column] = numeric.fillna(fill)

    artifact = {
        "schema_version": RANK_TZ_SCHEMA_VERSION,
        "target": "rank_tz",
        "feature_set": feature_set,
        "feature_order": feature_order,
        "input_schema": get_input_schema(feature_set),
        "fill_strategy": fill_strategy,
        "fill_values": fill_values,
        "allowed_missing": feature_order,
        "class_mapping": {
            "rank_to_class": {str(key): value for key, value in RANK_TO_CLASS_MAP.items()},
            "class_to_rank": {str(key): value for key, value in CLASS_TO_RANK_MAP.items()},
        },
        "training_rows": int(training_rows),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    return artifact, frame[feature_order]


def apply_preprocessor_artifact(
    payload: pd.DataFrame | dict[str, Any],
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Apply a stored preprocessor artifact to raw input."""
    _validate_preprocessor_artifact(artifact)
    feature_order = artifact["feature_order"]
    frame = ensure_feature_frame(payload, feature_order)

    fill_values: dict[str, Any] = artifact["fill_values"]
    for column in feature_order:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if column not in fill_values:
            raise ValueError(f"Missing fill value for feature '{column}' in preprocessor artifact")
        frame[column] = frame[column].fillna(fill_values[column])

    return frame[feature_order]


def ensure_feature_frame(
    payload: pd.DataFrame | dict[str, Any],
    feature_order: list[str],
) -> pd.DataFrame:
    """Create a feature frame with all required columns present."""
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])

    for column in feature_order:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[feature_order]


def add_rank_tz_engineered_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Add engineered tactical features when requested by the contract."""
    if feature_set != "enhanced_tactical":
        return df

    enriched = df.copy()

    if "fire_date" in enriched.columns:
        dates = pd.to_datetime(enriched["fire_date"], errors="coerce")
        enriched["day_of_week"] = dates.dt.dayofweek
        enriched["month"] = dates.dt.month
        enriched["season"] = enriched["month"].map(
            {
                12: 1,
                1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 2,
                6: 3,
                7: 3,
                8: 3,
                9: 4,
                10: 4,
                11: 4,
            }
        )
        enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(float)
    else:
        for column in ["day_of_week", "month", "season", "is_weekend"]:
            enriched[column] = np.nan

    if {"building_floors", "fire_floor"}.issubset(enriched.columns):
        floors = pd.to_numeric(enriched["building_floors"], errors="coerce")
        fire_floor = pd.to_numeric(enriched["fire_floor"], errors="coerce")
        enriched["is_high_rise"] = (floors > 9).astype(float)
        enriched["is_super_high_rise"] = (floors > 25).astype(float)
        enriched["basement_fire"] = (fire_floor < 0).astype(float)
        enriched["top_floor_fire"] = (fire_floor == floors).astype(float)
        ratio = fire_floor / floors.replace(0, np.nan)
        enriched["fire_floor_ratio"] = ratio.clip(-1, 2)
        enriched["building_floors_bin"] = pd.cut(
            floors,
            bins=[-np.inf, 2, 5, 9, 16, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)
    else:
        for column in [
            "is_high_rise",
            "is_super_high_rise",
            "basement_fire",
            "top_floor_fire",
            "fire_floor_ratio",
            "building_floors_bin",
        ]:
            enriched[column] = np.nan

    if "distance_to_station" in enriched.columns:
        distance = pd.to_numeric(enriched["distance_to_station"], errors="coerce")
        enriched["is_far"] = (distance > 5).astype(float)
        enriched["is_very_far"] = (distance > 10).astype(float)
        enriched["distance_category"] = pd.cut(
            distance,
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)
    else:
        for column in ["is_far", "is_very_far", "distance_category"]:
            enriched[column] = np.nan

    for column, indicator_name in [
        ("risk_category_code", "risk_category_missing"),
        ("fpo_class_code", "fpo_class_missing"),
    ]:
        if column in enriched.columns:
            enriched[indicator_name] = enriched[column].isna().astype(float)
        else:
            enriched[indicator_name] = 1.0

    if {"t_detect_min", "t_report_min"}.issubset(enriched.columns):
        detect = pd.to_numeric(enriched["t_detect_min"], errors="coerce")
        report = pd.to_numeric(enriched["t_report_min"], errors="coerce")
        enriched["delta_detect_to_report"] = report - detect
    else:
        enriched["delta_detect_to_report"] = np.nan

    if {"t_report_min", "t_arrival_min"}.issubset(enriched.columns):
        report = pd.to_numeric(enriched["t_report_min"], errors="coerce")
        arrival = pd.to_numeric(enriched["t_arrival_min"], errors="coerce")
        enriched["delta_report_to_arrival"] = arrival - report
    else:
        enriched["delta_report_to_arrival"] = np.nan

    if {"t_arrival_min", "t_first_hose_min"}.issubset(enriched.columns):
        arrival = pd.to_numeric(enriched["t_arrival_min"], errors="coerce")
        first_hose = pd.to_numeric(enriched["t_first_hose_min"], errors="coerce")
        enriched["delta_arrival_to_hose"] = first_hose - arrival
    else:
        enriched["delta_arrival_to_hose"] = np.nan

    return enriched


def get_manual_inference_feature_order() -> list[str]:
    """Legacy 6-field app-style inference projection for audit/comparison."""
    return FEATURE_SET_SPECS["basic"]["feature_order"]


def _validate_preprocessor_artifact(artifact: dict[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "target",
        "feature_set",
        "feature_order",
        "input_schema",
        "fill_strategy",
        "fill_values",
        "class_mapping",
    }
    missing = required_keys - set(artifact)
    if missing:
        raise ValueError(f"Preprocessor artifact is missing required keys: {sorted(missing)}")
