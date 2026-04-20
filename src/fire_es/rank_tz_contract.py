"""
Shared rank_tz ML contract for training, deployment, and UI.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd

RANK_TZ_SCHEMA_VERSION = "rank_tz_schema_v3"
PREPROCESSING_VERSION = 3

SEMANTIC_TARGET_RANK_TZ_VECTOR = "rank_tz_vector"
SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY = "rank_tz_count_proxy"
SEMANTIC_TARGET_RESEARCH_SEVERITY = "rank_ref_severity"

LABEL_SOURCE_HISTORICAL_VECTOR = "historical_vector"
LABEL_SOURCE_LPR_DECISION = "lpr_decision"
LABEL_SOURCE_PROXY_BOOTSTRAP = "proxy_bootstrap"
LABEL_SOURCE_RESEARCH_ONLY = "research_only"

AVAILABILITY_STAGE_DISPATCH = "dispatch_initial"
AVAILABILITY_STAGE_ARRIVAL = "arrival_update"
AVAILABILITY_STAGE_FIRST_HOSE = "first_hose_update"
AVAILABILITY_STAGE_RETROSPECTIVE = "retrospective"

DEPLOYMENT_ROLE_DISPATCH = "rank_tz_lpr_dispatch_production"
DEPLOYMENT_ROLE_ARRIVAL = "rank_tz_lpr_arrival_production"
DEPLOYMENT_ROLE_FIRST_HOSE = "rank_tz_lpr_first_hose_production"
OFFLINE_DEPLOYMENT_ROLE = "rank_tz_offline_benchmark"
PRODUCTION_DEPLOYMENT_ROLE = DEPLOYMENT_ROLE_DISPATCH
DEFAULT_LPR_FEATURE_SET = "dispatch_initial_safe"

LEGACY_FEATURE_SET_ALIASES = {
    "online_tactical": "first_hose_update_safe",
}

DISPATCH_FORBIDDEN_FIELDS = {
    "source_item_code",
    "t_arrival_min",
    "t_first_hose_min",
    "t_contained_min",
    "t_extinguished_min",
    "fatalities",
    "injuries",
    "direct_damage",
    "equipment_count",
    "nozzle_count",
    "rank_tz_count_proxy",
    "predicted_rank",
}

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
        "kind": "categorical",
        "min": 0,
        "max": 99999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "settlement_type_code": {
        "name": "settlement_type_code",
        "label": "Тип населенного пункта",
        "type": "int",
        "kind": "categorical",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "fire_protection_code": {
        "name": "fire_protection_code",
        "label": "Вид пожарной охраны",
        "type": "int",
        "kind": "categorical",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "enterprise_type_code": {
        "name": "enterprise_type_code",
        "label": "Тип предприятия",
        "type": "int",
        "kind": "categorical",
        "min": 0,
        "max": 999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "building_floors": {
        "name": "building_floors",
        "label": "Этажность здания",
        "type": "int",
        "kind": "ordinal",
        "min": 0,
        "max": 200,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "fire_floor": {
        "name": "fire_floor",
        "label": "Этаж пожара",
        "type": "int",
        "kind": "ordinal",
        "min": -1,
        "max": 200,
        "null_sentinel": -1,
        "allow_negative_input": True,
    },
    "fire_resistance_code": {
        "name": "fire_resistance_code",
        "label": "Степень огнестойкости",
        "type": "int",
        "kind": "categorical",
        "min": 0,
        "max": 100,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "source_item_code": {
        "name": "source_item_code",
        "label": "Источник зажигания",
        "type": "int",
        "kind": "categorical",
        "min": 0,
        "max": 999,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "distance_to_station": {
        "name": "distance_to_station",
        "label": "Расстояние до части",
        "type": "float",
        "kind": "numeric",
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
        "kind": "numeric",
        "min": 0,
        "max": 1000,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "injuries": {
        "name": "injuries",
        "label": "Пострадавшие",
        "type": "int",
        "kind": "numeric",
        "min": 0,
        "max": 5000,
        "null_sentinel": -1,
        "allow_negative_input": False,
    },
    "direct_damage": {
        "name": "direct_damage",
        "label": "Прямой ущерб",
        "type": "float",
        "kind": "numeric",
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
        "kind": "numeric",
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
        "kind": "numeric",
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
        "kind": "numeric",
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
        "kind": "numeric",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
        "suffix": " мин",
        "allow_negative_input": False,
    },
    "t_contained_min": {
        "name": "t_contained_min",
        "label": "Время локализации",
        "type": "int",
        "kind": "numeric",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
    },
    "t_extinguished_min": {
        "name": "t_extinguished_min",
        "label": "Время ликвидации",
        "type": "int",
        "kind": "numeric",
        "min": 0,
        "max": 10000,
        "null_sentinel": -1,
    },
}

FIELD_POLICY_DEFAULTS = {
    "optional_for_lpr": False,
    "known_at_dispatch": False,
    "known_at_arrival": False,
    "known_at_first_hose": False,
    "missing_indicator_always": False,
}

FIELD_POLICY_OVERRIDES: dict[str, dict[str, Any]] = {
    "region_code": {
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
    },
    "settlement_type_code": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "fire_protection_code": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "enterprise_type_code": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "building_floors": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "fire_floor": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "fire_resistance_code": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "source_item_code": {
        "optional_for_lpr": True,
        "known_at_dispatch": False,
        "known_at_arrival": False,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "distance_to_station": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "t_detect_min": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "t_report_min": {
        "optional_for_lpr": True,
        "known_at_dispatch": True,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "t_arrival_min": {
        "optional_for_lpr": False,
        "known_at_arrival": True,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
    },
    "t_first_hose_min": {
        "optional_for_lpr": False,
        "known_at_first_hose": True,
        "missing_indicator_always": True,
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
        "availability_stage": AVAILABILITY_STAGE_RETROSPECTIVE,
        "default_fill_strategy": "constant",
        "default_fill_value": 0.0,
        "label": "Basic (legacy baseline)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY,
    },
    "dispatch_initial_safe": {
        "feature_set": "dispatch_initial_safe",
        "feature_order": [
            "region_code",
            "settlement_type_code",
            "fire_protection_code",
            "enterprise_type_code",
            "building_floors",
            "fire_floor",
            "fire_resistance_code",
            "distance_to_station",
            "t_detect_min",
            "t_report_min",
        ],
        "deployment_role": DEPLOYMENT_ROLE_DISPATCH,
        "offline_only": False,
        "availability_stage": AVAILABILITY_STAGE_DISPATCH,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Dispatch initial safe (production)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
    },
    "arrival_update_safe": {
        "feature_set": "arrival_update_safe",
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
            "delta_report_to_arrival",
        ],
        "deployment_role": DEPLOYMENT_ROLE_ARRIVAL,
        "offline_only": False,
        "availability_stage": AVAILABILITY_STAGE_ARRIVAL,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Arrival update safe (production)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
    },
    "first_hose_update_safe": {
        "feature_set": "first_hose_update_safe",
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
            "delta_report_to_arrival",
            "t_first_hose_min",
            "delta_arrival_to_hose",
        ],
        "deployment_role": DEPLOYMENT_ROLE_FIRST_HOSE,
        "offline_only": False,
        "availability_stage": AVAILABILITY_STAGE_FIRST_HOSE,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "First hose update safe (production)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
    },
    "retrospective_benchmark": {
        "feature_set": "retrospective_benchmark",
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
        "availability_stage": AVAILABILITY_STAGE_RETROSPECTIVE,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Retrospective benchmark (offline)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
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
        "availability_stage": AVAILABILITY_STAGE_RETROSPECTIVE,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Enhanced tactical (offline experiment)",
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
    },
    "online_tactical": {
        "feature_set": "online_tactical",
        "legacy_alias_for": "first_hose_update_safe",
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
        "deployment_role": DEPLOYMENT_ROLE_FIRST_HOSE,
        "offline_only": False,
        "availability_stage": AVAILABILITY_STAGE_FIRST_HOSE,
        "default_fill_strategy": "median",
        "default_fill_value": None,
        "label": "Online tactical (legacy alias)",
        "legacy_alias": True,
        "semantic_target_default": SEMANTIC_TARGET_RANK_TZ_VECTOR,
    },
}


def get_feature_set_spec(feature_set: str) -> dict[str, Any]:
    if feature_set not in FEATURE_SET_SPECS:
        raise ValueError(f"Unknown rank_tz feature set: {feature_set}")
    return deepcopy(FEATURE_SET_SPECS[feature_set])


def get_input_schema(feature_set: str) -> list[dict[str, Any]]:
    spec = get_feature_set_spec(feature_set)
    schema: list[dict[str, Any]] = []
    for feature_name in spec["feature_order"]:
        field = deepcopy(FIELD_SPECS.get(feature_name, {"name": feature_name, "label": feature_name}))
        field.setdefault("type", "float")
        field.setdefault("kind", "numeric")
        field.setdefault("null_sentinel", -1.0)
        for key, value in FIELD_POLICY_DEFAULTS.items():
            field.setdefault(key, value)
        for key, value in FIELD_POLICY_OVERRIDES.get(feature_name, {}).items():
            field[key] = value
        schema.append(field)
    return schema


def get_default_lpr_feature_set() -> str:
    return DEFAULT_LPR_FEATURE_SET


def get_optional_lpr_fields(feature_set: str) -> list[str]:
    return [field["name"] for field in get_input_schema(feature_set) if field.get("optional_for_lpr")]


def get_feature_set_forbidden_violations(
    feature_order: list[str],
    *,
    availability_stage: str,
) -> list[str]:
    if availability_stage == AVAILABILITY_STAGE_DISPATCH:
        return sorted(DISPATCH_FORBIDDEN_FIELDS.intersection(feature_order))
    return []


def is_feature_set_production_safe(
    feature_order: list[str],
    *,
    availability_stage: str,
) -> bool:
    return not get_feature_set_forbidden_violations(feature_order, availability_stage=availability_stage)


def validate_stage_input_requirements(
    input_data: dict[str, Any],
    *,
    availability_stage: str,
) -> list[str]:
    missing: list[str] = []
    if availability_stage == AVAILABILITY_STAGE_ARRIVAL:
        if input_data.get("t_arrival_min") in (None, "") or pd.isna(input_data.get("t_arrival_min")):
            missing.append("t_arrival_min")
    elif availability_stage == AVAILABILITY_STAGE_FIRST_HOSE:
        for field in ("t_arrival_min", "t_first_hose_min"):
            if input_data.get(field) in (None, "") or pd.isna(input_data.get(field)):
                missing.append(field)
    return missing


def map_rank_series_to_classes(y: pd.Series) -> pd.Series:
    return y.map(lambda value: RANK_TO_CLASS_MAP.get(float(value)) if pd.notna(value) else np.nan)


def class_list_to_rank_values(classes: list[int]) -> list[float]:
    return [CLASS_TO_RANK_MAP.get(int(value), float(value)) for value in classes]


def ensure_feature_frame(
    payload: pd.DataFrame | dict[str, Any],
    feature_order: list[str],
) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])
    for column in feature_order:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _field_kind(feature_name: str) -> str:
    return FIELD_SPECS.get(feature_name, {}).get("kind", "numeric")


def _field_policy(feature_name: str, key: str, default: Any = None) -> Any:
    if feature_name in FIELD_POLICY_OVERRIDES and key in FIELD_POLICY_OVERRIDES[feature_name]:
        return FIELD_POLICY_OVERRIDES[feature_name][key]
    return FIELD_POLICY_DEFAULTS.get(key, default)


def _categorical_fill_token(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    return str(value)


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
    semantic_target: str = SEMANTIC_TARGET_RANK_TZ_VECTOR,
    label_source_policy: Optional[list[str]] = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build a serializable preprocessor artifact and transformed features."""
    frame = ensure_feature_frame(df, feature_order)
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    ordinal_features: list[str] = []
    missing_indicators: list[str] = []
    fill_values: dict[str, float] = {}
    category_maps: dict[str, Any] = {}
    categorical_encoding: dict[str, str] = {}
    transformed = pd.DataFrame(index=frame.index)
    feature_names_out: list[str] = []

    for column in feature_order:
        kind = _field_kind(column)
        raw = frame[column]
        missing_indicator_name = f"{column}__missing"
        if raw.isna().any() or bool(_field_policy(column, "missing_indicator_always", False)):
            transformed[missing_indicator_name] = raw.isna().astype(float)
            missing_indicators.append(missing_indicator_name)
            feature_names_out.append(missing_indicator_name)

        if kind in {"numeric", "ordinal"}:
            numeric = pd.to_numeric(raw, errors="coerce")
            if fill_strategy == "median":
                fill = numeric.median()
                if pd.isna(fill):
                    fill = 0.0
            elif fill_strategy == "constant":
                fill = 0.0 if fill_value is None else float(fill_value)
            else:
                raise ValueError(f"Unsupported fill_strategy for preprocessor artifact: {fill_strategy}")
            fill_values[column] = float(fill)
            transformed[column] = numeric.fillna(fill).astype(float)
            feature_names_out.append(column)
            if kind == "ordinal":
                ordinal_features.append(column)
            else:
                numeric_features.append(column)
            continue

        categorical_features.append(column)
        series = raw.map(_categorical_fill_token)
        unique_count = int(series.nunique(dropna=False))
        if unique_count <= 10:
            categories = sorted(series.unique().tolist())
            category_maps[column] = categories
            categorical_encoding[column] = "one_hot"
            unknown_column = f"{column}__unknown"
            for category in categories:
                feature_name = f"{column}__{category}"
                transformed[feature_name] = (series == category).astype(float)
                feature_names_out.append(feature_name)
            transformed[unknown_column] = 0.0
            feature_names_out.append(unknown_column)
        else:
            frequencies = (series.value_counts(normalize=True)).to_dict()
            category_maps[column] = {str(key): float(value) for key, value in frequencies.items()}
            categorical_encoding[column] = "frequency"
            transformed[column] = series.map(lambda value: category_maps[column].get(str(value), 0.0)).astype(float)
            feature_names_out.append(column)

    artifact = {
        "schema_version": RANK_TZ_SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target": "rank_tz",
        "semantic_target": semantic_target,
        "label_source_policy": sorted(set(label_source_policy or [LABEL_SOURCE_HISTORICAL_VECTOR])),
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
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ordinal_features": ordinal_features,
        "missing_indicators": missing_indicators,
        "categorical_encoding": categorical_encoding,
        "category_maps": category_maps,
        "unknown_category_policy": "one_hot_unknown_or_frequency_zero",
        "feature_names_out": feature_names_out,
        "missing_policy": "none_is_unknown",
        "optional_lpr_fields": [field for field in feature_order if _field_policy(field, "optional_for_lpr", False)],
    }
    return artifact, transformed[feature_names_out]


def apply_preprocessor_artifact(
    payload: pd.DataFrame | dict[str, Any],
    artifact: dict[str, Any],
) -> pd.DataFrame:
    _validate_preprocessor_artifact(artifact)
    if int(artifact.get("preprocessing_version", 2)) < 3:
        return _apply_legacy_preprocessor_artifact(payload, artifact)

    feature_order = artifact["feature_order"]
    frame = ensure_feature_frame(payload, feature_order)
    transformed = pd.DataFrame(index=frame.index)

    for indicator in artifact.get("missing_indicators", []):
        source_column = indicator.replace("__missing", "")
        transformed[indicator] = frame[source_column].isna().astype(float)

    for column in artifact.get("numeric_features", []) + artifact.get("ordinal_features", []):
        transformed[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(float(artifact["fill_values"].get(column, 0.0)))
            .astype(float)
        )

    category_maps = artifact.get("category_maps", {})
    categorical_encoding = artifact.get("categorical_encoding", {})
    for column in artifact.get("categorical_features", []):
        raw = frame[column].map(_categorical_fill_token)
        encoding = categorical_encoding.get(column, "frequency")
        if encoding == "one_hot":
            categories = category_maps.get(column, [])
            for category in categories:
                transformed[f"{column}__{category}"] = (raw == category).astype(float)
            unknown_column = f"{column}__unknown"
            transformed[unknown_column] = (~raw.isin(categories)).astype(float)
        else:
            frequency_map = {str(key): float(value) for key, value in category_maps.get(column, {}).items()}
            transformed[column] = raw.map(lambda value: frequency_map.get(str(value), 0.0)).astype(float)

    feature_names_out = artifact.get("feature_names_out", [])
    for column in feature_names_out:
        if column not in transformed.columns:
            transformed[column] = 0.0
    return transformed[feature_names_out]


def _apply_legacy_preprocessor_artifact(
    payload: pd.DataFrame | dict[str, Any],
    artifact: dict[str, Any],
) -> pd.DataFrame:
    feature_order = artifact["feature_order"]
    frame = ensure_feature_frame(payload, feature_order)
    fill_values: dict[str, Any] = artifact["fill_values"]
    for column in feature_order:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if column not in fill_values:
            raise ValueError(f"Missing fill value for feature '{column}' in preprocessor artifact")
        frame[column] = frame[column].fillna(fill_values[column])
    return frame[feature_order]


def prepare_feature_payload(
    payload: pd.DataFrame | dict[str, Any],
    *,
    feature_set: str,
    availability_stage: Optional[str] = None,
) -> pd.DataFrame:
    """Apply the shared stage-aware feature engineering pipeline."""
    if isinstance(payload, pd.DataFrame):
        enriched = payload.copy()
    else:
        enriched = pd.DataFrame([payload])

    resolved_feature_set = LEGACY_FEATURE_SET_ALIASES.get(feature_set, feature_set)
    availability_stage = availability_stage or get_feature_set_spec(feature_set).get("availability_stage")

    if resolved_feature_set not in {"enhanced_tactical", "arrival_update_safe", "first_hose_update_safe"}:
        return enriched

    if "fire_date" in enriched.columns:
        dates = pd.to_datetime(enriched["fire_date"], errors="coerce")
        enriched["day_of_week"] = dates.dt.dayofweek
        enriched["month"] = dates.dt.month
        enriched["season"] = enriched["month"].map(
            {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
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
        enriched["fire_floor_ratio"] = (fire_floor / floors.replace(0, np.nan)).clip(-1, 2)
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
        enriched[indicator_name] = enriched[column].isna().astype(float) if column in enriched.columns else 1.0

    if resolved_feature_set == "enhanced_tactical" and {"t_detect_min", "t_report_min"}.issubset(enriched.columns):
        detect = pd.to_numeric(enriched["t_detect_min"], errors="coerce")
        report = pd.to_numeric(enriched["t_report_min"], errors="coerce")
        enriched["delta_detect_to_report"] = report - detect
    elif resolved_feature_set == "enhanced_tactical":
        enriched["delta_detect_to_report"] = np.nan

    if availability_stage in {AVAILABILITY_STAGE_ARRIVAL, AVAILABILITY_STAGE_FIRST_HOSE, AVAILABILITY_STAGE_RETROSPECTIVE} and {"t_report_min", "t_arrival_min"}.issubset(enriched.columns):
        report = pd.to_numeric(enriched["t_report_min"], errors="coerce")
        arrival = pd.to_numeric(enriched["t_arrival_min"], errors="coerce")
        enriched["delta_report_to_arrival"] = arrival - report
    else:
        enriched["delta_report_to_arrival"] = np.nan

    if availability_stage in {AVAILABILITY_STAGE_FIRST_HOSE, AVAILABILITY_STAGE_RETROSPECTIVE} and {"t_arrival_min", "t_first_hose_min"}.issubset(enriched.columns):
        arrival = pd.to_numeric(enriched["t_arrival_min"], errors="coerce")
        first_hose = pd.to_numeric(enriched["t_first_hose_min"], errors="coerce")
        enriched["delta_arrival_to_hose"] = first_hose - arrival
    else:
        enriched["delta_arrival_to_hose"] = np.nan

    return enriched


def add_rank_tz_engineered_features(
    df: pd.DataFrame,
    feature_set: str,
    availability_stage: Optional[str] = None,
) -> pd.DataFrame:
    return prepare_feature_payload(
        df,
        feature_set=feature_set,
        availability_stage=availability_stage,
    )


def get_manual_inference_feature_order() -> list[str]:
    return FEATURE_SET_SPECS[DEFAULT_LPR_FEATURE_SET]["feature_order"]


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
