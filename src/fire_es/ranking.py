"""
Canonical rank assignment for TЗ mode 2.5.1.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from .equipment_parse import (
    build_resource_vector,
    normalize_vector,
    parse_equipment_field,
)
from .normatives import (
    get_normative_resource_vectors,
    load_rank_resource_normatives,
)
from .rank_tz_contract import (
    LABEL_SOURCE_HISTORICAL_VECTOR,
    LABEL_SOURCE_PROXY_BOOTSTRAP,
    SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY,
    SEMANTIC_TARGET_RANK_TZ_VECTOR,
)


def euclidean_distance(vec1: pd.Series, vec2: pd.Series) -> float:
    all_idx = vec1.index.union(vec2.index)
    v1 = vec1.reindex(all_idx, fill_value=0.0)
    v2 = vec2.reindex(all_idx, fill_value=0.0)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def calculate_rank_by_vector(
    equipment_vector: pd.Series | dict[str, int],
    normative_vectors: Optional[dict[float, dict[str, int]]] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Return the closest canonical rank in normalized vector space."""
    if equipment_vector is None:
        return None, None
    if isinstance(equipment_vector, dict):
        equipment_vector = build_resource_vector(equipment_vector)
    if equipment_vector.empty or float(equipment_vector.fillna(0).sum()) <= 0:
        return None, None

    normative_vectors = normative_vectors or get_normative_resource_vectors()
    max_values = {}
    for resources in normative_vectors.values():
        for category, count in resources.items():
            max_values[category] = max(max_values.get(category, 0), count)

    min_distance = float("inf")
    best_rank: Optional[float] = None
    equipment_norm = normalize_vector(equipment_vector, max_values)
    for rank, norm_vector in normative_vectors.items():
        norm_series = normalize_vector(pd.Series(norm_vector, dtype=float), max_values)
        distance = euclidean_distance(equipment_norm, norm_series)
        if distance < min_distance:
            min_distance = distance
            best_rank = float(rank)
    return best_rank, (None if best_rank is None else float(min_distance))


def calculate_rank_by_count(equipment_count: Any) -> tuple[Optional[float], Optional[float]]:
    """Auxiliary count proxy. Missing equipment never becomes rank 1."""
    if pd.isna(equipment_count) or equipment_count is None:
        return None, None
    count = float(equipment_count)
    if count < 1:
        return None, None
    if count == 1:
        return 1.0, 0.0
    if count == 2:
        return 1.5, 0.0
    if count == 3:
        return 2.0, 0.0
    if count == 4:
        return 3.0, 0.0
    if count == 5:
        return 4.0, 0.0
    return 5.0, float((count - 5) * 0.1)


def assign_rank_tz(
    df: pd.DataFrame,
    *,
    target_definition: str = "vector",
    equipment_count_col: str = "equipment_count",
    equipment_col: str = "equipment",
    method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assign canonical rank fields for either vector or count proxy mode.
    """
    result = df.copy()
    if method is not None:
        target_definition = "count_proxy" if method == "count" else "vector"
    normative_payload = load_rank_resource_normatives()
    normative_version = normative_payload["normative_version"]
    quality_flags: list[list[str]] = [[] for _ in range(len(result))]

    if target_definition == "vector":
        if equipment_col in result.columns:
            parsed_vectors = result[equipment_col].apply(parse_equipment_field)
        elif "equipment_vector" in result.columns:
            parsed_vectors = result["equipment_vector"]
        else:
            parsed_vectors = pd.Series([{} for _ in range(len(result))], index=result.index)

        assigned = parsed_vectors.apply(lambda value: calculate_rank_by_vector(build_resource_vector(value)))
        result["rank_tz_vector"] = assigned.apply(lambda value: value[0])
        result["rank_distance"] = assigned.apply(lambda value: value[1])
        result["rank_tz"] = result["rank_tz_vector"]
        result["rank_label_source"] = np.where(
            result["rank_tz_vector"].notna(),
            LABEL_SOURCE_HISTORICAL_VECTOR,
            None,
        )
        for idx, vector in enumerate(parsed_vectors.tolist()):
            if not vector:
                quality_flags[idx].append("missing_resources")
        result["usable_for_training"] = result.get("is_canonical_event_record", True) & result["rank_tz"].notna()
    elif target_definition == "count_proxy":
        assigned = result[equipment_count_col].apply(calculate_rank_by_count)
        result["rank_tz_count_proxy"] = assigned.apply(lambda value: value[0])
        result["rank_distance"] = assigned.apply(lambda value: value[1])
        result["rank_tz"] = result["rank_tz_count_proxy"]
        result["rank_label_source"] = np.where(
            result["rank_tz_count_proxy"].notna(),
            LABEL_SOURCE_PROXY_BOOTSTRAP,
            None,
        )
        for idx, value in enumerate(result[equipment_count_col].tolist()):
            if pd.isna(value) or value is None or float(value) < 1:
                quality_flags[idx].append("missing_resources")
        result["usable_for_training"] = result.get("is_canonical_event_record", True) & result["rank_tz"].notna()
    else:
        raise ValueError(f"Unknown target_definition: {target_definition}")

    result["rank_normative_version"] = normative_version
    result["rank_quality_flags"] = [
        json.dumps(flags, ensure_ascii=False) if flags else None for flags in quality_flags
    ]
    return result


def get_rank_description(rank: float) -> str:
    descriptions = {
        1.0: "Ранг 1 (1 единица техники)",
        1.5: "Ранг 1-бис (2 единицы техники)",
        2.0: "Ранг 2",
        3.0: "Ранг 3",
        4.0: "Ранг 4",
        5.0: "Ранг 5 (5+ единиц техники)",
    }
    return descriptions.get(rank, f"Ранг {rank}")


def validate_rank_distribution(df: pd.DataFrame) -> dict:
    if "rank_tz" not in df.columns:
        return {"error": "Колонка rank_tz отсутствует"}
    rank_counts = df["rank_tz"].value_counts(dropna=False).sort_index()
    return {
        "total": len(df),
        "distribution": rank_counts.to_dict(),
        "shares": (rank_counts / len(df)).to_dict(),
        "min_rank": float(rank_counts.index.min()) if len(rank_counts) else None,
        "max_rank": float(rank_counts.index.max()) if len(rank_counts) else None,
    }
