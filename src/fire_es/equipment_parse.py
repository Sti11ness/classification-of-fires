"""
Equipment parsing and resource-vector helpers.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from .normatives import get_normative_resource_vectors


EQUIPMENT_CODES = {
    11: "AC",
    49: "AC",
    52: "AC",
    53: "AC",
    54: "AC",
    55: "AC",
    56: "AC",
    57: "AC_T",
    58: "APS",
    59: "APS",
    60: "APS",
    61: "APS",
    45: "APP",
    14: "ANR",
    50: "ANVD",
    15: "APT",
    16: "AP",
    17: "AKT",
    35: "AGT",
    19: "AGVT",
    12: "PNS",
    44: "PPP",
    62: "PPP",
    18: "AA",
    63: "APS_T",
    64: "APS_TR",
    65: "APS_E",
    66: "PSAAM",
    67: "PNRK",
    68: "PANRK",
    69: "AC_L",
    23: "AL",
    24: "APK",
    70: "APKL",
    71: "ALC",
    72: "APKC",
    73: "APTM",
    43: "ASA",
    74: "AT",
    28: "ASO",
    27: "AG",
    33: "AR",
    26: "AD",
    75: "AOPT",
    51: "ABG",
    76: "PKS",
    29: "ASH",
    34: "ALP",
    77: "SPEKL",
    78: "AOS",
    79: "APRSS",
    80: "ADPT",
    25: "APTS",
    81: "PAKM",
    82: "AVZ",
    20: "TRAIN",
    21: "SHIP",
    22: "HELI",
    40: "PLANE",
    30: "MOTORPUMP",
    48: "ROBOT_G",
    13: "PNS_TRAILER",
    31: "CUSTOM",
    83: "ROBOT_A",
    84: "ROBOT_W",
    39: "SMOKE_EXTRACTOR",
    42: "TRACTOR",
    32: "OTHER",
    46: "RESCUE",
    47: "RESCUE_EQ",
}


def parse_equipment_field(value: Any) -> dict[str, int]:
    """Parse raw equipment field into category -> count mapping."""
    if pd.isna(value) or value is None:
        return {}

    result: dict[str, int] = {}
    value_str = str(value).strip()
    if not value_str or value_str.lower() == "nan":
        return {}

    parts = re.split(r"[,;]", value_str)
    abbrev_map = {
        "АЦ": "AC",
        "АЦЛ": "AC",
        "АПС": "APS",
        "АЛ": "AL",
        "АР": "AR",
        "АГ": "AG",
    }
    for part in parts:
        token = part.strip()
        if not token:
            continue
        code_match = re.match(r"^(\d+)", token)
        if code_match:
            code = int(code_match.group(1))
            category = EQUIPMENT_CODES.get(code)
            if category:
                result[category] = result.get(category, 0) + 1
                continue
        token_upper = token.upper()
        for abbrev, category in abbrev_map.items():
            if abbrev in token_upper:
                result[category] = result.get(category, 0) + 1
                break
    return result


def analyze_equipment_parse(
    value: Any,
    *,
    declared_count: Any = None,
) -> dict[str, Any]:
    """Parse equipment and return quality metadata for canonical vector labeling."""
    value_str = "" if pd.isna(value) or value is None else str(value).strip()
    if not value_str:
        return {
            "equipment_vector": {},
            "resource_parse_confidence": 0.0,
            "resource_parse_flags": ["missing_or_unparsed_resources"],
            "resource_vector_sum": 0,
            "resource_count_declared": None if pd.isna(declared_count) else float(declared_count),
            "resource_count_parsed": 0,
            "resource_count_conflict": None,
            "raw_equipment": value,
        }

    parts = [token.strip() for token in re.split(r"[,;]", value_str) if token.strip()]
    parsed = parse_equipment_field(value)
    parsed_count = int(sum(parsed.values()))
    recognized_ratio = float(parsed_count / len(parts)) if parts else 0.0
    flags: list[str] = []
    if parsed_count == 0:
        flags.append("missing_or_unparsed_resources")
    elif recognized_ratio < 1.0:
        flags.append("partially_unparsed_resources")

    declared = None
    if declared_count is not None and not pd.isna(declared_count):
        declared = float(declared_count)
    conflict_delta = None
    if declared is not None:
        conflict_delta = abs(parsed_count - declared)
        if conflict_delta > 0.01:
            flags.append("resource_parse_conflict")

    confidence = 0.0
    if parsed_count > 0:
        confidence = recognized_ratio
        if declared is not None and conflict_delta is not None and conflict_delta > 0.01:
            confidence *= 0.5

    return {
        "equipment_vector": parsed,
        "resource_parse_confidence": float(round(confidence, 4)),
        "resource_parse_flags": flags,
        "resource_vector_sum": parsed_count,
        "resource_count_declared": declared,
        "resource_count_parsed": parsed_count,
        "resource_count_conflict": conflict_delta,
        "raw_equipment": value,
    }


def get_all_resource_categories() -> list[str]:
    """Return the union of categories present in the canonical normative vectors."""
    categories: set[str] = set()
    for resource_vector in get_normative_resource_vectors().values():
        categories.update(resource_vector.keys())
    return sorted(categories)


def build_resource_vector(
    equipment_dict: dict[str, int],
    all_categories: Optional[list[str]] = None,
) -> pd.Series:
    """Build a dense resource vector aligned to canonical categories."""
    all_categories = all_categories or get_all_resource_categories()
    vector = {category: int(equipment_dict.get(category, 0)) for category in all_categories}
    return pd.Series(vector, dtype=float)


def parse_equipment_to_vector(value: Any) -> pd.Series:
    """Parse raw equipment text into a canonical resource vector."""
    parsed = parse_equipment_field(value)
    return build_resource_vector(parsed)


def normalize_vector(vector: pd.Series, max_values: Optional[dict[str, int]] = None) -> pd.Series:
    """Normalize resource counts to the [0, 1] interval."""
    if max_values is None:
        max_values = {}
        for resource_vector in get_normative_resource_vectors().values():
            for category, count in resource_vector.items():
                max_values[category] = max(max_values.get(category, 0), count)

    normalized = {}
    for category in sorted(set(vector.index.tolist()) | set(max_values.keys())):
        raw_value = float(vector.get(category, 0.0))
        max_value = max_values.get(category, 1)
        normalized[category] = min(raw_value / max_value, 1.0) if max_value > 0 else 0.0
    return pd.Series(normalized, dtype=float)


def process_equipment_column(df: pd.DataFrame, equipment_col: str = "equipment") -> pd.DataFrame:
    """Add parsed equipment vectors for downstream rank assignment."""
    result = df.copy()
    declared = (
        result["equipment_count"]
        if "equipment_count" in result.columns
        else pd.Series(index=result.index, dtype=float)
    )
    diagnostics = [
        analyze_equipment_parse(raw, declared_count=declared.loc[idx] if idx in declared.index else None)
        for idx, raw in result[equipment_col].items()
    ]
    result["equipment_vector"] = [item["equipment_vector"] for item in diagnostics]
    result["resource_parse_confidence"] = [item["resource_parse_confidence"] for item in diagnostics]
    result["resource_parse_flags"] = [item["resource_parse_flags"] for item in diagnostics]
    result["resource_vector_sum"] = [item["resource_vector_sum"] for item in diagnostics]
    result["resource_count_declared"] = [item["resource_count_declared"] for item in diagnostics]
    result["resource_count_parsed"] = [item["resource_count_parsed"] for item in diagnostics]
    result["resource_count_conflict"] = [item["resource_count_conflict"] for item in diagnostics]
    result["equipment_vector_norm"] = [
        normalize_vector(build_resource_vector(value)) for value in result["equipment_vector"].tolist()
    ]
    return result


def build_unparsed_equipment_report(
    df: pd.DataFrame,
    *,
    equipment_col: str = "equipment",
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the most frequent unparsed/low-confidence equipment strings."""
    diagnostics = [
        analyze_equipment_parse(raw, declared_count=row.get("equipment_count"))
        for _, row in df.iterrows()
        for raw in [row.get(equipment_col)]
    ]
    report_rows = []
    for item in diagnostics:
        if item["resource_parse_confidence"] >= 1.0 and not item["resource_parse_flags"]:
            continue
        report_rows.append(
            {
                "equipment": item["raw_equipment"],
                "resource_parse_confidence": item["resource_parse_confidence"],
                "resource_parse_flags": ",".join(item["resource_parse_flags"]),
            }
        )
    if not report_rows:
        return pd.DataFrame(columns=["equipment", "count", "resource_parse_confidence", "resource_parse_flags"])
    report = (
        pd.DataFrame(report_rows)
        .groupby(["equipment", "resource_parse_confidence", "resource_parse_flags"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return report
