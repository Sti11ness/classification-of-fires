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
    result["equipment_vector"] = result[equipment_col].apply(parse_equipment_field)
    result["equipment_vector_norm"] = result["equipment_vector"].apply(
        lambda value: normalize_vector(build_resource_vector(value))
    )
    return result
