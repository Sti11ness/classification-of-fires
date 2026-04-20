"""
Canonical normative table loader for rank/resource mapping.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


NORMATIVE_JSON_RELATIVE_PATH = Path("data") / "normatives" / "rank_resource_normatives.json"


def _default_normative_json_path() -> Path:
    """Resolve the canonical normative JSON in source and frozen runtimes."""
    module_path = Path(__file__).resolve()
    candidates: list[Path] = []

    # Source checkout: <repo>/src/fire_es/normatives.py -> <repo>/data/normatives/...
    candidates.append(module_path.parents[2] / NORMATIVE_JSON_RELATIVE_PATH)

    # PyInstaller one-folder: runtime data typically lands under _internal/data/...
    candidates.append(module_path.parents[1] / NORMATIVE_JSON_RELATIVE_PATH)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / NORMATIVE_JSON_RELATIVE_PATH)

    # Last-resort cwd-relative fallback for manually rearranged deployments.
    candidates.append(Path.cwd() / NORMATIVE_JSON_RELATIVE_PATH)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


NORMATIVE_JSON_PATH = _default_normative_json_path()


def load_rank_resource_normatives(path: Path | None = None) -> dict[str, Any]:
    """Load the canonical normative JSON payload."""
    json_path = path or NORMATIVE_JSON_PATH
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "normative_version" not in payload or "ranks" not in payload:
        raise ValueError("Invalid normative payload: missing required keys")
    return payload


def get_normative_hash(payload: dict[str, Any] | None = None) -> str:
    """Stable SHA256 hash of the canonical normative payload."""
    payload = payload or load_rank_resource_normatives()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_normative_rank_table(payload: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a rank-level dataframe suitable for UI and metadata."""
    payload = payload or load_rank_resource_normatives()
    rows = []
    for row in payload["ranks"]:
        rows.append(
            {
                "rank": float(row["rank"]),
                "label": row["label"],
                "display_name": row.get("display_name", row["label"]),
                "description": row.get("description", ""),
                "sort_order": int(row.get("sort_order", 0)),
                "min_equipment_count": row.get("min_equipment_count"),
                "resource_vector": row.get("resource_vector", {}),
            }
        )
    return pd.DataFrame(rows).sort_values("sort_order").reset_index(drop=True)


def get_normative_resource_vectors(payload: dict[str, Any] | None = None) -> dict[float, dict[str, int]]:
    """Return canonical rank -> resource vector mapping."""
    payload = payload or load_rank_resource_normatives()
    return {
        float(row["rank"]): {str(key): int(value) for key, value in row["resource_vector"].items()}
        for row in payload["ranks"]
    }


def get_rank_label_map(payload: dict[str, Any] | None = None) -> dict[float, str]:
    """Return canonical rank -> user-facing label mapping."""
    payload = payload or load_rank_resource_normatives()
    return {float(row["rank"]): str(row["label"]) for row in payload["ranks"]}
