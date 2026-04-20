"""
Leakage-safe split helpers for rank_tz training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


SPLIT_PROTOCOL_GROUP_SHUFFLE = "group_shuffle"
SPLIT_PROTOCOL_GROUP_KFOLD = "group_kfold"
SPLIT_PROTOCOL_TEMPORAL_HOLDOUT = "temporal_holdout"
SPLIT_PROTOCOL_SOURCE_HOLDOUT = "source_holdout"
SPLIT_PROTOCOL_ROW_RANDOM_LEGACY = "row_random_legacy"

PRODUCTION_SAFE_SPLIT_PROTOCOLS = {
    SPLIT_PROTOCOL_GROUP_SHUFFLE,
    SPLIT_PROTOCOL_GROUP_KFOLD,
    SPLIT_PROTOCOL_TEMPORAL_HOLDOUT,
}


@dataclass
class SplitResult:
    train_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict[str, Any]


def _as_datetime_series(df: pd.DataFrame) -> pd.Series:
    if "fire_date" in df.columns:
        series = pd.to_datetime(df["fire_date"], errors="coerce")
        if series.notna().any():
            return series
    if "year" in df.columns:
        return pd.to_datetime(df["year"], errors="coerce", format="%Y")
    return pd.Series(pd.NaT, index=df.index)


def _event_overlap_rate(train_groups: pd.Series, test_groups: pd.Series) -> float:
    train_set = {value for value in train_groups.dropna().tolist() if value != ""}
    test_set = {value for value in test_groups.dropna().tolist() if value != ""}
    if not test_set:
        return 0.0
    return float(len(train_set & test_set) / len(test_set))


def split_dataset(
    df: pd.DataFrame,
    *,
    y: pd.Series,
    split_protocol: str,
    test_size: float = 0.25,
    random_state: int = 42,
    event_id_column: str = "event_id",
    source_column: str = "source_sheet",
) -> SplitResult:
    """Return train/test indices and split passport metadata."""
    frame = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = frame[event_id_column] if event_id_column in frame.columns else pd.Series(index=frame.index, dtype=object)
    dates = _as_datetime_series(frame)

    if split_protocol == SPLIT_PROTOCOL_GROUP_SHUFFLE:
        if groups.isna().all():
            raise ValueError("Missing event_id for group_shuffle split")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(frame, y, groups=groups.fillna("__missing__")))
    elif split_protocol == SPLIT_PROTOCOL_GROUP_KFOLD:
        if groups.isna().all():
            raise ValueError("Missing event_id for group_kfold split")
        unique_groups = groups.fillna("__missing__").nunique()
        n_splits = max(2, min(5, int(unique_groups)))
        splitter = GroupKFold(n_splits=n_splits)
        train_idx, test_idx = next(splitter.split(frame, y, groups=groups.fillna("__missing__")))
    elif split_protocol == SPLIT_PROTOCOL_TEMPORAL_HOLDOUT:
        if dates.isna().all():
            raise ValueError("Missing fire_date/year for temporal_holdout split")
        order = dates.sort_values(kind="stable").index.to_numpy()
        split_at = max(1, int(len(order) * (1 - test_size)))
        train_idx = np.sort(order[:split_at])
        test_idx = np.sort(order[split_at:])
        if len(test_idx) == 0:
            raise ValueError("Temporal holdout produced an empty test split")
    elif split_protocol == SPLIT_PROTOCOL_SOURCE_HOLDOUT:
        if source_column not in frame.columns or frame[source_column].dropna().nunique() < 2:
            raise ValueError("source_holdout requires at least two sources")
        ordered_sources = sorted(frame[source_column].dropna().unique().tolist())
        held_out = ordered_sources[-1]
        test_idx = frame.index[frame[source_column] == held_out].to_numpy()
        train_idx = frame.index[frame[source_column] != held_out].to_numpy()
    elif split_protocol == SPLIT_PROTOCOL_ROW_RANDOM_LEGACY:
        train_idx, test_idx = train_test_split(
            frame.index.to_numpy(),
            test_size=test_size,
            random_state=random_state,
            stratify=y if y.nunique() > 1 else None,
        )
    else:
        raise ValueError(f"Unknown split protocol: {split_protocol}")

    train_groups = groups.iloc[train_idx] if not groups.empty else pd.Series(dtype=object)
    test_groups = groups.iloc[test_idx] if not groups.empty else pd.Series(dtype=object)
    metadata = {
        "split_protocol": split_protocol,
        "event_id_column": event_id_column if event_id_column in frame.columns else None,
        "event_overlap_rate": _event_overlap_rate(train_groups, test_groups),
        "train_event_count": int(train_groups.dropna().nunique()) if len(train_groups) else 0,
        "test_event_count": int(test_groups.dropna().nunique()) if len(test_groups) else 0,
        "train_date_min": _safe_iso(dates.iloc[train_idx].min()) if len(train_idx) else None,
        "train_date_max": _safe_iso(dates.iloc[train_idx].max()) if len(train_idx) else None,
        "test_date_min": _safe_iso(dates.iloc[test_idx].min()) if len(test_idx) else None,
        "test_date_max": _safe_iso(dates.iloc[test_idx].max()) if len(test_idx) else None,
        "production_safe": split_protocol in PRODUCTION_SAFE_SPLIT_PROTOCOLS,
    }
    return SplitResult(
        train_indices=np.asarray(train_idx, dtype=int),
        test_indices=np.asarray(test_idx, dtype=int),
        metadata=metadata,
    )


def _safe_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
