"""
Feature subset study for leakage-safe rank modeling.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fire_es.metrics import build_classification_metrics
from fire_es.model_selection import split_dataset
from fire_es.rank_tz_contract import CLASS_TO_RANK_MAP, map_rank_series_to_classes


def run_feature_subset_study(
    df: pd.DataFrame,
    *,
    feature_pool: list[str],
    semantic_target_column: str,
    split_protocol: str,
    max_subset_size: int = 4,
) -> pd.DataFrame:
    """Evaluate small feature subsets and return a comparison table."""
    rows: list[dict[str, Any]] = []
    y = map_rank_series_to_classes(df[semantic_target_column]).dropna().astype(int)
    working = df.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    for subset_size in range(2, min(max_subset_size, len(feature_pool)) + 1):
        for subset in itertools.combinations(feature_pool, subset_size):
            X = working[list(subset)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            split = split_dataset(working, y=y, split_protocol=split_protocol)
            model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
            model.fit(X.iloc[split.train_indices], y.iloc[split.train_indices])
            y_pred = model.predict(X.iloc[split.test_indices])
            y_proba = model.predict_proba(X.iloc[split.test_indices])
            metrics = build_classification_metrics(
                y_train=y.iloc[split.train_indices],
                y_test=y.iloc[split.test_indices],
                y_pred=y_pred,
                y_proba=y_proba,
                classes=model.classes_,
                class_to_rank_map=CLASS_TO_RANK_MAP,
                split_metadata=split.metadata,
            )
            rows.append(
                {
                    "subset_size": subset_size,
                    "features": ",".join(subset),
                    "semantic_target": semantic_target_column,
                    "split_protocol": split_protocol,
                    "f1_macro": metrics["f1_macro"],
                    "accuracy": metrics["accuracy"],
                    "event_overlap_rate": metrics["event_overlap_rate"],
                }
            )
    return pd.DataFrame(rows).sort_values(["subset_size", "f1_macro"], ascending=[True, False])
