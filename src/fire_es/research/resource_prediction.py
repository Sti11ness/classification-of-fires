"""
Rank-vs-resource research comparison.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fire_es.model_selection import split_dataset
from fire_es.predict import rank_from_resources
from fire_es.rank_tz_contract import map_rank_series_to_classes


def compare_rank_vs_resource_modes(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    rank_target_column: str,
    resource_target_column: str = "equipment_count",
    split_protocol: str = "group_shuffle",
) -> dict[str, Any]:
    """Compare rank-classification against direct resource prediction."""
    working = df.copy().reset_index(drop=True)
    X = working[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rank_y = map_rank_series_to_classes(working[rank_target_column]).dropna().astype(int)
    rank_df = working.loc[rank_y.index].reset_index(drop=True)
    rank_X = X.loc[rank_y.index].reset_index(drop=True)
    rank_y = rank_y.reset_index(drop=True)
    rank_split = split_dataset(rank_df, y=rank_y, split_protocol=split_protocol)
    rank_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rank_model.fit(rank_X.iloc[rank_split.train_indices], rank_y.iloc[rank_split.train_indices])
    predicted_rank_classes = rank_model.predict(rank_X.iloc[rank_split.test_indices])

    resource_df = working.dropna(subset=[resource_target_column]).reset_index(drop=True)
    resource_X = resource_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    resource_y = pd.to_numeric(resource_df[resource_target_column], errors="coerce").fillna(0.0)
    resource_split = split_dataset(resource_df, y=resource_y.rank(method="dense").astype(int), split_protocol=split_protocol)
    resource_model = RandomForestRegressor(n_estimators=100, random_state=42)
    resource_model.fit(resource_X.iloc[resource_split.train_indices], resource_y.iloc[resource_split.train_indices])
    predicted_resources = resource_model.predict(resource_X.iloc[resource_split.test_indices])

    return {
        "rank_accuracy": float(
            (predicted_rank_classes == rank_y.iloc[rank_split.test_indices].to_numpy()).mean()
        ),
        "resource_mae": float(
            (abs(predicted_resources - resource_y.iloc[resource_split.test_indices].to_numpy())).mean()
        ),
        "induced_rank_accuracy": float(
            (
                rank_from_resources(pd.Series(predicted_resources)).to_numpy()
                == resource_df[rank_target_column].iloc[resource_split.test_indices].to_numpy()
            ).mean()
        )
        if rank_target_column in resource_df.columns
        else 0.0,
        "split_protocol": split_protocol,
    }
