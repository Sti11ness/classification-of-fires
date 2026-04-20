"""
Simple digital twin profile and synthetic dataset generator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .distortions import apply_missingness, apply_numeric_noise


def build_statistical_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Build a lightweight statistical profile of the dataset."""
    numeric = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(exclude=["number"])
    return {
        "row_count": int(len(df)),
        "numeric_summary": numeric.describe(include="all").fillna(0.0).to_dict(),
        "categorical_summary": {
            column: categorical[column].value_counts(dropna=False).head(20).to_dict()
            for column in categorical.columns
        },
        "missingness": (df.isna().mean() * 100).round(4).to_dict(),
    }


def generate_synthetic_fire_dataset(
    df: pd.DataFrame,
    *,
    n_rows: int,
    missing_rate: float = 0.0,
    noise_scale: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a bootstrap-style synthetic dataset and its profile."""
    rng = np.random.default_rng(random_state)
    if df.empty:
        raise ValueError("Cannot build a synthetic dataset from an empty dataframe")
    sample_indices = rng.integers(0, len(df), size=int(n_rows))
    synthetic = df.iloc[sample_indices].reset_index(drop=True).copy()
    numeric_columns = synthetic.select_dtypes(include=["number"]).columns.tolist()
    if missing_rate > 0:
        synthetic = apply_missingness(synthetic, missing_rate=missing_rate, random_state=random_state)
    if noise_scale > 0 and numeric_columns:
        synthetic = apply_numeric_noise(
            synthetic,
            columns=numeric_columns,
            scale=noise_scale,
            random_state=random_state,
        )
    profile = build_statistical_profile(synthetic)
    return synthetic, profile
