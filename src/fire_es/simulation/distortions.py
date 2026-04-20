"""
Distortion helpers for synthetic fire datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_missingness(df: pd.DataFrame, *, missing_rate: float, random_state: int = 42) -> pd.DataFrame:
    """Inject random missingness into a dataframe."""
    rng = np.random.default_rng(random_state)
    result = df.copy()
    mask = rng.random(result.shape) < float(missing_rate)
    result = result.mask(mask)
    return result


def apply_numeric_noise(df: pd.DataFrame, *, columns: list[str], scale: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Inject Gaussian noise into numeric columns."""
    rng = np.random.default_rng(random_state)
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        std = numeric.std() or 1.0
        result[column] = numeric + rng.normal(0.0, std * scale, len(result))
    return result
