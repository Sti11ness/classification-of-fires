"""
Analyst-side cluster analysis block for exploratory segmentation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans


def run_cluster_analysis(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run a lightweight clustering workflow for analyst-side artifacts."""
    frame = df.copy()
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    clustered = frame.copy()
    clustered["cluster_label"] = labels
    return {
        "clustered_df": clustered,
        "cluster_centers": pd.DataFrame(model.cluster_centers_, columns=feature_columns),
        "n_clusters": n_clusters,
        "feature_columns": feature_columns,
        "inertia": float(model.inertia_),
    }
