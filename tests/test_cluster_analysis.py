import pandas as pd

from fire_es.cluster_analysis import run_cluster_analysis


def test_cluster_analysis_returns_labels_and_centers():
    df = pd.DataFrame(
        {
            "distance_to_station": [1.0, 2.0, 8.0, 9.0],
            "building_floors": [1, 2, 9, 10],
        }
    )
    result = run_cluster_analysis(df, feature_columns=["distance_to_station", "building_floors"], n_clusters=2)
    assert "cluster_label" in result["clustered_df"].columns
    assert len(result["cluster_centers"]) == 2
