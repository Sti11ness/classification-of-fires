import pandas as pd

from fire_es.research.resource_prediction import compare_rank_vs_resource_modes


def test_rank_vs_resource_study_returns_comparison():
    rows = []
    for idx in range(36):
        cls = (idx % 6) + 1
        rows.append(
            {
                "event_id": f"evt-{idx // 2}",
                "fire_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx),
                "region_code": 70 + cls,
                "building_floors": cls + 1,
                "fire_floor": cls,
                "distance_to_station": float(cls),
                "equipment_count": float(cls + 1),
                "rank_tz_vector": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}[cls],
            }
        )
    df = pd.DataFrame(rows)
    result = compare_rank_vs_resource_modes(
        df,
        feature_columns=["region_code", "building_floors", "fire_floor", "distance_to_station"],
        rank_target_column="rank_tz_vector",
        split_protocol="group_shuffle",
    )
    assert "rank_accuracy" in result
    assert "resource_mae" in result
