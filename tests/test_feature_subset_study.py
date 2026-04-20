import pandas as pd

from fire_es.research.feature_subset_study import run_feature_subset_study


def test_feature_subset_study_returns_results():
    rows = []
    for idx in range(30):
        cls = (idx % 6) + 1
        rows.append(
            {
                "event_id": f"evt-{idx // 2}",
                "fire_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx),
                "region_code": 70 + cls,
                "building_floors": cls + 1,
                "fire_floor": cls,
                "distance_to_station": float(cls),
                "rank_tz_vector": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}[cls],
            }
        )
    df = pd.DataFrame(rows)
    result = run_feature_subset_study(
        df,
        feature_pool=["region_code", "building_floors", "fire_floor", "distance_to_station"],
        semantic_target_column="rank_tz_vector",
        split_protocol="group_shuffle",
        max_subset_size=3,
    )
    assert not result.empty
    assert {"subset_size", "features", "f1_macro"}.issubset(result.columns)
