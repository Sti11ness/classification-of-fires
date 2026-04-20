import pandas as pd

from fire_es.rank_tz_contract import apply_preprocessor_artifact, build_preprocessor_artifact, get_feature_set_spec


def test_preprocessor_v3_handles_unknown_category_and_shape():
    spec = get_feature_set_spec("dispatch_initial_safe")
    df = pd.DataFrame(
        {
            "region_code": [77, 78, 79],
            "settlement_type_code": [1, 2, 1],
            "fire_protection_code": [1, 1, 2],
            "enterprise_type_code": [11, 12, 11],
            "building_floors": [5, 10, 3],
            "fire_floor": [2, 3, 1],
            "fire_resistance_code": [2, 3, 2],
            "source_item_code": [10, 11, 12],
            "distance_to_station": [2.5, 4.1, None],
            "t_detect_min": [10, 15, 20],
            "t_report_min": [12, 18, None],
        }
    )
    artifact, transformed = build_preprocessor_artifact(
        df,
        feature_order=spec["feature_order"],
        feature_set=spec["feature_set"],
        fill_strategy=spec["default_fill_strategy"],
        fill_value=spec["default_fill_value"],
        training_rows=len(df),
        test_size=0.25,
        random_state=42,
    )
    payload = {
        "region_code": 999999,
        "settlement_type_code": 9,
        "fire_protection_code": 9,
        "enterprise_type_code": 999,
        "building_floors": 12,
        "fire_floor": 3,
        "fire_resistance_code": 9,
        "source_item_code": 999,
        "distance_to_station": 4.5,
        "t_detect_min": 11,
        "t_report_min": 16,
    }
    applied = apply_preprocessor_artifact(payload, artifact)
    assert artifact["preprocessing_version"] >= 3
    assert list(applied.columns) == artifact["feature_names_out"]
    assert applied.shape[1] == transformed.shape[1]
    assert applied.isna().sum().sum() == 0
