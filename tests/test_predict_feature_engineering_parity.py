import pandas as pd

from fire_es.rank_tz_contract import (
    apply_preprocessor_artifact,
    build_preprocessor_artifact,
    get_feature_set_spec,
    prepare_feature_payload,
)


def test_arrival_feature_engineering_parity_between_train_and_predict():
    spec = get_feature_set_spec("arrival_update_safe")
    train_raw = pd.DataFrame(
        {
            "region_code": [77, 78],
            "settlement_type_code": [1, 2],
            "fire_protection_code": [1, 1],
            "enterprise_type_code": [11, 12],
            "building_floors": [5, 6],
            "fire_floor": [2, 3],
            "fire_resistance_code": [2, 3],
            "distance_to_station": [2.0, 3.0],
            "t_detect_min": [10, 12],
            "t_report_min": [15, 18],
            "t_arrival_min": [22, 24],
        }
    )
    engineered_train = prepare_feature_payload(
        train_raw,
        feature_set=spec["feature_set"],
        availability_stage=spec["availability_stage"],
    )
    artifact, _ = build_preprocessor_artifact(
        engineered_train,
        feature_order=spec["feature_order"],
        feature_set=spec["feature_set"],
        fill_strategy=spec["default_fill_strategy"],
        fill_value=spec["default_fill_value"],
        training_rows=len(train_raw),
        test_size=0.25,
        random_state=42,
    )

    predict_raw = {
        "region_code": 79,
        "settlement_type_code": 1,
        "fire_protection_code": 1,
        "enterprise_type_code": 13,
        "building_floors": 7,
        "fire_floor": 4,
        "fire_resistance_code": 2,
        "distance_to_station": 4.0,
        "t_detect_min": 9,
        "t_report_min": 14,
        "t_arrival_min": 20,
    }
    engineered_predict = prepare_feature_payload(
        predict_raw,
        feature_set=spec["feature_set"],
        availability_stage=spec["availability_stage"],
    )
    applied = apply_preprocessor_artifact(engineered_predict, artifact)

    assert list(applied.columns) == artifact["feature_names_out"]
    assert applied.iloc[0]["delta_report_to_arrival"] == 6.0


def test_dispatch_feature_engineering_does_not_create_future_deltas():
    spec = get_feature_set_spec("dispatch_initial_safe")
    engineered = prepare_feature_payload(
        {"region_code": 77, "t_detect_min": 10, "t_report_min": 14},
        feature_set=spec["feature_set"],
        availability_stage=spec["availability_stage"],
    )
    assert "delta_report_to_arrival" not in engineered.columns or engineered["delta_report_to_arrival"].isna().all()
    assert "delta_arrival_to_hose" not in engineered.columns or engineered["delta_arrival_to_hose"].isna().all()
