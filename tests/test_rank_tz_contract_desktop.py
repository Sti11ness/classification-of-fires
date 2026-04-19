import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fire_es.rank_tz_contract import (
    OFFLINE_DEPLOYMENT_ROLE,
    PRODUCTION_DEPLOYMENT_ROLE,
    add_rank_tz_engineered_features,
    apply_preprocessor_artifact,
    build_preprocessor_artifact,
    get_feature_set_spec,
)
from fire_es_desktop.infra import ModelRegistry
from fire_es_desktop.use_cases import BatchPredictExportUseCase, PredictUseCase


def make_rank_df(rows: int = 60) -> pd.DataFrame:
    base = []
    for idx in range(rows):
        cls = (idx % 6) + 1
        base.append(
            {
                "region_code": 70 + cls,
                "settlement_type_code": cls % 3 + 1,
                "fire_protection_code": cls % 4 + 1,
                "enterprise_type_code": 10 + cls,
                "building_floors": cls + 1,
                "fire_floor": cls,
                "fire_resistance_code": cls % 5 + 1,
                "source_item_code": 20 + cls,
                "distance_to_station": float(cls * 1.5),
                "t_detect_min": 5 * cls,
                "t_report_min": 5 * cls + 3,
                "t_arrival_min": 5 * cls + 9,
                "t_first_hose_min": 5 * cls + 15,
                "rank_tz": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}[cls],
                "fire_date": "2025-01-01",
            }
        )
    return pd.DataFrame(base)


def create_production_bundle(tmp_path: Path) -> Path:
    models_path = tmp_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)

    spec = get_feature_set_spec("online_tactical")
    df = make_rank_df()
    raw_X = df[spec["feature_order"]]
    y = pd.Series([(idx % 6) + 1 for idx in range(len(df))], dtype=int)
    artifact, X_train = build_preprocessor_artifact(
        raw_X,
        feature_order=spec["feature_order"],
        feature_set=spec["feature_set"],
        fill_strategy=spec["default_fill_strategy"],
        fill_value=spec["default_fill_value"],
        training_rows=len(df),
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y)

    model_id = "prod1234"
    artifact_path = models_path / f"model_{model_id}.joblib"
    preprocessor_path = models_path / f"model_{model_id}_preprocessor.json"
    metadata_path = models_path / f"model_{model_id}_meta.json"
    joblib.dump(model, artifact_path)
    preprocessor_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_name": "rf_online_tactical_test",
                "feature_set": spec["feature_set"],
                "features": spec["feature_order"],
                "input_schema": artifact["input_schema"],
                "fill_strategy": artifact["fill_strategy"],
                "fill_values": artifact["fill_values"],
                "allowed_missing": artifact["allowed_missing"],
                "class_mapping": artifact["class_mapping"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id=model_id,
        name="rf_online_tactical_test",
        model_type="random_forest",
        target="rank_tz",
        features=spec["feature_order"],
        metrics={"f1_macro": 0.5},
        params={"test_size": 0.25, "class_weight": "balanced", "random_state": 42},
        dataset_info={"samples": len(df), "features_count": len(spec["feature_order"])},
        extra={
            "artifact_path": artifact_path.name,
            "metadata_path": metadata_path.name,
            "preprocessor_path": preprocessor_path.name,
            "deployment_role": PRODUCTION_DEPLOYMENT_ROLE,
            "offline_only": False,
            "feature_set": spec["feature_set"],
            "input_schema": artifact["input_schema"],
            "fill_strategy": artifact["fill_strategy"],
            "fill_values": artifact["fill_values"],
            "allowed_missing": artifact["allowed_missing"],
            "class_mapping": artifact["class_mapping"],
        },
    )
    assert registry.set_active_model(model_id) is True
    return models_path


def test_preprocessor_artifact_median_fill_and_apply():
    spec = get_feature_set_spec("online_tactical")
    df = pd.DataFrame(
        {
            "region_code": [77, None, 79],
            "settlement_type_code": [1, 2, None],
            "fire_protection_code": [2, None, 3],
            "enterprise_type_code": [11, 12, None],
            "building_floors": [9, None, 3],
            "fire_floor": [2, None, 1],
            "fire_resistance_code": [2, 3, None],
            "source_item_code": [10, None, 11],
            "distance_to_station": [2.5, None, 4.0],
            "t_detect_min": [10, None, 5],
            "t_report_min": [15, 20, None],
            "t_arrival_min": [25, None, 12],
            "t_first_hose_min": [30, 35, None],
        }
    )
    artifact, transformed = build_preprocessor_artifact(
        df,
        feature_order=spec["feature_order"],
        feature_set=spec["feature_set"],
        fill_strategy="median",
        fill_value=None,
        training_rows=len(df),
        test_size=0.25,
        random_state=42,
    )
    assert artifact["fill_strategy"] == "median"
    assert transformed.isna().sum().sum() == 0

    payload = {"region_code": 99, "building_floors": 12, "fire_floor": 3}
    applied = apply_preprocessor_artifact(payload, artifact)
    assert list(applied.columns) == spec["feature_order"]
    assert applied.isna().sum().sum() == 0
    assert applied.iloc[0]["settlement_type_code"] == artifact["fill_values"]["settlement_type_code"]


def test_model_registry_blocks_offline_rank_tz_activation(tmp_path: Path):
    models_path = tmp_path / "models"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="offline1",
        name="extended_offline",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code"],
        metrics={"f1_macro": 0.8},
        params={"test_size": 0.25, "class_weight": "balanced"},
        dataset_info={"samples": 10, "features_count": 1},
        extra={
            "deployment_role": OFFLINE_DEPLOYMENT_ROLE,
            "offline_only": True,
        },
    )
    assert registry.set_active_model("offline1") is False
    assert registry.get_active_model_id() is None


def test_predict_use_case_requires_production_bundle(tmp_path: Path):
    models_path = create_production_bundle(tmp_path)
    use_case = PredictUseCase(models_path)
    result = use_case.execute(
        input_data={
            "region_code": 77,
            "building_floors": 9,
            "fire_floor": 2,
            "distance_to_station": 2.5,
            "t_detect_min": 10,
        },
        top_k=3,
    )
    assert result.success is True
    assert result.data["deployment_role"] == PRODUCTION_DEPLOYMENT_ROLE
    assert len(result.data["top_k_ranks"]) >= 1


def test_batch_predict_uses_same_production_contract(tmp_path: Path):
    models_path = create_production_bundle(tmp_path)
    reports_path = tmp_path / "reports"
    reports_path.mkdir()

    df_input = pd.DataFrame(
        [
            {
                "region_code": 77,
                "settlement_type_code": 1,
                "building_floors": 9,
                "fire_floor": 3,
                "distance_to_station": 2.5,
                "t_detect_min": 10,
                "t_report_min": 15,
            },
            {
                "region_code": 78,
                "enterprise_type_code": 12,
                "building_floors": 4,
                "fire_floor": 2,
                "distance_to_station": 4.5,
            },
        ]
    )
    input_path = tmp_path / "batch.xlsx"
    df_input.to_excel(input_path, index=False)

    use_case = BatchPredictExportUseCase(models_path, reports_path)
    result = use_case.execute(
        input_source="excel",
        input_path=input_path,
        output_format="csv",
        top_k=3,
        use_bootstrap=False,
        n_bootstrap=5,
    )
    assert result.success is True
    assert Path(result.data["output_path"]).exists()
    assert result.data["predictions_count"] == 2
    assert result.data["model_info"]["deployment_role"] == PRODUCTION_DEPLOYMENT_ROLE


def test_enhanced_tactical_engineering_adds_expected_columns():
    df = make_rank_df(10)
    enriched = add_rank_tz_engineered_features(df, "enhanced_tactical")
    for column in [
        "day_of_week",
        "season",
        "is_weekend",
        "fire_floor_ratio",
        "distance_category",
        "risk_category_missing",
        "delta_detect_to_report",
        "delta_arrival_to_hose",
    ]:
        assert column in enriched.columns
