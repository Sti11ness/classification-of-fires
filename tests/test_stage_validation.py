import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fire_es.rank_tz_contract import (
    SEMANTIC_TARGET_RANK_TZ_VECTOR,
    build_preprocessor_artifact,
    get_feature_set_spec,
)
from fire_es_desktop.infra import ModelRegistry
from fire_es_desktop.use_cases import PredictUseCase


def _create_stage_bundle(tmp_path: Path, feature_set: str) -> Path:
    models_path = tmp_path / f"models_{feature_set}"
    models_path.mkdir(parents=True, exist_ok=True)
    spec = get_feature_set_spec(feature_set)
    rows = []
    for idx in range(24):
        cls = (idx % 6) + 1
        row = {
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
            "delta_report_to_arrival": 6,
            "delta_arrival_to_hose": 6,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
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
    model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42, class_weight="balanced")
    model.fit(X_train, y)

    model_id = f"stage_{feature_set}"
    artifact_path = models_path / f"model_{model_id}.joblib"
    preprocessor_path = models_path / f"model_{model_id}_preprocessor.json"
    joblib.dump(model, artifact_path)
    preprocessor_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id=model_id,
        name=model_id,
        model_type="random_forest",
        target="rank_tz",
        features=spec["feature_order"],
        metrics={"f1_macro": 0.5},
        params={},
        dataset_info={},
        extra={
            "artifact_path": artifact_path.name,
            "preprocessor_path": preprocessor_path.name,
            "deployment_role": spec["deployment_role"],
            "offline_only": False,
            "feature_set": spec["feature_set"],
            "input_schema": artifact["input_schema"],
            "semantic_target": SEMANTIC_TARGET_RANK_TZ_VECTOR,
            "availability_stage": spec["availability_stage"],
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "metric_primary": "f1_macro",
            "normative_version": "rank_resource_normatives_v1",
            "training_schema_version": artifact["schema_version"],
        },
    )
    assert registry.set_active_model(model_id) is True
    return models_path


def test_arrival_stage_requires_arrival_time(tmp_path: Path):
    models_path = _create_stage_bundle(tmp_path, "arrival_update_safe")
    use_case = PredictUseCase(models_path)
    result = use_case.execute(
        input_data={
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 1,
            "enterprise_type_code": 11,
            "building_floors": 5,
            "fire_floor": 2,
            "fire_resistance_code": 2,
            "distance_to_station": 2.0,
            "t_detect_min": 10,
            "t_report_min": 15,
        }
    )
    assert result.success is False
    assert "t_arrival_min" in result.message


def test_first_hose_stage_requires_hose_time(tmp_path: Path):
    models_path = _create_stage_bundle(tmp_path, "first_hose_update_safe")
    use_case = PredictUseCase(models_path)
    result = use_case.execute(
        input_data={
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 1,
            "enterprise_type_code": 11,
            "building_floors": 5,
            "fire_floor": 2,
            "fire_resistance_code": 2,
            "distance_to_station": 2.0,
            "t_detect_min": 10,
            "t_report_min": 15,
            "t_arrival_min": 22,
        }
    )
    assert result.success is False
    assert "t_first_hose_min" in result.message
