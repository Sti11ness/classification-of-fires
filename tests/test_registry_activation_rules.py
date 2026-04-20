from pathlib import Path

from fire_es_desktop.infra import ModelRegistry


def test_registry_blocks_non_vector_or_leaky_models(tmp_path: Path):
    models_path = tmp_path / "models"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="unsafe1",
        name="proxy_model",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code"],
        metrics={"f1_macro": 0.4},
        params={},
        dataset_info={},
        extra={
            "deployment_role": "rank_tz_lpr_dispatch_production",
            "offline_only": False,
            "semantic_target": "rank_tz_count_proxy",
            "availability_stage": "dispatch_initial",
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "preprocessor_path": "model_unsafe1_preprocessor.json",
            "training_schema_version": "rank_tz_schema_v3",
            "normative_version": "rank_resource_normatives_v1",
            "input_schema": [{"name": "region_code"}],
        },
    )
    assert registry.set_active_model("unsafe1") is False


def test_registry_blocks_dispatch_model_with_forbidden_field(tmp_path: Path):
    models_path = tmp_path / "models_dispatch_forbidden"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="unsafe2",
        name="dispatch_with_source_item",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code", "source_item_code"],
        metrics={"f1_macro": 0.5},
        params={},
        dataset_info={},
        extra={
            "deployment_role": "rank_tz_lpr_dispatch_production",
            "offline_only": False,
            "semantic_target": "rank_tz_vector",
            "availability_stage": "dispatch_initial",
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "preprocessor_path": "model_unsafe2_preprocessor.json",
            "training_schema_version": "rank_tz_schema_v3",
            "normative_version": "rank_resource_normatives_v1",
            "input_schema": [{"name": "region_code"}, {"name": "source_item_code"}],
        },
    )
    assert registry.set_active_model("unsafe2") is False


def test_registry_blocks_legacy_online_tactical_model(tmp_path: Path):
    models_path = tmp_path / "models_legacy_online"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="unsafe3",
        name="legacy_online_tactical",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code", "t_arrival_min", "t_first_hose_min"],
        metrics={"f1_macro": 0.5},
        params={},
        dataset_info={},
        extra={
            "deployment_role": "rank_tz_lpr_first_hose_production",
            "offline_only": False,
            "semantic_target": "rank_tz_vector",
            "availability_stage": "first_hose_update",
            "feature_set": "online_tactical",
            "legacy_alias": True,
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "preprocessor_path": "model_unsafe3_preprocessor.json",
            "training_schema_version": "rank_tz_schema_v3",
            "normative_version": "rank_resource_normatives_v1",
            "input_schema": [{"name": "region_code"}],
        },
    )
    assert registry.set_active_model("unsafe3") is False


def test_registry_blocks_arrival_model_from_canonical_lpr_activation(tmp_path: Path):
    models_path = tmp_path / "models_arrival"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="unsafe4",
        name="arrival_model",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code", "t_arrival_min", "delta_report_to_arrival"],
        metrics={"f1_macro": 0.6},
        params={},
        dataset_info={},
        extra={
            "deployment_role": "rank_tz_lpr_arrival_production",
            "offline_only": False,
            "semantic_target": "rank_tz_vector",
            "availability_stage": "arrival_update",
            "feature_set": "arrival_update_safe",
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "preprocessor_path": "model_unsafe4_preprocessor.json",
            "training_schema_version": "rank_tz_schema_v3",
            "normative_version": "rank_resource_normatives_v1",
            "input_schema": [{"name": "region_code"}, {"name": "t_arrival_min"}],
        },
    )
    assert registry.set_active_model("unsafe4") is False


def test_registry_blocks_first_hose_model_from_canonical_lpr_activation(tmp_path: Path):
    models_path = tmp_path / "models_first_hose"
    models_path.mkdir()
    registry = ModelRegistry(models_path)
    registry.register_model(
        model_id="unsafe5",
        name="first_hose_model",
        model_type="random_forest",
        target="rank_tz",
        features=["region_code", "t_arrival_min", "t_first_hose_min", "delta_arrival_to_hose"],
        metrics={"f1_macro": 0.6},
        params={},
        dataset_info={},
        extra={
            "deployment_role": "rank_tz_lpr_first_hose_production",
            "offline_only": False,
            "semantic_target": "rank_tz_vector",
            "availability_stage": "first_hose_update",
            "feature_set": "first_hose_update_safe",
            "split_protocol": "group_shuffle",
            "event_overlap_rate": 0.0,
            "preprocessor_path": "model_unsafe5_preprocessor.json",
            "training_schema_version": "rank_tz_schema_v3",
            "normative_version": "rank_resource_normatives_v1",
            "input_schema": [{"name": "region_code"}, {"name": "t_first_hose_min"}],
        },
    )
    assert registry.set_active_model("unsafe5") is False
