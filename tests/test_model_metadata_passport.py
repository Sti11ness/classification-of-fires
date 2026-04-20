import json
from pathlib import Path

from tests.test_metrics_passport import _seed_training_db
from fire_es_desktop.use_cases import TrainModelUseCase


def test_model_metadata_contains_followup_passport_fields(tmp_path: Path):
    db_path = tmp_path / "passport.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    _seed_training_db(db_path)
    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(model_type="random_forest")
    assert result.success is True
    metadata = json.loads(Path(result.data["metadata_path"]).read_text(encoding="utf-8"))
    for key in [
        "semantic_target",
        "label_source_policy",
        "availability_stage",
        "feature_set",
        "deployment_role",
        "split_protocol",
        "event_overlap_rate",
        "duplicate_policy",
        "canonical_only",
        "normative_version",
        "normative_hash",
        "preprocessing_version",
        "calibration_status",
        "forbidden_feature_check_passed",
        "missing_policy",
        "optional_lpr_fields",
        "metric_primary",
        "train_event_count",
        "test_event_count",
    ]:
        assert key in metadata
