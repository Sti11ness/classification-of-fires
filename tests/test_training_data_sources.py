from datetime import datetime
from pathlib import Path

import json
import sqlite3

import pandas as pd

from fire_es.db import init_db
from fire_es_desktop.infra import DbRepository, TrainingDataStore
from fire_es_desktop.use_cases import TrainModelUseCase
from tests.test_metrics_passport import _seed_training_db


def _seed_imbalanced_historical(db_path: Path) -> None:
    store = TrainingDataStore(db_path)
    rows = []
    classes = [1.0] * 12 + [2.0] * 8 + [3.0] * 6
    for idx, rank in enumerate(classes):
        rows.append(
            {
                "row_id": idx + 1,
                "source_sheet": "HIST",
                "fire_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx),
                "year": 2025,
                "month": 1,
                "region_code": 70 + (idx % 5),
                "settlement_type_code": idx % 3 + 1,
                "fire_protection_code": idx % 4 + 1,
                "enterprise_type_code": 10 + (idx % 6),
                "building_floors": idx % 9 + 1,
                "fire_floor": idx % 5,
                "fire_resistance_code": idx % 5 + 1,
                "distance_to_station": float((idx % 7) + 1),
                "t_detect_min": 5 + idx,
                "t_report_min": 7 + idx,
                "rank_tz": rank,
                "rank_tz_vector": rank,
                "rank_label_source": "historical_vector",
                "rank_normative_version": "rank_resource_normatives_v1",
                "usable_for_training": True,
                "is_canonical_event_record": True,
                "event_id": f"evt-h-{idx}",
            }
        )
    store.insert_historical_records(rows)
    store.close()


def test_sync_new_lpr_decisions_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "sync.sqlite"
    init_db(str(db_path))
    repo = DbRepository(db_path)
    fire_id = repo.add_fire(
        {
            "row_id": 1,
            "source_sheet": "LPR_MANUAL_INPUT",
            "fire_date": pd.Timestamp("2025-01-01"),
            "year": 2025,
            "month": 1,
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 2,
            "enterprise_type_code": 11,
            "building_floors": 5,
            "fire_floor": 2,
            "fire_resistance_code": 3,
            "distance_to_station": 1.5,
            "rank_tz": 2.0,
            "rank_tz_vector": 2.0,
            "rank_label_source": "lpr_decision",
            "human_verified": True,
            "usable_for_training": True,
            "event_id": "evt-lpr-1",
            "is_canonical_event_record": True,
        }
    )
    repo.add_lpr_decision(
        {
            "fire_id": fire_id,
            "decision_rank": 2.0,
            "decision_resources": {"selected_rank": "2"},
            "predicted_rank": 2.0,
            "predicted_probabilities": [],
            "comment": "ok",
            "save_to_db": True,
            "created_at": datetime.utcnow(),
        }
    )
    store = TrainingDataStore(db_path)
    first = store.sync_new_lpr_decisions(promoted_by="analyst")
    second = store.sync_new_lpr_decisions(promoted_by="analyst")
    counts = store.get_source_counts()
    store.close()
    repo.close()

    assert first["added"] == 1
    assert second["added"] == 0
    assert counts["lpr_total"] == 1
    assert counts["new_lpr_candidates"] == 0


def test_train_model_uses_historical_sources_and_persists_synthetic_batch(tmp_path: Path):
    db_path = tmp_path / "synthetic.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    init_db(str(db_path))
    _seed_imbalanced_historical(db_path)

    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(
        model_type="random_forest",
        synthetic_method="smote",
        synthetic_k_neighbors=3,
    )

    assert result.success is True
    assert result.data["historical_selected_rows"] > 0
    assert result.data["synthetic_rows_added"] > 0
    assert result.data["synthetic_method"] == "smote"

    metadata = json.loads(Path(result.data["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["dataset_info"]["synthetic_rows_added"] > 0
    assert metadata["dataset_info"]["synthetic_method"] == "smote"

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT COUNT(*) FROM train_synthetic").fetchone()[0]
    conn.close()
    assert rows > 0


def test_train_model_respects_requested_train_size_after_synthetic(tmp_path: Path):
    db_path = tmp_path / "synthetic_target.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    init_db(str(db_path))
    _seed_imbalanced_historical(db_path)

    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(
        model_type="random_forest",
        synthetic_method="smote",
        synthetic_k_neighbors=3,
        synthetic_target_total_rows=24,
    )

    assert result.success is True
    assert result.data["synthetic_method"] == "smote"
    assert result.data["synthetic_target_total_rows"] == 24
    assert result.data["train_rows_after_synthetic"] == 24
    assert result.data["test_rows_real"] > 0


def test_optuna_tuning_runs_for_supported_models(tmp_path: Path):
    db_path = tmp_path / "tuning.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    _seed_training_db(db_path)

    for model_type in ("decision_tree", "random_forest", "gradient_boosting"):
        use_case = TrainModelUseCase(db_path, models_path)
        result = use_case.execute(
            model_type=model_type,
            tuning_enabled=True,
            tuning_trials=2,
        )
        assert result.success is True
        assert result.data["tuning_enabled"] is True
        assert result.data["tuning_trials_completed"] == 2
        assert result.data["tuning_metric"] == "f1_macro"
        assert isinstance(result.data["best_params"], dict)
        assert "class_weight" in result.data["best_params"]


def test_optuna_rejects_temporal_holdout_in_v1(tmp_path: Path):
    db_path = tmp_path / "tuning_temporal.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    _seed_training_db(db_path)

    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(
        model_type="random_forest",
        tuning_enabled=True,
        tuning_trials=2,
        split_protocol="temporal_holdout",
    )

    assert result.success is False
    assert "автоподбор" in result.message.lower()


def test_safe_neighbor_count_is_clamped_for_small_classes(tmp_path: Path):
    db_path = tmp_path / "neighbors.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    use_case = TrainModelUseCase(db_path, models_path)
    y_train = pd.Series([1, 1, 1, 2, 2, 3, 3])
    safe_k = use_case._safe_neighbor_count(  # noqa: SLF001 - regression on internal guard
        y_train=y_train,
        sampling_strategy="not majority",
        requested_neighbors=5,
    )
    assert safe_k == 1


def test_train_model_supports_smotenc_on_categorical_feature_set(tmp_path: Path):
    db_path = tmp_path / "smotenc.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    _seed_training_db(db_path)

    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(
        model_type="random_forest",
        synthetic_method="smotenc",
        synthetic_k_neighbors=2,
        synthetic_target_total_rows=20,
    )

    assert result.success is True
    assert result.data["synthetic_method"] == "smotenc"
    assert result.data["synthetic_rows_added"] >= 0
