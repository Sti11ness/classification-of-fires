import json
from pathlib import Path

import pandas as pd

from fire_es.db import init_db
from fire_es_desktop.infra import TrainingDataStore
from fire_es_desktop.use_cases import TrainModelUseCase


def _seed_training_db(db_path: Path) -> None:
    db = init_db(str(db_path))
    store = TrainingDataStore(db_path)
    rows = []
    for idx in range(24):
        cls = (idx % 6) + 1
        rows.append(
            {
                "row_id": idx + 1,
                "source_sheet": "TEST",
                "fire_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx),
                "year": 2025,
                "month": 1,
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
                "rank_tz": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}[cls],
                "rank_tz_vector": {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}[cls],
                "rank_label_source": "historical_vector",
                "rank_normative_version": "rank_resource_normatives_v1",
                "usable_for_training": True,
                "is_canonical_event_record": True,
                "event_id": f"evt-{idx // 2}",
            }
        )
    store.insert_historical_records(rows)
    store.close()
    db.close()


def test_train_model_writes_metrics_passport(tmp_path: Path):
    db_path = tmp_path / "train.sqlite"
    models_path = tmp_path / "models"
    models_path.mkdir()
    _seed_training_db(db_path)
    use_case = TrainModelUseCase(db_path, models_path)
    result = use_case.execute(model_type="decision_tree")
    assert result.success is True
    meta_path = Path(result.data["metadata_path"])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["semantic_target"] == "rank_tz_vector"
    assert metadata["split_protocol"] == "group_shuffle"
    assert metadata["event_overlap_rate"] == 0.0
    assert metadata["metric_primary"] == "f1_macro"
