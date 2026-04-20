from pathlib import Path

import pandas as pd

from fire_es.db import DatabaseManager, init_db
from fire_es_desktop.use_cases import SaveDecisionUseCase


def test_lpr_decision_becomes_training_target(tmp_path: Path):
    db_path = tmp_path / "feedback.sqlite"
    init_db(str(db_path))
    use_case = SaveDecisionUseCase(db_path)

    result = use_case.execute(
        input_data={
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 1,
            "enterprise_type_code": 11,
            "building_floors": 9,
            "fire_floor": 3,
            "fire_resistance_code": 2,
            "source_item_code": 12,
            "distance_to_station": 3.0,
            "t_detect_min": 10,
            "t_report_min": 14,
        },
        prediction_data={
            "top_k_ranks": [
                {"rank": "2", "probability": 0.7},
                {"rank": "3", "probability": 0.2},
            ]
        },
        decision_rank="3",
    )

    assert result.success is True
    db = DatabaseManager(str(db_path))
    df = pd.read_sql("SELECT rank_tz, rank_tz_vector, rank_label_source, usable_for_training, predicted_rank_at_decision, event_id FROM fires", db.engine)
    assert df.iloc[0]["rank_tz"] == 3.0
    assert df.iloc[0]["rank_tz_vector"] == 3.0
    assert df.iloc[0]["rank_label_source"] == "lpr_decision"
    assert df.iloc[0]["usable_for_training"] == 1
    assert df.iloc[0]["predicted_rank_at_decision"] == 2.0
    assert df.iloc[0]["event_id"] is not None
    db.close()


def test_lpr_decision_with_sparse_dispatch_input_still_builds_event_identity(tmp_path: Path):
    db_path = tmp_path / "feedback_sparse.sqlite"
    init_db(str(db_path))
    use_case = SaveDecisionUseCase(db_path)

    result = use_case.execute(
        input_data={
            "region_code": 77,
            "settlement_type_code": 1,
            "enterprise_type_code": 11,
            "t_report_min": 14,
        },
        prediction_data={
            "top_k_ranks": [
                {"rank": "2", "probability": 0.7},
                {"rank": "3", "probability": 0.2},
            ]
        },
        decision_rank="3",
    )

    assert result.success is True
    db = DatabaseManager(str(db_path))
    df = pd.read_sql("SELECT event_id, duplicate_group_id, rank_tz_vector FROM fires", db.engine)
    assert df.iloc[0]["event_id"] is not None
    assert df.iloc[0]["duplicate_group_id"] is not None
    assert df.iloc[0]["rank_tz_vector"] == 3.0
    db.close()
