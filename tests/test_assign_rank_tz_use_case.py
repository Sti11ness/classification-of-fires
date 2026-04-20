from pathlib import Path

import pandas as pd

from fire_es.db import DatabaseManager, init_db
from fire_es_desktop.use_cases import AssignRankTzUseCase


def test_assign_rank_tz_skips_human_verified_rows(tmp_path: Path):
    db_path = tmp_path / "assign_rank.sqlite"
    init_db(str(db_path))
    db = DatabaseManager(str(db_path))
    db.add_fire(
        {
            "row_id": 1,
            "fire_date": pd.Timestamp("2025-01-01"),
            "year": 2025,
            "month": 1,
            "equipment": "11, 23",
            "equipment_count": 2,
            "rank_tz": 3.0,
            "rank_tz_vector": 3.0,
            "rank_label_source": "lpr_decision",
            "human_verified": True,
            "usable_for_training": True,
            "is_canonical_event_record": True,
        }
    )
    db.add_fire(
        {
            "row_id": 2,
            "fire_date": pd.Timestamp("2025-01-02"),
            "year": 2025,
            "month": 1,
            "equipment": "11, 23",
            "equipment_count": 2,
            "usable_for_training": True,
            "is_canonical_event_record": True,
        }
    )
    db.close()

    use_case = AssignRankTzUseCase(db_path)
    result = use_case.execute()
    assert result.success is True
    assert result.data["human_verified_skipped_count"] == 1
    assert result.data["lpr_decision_skipped_count"] == 1

    db = DatabaseManager(str(db_path))
    df = pd.read_sql("SELECT row_id, rank_tz_vector, rank_label_source FROM fires ORDER BY row_id", db.engine)
    assert df.loc[0, "rank_tz_vector"] == 3.0
    assert df.loc[0, "rank_label_source"] == "lpr_decision"
    assert df.loc[1, "rank_label_source"] == "historical_vector"
    db.close()
