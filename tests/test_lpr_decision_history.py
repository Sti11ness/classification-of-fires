from datetime import datetime
from pathlib import Path

from fire_es.db import DatabaseManager, init_db
from fire_es_desktop.infra import DbRepository
from fire_es_desktop.viewmodels import LPRDecisionHistoryViewModel


def create_history_fixture(db_path: Path) -> int:
    db = DatabaseManager(str(db_path))
    fire = db.add_fire(
        {
            "row_id": 1,
            "source_sheet": "LPR_MANUAL_INPUT",
            "fire_date": datetime(2025, 1, 1, 12, 0, 0),
            "year": 2025,
            "month": 1,
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 2,
            "enterprise_type_code": 11,
            "building_floors": 9,
            "fire_floor": 3,
            "fire_resistance_code": 2,
            "source_item_code": 12,
            "distance_to_station": 2.5,
            "t_detect_min": 15,
            "t_report_min": 25,
            "t_arrival_min": 35,
            "t_first_hose_min": 45,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
    )
    decision = db.add_lpr_decision(
        fire_id=fire.id,
        decision_rank=2.0,
        predicted_rank=1.5,
        predicted_probabilities=[
            {"rank": "1", "probability": 0.37},
            {"rank": "1-бис", "probability": 0.25},
        ],
        comment="Initial comment",
        save_to_db=True,
    )
    db.close()
    return decision.id


def test_db_repository_returns_decision_history_detail(tmp_path: Path):
    db_path = tmp_path / "history.sqlite"
    init_db(str(db_path))
    decision_id = create_history_fixture(db_path)

    repo = DbRepository(db_path)
    summaries = repo.get_lpr_decision_summaries()
    assert len(summaries) == 1
    assert summaries[0]["decision_id"] == decision_id
    assert summaries[0]["comment_preview"] == "Initial comment"

    detail = repo.get_lpr_decision_detail(decision_id)
    assert detail is not None
    assert detail["decision_rank"] == 2.0
    assert detail["predicted_rank"] == 1.5
    assert detail["fire_snapshot"]["region_code"] == 77
    assert detail["predicted_probabilities"][0]["rank"] == "1"
    repo.close()


def test_history_viewmodel_updates_only_rank_and_comment(tmp_path: Path):
    db_path = tmp_path / "history.sqlite"
    init_db(str(db_path))
    decision_id = create_history_fixture(db_path)

    vm = LPRDecisionHistoryViewModel(db_path)
    vm.load_decisions()
    assert len(vm.state.decisions) == 1
    assert vm.state.decisions[0]["decision_rank_label"] == "2"

    vm.select_decision(decision_id)
    assert vm.state.selected_detail is not None
    assert vm.state.selected_detail["comment"] == "Initial comment"
    assert vm.state.selected_detail["predicted_rank"] == 1.5

    updated = vm.update_selected_decision("3", "Updated comment")
    assert updated is True

    vm.select_decision(decision_id)
    assert vm.state.selected_detail["decision_rank"] == 3.0
    assert vm.state.selected_detail["decision_rank_label"] == "3"
    assert vm.state.selected_detail["comment"] == "Updated comment"
    assert vm.state.selected_detail["predicted_rank"] == 1.5
    vm.close()
