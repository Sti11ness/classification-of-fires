import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from fire_es.db import init_db
from fire_es_desktop.ui.pages.lpr_predict_page import LPRPredictPage
from tests.test_rank_tz_contract_desktop import create_production_bundle


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_lpr_page_defaults_to_dispatch_schema(qt_app):
    page = LPRPredictPage()
    names = [field["name"] for field in page.input_schema]
    assert "t_arrival_min" not in names
    assert "t_first_hose_min" not in names
    assert "source_item_code" not in names


def test_lpr_page_uses_active_model_schema_or_dispatch_fallback(tmp_path: Path, qt_app):
    db_path = tmp_path / "page.sqlite"
    init_db(str(db_path))
    models_path = create_production_bundle(tmp_path)
    page = LPRPredictPage()
    page.set_paths(db_path, models_path)
    names = [field["name"] for field in page.input_schema]
    assert names == [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "distance_to_station",
        "t_detect_min",
        "t_report_min",
    ]
