import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from fire_es.rank_tz_contract import apply_preprocessor_artifact, build_preprocessor_artifact, get_feature_set_spec
from fire_es_desktop.ui.pages.lpr_predict_page import LPRPredictPage
import pandas as pd


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_blank_lpr_inputs_stay_none(qt_app):
    page = LPRPredictPage()
    payload = page._collect_input_data()
    assert payload["building_floors"] is None
    assert payload["distance_to_station"] is None


def test_optional_dispatch_fields_always_emit_missing_indicators():
    spec = get_feature_set_spec("dispatch_initial_safe")
    df = pd.DataFrame(
        {
            "region_code": [77, 78],
            "settlement_type_code": [1, 2],
            "fire_protection_code": [1, 2],
            "enterprise_type_code": [11, 12],
            "building_floors": [5, 6],
            "fire_floor": [2, 3],
            "fire_resistance_code": [2, 3],
            "distance_to_station": [1.5, 2.5],
            "t_detect_min": [10, 12],
            "t_report_min": [14, 16],
        }
    )
    artifact, _ = build_preprocessor_artifact(
        df,
        feature_order=spec["feature_order"],
        feature_set=spec["feature_set"],
        fill_strategy=spec["default_fill_strategy"],
        fill_value=spec["default_fill_value"],
        training_rows=len(df),
        test_size=0.25,
        random_state=42,
    )
    applied = apply_preprocessor_artifact({"region_code": 77}, artifact)
    assert applied.iloc[0]["building_floors__missing"] == 1.0
    assert applied.iloc[0]["distance_to_station__missing"] == 1.0
