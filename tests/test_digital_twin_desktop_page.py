import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from fire_es_desktop.ui.main_window import MainWindow
from fire_es_desktop.ui.pages.digital_twin_page import DigitalTwinPage


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _navigation_texts(window: MainWindow) -> list[str]:
    return [window.navigation.item(idx).text() for idx in range(window.navigation.count())]


def test_digital_twin_page_visible_only_for_analyst(qt_app):
    analyst_window = MainWindow(role="analyst")
    try:
        assert "Цифровой двойник" in _navigation_texts(analyst_window)
    finally:
        analyst_window.close()

    lpr_window = MainWindow(role="lpr")
    try:
        assert "Цифровой двойник" not in _navigation_texts(lpr_window)
    finally:
        lpr_window.close()


def test_digital_twin_page_receives_reports_path(tmp_path, qt_app):
    page = DigitalTwinPage()
    reports_path = tmp_path / "reports"
    reports_path.mkdir()
    page.set_paths(None, reports_path)
    assert page.viewmodel is not None
    assert page.viewmodel.reports_path == reports_path
