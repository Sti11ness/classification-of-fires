from pathlib import Path

import pandas as pd

from fire_es.db import init_db
from fire_es_desktop.viewmodels.import_data_viewmodel import ImportDataViewModel


def _build_preview_workbook(path: Path) -> None:
    rows = [
        ["meta", "", ""],
        ["meta2", "", ""],
        ["meta3", "", ""],
        ["N п/п", "Субъекты РФ", "Дата возникновения пожара"],
        [1, "77 (Москва)", "2020-01-01"],
        [2, "78 (Санкт-Петербург)", "2020-02-01"],
    ]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="БД-1 тест", header=False, index=False)


def test_import_preview_supports_raw_and_cleaned_modes(tmp_path: Path):
    db_path = tmp_path / "preview.sqlite"
    init_db(str(db_path))
    excel_path = tmp_path / "preview.xlsx"
    _build_preview_workbook(excel_path)

    vm = ImportDataViewModel(db_path)
    assert vm.select_file(excel_path) is True

    vm.set_preview_mode("raw")
    assert vm.load_preview(sheet_name="БД-1 тест", limit=10) is True
    raw = vm.get_preview_data()
    assert raw is not None
    assert "Лист" in raw.columns
    assert "A" in raw.columns
    assert "N п/п" not in raw.columns

    vm.set_preview_mode("cleaned")
    vm.set_clean_option(True)
    assert vm.load_preview(sheet_name="БД-1 тест", limit=10) is True
    cleaned = vm.get_preview_data()
    assert cleaned is not None
    assert "Источник" in cleaned.columns
    assert "N п/п" in cleaned.columns
    assert "Субъекты РФ" in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns
    assert "rank_label_source" not in cleaned.columns
    assert "event_id" not in cleaned.columns
