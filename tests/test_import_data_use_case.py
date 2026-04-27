from pathlib import Path

import pandas as pd

from fire_es.db import init_db
from fire_es_desktop.use_cases.import_data_use_case import ImportDataUseCase


def _build_import_workbook(path: Path) -> None:
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


def test_import_data_use_case_skips_existing_events_on_reimport(tmp_path: Path):
    db_path = tmp_path / "import.sqlite"
    init_db(str(db_path))
    excel_path = tmp_path / "import.xlsx"
    _build_import_workbook(excel_path)

    use_case = ImportDataUseCase(db_path)
    first = use_case.execute(
        excel_path=excel_path,
        sheet_name="БД-1 тест",
        clean=True,
        save_to_db=True,
        skip_existing_events=True,
    )
    second = use_case.execute(
        excel_path=excel_path,
        sheet_name="БД-1 тест",
        clean=True,
        save_to_db=True,
        skip_existing_events=True,
    )

    assert first.success is True
    assert second.success is True
    assert first.data["added_to_db"] == 2
    assert second.data["added_to_db"] == 0
    assert second.data["skipped_existing_duplicates"] == 2
