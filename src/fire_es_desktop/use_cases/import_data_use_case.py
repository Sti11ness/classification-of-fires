# src/fire_es_desktop/use_cases/import_data_use_case.py
"""
ImportDataUseCase — импорт данных из Excel.

Согласно spec_first.md раздел 5.1 и spec_second.md раздел 11.2:
- Массовая загрузка Excel
- Очистка и валидация
- Запись в БД
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys
from pathlib import Path as PPath
sys.path.insert(0, str(PPath(__file__).parent.parent.parent / "fire_es"))

from fire_es.cleaning import load_fact_sheet, clean_fire_data
from fire_es.db import DatabaseManager

logger = logging.getLogger("ImportDataUseCase")


def _normalize_db_value(value: Any) -> Any:
    """Преобразовать pandas/numpy missing в None перед вставкой в SQLite."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # numpy scalar -> python scalar
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.item()
        except Exception:
            pass

    return value


class ImportDataUseCase(BaseUseCase):
    """
    Сценарий импорта данных из Excel.

    Шаги:
    1. Загрузка Excel
    2. Очистка и валидация
    3. Отчёт о качестве
    4. Запись в БД
    """

    def __init__(self, db_path: Path):
        super().__init__(
            name="ImportData",
            description="Импорт данных из Excel в БД"
        )
        self.db_path = db_path

    def execute(
        self,
        excel_path: Path,
        sheet_name: Optional[str] = None,
        sheet_names: Optional[List[str]] = None,
        clean: bool = True,
        save_to_db: bool = True
    ) -> UseCaseResult:
        """
        Выполнить импорт данных.

        Args:
            excel_path: Путь к Excel файлу.
            sheet_name: Имя листа (None = все листы).
            sheet_names: Список листов для импорта (приоритетнее sheet_name).
            clean: Выполнять очистку.
            save_to_db: Сохранять в БД.

        Returns:
            Результат импорта.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings = []

        try:
            # Шаг 1: Загрузка Excel
            self.report_progress(1, 4, "Загрузка Excel файла")
            self.check_cancelled()

            if not excel_path.exists():
                return UseCaseResult(
                    success=False,
                    message=f"Файл не найден: {excel_path}",
                    warnings=warnings
                )

            all_dfs = []
            with pd.ExcelFile(str(excel_path)) as xl:
                # Загрузка данных
                if sheet_names:
                    sheets = sheet_names
                elif sheet_name:
                    sheets = [sheet_name]
                else:
                    # Загрузить все листы
                    sheets = xl.sheet_names

                for i, sheet in enumerate(sheets):
                    self.check_cancelled()
                    df = load_fact_sheet(sheet, xl)
                    df['source_sheet'] = sheet
                    all_dfs.append(df)
                    logger.info(f"Loaded sheet '{sheet}': {len(df)} rows")

            if not all_dfs:
                return UseCaseResult(
                    success=False,
                    message="Нет данных для загрузки",
                    warnings=warnings
                )

            # Склейка листов
            raw_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Total rows after concat: {len(raw_df)}")

            # Шаг 2: Очистка и валидация
            if clean:
                self.report_progress(2, 4, "Очистка и валидация данных")
                self.check_cancelled()

                clean_df, quality_report = clean_fire_data(raw_df)

                # Предупреждения о качестве
                if quality_report:
                    duplicates = quality_report.get("duplicates_count", 0)
                    if duplicates > 0:
                        warnings.append(f"Найдено дубликатов: {duplicates}")

                    invalid_dates = quality_report.get("invalid_dates_count", 0)
                    if invalid_dates > 0:
                        warnings.append(f"Некорректных дат: {invalid_dates}")

                    outliers = quality_report.get("outliers_count", 0)
                    if outliers > 0:
                        warnings.append(f"Выбросов (этажность): {outliers}")

                logger.info(f"Clean data: {len(clean_df)} rows")
            else:
                clean_df = raw_df
                quality_report = {}

            # Шаг 3: Сохранение в БД
            if save_to_db:
                self.report_progress(3, 4, "Запись данных в БД")
                self.check_cancelled()

                db = DatabaseManager(str(self.db_path))
                db.create_tables()

                # Конвертировать DataFrame в список словарей
                records = clean_df.to_dict('records')

                # Добавить флаги источника
                for record in records:
                    record['source_file'] = str(excel_path)

                # Пакетная вставка
                added_count = 0
                if records:
                    # Используем SQLAlchemy bulk insert
                    from fire_es.db import Fire
                    from sqlalchemy.orm import sessionmaker

                    engine = db.engine
                    Session = sessionmaker(bind=engine)
                    fire_columns = {c.name for c in Fire.__table__.columns}

                    with Session() as session:
                        # Создать объекты Fire
                        fires = []
                        for record in records:
                            clean_record = {}
                            for k, v in record.items():
                                if k not in fire_columns:
                                    continue
                                clean_record[k] = _normalize_db_value(v)
                            fire = Fire(**clean_record)
                            fires.append(fire)

                        session.bulk_save_objects(fires)
                        session.commit()
                        added_count = len(fires)

                self.report_progress(4, 4, "Импорт завершён")

                return UseCaseResult(
                    success=True,
                    message=f"Импортировано {added_count} записей",
                    data={
                        "loaded_rows": len(raw_df),
                        "clean_rows": len(clean_df),
                        "added_to_db": added_count,
                        "quality_report": quality_report
                    },
                    warnings=warnings
                )
            else:
                # Не сохранять в БД, вернуть данные
                self.report_progress(4, 4, "Загрузка завершена (без сохранения)")

                return UseCaseResult(
                    success=True,
                    message=f"Загружено {len(clean_df)} записей",
                    data={
                        "loaded_rows": len(raw_df),
                        "clean_rows": len(clean_df),
                        "dataframe": clean_df,
                        "quality_report": quality_report
                    },
                    warnings=warnings
                )

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            self.status = UseCaseStatus.FAILED

            return UseCaseResult(
                success=False,
                message=f"Ошибка импорта: {str(e)}",
                error=str(e),
                warnings=warnings
            )
