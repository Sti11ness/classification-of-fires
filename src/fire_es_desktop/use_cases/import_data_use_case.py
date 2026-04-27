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
from sqlalchemy import bindparam, text

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys
from pathlib import Path as PPath
sys.path.insert(0, str(PPath(__file__).parent.parent.parent / "fire_es"))

from fire_es.cleaning import load_fact_sheet, clean_fire_data
from fire_es.db import DatabaseManager, FIRES_HISTORICAL_TABLE
from ..infra import TrainingDataStore

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
        save_to_db: bool = True,
        skip_existing_events: bool = True,
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
                dropped_internal_duplicates = 0
                if "is_canonical_event_record" in clean_df.columns:
                    before_dedup = len(clean_df)
                    clean_df = clean_df.loc[
                        clean_df["is_canonical_event_record"].fillna(True).astype(bool)
                    ].reset_index(drop=True)
                    dropped_internal_duplicates = before_dedup - len(clean_df)
                    quality_report["dropped_internal_duplicates"] = dropped_internal_duplicates
                    quality_report["canonical_rows_after_cleanup"] = int(len(clean_df))

                # Предупреждения о качестве
                if quality_report:
                    duplicates = quality_report.get("duplicates_count", quality_report.get("duplicate_rows", 0))
                    if duplicates > 0:
                        warnings.append(f"Найдено дубликатов: {duplicates}")
                    if dropped_internal_duplicates > 0:
                        warnings.append(
                            f"Удалено внутренних дублей при очистке: {dropped_internal_duplicates}"
                        )

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
                training_store = TrainingDataStore(self.db_path)
                skipped_existing_duplicates = 0

                # Конвертировать DataFrame в список словарей
                records = clean_df.to_dict('records')

                # Добавить флаги источника
                for record in records:
                    record['source_file'] = str(excel_path)

                # Пакетная вставка
                added_count = 0
                if records:
                    engine = db.engine
                    from sqlalchemy.orm import sessionmaker
                    Session = sessionmaker(bind=engine)
                    fire_columns = {c.name for c in FIRES_HISTORICAL_TABLE.columns}

                    with Session() as session:
                        if skip_existing_events and records:
                            records, skipped_existing_duplicates = self._filter_existing_events(
                                session,
                                records,
                            )
                            if skipped_existing_duplicates:
                                warnings.append(
                                    f"Пропущено уже известных событий: {skipped_existing_duplicates}"
                                )

                        historical_records = []
                        for record in records:
                            clean_record = {}
                            for k, v in record.items():
                                if k not in fire_columns:
                                    continue
                                clean_record[k] = _normalize_db_value(v)
                            historical_records.append(clean_record)

                        added_count = training_store.insert_historical_records(historical_records)

                self.report_progress(4, 4, "Импорт завершён")
                training_store.close()
                db.close()

                return UseCaseResult(
                    success=True,
                    message=f"Импортировано {added_count} записей",
                    data={
                        "loaded_rows": len(raw_df),
                        "clean_rows": len(clean_df),
                        "added_to_db": added_count,
                        "skipped_existing_duplicates": skipped_existing_duplicates if save_to_db else 0,
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

    @staticmethod
    def _chunked(values: list[str], chunk_size: int = 800) -> list[list[str]]:
        return [values[index:index + chunk_size] for index in range(0, len(values), chunk_size)]

    def _filter_existing_events(
        self,
        session,
        records: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int]:
        """Skip rows whose event identity is already present in the workspace."""
        incoming_event_ids = sorted(
            {
                str(record["event_id"])
                for record in records
                if record.get("event_id") not in (None, "")
            }
        )
        incoming_fingerprints = sorted(
            {
                str(record["event_fingerprint"])
                for record in records
                if record.get("event_fingerprint") not in (None, "")
            }
        )
        existing_event_ids: set[str] = set()
        existing_fingerprints: set[str] = set()

        for chunk in self._chunked(incoming_event_ids):
            rows = session.execute(
                text(
                    "SELECT event_id FROM fires_historical "
                    "WHERE event_id IN :event_ids"
                ).bindparams(bindparam("event_ids", expanding=True)),
                {"event_ids": chunk},
            ).all()
            existing_event_ids.update(value for value, in rows if value)

        for chunk in self._chunked(incoming_fingerprints):
            rows = session.execute(
                text(
                    "SELECT event_fingerprint FROM fires_historical "
                    "WHERE event_fingerprint IN :event_fingerprints"
                ).bindparams(bindparam("event_fingerprints", expanding=True)),
                {"event_fingerprints": chunk},
            ).all()
            existing_fingerprints.update(value for value, in rows if value)

        filtered_records: list[Dict[str, Any]] = []
        skipped_count = 0
        for record in records:
            event_id = record.get("event_id")
            fingerprint = record.get("event_fingerprint")
            if (event_id and event_id in existing_event_ids) or (
                fingerprint and fingerprint in existing_fingerprints
            ):
                skipped_count += 1
                continue
            filtered_records.append(record)

        return filtered_records, skipped_count
