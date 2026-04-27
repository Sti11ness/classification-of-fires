# src/fire_es_desktop/viewmodels/import_data_viewmodel.py
"""
ImportDataViewModel — ViewModel для экрана импорта данных.

Согласно spec_second.md раздел 11.2:
- Выбор файлов Excel
- Предпросмотр
- Очистка и валидация
- Запись в БД
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import pandas as pd

from ..use_cases import ImportDataUseCase, UseCaseResult
from fire_es.cleaning import load_fact_sheet, clean_fire_data
from fire_es.schema import RU_COLS, RU_TO_EN

logger = logging.getLogger("ImportDataViewModel")


class ImportDataViewModel:
    """
    ViewModel для экрана импорта.

    Управляет загрузкой Excel и валидацией данных.
    """

    def __init__(self, db_path: Path):
        """
        Инициализировать ViewModel.

        Args:
            db_path: Путь к БД.
        """
        self.db_path = db_path
        self.import_use_case = ImportDataUseCase(db_path)

        # Состояние
        self.is_importing = False
        self.selected_file: Optional[Path] = None
        self.available_sheets: List[str] = []
        self.selected_sheets: List[str] = []
        self.preview_data: Optional[pd.DataFrame] = None
        self.import_result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        self.warnings: List[str] = []

        # Опции
        self.do_clean = True
        self.do_save_to_db = True
        self.skip_existing_events = True
        self.preview_mode = "cleaned"

        # Callbacks
        self.on_import_started: Optional[Callable[[], None]] = None
        self.on_import_complete: Optional[Callable[[Dict], None]] = None
        self.on_import_failed: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[int, int, str], None]] = None

    def select_file(self, file_path: Path) -> bool:
        """
        Выбрать файл для импорта.

        Args:
            file_path: Путь к Excel файлу.

        Returns:
            True если успешно.
        """
        try:
            if not file_path.exists():
                self.error_message = f"Файл не найден: {file_path}"
                return False

            if not file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.error_message = "Выберите файл Excel (.xlsx)"
                return False

            self.selected_file = file_path
            self.error_message = None

            # Получить список листов
            xl = pd.ExcelFile(file_path)
            self.available_sheets = xl.sheet_names
            self.selected_sheets = []  # По умолчанию не выбрано ни одного
            xl.close()

            logger.info(f"Selected file: {file_path}, sheets: {self.available_sheets}")
            return True

        except Exception as e:
            logger.error(f"Select file failed: {e}", exc_info=True)
            self.error_message = f"Ошибка: {str(e)}"
            return False

    def set_selected_sheets(self, sheets: List[str]) -> None:
        """Установить выбранные листы."""
        self.selected_sheets = sheets

    def set_clean_option(self, do_clean: bool) -> None:
        """Установить опцию очистки."""
        self.do_clean = do_clean

    def set_save_to_db_option(self, do_save: bool) -> None:
        """Установить опцию сохранения в БД."""
        self.do_save_to_db = do_save

    def set_skip_existing_events_option(self, skip_existing_events: bool) -> None:
        """Установить опцию защиты от повторного импорта уже известных событий."""
        self.skip_existing_events = skip_existing_events

    def set_preview_mode(self, mode: str) -> None:
        """Установить режим предпросмотра: raw или cleaned."""
        self.preview_mode = mode

    def load_preview(
        self,
        sheet_name: Optional[str] = None,
        sheet_names: Optional[List[str]] = None,
        limit: int = 100,
        mode: Optional[str] = None,
    ) -> bool:
        """
        Загрузить предпросмотр данных.

        Args:
            sheet_name: Имя листа (None = первый лист).
            limit: Ограничить количество строк.

        Returns:
            True если успешно.
        """
        if not self.selected_file:
            self.error_message = "Файл не выбран"
            return False

        try:
            selected = list(sheet_names or ([] if sheet_name is None else [sheet_name]))
            if not selected and self.available_sheets:
                selected = [self.available_sheets[0]]
            preview_mode = mode or self.preview_mode

            if preview_mode == "raw":
                df = self._load_raw_preview(selected, limit=limit)
            else:
                df = self._load_cleaned_preview(selected, limit=limit)
            self.preview_data = df

            logger.info(
                "Loaded preview: %s rows from sheets %s (mode=%s)",
                len(df),
                selected,
                preview_mode,
            )
            return True

        except Exception as e:
            logger.error(f"Load preview failed: {e}", exc_info=True)
            self.error_message = f"Ошибка предпросмотра: {str(e)}"
            return False

    def import_data(self) -> None:
        """
        Запустить импорт.

        Асинхронный вызов через TaskRunner должен быть снаружи.
        """
        if not self.selected_file:
            self.error_message = "Файл не выбран"
            if self.on_import_failed:
                self.on_import_failed(self.error_message)
            return

        self.is_importing = True
        self.import_result = None
        self.error_message = None
        self.warnings = []

        if self.on_import_started:
            self.on_import_started()

        # Установить callback прогресса
        def on_progress(current, total, description):
            if self.on_progress:
                self.on_progress(current, total, description)

        self.import_use_case.set_progress_callback(on_progress)

        try:
            # Выполнить импорт
            sheet_name = self.selected_sheets[0] if len(self.selected_sheets) == 1 else None
            sheet_names = self.selected_sheets if len(self.selected_sheets) > 1 else None

            result = self.import_use_case.execute(
                excel_path=self.selected_file,
                sheet_name=sheet_name,
                sheet_names=sheet_names,
                clean=self.do_clean,
                save_to_db=self.do_save_to_db,
                skip_existing_events=self.skip_existing_events,
            )

            if result.success:
                self.import_result = result.data
                self.warnings = result.warnings

                if self.on_import_complete:
                    self.on_import_complete(result.data)
            else:
                self.error_message = result.message
                if self.on_import_failed:
                    self.on_import_failed(result.message)

        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            self.error_message = f"Ошибка импорта: {str(e)}"
            if self.on_import_failed:
                self.on_import_failed(self.error_message)

        finally:
            self.is_importing = False

    def get_preview_data(self) -> Optional[pd.DataFrame]:
        """Получить данные предпросмотра."""
        return self.preview_data

    def get_quality_report(self) -> Optional[Dict[str, Any]]:
        """Получить отчёт о качестве."""
        if self.import_result:
            return self.import_result.get("quality_report", {})
        return None

    def _load_raw_preview(self, sheets: List[str], *, limit: int) -> pd.DataFrame:
        """Показать Excel почти как есть, без угадывания заголовка."""
        remaining = limit
        raw_parts: List[pd.DataFrame] = []
        for sheet in sheets:
            if remaining <= 0:
                break
            df = pd.read_excel(self.selected_file, sheet_name=sheet, header=None, nrows=remaining)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "Лист", sheet)
            df.columns = ["Лист", *[self._excel_column_name(i) for i in range(df.shape[1] - 1)]]
            raw_parts.append(df)
            remaining -= len(df)
        return pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()

    def _load_cleaned_preview(self, sheets: List[str], *, limit: int) -> pd.DataFrame:
        """Показать данные в том виде, как они реально проходят в систему."""
        remaining = limit
        parts: List[pd.DataFrame] = []
        reverse_map = {value: key for key, value in RU_TO_EN.items()}
        with pd.ExcelFile(self.selected_file) as xl:
            for sheet in sheets:
                if remaining <= 0:
                    break
                df = load_fact_sheet(sheet, xl, nrows=remaining)
                if self.do_clean:
                    df, _ = clean_fire_data(df)
                    rename_map = {column: reverse_map.get(column, column) for column in df.columns}
                    df = df.rename(columns=rename_map)
                    df = self._prepare_display_preview(df)
                parts.append(df.head(remaining))
                remaining -= min(len(df), remaining)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _prepare_display_preview(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оставить только человекочитаемые очищенные поля для экрана импорта."""
        ordered_columns: List[str] = []
        if "source_sheet" in df.columns:
            ordered_columns.append("source_sheet")
        ordered_columns.extend(column for column in RU_COLS if column in df.columns)
        preview_df = df.loc[:, ordered_columns].copy()
        if "source_sheet" in preview_df.columns:
            preview_df = preview_df.rename(columns={"source_sheet": "Источник"})

        for column in preview_df.columns:
            series = preview_df[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                preview_df[column] = series.dt.strftime("%Y-%m-%d").fillna("")
            else:
                preview_df[column] = series.where(series.notna(), "")

        return preview_df

    @staticmethod
    def _excel_column_name(index: int) -> str:
        name = ""
        current = index + 1
        while current:
            current, remainder = divmod(current - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def reset(self) -> None:
        """Сбросить состояние."""
        self.selected_file = None
        self.available_sheets = []
        self.selected_sheets = []
        self.preview_data = None
        self.import_result = None
        self.error_message = None
        self.warnings = []
        self.is_importing = False
