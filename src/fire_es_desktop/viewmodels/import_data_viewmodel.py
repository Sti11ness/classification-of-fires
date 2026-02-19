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

    def load_preview(self, sheet_name: Optional[str] = None, limit: int = 100) -> bool:
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
            if sheet_name is None and self.available_sheets:
                sheet_name = self.available_sheets[0]

            # Загрузить данные для предпросмотра
            df = pd.read_excel(self.selected_file, sheet_name=sheet_name, nrows=limit)
            self.preview_data = df

            logger.info(f"Loaded preview: {len(df)} rows from sheet '{sheet_name}'")
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
                save_to_db=self.do_save_to_db
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
