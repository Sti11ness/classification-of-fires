# src/fire_es_desktop/ui/pages/import_page.py
"""
ImportPage — страница импорта данных из Excel.

Согласно spec_second.md раздел 11.2:
- Выбор файлов Excel
- Предпросмотр
- Очистка и валидация
- Запись в БД
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QGroupBox,
    QCheckBox, QListWidget, QMessageBox, QProgressBar,
    QComboBox,
    QAbstractItemView, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlite3

from ...viewmodels import ImportDataViewModel
from ..theme import (
    configure_table,
    create_hint,
    create_page_header,
    create_scrollable_page,
    create_status_label,
    style_button,
    style_label,
)


class ImportWorker(QThread):
    """Рабочий поток для импорта."""

    progress = Signal(int, int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel: ImportDataViewModel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        """Выполнить импорт."""
        def on_progress(current, total, description):
            self.progress.emit(current, total, description)

        self.viewmodel.on_progress = on_progress

        try:
            self.viewmodel.import_data()

            if self.viewmodel.error_message:
                self.error.emit(self.viewmodel.error_message)
            else:
                self.complete.emit(self.viewmodel.import_result or {})
        except Exception as e:
            self.error.emit(str(e))


class ImportPage(QWidget):
    """Страница импорта."""

    def __init__(self):
        super().__init__()
        self.viewmodel: Optional[ImportDataViewModel] = None
        self.worker: Optional[ImportWorker] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        _, _, _, layout = create_scrollable_page(self)

        # Заголовок
        layout.addWidget(
            create_page_header(
                "Импорт данных из Excel",
                "Загрузка карточек пожаров, выбор листов, предпросмотр и сохранение очищенных данных в рабочее пространство.",
            )
        )

        # Выбор файла
        file_group = QGroupBox("Файл Excel")
        file_layout = QHBoxLayout(file_group)

        self.file_path_label = QLabel("Файл не выбран")
        style_label(self.file_path_label, "muted", word_wrap=True)
        file_layout.addWidget(self.file_path_label, 1)

        self.select_file_btn = QPushButton("Выбрать файл")
        style_button(self.select_file_btn, "primary")
        file_layout.addWidget(self.select_file_btn)

        layout.addWidget(file_group)

        # Выбор листов
        sheets_group = QGroupBox("Листы и таблицы")
        sheets_layout = QVBoxLayout(sheets_group)

        self.sheets_list = QListWidget()
        self.sheets_list.setMinimumHeight(180)
        self.sheets_list.setSelectionMode(QListWidget.MultiSelection)
        sheets_layout.addWidget(self.sheets_list)

        actions_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Выбрать все")
        style_button(self.select_all_btn, "ghost")
        actions_layout.addWidget(self.select_all_btn)

        self.preview_btn = QPushButton("Предпросмотр")
        style_button(self.preview_btn, "ghost")
        actions_layout.addWidget(self.preview_btn)
        actions_layout.addStretch()
        sheets_layout.addLayout(actions_layout)
        sheets_layout.addWidget(
            create_hint(
                "Можно выбрать одну или несколько вкладок. Предпросмотр показывает либо данные как в Excel, либо уже подготовленный вариант для системы."
            )
        )

        layout.addWidget(sheets_group)

        # Опции
        options_group = QGroupBox("Опции")
        options_layout = QHBoxLayout(options_group)

        self.clean_checkbox = QCheckBox("Выполнять очистку данных")
        self.clean_checkbox.setChecked(True)
        options_layout.addWidget(self.clean_checkbox)

        self.save_db_checkbox = QCheckBox("Сохранять в БД")
        self.save_db_checkbox.setChecked(True)
        options_layout.addWidget(self.save_db_checkbox)

        self.skip_duplicates_checkbox = QCheckBox("Пропускать уже загруженные события")
        self.skip_duplicates_checkbox.setChecked(True)
        options_layout.addWidget(self.skip_duplicates_checkbox)

        options_layout.addWidget(QLabel("Режим предпросмотра:"))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItem("Очищенные данные", "cleaned")
        self.preview_mode_combo.addItem("Сырой лист", "raw")
        options_layout.addWidget(self.preview_mode_combo)

        options_layout.addStretch()

        layout.addWidget(options_group)

        # Предпросмотр
        preview_group = QGroupBox("Предпросмотр данных")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.preview_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.preview_table.setWordWrap(False)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        configure_table(self.preview_table, min_height=360)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        # Прогресс и кнопки
        control_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar, 1)

        self.import_btn = QPushButton("Импортировать")
        style_button(self.import_btn, "success", large=True)
        control_layout.addWidget(self.import_btn)

        layout.addLayout(control_layout)

        # Статус
        self.status_label = create_status_label()
        self.status_label.setText("Выберите Excel-файл и один или несколько листов для импорта.")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.select_file_btn.clicked.connect(self._on_select_file)
        self.select_all_btn.clicked.connect(self._on_select_all)
        self.preview_btn.clicked.connect(self._on_preview)
        self.import_btn.clicked.connect(self._on_import)

    def set_db_path(self, db_path: Optional[Path]) -> None:
        """Установить путь к БД."""
        if db_path and db_path.exists():
            self.viewmodel = ImportDataViewModel(db_path)
            self.import_btn.setEnabled(True)
        else:
            self.viewmodel = None
            self.import_btn.setEnabled(False)

    def _on_select_file(self) -> None:
        """Выбрать файл."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите Excel файл",
            "",
            "Excel Files (*.xlsx *.xls)"
        )

        if file_path and self.viewmodel:
            success = self.viewmodel.select_file(Path(file_path))
            if success:
                self.file_path_label.setText(file_path)
                self.status_label.setText("Файл выбран. Выберите одну или несколько вкладок и откройте предпросмотр.")

                # Заполнить список листов
                self.sheets_list.clear()
                for sheet in self.viewmodel.available_sheets:
                    self.sheets_list.addItem(sheet)
            else:
                QMessageBox.critical(
                    self, "Ошибка",
                    self.viewmodel.error_message or "Ошибка выбора файла"
                )

    def _on_preview(self) -> None:
        """Предпросмотр данных."""
        if not self.viewmodel:
            return

        # Получить выбранные листы
        selected_items = self.sheets_list.selectedItems()
        sheet_names = [item.text() for item in selected_items]
        sheet_name = sheet_names[0] if len(sheet_names) == 1 else None
        self.viewmodel.set_preview_mode(self.preview_mode_combo.currentData())

        success = self.viewmodel.load_preview(
            sheet_name=sheet_name,
            sheet_names=sheet_names if len(sheet_names) > 1 else None,
            limit=100,
        )
        if success and self.viewmodel.preview_data is not None:
            self._show_preview(self.viewmodel.preview_data)
            mode_label = self.preview_mode_combo.currentText().lower()
            selected_count = len(sheet_names) if sheet_names else 1
            if selected_count > 1:
                self.status_label.setText(
                    f"Показан объединенный {mode_label} предпросмотр по {selected_count} вкладкам."
                )
            else:
                self.status_label.setText(f"Показан {mode_label} предпросмотр выбранной вкладки.")
        else:
            QMessageBox.critical(
                self, "Ошибка",
                self.viewmodel.error_message or "Ошибка предпросмотра"
            )

    def _show_preview(self, df: pd.DataFrame) -> None:
        """Показать предпросмотр."""
        self.preview_table.setUpdatesEnabled(False)
        try:
            self.preview_table.clear()
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
            self.preview_table.setRowCount(min(len(df), 100))

            for i, row in df.head(100).iterrows():
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.preview_table.setItem(i, j, item)

            for col, column_name in enumerate(df.columns):
                estimated = max(120, min(240, 14 + len(str(column_name)) * 9))
                self.preview_table.setColumnWidth(col, estimated)
        finally:
            self.preview_table.setUpdatesEnabled(True)

    def _on_import(self) -> None:
        """Импортировать данные."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
            return

        # Настройки
        selected_items = self.sheets_list.selectedItems()
        self.viewmodel.set_selected_sheets(
            [item.text() for item in selected_items]
        )
        self.viewmodel.set_clean_option(self.clean_checkbox.isChecked())
        self.viewmodel.set_save_to_db_option(self.save_db_checkbox.isChecked())
        self.viewmodel.set_skip_existing_events_option(self.skip_duplicates_checkbox.isChecked())

        # Запустить в фоне
        self.worker = ImportWorker(self.viewmodel)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)

        self.import_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Импорт данных...")

        self.worker.start()

    def _on_progress(self, current: int, total: int, description: str) -> None:
        """Обновление прогресса."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
        self.status_label.setText(description or "Импорт...")

    def _on_complete(self, result: dict) -> None:
        """Импорт завершён."""
        self.import_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        rows = result.get("added_to_db", result.get("clean_rows", 0))
        skipped_existing = result.get("skipped_existing_duplicates", 0)
        total_rows = self._get_total_rows_in_workspace()
        status_parts = [f"Добавлено {rows} записей"]
        if skipped_existing:
            status_parts.append(f"пропущено как уже известные {skipped_existing}")
        status_parts.append(f"всего строк в рабочем пространстве {total_rows}")
        self.status_label.setText(". ".join(status_parts) + ".")

        info_lines = [
            "Импорт завершён.",
            "",
            f"Добавлено записей: {rows}",
        ]
        if skipped_existing:
            info_lines.append(f"Пропущено уже известных событий: {skipped_existing}")
        info_lines.append(f"Всего строк в рабочем пространстве: {total_rows}")
        QMessageBox.information(
            self, "Импорт завершён",
            "\n".join(info_lines),
        )

    def _on_error(self, message: str) -> None:
        """Ошибка импорта."""
        self.import_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка импорта")

        QMessageBox.critical(self, "Ошибка импорта", message)

    def _on_select_all(self) -> None:
        for index in range(self.sheets_list.count()):
            item = self.sheets_list.item(index)
            item.setSelected(True)

    def _get_total_rows_in_workspace(self) -> int:
        if not self.viewmodel:
            return 0
        conn = None
        try:
            conn = sqlite3.connect(self.viewmodel.db_path)
            row = conn.execute("SELECT COUNT(*) FROM fires_historical").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0
        finally:
            try:
                conn.close()
            except Exception:
                pass
