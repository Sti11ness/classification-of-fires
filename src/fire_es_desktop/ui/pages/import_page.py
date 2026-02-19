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
    QAbstractItemView, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Optional

import pandas as pd

from ...viewmodels import ImportDataViewModel


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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Импорт данных из Excel")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)

        # Выбор файла
        file_group = QGroupBox("Файл Excel")
        file_layout = QHBoxLayout(file_group)

        self.file_path_label = QLabel("Файл не выбран")
        self.file_path_label.setStyleSheet("color: white;")
        file_layout.addWidget(self.file_path_label, 1)

        self.select_file_btn = QPushButton("Выбрать файл")
        self.select_file_btn.setFixedWidth(150)
        file_layout.addWidget(self.select_file_btn)

        layout.addWidget(file_group)

        # Выбор листов
        sheets_group = QGroupBox("Листы")
        sheets_layout = QVBoxLayout(sheets_group)

        self.sheets_list = QListWidget()
        self.sheets_list.setFixedHeight(150)
        self.sheets_list.setSelectionMode(QListWidget.MultiSelection)
        sheets_layout.addWidget(self.sheets_list)

        self.preview_btn = QPushButton("Предпросмотр")
        self.preview_btn.setFixedWidth(150)
        sheets_layout.addWidget(self.preview_btn, alignment=Qt.AlignLeft)

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

        options_layout.addStretch()

        layout.addWidget(options_group)

        # Предпросмотр
        preview_group = QGroupBox("Предпросмотр данных")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setFixedHeight(300)
        self.preview_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.preview_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.preview_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.preview_table.setWordWrap(False)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        # Прогресс и кнопки
        control_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar, 1)

        self.import_btn = QPushButton("Импортировать")
        self.import_btn.setFixedHeight(40)
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.import_btn)

        layout.addLayout(control_layout)

        # Статус
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: white;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.select_file_btn.clicked.connect(self._on_select_file)
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
                self.file_path_label.setStyleSheet("color: white;")

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

        # Получить выбранный лист
        selected_items = self.sheets_list.selectedItems()
        sheet_name = selected_items[0].text() if selected_items else None

        success = self.viewmodel.load_preview(sheet_name=sheet_name, limit=100)
        if success and self.viewmodel.preview_data is not None:
            self._show_preview(self.viewmodel.preview_data)
        else:
            QMessageBox.critical(
                self, "Ошибка",
                self.viewmodel.error_message or "Ошибка предпросмотра"
            )

    def _show_preview(self, df: pd.DataFrame) -> None:
        """Показать предпросмотр."""
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
        self.preview_table.setRowCount(min(len(df), 100))

        for i, row in df.head(100).iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)

        self.preview_table.resizeColumnsToContents()
        for col in range(self.preview_table.columnCount()):
            self.preview_table.setColumnWidth(
                col, max(120, min(280, self.preview_table.columnWidth(col)))
            )

    def _on_import(self) -> None:
        """Импортировать данные."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
            return

        # Настройки
        selected_items = self.sheets_list.selectedItems()
        self.viewmodel.set_selected_sheets(
            [item.text() for item in selected_items]
        )
        self.viewmodel.set_clean_option(self.clean_checkbox.isChecked())
        self.viewmodel.set_save_to_db_option(self.save_db_checkbox.isChecked())

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
        self.status_label.setText(f"Импортировано {rows} записей")
        self.status_label.setStyleSheet("color: white;")

        QMessageBox.information(
            self, "Импорт завершён",
            f"Успешно импортировано {rows} записей"
        )

    def _on_error(self, message: str) -> None:
        """Ошибка импорта."""
        self.import_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка импорта")
        self.status_label.setStyleSheet("color: white;")

        QMessageBox.critical(self, "Ошибка импорта", message)
