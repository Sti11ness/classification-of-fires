# src/fire_es_desktop/ui/pages/log_page.py
"""
LogPage — страница журнала операций.

Согласно spec_second.md раздел 11.8 и раздел 13:
- Лента событий (операции, предупреждения, ошибки)
- Фильтры
- Экспорт
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QComboBox,
    QFileDialog, QMessageBox, QHeaderView, QTextEdit
)
from PySide6.QtCore import Qt
from pathlib import Path
from typing import Optional, List, Dict, Any

import json

from ..theme import (
    configure_table,
    configure_text_panel,
    create_page_header,
    create_static_page,
    style_button,
)


class LogPage(QWidget):
    """Страница журнала."""

    def __init__(self):
        super().__init__()
        self.log_store = None
        self.logs_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        layout = create_static_page(self)

        # Заголовок
        layout.addWidget(
            create_page_header(
                "Журнал операций",
                "Фильтрация, просмотр и экспорт событий приложения, фоновых задач и сервисных операций.",
            )
        )

        # Фильтры
        filter_group = QGroupBox("Фильтры")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Уровень:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems([
            "Все", "INFO", "WARNING", "ERROR", "CRITICAL"
        ])
        filter_layout.addWidget(self.level_combo)

        filter_layout.addWidget(QLabel("Источник:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Все")
        filter_layout.addWidget(self.source_combo)

        self.filter_btn = QPushButton("Применить")
        filter_layout.addWidget(self.filter_btn)

        layout.addWidget(filter_group)

        # Список записей
        logs_group = QGroupBox("Записи журнала")
        logs_layout = QVBoxLayout(logs_group)

        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(5)
        self.logs_table.setHorizontalHeaderLabels([
            "Время", "Уровень", "Источник", "Сообщение", "Данные"
        ])
        configure_table(self.logs_table, min_height=360)
        self._configure_table_columns()
        logs_layout.addWidget(self.logs_table)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.refresh_btn = QPushButton("Обновить")
        style_button(self.refresh_btn, "ghost")
        buttons_layout.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Экспорт")
        style_button(self.export_btn, "primary")
        buttons_layout.addWidget(self.export_btn)

        self.open_log_btn = QPushButton("Открыть лог")
        style_button(self.open_log_btn, "ghost")
        buttons_layout.addWidget(self.open_log_btn)

        buttons_layout.addStretch()

        logs_layout.addLayout(buttons_layout)

        layout.addWidget(logs_group)

        # Детали
        details_group = QGroupBox("Детали записи")
        details_layout = QVBoxLayout(details_group)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        configure_text_panel(self.details_text, min_height=180)
        details_layout.addWidget(self.details_text)

        layout.addWidget(details_group)

        layout.addStretch()

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.filter_btn.clicked.connect(self._on_filter)
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.export_btn.clicked.connect(self._on_export)
        self.open_log_btn.clicked.connect(self._on_open_log)
        self.logs_table.cellClicked.connect(self._on_cell_clicked)

    def set_logs_path(self, logs_path: Optional[Path]) -> None:
        """Установить путь к логам."""
        self.logs_path = logs_path
        if logs_path and logs_path.exists():
            from ...infra import LogStore
            self.log_store = LogStore(logs_path)
            self._load_logs()
        else:
            self.log_store = None
            self.logs_table.setRowCount(0)
            self.details_text.clear()

    def _load_logs(self) -> None:
        """Загрузить журнал."""
        if not self.log_store:
            return

        # Получить уровень фильтра
        level_filter = self.level_combo.currentText()
        if level_filter == "Все":
            level_filter = None

        operations = self.log_store.get_operations(limit=100)

        self.logs_table.setUpdatesEnabled(False)
        try:
            self.logs_table.setRowCount(0)
            self._configure_table_columns()

            for entry in reversed(operations):
                # Применить фильтр по уровню
                if level_filter and entry.level.value != level_filter.lower():
                    continue

                row = self.logs_table.rowCount()
                self.logs_table.insertRow(row)

                # Время
                item = QTableWidgetItem(entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.logs_table.setItem(row, 0, item)

                # Уровень
                item = QTableWidgetItem(entry.level.value.upper())
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.logs_table.setItem(row, 1, item)

                # Источник
                item = QTableWidgetItem(entry.source)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.logs_table.setItem(row, 2, item)

                # Сообщение
                item = QTableWidgetItem(entry.message)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.logs_table.setItem(row, 3, item)

                # Данные
                data_str = json.dumps(entry.data, ensure_ascii=False) if entry.data else ""
                item = QTableWidgetItem(data_str)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.logs_table.setItem(row, 4, item)
        finally:
            self.logs_table.setUpdatesEnabled(True)

    def _configure_table_columns(self) -> None:
        """Зафиксировать стабильную геометрию колонок журнала."""
        header = self.logs_table.horizontalHeader()
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.logs_table.setColumnWidth(0, 160)
        self.logs_table.setColumnWidth(1, 100)
        self.logs_table.setColumnWidth(2, 160)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Клик по ячейке."""
        if row >= 0 and col >= 0:
            # Показать детали
            time_item = self.logs_table.item(row, 0)
            level_item = self.logs_table.item(row, 1)
            source_item = self.logs_table.item(row, 2)
            message_item = self.logs_table.item(row, 3)
            data_item = self.logs_table.item(row, 4)

            if all([time_item, level_item, source_item, message_item]):
                details = (
                    f"Время: {time_item.text()}\n"
                    f"Уровень: {level_item.text()}\n"
                    f"Источник: {source_item.text()}\n\n"
                    f"Сообщение:\n{message_item.text()}\n\n"
                    f"Данные:\n{data_item.text() if data_item else '{}'}"
                )
                self.details_text.setText(details)

    def _on_filter(self) -> None:
        """Применить фильтры."""
        self._load_logs()

    def _on_refresh(self) -> None:
        """Обновить журнал."""
        self._load_logs()

    def _on_export(self) -> None:
        """Экспортировать журнал."""
        if not self.log_store:
            QMessageBox.warning(
                self, "Предупреждение",
                "Журнал не загружен"
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт журнала",
            "",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if file_path:
            path = Path(file_path)
            fmt = "json" if path.suffix == ".json" else "csv"

            try:
                self.log_store.export_operations(path, fmt=fmt)
                QMessageBox.information(
                    self, "Экспорт",
                    f"Журнал экспортирован в {path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка экспорта",
                    str(e)
                )

    def _on_open_log(self) -> None:
        """Открыть файл лога."""
        if not self.logs_path:
            return

        log_file = self.logs_path / "app.log"
        if log_file.exists():
            import os
            os.startfile(str(log_file))
        else:
            QMessageBox.warning(
                self, "Предупреждение",
                "Файл лога не найден"
            )
