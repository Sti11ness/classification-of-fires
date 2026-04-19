# src/fire_es_desktop/ui/pages/models_page.py
"""
ModelsPage — страница управления моделями.

Согласно spec_second.md раздел 11.7:
- Список моделей
- Подробности (метрики, цель, признаки)
- Активация модели
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QTextEdit,
    QHeaderView, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt
from pathlib import Path
from typing import Optional, Dict, Any


class ModelsPage(QWidget):
    """Страница моделей."""

    def __init__(self):
        super().__init__()
        self.model_registry = None
        self.models_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Модели и версии")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)

        # Список моделей
        models_group = QGroupBox("Зарегистрированные модели")
        models_layout = QVBoxLayout(models_group)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(6)
        self.models_table.setHorizontalHeaderLabels([
            "ID", "Название", "Тип", "Цель", "Метрика (F1)", "Активна"
        ])
        self.models_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.models_table.setAlternatingRowColors(True)
        models_layout.addWidget(self.models_table)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.refresh_btn = QPushButton("Обновить")
        buttons_layout.addWidget(self.refresh_btn)

        self.activate_btn = QPushButton("Сделать активной")
        self.activate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
            }
        """)
        buttons_layout.addWidget(self.activate_btn)

        self.open_folder_btn = QPushButton("Открыть папку")
        buttons_layout.addWidget(self.open_folder_btn)

        buttons_layout.addStretch()

        models_layout.addLayout(buttons_layout)

        layout.addWidget(models_group)

        # Информация о модели
        info_group = QGroupBox("Информация о модели")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(200)
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)

        layout.addStretch()

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.activate_btn.clicked.connect(self._on_activate)
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        self.models_table.cellClicked.connect(self._on_cell_clicked)

    def set_models_path(self, models_path: Optional[Path]) -> None:
        """Установить путь к моделям."""
        self.models_path = models_path
        if models_path and models_path.exists():
            from ...infra import ModelRegistry
            self.model_registry = ModelRegistry(models_path)
            self._load_models()
        else:
            self.model_registry = None
            self.models_table.setRowCount(0)
            self.info_text.clear()

    def _load_models(self) -> None:
        """Загрузить список моделей."""
        if not self.model_registry:
            return

        self.models_table.setRowCount(0)
        models = self.model_registry.list_models()

        active_id = self.model_registry.get_active_model_id()

        for model in models:
            row = self.models_table.rowCount()
            self.models_table.insertRow(row)

            # ID
            item = QTableWidgetItem(model.get("model_id", ""))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.models_table.setItem(row, 0, item)

            # Название
            item = QTableWidgetItem(model.get("name", ""))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.models_table.setItem(row, 1, item)

            # Тип
            item = QTableWidgetItem(model.get("model_type", ""))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.models_table.setItem(row, 2, item)

            # Цель
            item = QTableWidgetItem(model.get("target", ""))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.models_table.setItem(row, 3, item)

            # Метрика
            metrics = model.get("metrics", {})
            f1 = metrics.get("macro_f1", metrics.get("f1_macro", 0))
            item = QTableWidgetItem(f"{f1:.4f}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.models_table.setItem(row, 4, item)

            # Активна
            is_active = model.get("model_id") == active_id
            item = QTableWidgetItem("✓" if is_active else "")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self.models_table.setItem(row, 5, item)

        self.models_table.resizeColumnsToContents()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Клик по ячейке."""
        if not self.model_registry:
            return

        model_id = self.models_table.item(row, 0).text()
        model_info = self.model_registry.get_model_info(model_id)

        if model_info:
            info_text = f"=== {model_info.get('name', '')} ===\n\n"
            info_text += f"ID: {model_info.get('model_id', '')}\n"
            info_text += f"Тип: {model_info.get('model_type', '')}\n"
            info_text += f"Цель: {model_info.get('target', '')}\n"
            info_text += f"Deployment role: {model_info.get('deployment_role', 'n/a')}\n"
            info_text += f"Offline only: {model_info.get('offline_only', False)}\n"
            info_text += f"Feature set: {model_info.get('feature_set', '')}\n"
            info_text += f"Создана: {model_info.get('created_at', '')}\n\n"

            info_text += "Метрики:\n"
            for k, v in model_info.get("metrics", {}).items():
                if isinstance(v, float):
                    info_text += f"  {k}: {v:.4f}\n"
                else:
                    info_text += f"  {k}: {v}\n"

            info_text += f"\nПризнаков: {len(model_info.get('features', []))}\n"
            info_text += f"Выборка: {model_info.get('dataset_info', {}).get('samples', 0)}\n"

            self.info_text.setText(info_text)

    def _on_refresh(self) -> None:
        """Обновить список."""
        self._load_models()

    def _on_activate(self) -> None:
        """Активировать модель."""
        selected_rows = self.models_table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(
                self, "Предупреждение",
                "Выберите модель"
            )
            return

        row = selected_rows[0].row()
        model_id = self.models_table.item(row, 0).text()

        if self.model_registry:
            success = self.model_registry.set_active_model(model_id)
            if success:
                QMessageBox.information(
                    self, "Активация",
                    f"Модель {model_id} активирована"
                )
                self._load_models()
            else:
                QMessageBox.critical(
                    self, "Ошибка",
                    "Не удалось активировать модель. Для rank_tz разрешены только production-safe модели."
                )

    def _on_open_folder(self) -> None:
        """Открыть папку моделей."""
        if self.models_path and self.models_path.exists():
            import os
            os.startfile(str(self.models_path))
        else:
            QMessageBox.warning(
                self, "Предупреждение",
                "Папка моделей не найдена"
            )
