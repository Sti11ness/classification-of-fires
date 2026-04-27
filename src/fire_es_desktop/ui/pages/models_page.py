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

from ..theme import (
    configure_table,
    configure_text_panel,
    create_hint,
    create_page_header,
    create_static_page,
    style_button,
)


class ModelsPage(QWidget):
    """Страница моделей."""

    def __init__(self):
        super().__init__()
        self.model_registry = None
        self.models_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        layout = create_static_page(self)

        # Заголовок
        layout.addWidget(
            create_page_header(
                "Модели и версии",
                "Реестр обученных моделей, качество обучения и готовность к использованию на экране ЛПР.",
            )
        )

        # Список моделей
        models_group = QGroupBox("Зарегистрированные модели")
        models_layout = QVBoxLayout(models_group)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(7)
        self.models_table.setHorizontalHeaderLabels([
            "ID", "Название", "Тип", "Дата", "Метрика (F1)", "Для ЛПР", "Активна"
        ])
        configure_table(self.models_table, min_height=360, sortable=True)
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        self.models_table.setColumnWidth(1, 420)
        self.models_table.setColumnWidth(6, 84)
        models_layout.addWidget(self.models_table)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.refresh_btn = QPushButton("Обновить")
        style_button(self.refresh_btn, "ghost")
        buttons_layout.addWidget(self.refresh_btn)

        self.activate_btn = QPushButton("Сделать активной")
        style_button(self.activate_btn, "success")
        buttons_layout.addWidget(self.activate_btn)

        self.delete_btn = QPushButton("Удалить")
        style_button(self.delete_btn, "danger")
        buttons_layout.addWidget(self.delete_btn)

        self.open_folder_btn = QPushButton("Открыть папку")
        style_button(self.open_folder_btn, "ghost")
        buttons_layout.addWidget(self.open_folder_btn)

        buttons_layout.addStretch()

        models_layout.addLayout(buttons_layout)

        layout.addWidget(models_group)

        # Информация о модели
        info_group = QGroupBox("Информация о модели")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        configure_text_panel(self.info_text, min_height=260)
        info_layout.addWidget(self.info_text)
        info_layout.addWidget(
            create_hint(
                "Здесь показываются дата обучения, качество модели и объяснение, подходит ли она для экрана ЛПР."
            )
        )

        layout.addWidget(info_group)

        layout.addStretch()

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.activate_btn.clicked.connect(self._on_activate)
        self.delete_btn.clicked.connect(self._on_delete)
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

        models = self.model_registry.list_models()
        active_id = self.model_registry.get_active_model_id()

        self.models_table.setUpdatesEnabled(False)
        try:
            self.models_table.setSortingEnabled(False)
            self.models_table.setRowCount(0)

            for model in models:
                row = self.models_table.rowCount()
                self.models_table.insertRow(row)

                # ID
                item = SortableTableWidgetItem(model.get("model_id", ""))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.models_table.setItem(row, 0, item)

                # Название
                item = SortableTableWidgetItem(model.get("name", ""))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(model.get("name", ""))
                self.models_table.setItem(row, 1, item)

                # Тип
                item = SortableTableWidgetItem(model.get("model_type", ""))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.models_table.setItem(row, 2, item)

                # Дата
                created_at = self._format_created_at(model.get("created_at"))
                item = SortableTableWidgetItem(created_at)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.models_table.setItem(row, 3, item)

                # Метрика
                metrics = model.get("metrics", {})
                f1 = metrics.get("macro_f1", metrics.get("f1_macro", 0))
                item = SortableTableWidgetItem(f"{f1:.4f}")
                item.set_sort_key(float(f1))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.models_table.setItem(row, 4, item)

                # Совместимость с ЛПР
                compatible = self.model_registry.is_model_production_safe(model)
                item = SortableTableWidgetItem("Да" if compatible else "Нет")
                item.set_sort_key(1 if compatible else 0)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                self.models_table.setItem(row, 5, item)

                # Активна
                is_active = model.get("model_id") == active_id
                item = SortableTableWidgetItem("✓" if is_active else "")
                item.set_sort_key(1 if is_active else 0)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                self.models_table.setItem(row, 6, item)
        finally:
            self.models_table.setSortingEnabled(True)
            self.models_table.setUpdatesEnabled(True)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """Клик по ячейке."""
        if not self.model_registry:
            return

        model_id = self.models_table.item(row, 0).text()
        model_info = self.model_registry.get_model_info(model_id)

        if model_info:
            info_text = f"=== {model_info.get('name', '')} ===\n\n"
            info_text += f"ID: {model_info.get('model_id', '')}\n"
            info_text += f"Тип модели: {model_info.get('model_type', '')}\n"
            info_text += f"Дата обучения: {self._format_created_at(model_info.get('created_at'))}\n"
            info_text += f"Целевая колонка: {model_info.get('target', '')}\n"
            info_text += f"Набор признаков: {model_info.get('feature_set', '')}\n"
            info_text += f"Стадия данных: {self._format_stage_value(model_info.get('availability_stage', ''))}\n"
            info_text += f"Тип разметки: {model_info.get('semantic_target', '')}\n"
            info_text += f"Схема проверки: {self._format_split_value(model_info.get('split_protocol', ''))}\n"
            info_text += f"Версия нормативов: {model_info.get('normative_version', '—')}\n"
            info_text += f"Записей в обучении: {model_info.get('dataset_info', {}).get('samples', 0)}\n\n"

            info_text += "Качество модели:\n"
            for k, v in model_info.get("metrics", {}).items():
                if isinstance(v, float):
                    info_text += f"  {k}: {v:.4f}\n"
                else:
                    info_text += f"  {k}: {v}\n"

            reasons = self.model_registry.get_production_unsafe_reasons(model_info)
            compatibility = "Да" if not reasons else "Нет"
            info_text += f"\nСовместима с экраном ЛПР: {compatibility}\n"
            if reasons:
                info_text += "Почему нельзя сделать рабочей:\n"
                for reason in reasons:
                    info_text += f"  - {reason}\n"
            else:
                info_text += "Модель можно использовать на рабочем экране ЛПР.\n"

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
                model_info = self.model_registry.get_model_info(model_id)
                reasons = self.model_registry.get_production_unsafe_reasons(model_info or {})
                QMessageBox.critical(
                    self, "Ошибка",
                    "Не удалось сделать модель рабочей для ЛПР.\n\n"
                    + ("\n".join(f"• {reason}" for reason in reasons) if reasons else "Причина не определена.")
                )

    def _on_delete(self) -> None:
        """Удалить модель и связанные файлы."""
        selected_rows = self.models_table.selectedItems()
        if not selected_rows or not self.model_registry:
            QMessageBox.warning(self, "Предупреждение", "Выберите модель")
            return

        row = selected_rows[0].row()
        model_id = self.models_table.item(row, 0).text()
        model_name = self.models_table.item(row, 1).text()
        answer = QMessageBox.question(
            self,
            "Удаление модели",
            f"Удалить модель «{model_name}» и связанные файлы?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        if self.model_registry.delete_model(model_id):
            QMessageBox.information(self, "Удаление", "Модель удалена")
            self.info_text.clear()
            self._load_models()
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось удалить модель")

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

    @staticmethod
    def _format_created_at(value: Optional[str]) -> str:
        if not value:
            return "—"
        return str(value).replace("T", " ")[:19]

    @staticmethod
    def _format_stage_value(value: Optional[str]) -> str:
        mapping = {
            "dispatch_initial": "до прибытия подразделения",
            "arrival_update": "после прибытия подразделения",
            "first_hose_update": "после подачи первого ствола",
            "retrospective": "архивный режим",
        }
        return mapping.get(str(value), str(value))

    @staticmethod
    def _format_split_value(value: Optional[str]) -> str:
        mapping = {
            "group_shuffle": "групповое разделение без пересечения событий",
            "group_kfold": "групповая перекрестная проверка",
            "temporal_holdout": "разделение по времени",
            "row_random_legacy": "устаревшее случайное разделение",
            "": "—",
            None: "—",
        }
        return mapping.get(value, str(value))


class SortableTableWidgetItem(QTableWidgetItem):
    """Элемент таблицы с управляемым ключом сортировки."""

    def __init__(self, text: str):
        super().__init__(text)
        self._sort_key: Any = text

    def set_sort_key(self, value: Any) -> None:
        self._sort_key = value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTableWidgetItem):
            return self._sort_key < other._sort_key
        return super().__lt__(other)
