# src/fire_es_desktop/ui/pages/project_page.py
"""
ProjectPage — страница управления проектом (Workspace).

Согласно spec_second.md раздел 11.1:
- Создать/открыть Workspace
- Сводка о состоянии
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QFrame, QGroupBox, QGridLayout
)
from PySide6.QtCore import Qt
from pathlib import Path

from ...viewmodels import ProjectViewModel


class ProjectPage(QWidget):
    """Страница проекта."""

    def __init__(self, project_vm: ProjectViewModel):
        super().__init__()
        self.project_vm = project_vm

        self._init_ui()
        self._connect_signals()
        self._update_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Заголовок
        title = QLabel("Проект (Workspace)")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)

        # Кнопки управления
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.create_btn = QPushButton("Создать Workspace")
        self.create_btn.setFixedHeight(40)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        buttons_layout.addWidget(self.create_btn)

        self.open_btn = QPushButton("Открыть Workspace")
        self.open_btn.setFixedHeight(40)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2085d9;
            }
        """)
        buttons_layout.addWidget(self.open_btn)

        self.close_btn = QPushButton("Закрыть Workspace")
        self.close_btn.setFixedHeight(40)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da3c30;
            }
        """)
        buttons_layout.addWidget(self.close_btn)

        layout.addLayout(buttons_layout)

        # Информация о Workspace
        self.workspace_group = QGroupBox("Информация о Workspace")
        ws_layout = QGridLayout(self.workspace_group)
        ws_layout.setSpacing(10)

        self.ws_path_label = QLabel("Путь:")
        ws_layout.addWidget(self.ws_path_label, 0, 0)
        self.ws_path_value = QLabel("")
        ws_layout.addWidget(self.ws_path_value, 0, 1)

        self.ws_db_label = QLabel("База данных:")
        ws_layout.addWidget(self.ws_db_label, 1, 0)
        self.ws_db_value = QLabel("")
        ws_layout.addWidget(self.ws_db_value, 1, 1)

        self.ws_valid_label = QLabel("Статус:")
        ws_layout.addWidget(self.ws_valid_label, 2, 0)
        self.ws_valid_value = QLabel("")
        ws_layout.addWidget(self.ws_valid_value, 2, 1)

        layout.addWidget(self.workspace_group)

        # Статистика
        self.stats_group = QGroupBox("Статистика проекта")
        stats_layout = QGridLayout(self.stats_group)
        stats_layout.setSpacing(15)

        self.fires_count_label = QLabel("Пожаров: 0")
        self.fires_count_label.setStyleSheet("font-size: 16px;")
        stats_layout.addWidget(self.fires_count_label, 0, 0)

        self.decisions_count_label = QLabel("Решений ЛПР: 0")
        self.decisions_count_label.setStyleSheet("font-size: 16px;")
        stats_layout.addWidget(self.decisions_count_label, 0, 1)

        self.models_count_label = QLabel("Моделей: 0")
        self.models_count_label.setStyleSheet("font-size: 16px;")
        stats_layout.addWidget(self.models_count_label, 1, 0)

        self.active_model_label = QLabel("Активная модель: Нет")
        self.active_model_label.setStyleSheet("font-size: 16px;")
        stats_layout.addWidget(self.active_model_label, 1, 1)

        layout.addWidget(self.stats_group)

        layout.addStretch()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.create_btn.clicked.connect(self._on_create)
        self.open_btn.clicked.connect(self._on_open)
        self.close_btn.clicked.connect(self._on_close)

    def _connect_vm_signals(self) -> None:
        """Подключить сигналы VM."""
        # Уже подключены в MainWindow

    def _update_ui(self) -> None:
        """Обновить UI."""
        is_open = self.project_vm.is_workspace_open

        self.create_btn.setEnabled(not is_open)
        self.open_btn.setEnabled(not is_open)
        self.close_btn.setEnabled(is_open)
        self.workspace_group.setEnabled(is_open)
        self.stats_group.setEnabled(is_open)

        if is_open:
            path = self.project_vm.get_workspace_path()
            db_path = self.project_vm.get_db_path()
            valid, msg = self.project_vm.validate_workspace()

            self.ws_path_value.setText(str(path) if path else "")
            self.ws_db_value.setText(str(db_path) if db_path else "")
            self.ws_valid_value.setText(
                "✓ Валиден" if valid else f"✗ {msg}"
            )
            self.ws_valid_value.setStyleSheet("color: white;")

            # Статистика
            stats = self.project_vm.get_stats()
            self.fires_count_label.setText(f"Пожаров: {stats.get('fires_count', 0)}")
            self.decisions_count_label.setText(
                f"Решений ЛПР: {stats.get('lpr_decisions_count', 0)}"
            )
            self.models_count_label.setText(f"Моделей: {stats.get('models_count', 0)}")

            model_info = self.project_vm.get_active_model_info()
            if model_info:
                self.active_model_label.setText(
                    f"Активная модель: {model_info.get('name', 'Нет')}"
                )
            else:
                self.active_model_label.setText("Активная модель: Нет")
        else:
            self.ws_path_value.setText("")
            self.ws_db_value.setText("")
            self.ws_valid_value.setText("")
            self.fires_count_label.setText("Пожаров: 0")
            self.decisions_count_label.setText("Решений ЛПР: 0")
            self.models_count_label.setText("Моделей: 0")
            self.active_model_label.setText("Активная модель: Нет")

    def _on_create(self) -> None:
        """Создать Workspace."""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)

        if dialog.exec():
            path = Path(dialog.selectedFiles()[0])
            # Выбрать подпапку или создать новую
            workspace_path = path / "fire_es_workspace"

            success = self.project_vm.create_workspace(workspace_path)
            if success:
                self._update_ui()

    def _on_open(self) -> None:
        """Открыть Workspace."""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)

        if dialog.exec():
            path = Path(dialog.selectedFiles()[0])
            success = self.project_vm.open_workspace(path)
            if success:
                self._update_ui()

    def _on_close(self) -> None:
        """Закрыть Workspace."""
        self.project_vm.close_workspace()
        self._update_ui()
