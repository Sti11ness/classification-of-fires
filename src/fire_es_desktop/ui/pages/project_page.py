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
from ..theme import (
    configure_grid_layout,
    create_page_header,
    create_static_page,
    create_status_label,
    style_button,
    style_label,
)


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
        layout = create_static_page(self)

        # Заголовок
        layout.addWidget(
            create_page_header(
                "Рабочее пространство",
                "Создание, открытие и контроль локальной рабочей папки с базой, моделями и журналом.",
            )
        )

        # Кнопки управления
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.create_btn = QPushButton("Создать рабочее пространство")
        style_button(self.create_btn, "success")
        buttons_layout.addWidget(self.create_btn)

        self.open_btn = QPushButton("Открыть рабочее пространство")
        style_button(self.open_btn, "primary")
        buttons_layout.addWidget(self.open_btn)

        self.close_btn = QPushButton("Закрыть рабочее пространство")
        style_button(self.close_btn, "danger")
        buttons_layout.addWidget(self.close_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Информация о Workspace
        self.workspace_group = QGroupBox("Сведения о рабочем пространстве")
        ws_layout = QGridLayout(self.workspace_group)
        configure_grid_layout(ws_layout)

        self.ws_path_label = QLabel("Путь:")
        style_label(self.ws_path_label, "metric", word_wrap=False)
        ws_layout.addWidget(self.ws_path_label, 0, 0)
        self.ws_path_value = QLabel("")
        style_label(self.ws_path_value, "value", word_wrap=True)
        ws_layout.addWidget(self.ws_path_value, 0, 1)

        self.ws_db_label = QLabel("База данных:")
        style_label(self.ws_db_label, "metric", word_wrap=False)
        ws_layout.addWidget(self.ws_db_label, 1, 0)
        self.ws_db_value = QLabel("")
        style_label(self.ws_db_value, "value", word_wrap=True)
        ws_layout.addWidget(self.ws_db_value, 1, 1)

        self.ws_valid_label = QLabel("Статус:")
        style_label(self.ws_valid_label, "metric", word_wrap=False)
        ws_layout.addWidget(self.ws_valid_label, 2, 0)
        self.ws_valid_value = QLabel("")
        style_label(self.ws_valid_value, "value", word_wrap=True)
        ws_layout.addWidget(self.ws_valid_value, 2, 1)

        layout.addWidget(self.workspace_group)

        # Статистика
        self.stats_group = QGroupBox("Статистика проекта")
        stats_layout = QGridLayout(self.stats_group)
        configure_grid_layout(stats_layout)

        self.fires_count_label = QLabel("Исторических записей: 0")
        style_label(self.fires_count_label, "metric", word_wrap=False)
        stats_layout.addWidget(self.fires_count_label, 0, 0)

        self.decisions_count_label = QLabel("Решений ЛПР: 0")
        style_label(self.decisions_count_label, "metric", word_wrap=False)
        stats_layout.addWidget(self.decisions_count_label, 0, 1)

        self.models_count_label = QLabel("Моделей: 0")
        style_label(self.models_count_label, "metric", word_wrap=False)
        stats_layout.addWidget(self.models_count_label, 1, 0)

        self.active_model_label = QLabel("Активная модель в реестре: нет")
        style_label(self.active_model_label, "value", word_wrap=True)
        stats_layout.addWidget(self.active_model_label, 1, 1)

        self.working_model_label = QLabel("Рабочая модель для прогноза ЛПР: не выбрана")
        style_label(self.working_model_label, "value", word_wrap=True)
        stats_layout.addWidget(self.working_model_label, 2, 0, 1, 2)

        layout.addWidget(self.stats_group)

        self.status_label = create_status_label()
        self.status_label.hide()
        layout.addWidget(self.status_label)

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
            style_label(self.ws_valid_value, "ok" if valid else "problem", word_wrap=True)
            self.status_label.setText("Рабочее пространство подключено и готово к работе.")
            self.status_label.show()

            # Статистика
            stats = self.project_vm.get_stats()
            self.fires_count_label.setText(f"Исторических записей: {stats.get('fires_count', 0)}")
            self.decisions_count_label.setText(
                f"Решений ЛПР: {stats.get('lpr_decisions_count', 0)}"
            )
            self.models_count_label.setText(f"Моделей: {stats.get('models_count', 0)}")

            self.active_model_label.setText(
                "Активная модель в реестре: "
                f"{stats.get('active_registry_model_name') or 'нет'}"
            )
            self.working_model_label.setText(
                "Рабочая модель для прогноза ЛПР: "
                f"{stats.get('working_model_name') or 'не выбрана'}"
            )
        else:
            self.ws_path_value.setText("")
            self.ws_db_value.setText("")
            self.ws_valid_value.setText("")
            self.fires_count_label.setText("Исторических записей: 0")
            self.decisions_count_label.setText("Решений ЛПР: 0")
            self.models_count_label.setText("Моделей: 0")
            self.active_model_label.setText("Активная модель в реестре: нет")
            self.working_model_label.setText("Рабочая модель для прогноза ЛПР: не выбрана")
            self.status_label.setText("Откройте существующее рабочее пространство или создайте новое.")
            self.status_label.show()

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
