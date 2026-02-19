# src/fire_es_desktop/ui/main_window.py
"""
Main Window — главное окно приложения Fire ES Desktop.

Согласно spec_second.md раздел 9:
- Левая навигация
- Центральная рабочая область
- Правая контекст-панель
- Нижняя строка состояния
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QListWidget, QListWidgetItem, QLabel, QFrame, QSplitter,
    QStatusBar, QProgressBar, QPushButton, QMessageBox, QFileDialog,
    QComboBox
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QSignalBlocker
from PySide6.QtGui import QFont, QIcon

from ..viewmodels import ProjectViewModel
from .pages import (
    ProjectPage,
    ImportPage,
    TrainingPage,
    LPRPredictPage,
    ModelsPage,
    LogPage,
    BatchPredictPage
)

logger = logging.getLogger("MainWindow")


class NavigationList(QListWidget):
    """Список навигации."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(200)
        self.setSpacing(2)
        self.setStyleSheet("""
            QListWidget {
                background-color: #5a5a5a;
                border: 1px solid black;
                color: white;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 4px;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #7a7a7a;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #6a6a6a;
            }
        """)


class ContextPanel(QFrame):
    """Правая контекст-панель."""
    role_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(250)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #5a5a5a;
                border-left: 1px solid black;
                color: white;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Контекст")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # Workspace
        layout.addWidget(QLabel("Workspace:"))
        self.workspace_label = QLabel("Не открыт")
        self.workspace_label.setWordWrap(True)
        layout.addWidget(self.workspace_label)

        # Активная модель
        layout.addWidget(QLabel("Активная модель:"))
        self.model_label = QLabel("Нет")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        # Роль
        layout.addWidget(QLabel("Роль:"))
        self.role_combo = QComboBox()
        self.role_combo.addItem("Аналитик", "analyst")
        self.role_combo.addItem("ЛПР", "lpr")
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        layout.addWidget(self.role_combo)

        # Статистика
        layout.addWidget(QLabel("Статистика:"))
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def update_context(self, workspace_path: Optional[Path],
                       model_info: Optional[Dict], role: str,
                       stats: Optional[Dict]) -> None:
        """Обновить контекст-панель."""
        if workspace_path:
            self.workspace_label.setText(str(workspace_path))
        else:
            self.workspace_label.setText("Не открыт")

        if model_info:
            self.model_label.setText(model_info.get("name", "Нет"))
        else:
            self.model_label.setText("Нет")

        self.set_role(role)

        if stats:
            stats_text = (
                f"Пожаров: {stats.get('fires_count', 0)}\n"
                f"Решений ЛПР: {stats.get('lpr_decisions_count', 0)}\n"
                f"Моделей: {stats.get('models_count', 0)}"
            )
            self.stats_label.setText(stats_text)
        else:
            self.stats_label.setText("")

    def set_role(self, role: str) -> None:
        """Установить роль в комбобоксе без повторного сигнала."""
        role_index = self.role_combo.findData(role)
        if role_index < 0:
            return
        with QSignalBlocker(self.role_combo):
            self.role_combo.setCurrentIndex(role_index)

    def _on_role_changed(self) -> None:
        """Прокинуть изменение роли наружу."""
        role = self.role_combo.currentData()
        if role:
            self.role_changed.emit(role)


class StatusBar(QStatusBar):
    """Нижняя строка состояния."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(30)

        # Индикатор задачи
        self.task_label = QLabel("")
        self.addWidget(self.task_label, 1)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.addPermanentWidget(self.progress_bar)

        # Кнопка журнала
        self.log_button = QPushButton("Журнал")
        self.log_button.setFixedWidth(80)
        self.addPermanentWidget(self.log_button)

    def set_task_status(self, message: str) -> None:
        """Установить статус задачи."""
        self.task_label.setText(message)

    def set_progress(self, value: int, visible: bool = True) -> None:
        """Установить прогресс."""
        self.progress_bar.setVisible(visible)
        if visible:
            self.progress_bar.setValue(value)

    def clear_task(self) -> None:
        """Очистить статус задачи."""
        self.task_label.setText("")
        self.set_progress(0, False)


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self, role: str = "analyst"):
        super().__init__()
        self.role = role  # "analyst" или "lpr"
        self._nav_page_indices: list[int] = []
        self.project_vm = ProjectViewModel()

        self.setWindowTitle("Fire ES — Экспертная система классификации пожаров")
        self.setMinimumSize(1200, 800)

        self._init_ui()
        self._connect_signals()
        self._update_navigation_for_role()

        logger.info(f"MainWindow initialized, role: {self.role}")

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Левая навигация
        self.navigation = NavigationList()
        main_layout.addWidget(self.navigation)

        # Центральная область
        self.pages_stack = QStackedWidget()
        main_layout.addWidget(self.pages_stack, 1)

        # Правая контекст-панель
        self.context_panel = ContextPanel()
        main_layout.addWidget(self.context_panel)

        # Статус бар
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        # Создать страницы
        self._create_pages()

    def _create_pages(self) -> None:
        """Создать страницы."""
        # Страница проекта
        self.project_page = ProjectPage(self.project_vm)
        self.pages_stack.addWidget(self.project_page)

        # Страница импорта
        self.import_page = ImportPage()
        self.pages_stack.addWidget(self.import_page)

        # Страница обучения
        self.training_page = TrainingPage()
        self.pages_stack.addWidget(self.training_page)

        # Страница пакетного прогноза
        self.batch_predict_page = BatchPredictPage()
        self.pages_stack.addWidget(self.batch_predict_page)

        # Страница прогноза ЛПР
        self.lpr_page = LPRPredictPage()
        self.pages_stack.addWidget(self.lpr_page)

        # Страница моделей
        self.models_page = ModelsPage()
        self.pages_stack.addWidget(self.models_page)

        # Страница журнала
        self.log_page = LogPage()
        self.pages_stack.addWidget(self.log_page)

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        # Навигация
        self.navigation.currentRowChanged.connect(self._on_navigation_changed)
        self.context_panel.role_changed.connect(self._on_role_changed)

        # Project VM
        self.project_vm.on_workspace_changed = self._on_workspace_changed
        self.project_vm.on_error = self._on_error

        # Status bar log button
        self.status_bar.log_button.clicked.connect(self._show_log_page)

    def _on_workspace_changed(self, workspace_path: Optional[Path]) -> None:
        """Обработчик изменения Workspace."""
        if workspace_path:
            # Обновить контекст-панель
            stats = self.project_vm.get_stats()
            model_info = self.project_vm.get_active_model_info()
            self.context_panel.update_context(
                workspace_path=workspace_path,
                model_info=model_info,
                role=self.role,
                stats=stats
            )

            # Обновить страницы
            self.import_page.set_db_path(self.project_vm.get_db_path())
            self.training_page.set_paths(
                db_path=self.project_vm.get_db_path(),
                models_path=self.project_vm.get_reports_path() / "models"
                if self.project_vm.get_reports_path() else None
            )
            self.batch_predict_page.set_paths(
                db_path=self.project_vm.get_db_path(),
                models_path=self.project_vm.get_reports_path() / "models"
                if self.project_vm.get_reports_path() else None,
                reports_path=self.project_vm.get_reports_path()
                if self.project_vm.get_reports_path() else None
            )
            self.lpr_page.set_paths(
                db_path=self.project_vm.get_db_path(),
                models_path=self.project_vm.get_reports_path() / "models"
                if self.project_vm.get_reports_path() else None
            )
            self.models_page.set_models_path(
                self.project_vm.get_reports_path() / "models"
                if self.project_vm.get_reports_path() else None
            )
            self.log_page.set_logs_path(self.project_vm.get_logs_path())

    def _on_error(self, message: str) -> None:
        """Обработчик ошибки."""
        QMessageBox.critical(self, "Ошибка", message)

    def _update_navigation_for_role(self) -> None:
        """Обновить навигацию для роли."""
        self.navigation.clear()

        if self.role == "analyst":
            items = [
                ("Проект", 0),
                ("Импорт данных", 1),
                ("Обучение модели", 2),
                ("Пакетный прогноз", 3),
                ("Прогноз (ЛПР)", 4),
                ("Модели", 5),
                ("Журнал", 6)
            ]
        else:  # lpr
            items = [
                ("Прогноз (ЛПР)", 4),
                ("Журнал", 6)
            ]

        self._nav_page_indices = [page_index for _, page_index in items]

        for text, page_index in items:
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(200, 40))
            item.setData(Qt.UserRole, page_index)
            self.navigation.addItem(item)

        self.context_panel.set_role(self.role)
        self.navigation.setCurrentRow(0)

    def _on_navigation_changed(self, row: int) -> None:
        """Переключить страницу по карте навигации."""
        if row < 0 or row >= len(self._nav_page_indices):
            return
        self.pages_stack.setCurrentIndex(self._nav_page_indices[row])

    def _on_role_changed(self, role: str) -> None:
        """Переключить режим приложения без перезапуска."""
        if role == self.role:
            return
        self.role = role
        self._update_navigation_for_role()
        self.context_panel.update_context(
            workspace_path=self.project_vm.get_workspace_path(),
            model_info=self.project_vm.get_active_model_info(),
            role=self.role,
            stats=self.project_vm.get_stats()
        )
        self.status_bar.set_task_status(f"Режим: {self.role}")
        logger.info(f"Role switched to: {self.role}")

    def _show_log_page(self) -> None:
        """Показать страницу журнала."""
        log_page_index = 6
        if log_page_index not in self._nav_page_indices:
            QMessageBox.information(self, "Журнал", "Журнал недоступен в этом режиме")
            return
        nav_row = self._nav_page_indices.index(log_page_index)
        self.navigation.setCurrentRow(nav_row)

    def closeEvent(self, event) -> None:
        """Обработчик закрытия окна."""
        # Закрыть Workspace
        self.project_vm.close_workspace()
        event.accept()
