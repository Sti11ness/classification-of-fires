# src/fire_es_desktop/ui/main_window.py
"""
Main Window — главное окно приложения Fire ES Desktop.

Согласно spec_second.md раздел 9:
- Левая навигация
- Центральная рабочая область
- Правая контекст-панель
- Нижняя строка состояния
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..viewmodels import ProjectViewModel
from .pages import (
    BatchPredictPage,
    DigitalTwinPage,
    ImportPage,
    LogPage,
    LPRDecisionHistoryPage,
    LPRPredictPage,
    ModelsPage,
    ProjectPage,
    TrainingPage,
)
from .theme import create_page_header, style_label

logger = logging.getLogger("MainWindow")

PAGE_PROJECT = "project"
PAGE_IMPORT = "import"
PAGE_TRAINING = "training"
PAGE_BATCH_PREDICT = "batch_predict"
PAGE_LPR_PREDICT = "lpr_predict"
PAGE_LPR_HISTORY = "lpr_history"
PAGE_MODELS = "models"
PAGE_DIGITAL_TWIN = "digital_twin"
PAGE_LOG = "log"


class NavigationList(QListWidget):
    """Список навигации."""

    def __init__(self):
        super().__init__()
        self.setObjectName("NavigationList")
        self.setMinimumWidth(190)
        self.setMaximumWidth(190)
        self.setSpacing(6)
        self.setFrameShape(QFrame.NoFrame)
        self.setUniformItemSizes(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)


class ContextPanel(QFrame):
    """Правая контекст-панель."""
    role_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("ContextPanel")
        self.setMinimumWidth(260)
        self.setMaximumWidth(300)
        self.setFrameStyle(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Заголовок
        layout.addWidget(create_page_header("Состояние", "Текущее рабочее пространство, выбранная модель и режим работы."))

        # Рабочее пространство
        workspace_title = QLabel("Рабочее пространство")
        style_label(workspace_title, "metric", word_wrap=False)
        layout.addWidget(workspace_title)
        self.workspace_label = QLabel("Не открыт")
        self.workspace_label.setWordWrap(True)
        style_label(self.workspace_label, "value", word_wrap=True)
        layout.addWidget(self.workspace_label)

        # Модель
        model_title = QLabel("Модели")
        style_label(model_title, "metric", word_wrap=False)
        layout.addWidget(model_title)
        self.model_label = QLabel("Активная модель в реестре: нет\nРабочая модель для прогноза: нет")
        self.model_label.setWordWrap(True)
        style_label(self.model_label, "value", word_wrap=True)
        layout.addWidget(self.model_label)

        # Роль
        role_title = QLabel("Роль")
        style_label(role_title, "metric", word_wrap=False)
        layout.addWidget(role_title)
        self.role_combo = QComboBox()
        self.role_combo.addItem("Аналитик", "analyst")
        self.role_combo.addItem("ЛПР", "lpr")
        self.role_combo.currentIndexChanged.connect(self._on_role_changed)
        layout.addWidget(self.role_combo)

        # Статистика
        stats_title = QLabel("Статистика")
        style_label(stats_title, "metric", word_wrap=False)
        layout.addWidget(stats_title)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        style_label(self.stats_label, "value", word_wrap=True)
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def update_context(self, workspace_path: Optional[Path],
                       model_info: Optional[dict], role: str,
                       stats: Optional[dict]) -> None:
        """Обновить контекст-панель."""
        if workspace_path:
            self.workspace_label.setText(str(workspace_path))
        else:
            self.workspace_label.setText("Не открыт")

        active_registry_model = stats.get("active_registry_model_name") if stats else None
        working_model = stats.get("working_model_name") if stats else None
        self.model_label.setText(
            "Активная модель в реестре: "
            f"{active_registry_model or 'нет'}\n"
            "Рабочая модель для прогноза: "
            f"{working_model or 'не выбрана'}"
        )

        self.set_role(role)

        if stats:
            stats_text = (
                f"Исторических записей: {stats.get('fires_count', 0)}\n"
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
        self.progress_bar.setFixedWidth(240)
        self.progress_bar.setVisible(False)
        self.addPermanentWidget(self.progress_bar)

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

    CONTEXT_PANEL_COLLAPSE_WIDTH = 1540

    def __init__(self, role: str = "analyst"):
        super().__init__()
        self.role = role  # "analyst" или "lpr"
        self._nav_page_keys: list[str] = []
        self.page_indices: dict[str, int] = {}
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

        self.shell_splitter = QSplitter(Qt.Horizontal)
        self.shell_splitter.setChildrenCollapsible(False)
        self.shell_splitter.setHandleWidth(0)

        # Левая навигация
        self.navigation = NavigationList()
        self.shell_splitter.addWidget(self.navigation)

        # Центральная область
        self.pages_stack = QStackedWidget()
        self.pages_stack.setObjectName("PagesStack")
        self.shell_splitter.addWidget(self.pages_stack)

        # Правая контекст-панель
        self.context_panel = ContextPanel()
        self.shell_splitter.addWidget(self.context_panel)
        self.shell_splitter.setStretchFactor(0, 0)
        self.shell_splitter.setStretchFactor(1, 1)
        self.shell_splitter.setStretchFactor(2, 0)
        self.shell_splitter.setSizes([190, 1030, 280])
        main_layout.addWidget(self.shell_splitter)

        # Статус бар
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        # Создать страницы
        self._create_pages()
        self._update_shell_layout()

    def _create_pages(self) -> None:
        """Создать страницы."""
        # Страница проекта
        self.project_page = ProjectPage(self.project_vm)
        self._register_page(PAGE_PROJECT, self.project_page)

        # Страница импорта
        self.import_page = ImportPage()
        self._register_page(PAGE_IMPORT, self.import_page)

        # Страница обучения
        self.training_page = TrainingPage()
        self._register_page(PAGE_TRAINING, self.training_page)

        # Страница пакетного прогноза
        self.batch_predict_page = BatchPredictPage()
        self._register_page(PAGE_BATCH_PREDICT, self.batch_predict_page)

        # Страница прогноза ЛПР
        self.lpr_page = LPRPredictPage()
        self._register_page(PAGE_LPR_PREDICT, self.lpr_page)

        # История решений ЛПР
        self.lpr_history_page = LPRDecisionHistoryPage()
        self._register_page(PAGE_LPR_HISTORY, self.lpr_history_page)

        # Страница моделей
        self.models_page = ModelsPage()
        self._register_page(PAGE_MODELS, self.models_page)

        # Цифровой двойник среды
        self.digital_twin_page = DigitalTwinPage()
        self._register_page(PAGE_DIGITAL_TWIN, self.digital_twin_page)

        # Страница журнала
        self.log_page = LogPage()
        self._register_page(PAGE_LOG, self.log_page)

    def _register_page(self, page_key: str, widget: QWidget) -> None:
        """Зарегистрировать страницу и сохранить ее индекс в stacked widget."""
        self.page_indices[page_key] = self.pages_stack.addWidget(widget)

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        # Навигация
        self.navigation.currentRowChanged.connect(self._on_navigation_changed)
        self.context_panel.role_changed.connect(self._on_role_changed)

        # Project VM
        self.project_vm.on_workspace_changed = self._on_workspace_changed
        self.project_vm.on_error = self._on_error

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
        else:
            self.context_panel.update_context(
                workspace_path=None,
                model_info=None,
                role=self.role,
                stats=None,
            )

        self._propagate_workspace_state()

    def _on_error(self, message: str) -> None:
        """Обработчик ошибки."""
        QMessageBox.critical(self, "Ошибка", message)

    def _update_navigation_for_role(self) -> None:
        """Обновить навигацию для роли."""
        self.navigation.clear()

        if self.role == "analyst":
            items = [
                ("Рабочее пространство", PAGE_PROJECT),
                ("Импорт данных", PAGE_IMPORT),
                ("Обучение модели", PAGE_TRAINING),
                ("Пакетный прогноз", PAGE_BATCH_PREDICT),
                ("Прогноз (ЛПР)", PAGE_LPR_PREDICT),
                ("История решений ЛПР", PAGE_LPR_HISTORY),
                ("Модели", PAGE_MODELS),
                ("Цифровой двойник", PAGE_DIGITAL_TWIN),
                ("Журнал", PAGE_LOG),
            ]
        else:  # lpr
            items = [
                ("Рабочее пространство", PAGE_PROJECT),
                ("Прогноз (ЛПР)", PAGE_LPR_PREDICT),
                ("История решений ЛПР", PAGE_LPR_HISTORY),
                ("Журнал", PAGE_LOG),
            ]

        self._nav_page_keys = [page_key for _, page_key in items]

        for text, page_key in items:
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(200, 40))
            item.setData(Qt.UserRole, page_key)
            self.navigation.addItem(item)

        self.context_panel.set_role(self.role)
        self.navigation.setCurrentRow(0)

    def _on_navigation_changed(self, row: int) -> None:
        """Переключить страницу по карте навигации."""
        if row < 0 or row >= len(self._nav_page_keys):
            return
        self.project_vm.refresh_stats()
        self.context_panel.update_context(
            workspace_path=self.project_vm.get_workspace_path(),
            model_info=self.project_vm.get_active_model_info(),
            role=self.role,
            stats=self.project_vm.get_stats()
        )
        page_key = self._nav_page_keys[row]
        if page_key == PAGE_PROJECT:
            self.project_page._update_ui()
        elif page_key == PAGE_TRAINING and self.project_vm.get_db_path():
            self.training_page._update_source_info(self.project_vm.get_db_path())
        self.pages_stack.setCurrentIndex(self.page_indices[page_key])

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._update_shell_layout()

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
        role_label = "ЛПР" if self.role == "lpr" else "Аналитик"
        self.status_bar.set_task_status(f"Режим: {role_label}")
        logger.info(f"Role switched to: {self.role}")

    def _propagate_workspace_state(self) -> None:
        """Передать текущие workspace-зависимости во все страницы."""
        db_path = self.project_vm.get_db_path()
        reports_path = self.project_vm.get_reports_path()
        models_path = reports_path / "models" if reports_path else None
        logs_path = self.project_vm.get_logs_path()

        self.import_page.set_db_path(db_path)
        self.training_page.set_paths(
            db_path=db_path,
            models_path=models_path,
        )
        self.batch_predict_page.set_paths(
            db_path=db_path,
            models_path=models_path,
            reports_path=reports_path,
        )
        self.lpr_page.set_paths(
            db_path=db_path,
            models_path=models_path,
        )
        self.lpr_history_page.set_db_path(db_path)
        self.models_page.set_models_path(models_path)
        self.digital_twin_page.set_paths(db_path, reports_path)
        self.log_page.set_logs_path(logs_path)

    def _update_shell_layout(self) -> None:
        """Подстраивать shell под узкое окно."""
        compact = self.width() < self.CONTEXT_PANEL_COLLAPSE_WIDTH
        self.context_panel.setVisible(not compact)
        if compact:
            self.shell_splitter.setSizes([190, max(780, self.width() - 210), 0])
        else:
            self.shell_splitter.setSizes([190, max(900, self.width() - 490), 280])

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Обработчик закрытия окна."""
        # Закрыть Workspace
        self.project_vm.close_workspace()
        event.accept()
