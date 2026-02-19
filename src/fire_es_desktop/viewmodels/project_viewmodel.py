# src/fire_es_desktop/viewmodels/project_viewmodel.py
"""
ProjectViewModel — ViewModel для управления проектом (Workspace).

Согласно spec_second.md раздел 11.1:
- Создание/открытие Workspace
- Сводка о состоянии проекта
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from ..workspace.workspace_manager import WorkspaceManager
from ..infra import DbRepository, ModelRegistry, LogStore

logger = logging.getLogger("ProjectViewModel")


class ProjectViewModel:
    """
    ViewModel для экрана проекта.

    Управляет Workspace и предоставляет информацию о проекте.
    """

    def __init__(self):
        """Инициализировать ViewModel."""
        self.workspace_manager = WorkspaceManager()
        self.db_repo: Optional[DbRepository] = None
        self.model_registry: Optional[ModelRegistry] = None
        self.log_store: Optional[LogStore] = None

        # Callbacks
        self.on_workspace_changed: Optional[Callable[[Optional[Path]], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Состояние
        self.is_workspace_open = False
        self.workspace_path: Optional[Path] = None
        self.project_stats: Dict[str, Any] = {}

    def create_workspace(self, path: Path) -> bool:
        """
        Создать новый Workspace.

        Args:
            path: Путь к папке Workspace.

        Returns:
            True если успешно.
        """
        try:
            success = self.workspace_manager.create_workspace(path)
            if success:
                self._open_workspace_internal(path)
                logger.info(f"Created workspace: {path}")
                return True
            else:
                if self.on_error:
                    self.on_error("Не удалось создать Workspace")
                return False
        except Exception as e:
            logger.error(f"Create workspace failed: {e}", exc_info=True)
            if self.on_error:
                self.on_error(f"Ошибка: {str(e)}")
            return False

    def open_workspace(self, path: Path) -> bool:
        """
        Открыть существующий Workspace.

        Args:
            path: Путь к папке Workspace.

        Returns:
            True если успешно.
        """
        try:
            success = self.workspace_manager.open_workspace(path)
            if success:
                self._open_workspace_internal(path)
                logger.info(f"Opened workspace: {path}")
                return True
            else:
                if self.on_error:
                    self.on_error("Не удалось открыть Workspace")
                return False
        except Exception as e:
            logger.error(f"Open workspace failed: {e}", exc_info=True)
            if self.on_error:
                self.on_error(f"Ошибка: {str(e)}")
            return False

    def _open_workspace_internal(self, path: Path) -> None:
        """
        Внутренний метод открытия Workspace.

        Args:
            path: Путь к Workspace.
        """
        self.workspace_path = path
        self.is_workspace_open = True

        # Инициализировать репозитории
        db_path = self.workspace_manager.get_db_path()
        if db_path:
            self.db_repo = DbRepository(db_path)

        reports_path = self.workspace_manager.get_reports_path()
        if reports_path:
            models_path = reports_path / "models"
            self.model_registry = ModelRegistry(models_path)

        logs_path = self.workspace_manager.get_logs_path()
        if logs_path:
            self.log_store = LogStore(logs_path)

        # Обновить статистику
        self.refresh_stats()

        # Callback
        if self.on_workspace_changed:
            self.on_workspace_changed(path)

    def close_workspace(self) -> None:
        """Закрыть текущий Workspace."""
        if self.db_repo:
            self.db_repo.close()

        self.workspace_path = None
        self.is_workspace_open = False
        self.db_repo = None
        self.model_registry = None
        self.log_store = None
        self.project_stats = {}

        if self.on_workspace_changed:
            self.on_workspace_changed(None)

        logger.info("Workspace closed")

    def refresh_stats(self) -> None:
        """Обновить статистику проекта."""
        if not self.is_workspace_open or not self.db_repo:
            return

        try:
            stats = self.db_repo.get_stats()
            self.project_stats = stats
        except Exception as e:
            logger.error(f"Refresh stats failed: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику проекта.

        Returns:
            Словарь со статистикой.
        """
        return self.project_stats.copy()

    def get_workspace_path(self) -> Optional[Path]:
        """Получить путь к Workspace."""
        return self.workspace_path

    def get_db_path(self) -> Optional[Path]:
        """Получить путь к БД."""
        if self.workspace_manager:
            return self.workspace_manager.get_db_path()
        return None

    def get_reports_path(self) -> Optional[Path]:
        """Получить путь к reports."""
        if self.workspace_manager:
            return self.workspace_manager.get_reports_path()
        return None

    def get_logs_path(self) -> Optional[Path]:
        """Получить путь к logs."""
        if self.workspace_manager:
            return self.workspace_manager.get_logs_path()
        return None

    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """Получить информацию об активной модели."""
        if not self.model_registry:
            return None
        return self.model_registry.get_active_model_info()

    def validate_workspace(self) -> tuple[bool, str]:
        """
        Проверить целостность Workspace.

        Returns:
            (валидно, сообщение).
        """
        if not self.workspace_manager:
            return False, "Workspace не открыт"

        return self.workspace_manager.validate_workspace()
