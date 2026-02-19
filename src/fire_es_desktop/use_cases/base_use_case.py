# src/fire_es_desktop/use_cases/base_use_case.py
"""
BaseUseCase — базовый класс для всех сценариев использования.

Согласно spec_second.md раздел 8.2:
- Application слой (Use Cases)
- Сценарии: импорт, очистка, локализация, разметка, обучение, прогноз
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from enum import Enum


class UseCaseStatus(Enum):
    """Статус выполнения Use Case."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UseCaseResult:
    """Результат выполнения Use Case."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    warnings: list = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class BaseUseCase(ABC):
    """
    Базовый класс Use Case.

    Все сценарии использования наследуются от этого класса.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = UseCaseStatus.PENDING
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
        self._cancel_requested = False

    def set_progress_callback(
        self,
        callback: Callable[[int, int, str], None]
    ) -> None:
        """
        Установить callback для обновления прогресса.

        Args:
            callback: Функция(current, total, description).
        """
        self._progress_callback = callback

    def report_progress(
        self,
        current: int,
        total: int,
        description: str = ""
    ) -> None:
        """
        Сообщить о прогрессе.

        Args:
            current: Текущий шаг.
            total: Всего шагов.
            description: Описание шага.
        """
        if self._progress_callback:
            self._progress_callback(current, total, description)

    def cancel(self) -> None:
        """Запросить отмену выполнения."""
        self._cancel_requested = True

    def check_cancelled(self) -> None:
        """Проверить отмену и выбросить исключение."""
        if self._cancel_requested:
            self.status = UseCaseStatus.CANCELLED
            raise UseCaseCancelledError(f"UseCase {self.name} was cancelled")

    @abstractmethod
    def execute(self, **kwargs) -> UseCaseResult:
        """
        Выполнить сценарий.

        Args:
            **kwargs: Параметры сценария.

        Returns:
            Результат выполнения.
        """
        pass


class UseCaseCancelledError(Exception):
    """Исключение отмены Use Case."""
    pass
