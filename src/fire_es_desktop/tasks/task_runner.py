# src/fire_es_desktop/tasks/task_runner.py
"""
TaskRunner — выполнение фоновых задач с прогрессом, логированием и отменой.

Согласно spec_second.md раздел 8.3:
- Любая операция > 200-300 мс выполняется как фоновая задача
- Прогресс по шагам
- Лог-стрим
- Кооперативная отмена
- Результат (успех/ошибка)
- Запись события в журнал
"""

import logging
import threading
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path


class TaskStatus(Enum):
    """Статус фоновой задачи."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Прогресс выполнения задачи."""
    current_step: int = 0
    total_steps: int = 1
    description: str = ""
    percent: float = 0.0

    def update(self, current: int, total: int, description: str = "") -> None:
        self.current_step = current
        self.total_steps = total
        self.percent = (current / total * 100) if total > 0 else 0
        if description:
            self.description = description


@dataclass
class TaskResult:
    """Результат выполнения задачи."""
    task_id: str
    task_name: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    data: Any = None
    log_messages: List[str] = field(default_factory=list)


class Task(ABC):
    """Базовый класс фоновой задачи."""

    def __init__(self, task_id: str, name: str):
        self.task_id = task_id
        self.name = name
        self.status = TaskStatus.PENDING
        self.progress = TaskProgress()
        self._cancel_requested = False
        self._logger = logging.getLogger(f"Task.{task_id}")
        self.result: Optional[TaskResult] = None

    @abstractmethod
    def execute(self, on_progress: Callable[[TaskProgress], None]) -> Any:
        """
        Выполнить задачу.

        Args:
            on_progress: Callback для обновления прогресса.

        Returns:
            Результат выполнения задачи.

        Raises:
            TaskCancelledError: Если запрошена отмена.
        """
        pass

    def cancel(self) -> None:
        """Запросить отмену задачи."""
        self._cancel_requested = True
        self._logger.info("Cancel requested")

    def check_cancelled(self) -> None:
        """Проверить, запрошена ли отмена."""
        if self._cancel_requested:
            self.status = TaskStatus.CANCELLED
            raise TaskCancelledError(f"Task {self.task_id} was cancelled")

    def log(self, message: str) -> None:
        """Добавить сообщение в лог задачи."""
        self._logger.info(message)


class TaskCancelledError(Exception):
    """Исключение отмены задачи."""
    pass


class TaskRunner:
    """
    Менеджер фоновых задач.

    Запускает задачи в отдельных потоках, отслеживает прогресс,
    обрабатывает отмену и результаты.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._tasks: Dict[str, Task] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._callbacks: Dict[str, List[Callable[[TaskResult], None]]] = {}
        self._progress_callbacks: Dict[str, List[Callable[[TaskProgress], None]]] = {}
        self._lock = threading.Lock()

        # Настройка логирования
        self._log_path = log_path
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Настроить логирование задач."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        if self._log_path:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(self._log_path / "tasks.log", encoding="utf-8"),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(level=logging.INFO, format=log_format)

    def register_task(self, task: Task) -> None:
        """
        Зарегистрировать задачу для выполнения.

        Args:
            task: Задача для регистрации.
        """
        with self._lock:
            self._tasks[task.task_id] = task
            self._callbacks[task.task_id] = []
            self._progress_callbacks[task.task_id] = []

    def on_task_complete(self, task_id: str, callback: Callable[[TaskResult], None]) -> None:
        """
        Зарегистрировать callback по завершении задачи.

        Args:
            task_id: ID задачи.
            callback: Callback функция.
        """
        with self._lock:
            if task_id in self._callbacks:
                self._callbacks[task_id].append(callback)

    def on_task_progress(self, task_id: str, callback: Callable[[TaskProgress], None]) -> None:
        """
        Зарегистрировать callback прогресса задачи.

        Args:
            task_id: ID задачи.
            callback: Callback функция.
        """
        with self._lock:
            if task_id in self._progress_callbacks:
                self._progress_callbacks[task_id].append(callback)

    def start_task(self, task_id: str) -> None:
        """
        Запустить задачу в фоновом потоке.

        Args:
            task_id: ID задачи для запуска.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not registered")

            task = self._tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                return

            task.status = TaskStatus.RUNNING
            thread = threading.Thread(
                target=self._run_task,
                args=(task_id,),
                daemon=True
            )
            self._threads[task_id] = thread
            thread.start()

    def _run_task(self, task_id: str) -> None:
        """
        Выполнить задачу в потоке.

        Args:
            task_id: ID задачи.
        """
        task = self._tasks[task_id]
        started_at = datetime.now()

        try:
            self._logger.info(f"Starting task {task_id}: {task.name}")

            def on_progress(progress: TaskProgress) -> None:
                task.progress = progress
                with self._lock:
                    for callback in self._progress_callbacks.get(task_id, []):
                        callback(progress)

            result_data = task.execute(on_progress)

            task.status = TaskStatus.COMPLETED
            task.result = TaskResult(
                task_id=task_id,
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(),
                data=result_data
            )

            self._logger.info(f"Task {task_id} completed successfully")

        except TaskCancelledError:
            task.status = TaskStatus.CANCELLED
            task.result = TaskResult(
                task_id=task_id,
                task_name=task.name,
                status=TaskStatus.CANCELLED,
                started_at=started_at,
                completed_at=datetime.now()
            )
            self._logger.info(f"Task {task_id} was cancelled")

        except Exception as e:
            task.status = TaskStatus.FAILED
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            task.result = TaskResult(
                task_id=task_id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(),
                error=error_msg
            )
            self._logger.error(f"Task {task_id} failed: {error_msg}")

        # Уведомить callbacks
        with self._lock:
            for callback in self._callbacks.get(task_id, []):
                try:
                    callback(task.result)
                except Exception as cb_error:
                    self._logger.error(f"Callback error for task {task_id}: {cb_error}")

    def cancel_task(self, task_id: str) -> None:
        """
        Отменить задачу.

        Args:
            task_id: ID задачи для отмены.
        """
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return

            task.cancel()

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Получить статус задачи.

        Args:
            task_id: ID задачи.

        Returns:
            Статус задачи или None.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return task.status if task else None

    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """
        Получить прогресс задачи.

        Args:
            task_id: ID задачи.

        Returns:
            Прогресс задачи или None.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return task.progress if task else None

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Получить результат задачи.

        Args:
            task_id: ID задачи.

        Returns:
            Результат задачи или None.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            return task.result if task else None

    def is_task_running(self, task_id: str) -> bool:
        """
        Проверить, выполняется ли задача.

        Args:
            task_id: ID задачи.

        Returns:
            True если задача выполняется.
        """
        status = self.get_task_status(task_id)
        return status == TaskStatus.RUNNING
