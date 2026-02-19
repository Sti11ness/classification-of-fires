# src/fire_es_desktop/tasks/__init__.py
"""
Tasks — фоновые задачи с прогрессом и отменой.

Согласно spec_second.md раздел 8.3:
- Любая операция > 200-300 мс выполняется как фоновая задача
- Прогресс по шагам
- Лог-стрим
- Кооперативная отмена
"""

from .task_runner import (
    Task,
    TaskRunner,
    TaskStatus,
    TaskProgress,
    TaskResult,
    TaskCancelledError,
)

__all__ = [
    "Task",
    "TaskRunner",
    "TaskStatus",
    "TaskProgress",
    "TaskResult",
    "TaskCancelledError",
]
