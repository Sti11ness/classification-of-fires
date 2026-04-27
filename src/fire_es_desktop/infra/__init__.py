# src/fire_es_desktop/infra/__init__.py
"""
Infrastructure — инфраструктурный слой.

Согласно spec_second.md раздел 8.1:
- Доступ к SQLite
- Файловая система
- Артефакты
- Реестр моделей
- Логирование
"""

from .db_repository import DbRepository
from .artifact_store import ArtifactStore
from .model_registry import ModelRegistry
from .log_store import LogStore, LogLevel, LogEntry
from .training_data_store import TrainingDataStore

__all__ = [
    "DbRepository",
    "ArtifactStore",
    "ModelRegistry",
    "LogStore",
    "LogLevel",
    "LogEntry",
    "TrainingDataStore",
]
