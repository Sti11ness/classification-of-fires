# src/fire_es_desktop/infra/log_store.py
"""
LogStore — хранилище логов и журнал операций.

Согласно spec_second.md раздел 13:
- Журналирование операций
- Трассируемость
- Логи приложения и задач
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Уровень лог-сообщения."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEntry:
    """Запись в журнале."""

    def __init__(
        self,
        timestamp: datetime,
        level: LogLevel,
        message: str,
        source: str = "",
        data: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.source = source
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "data": self.data
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LogEntry":
        return LogEntry(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=LogLevel(data["level"]),
            message=data["message"],
            source=data.get("source", ""),
            data=data.get("data", {})
        )


class LogStore:
    """
    Хранилище логов и журнал операций.

    Предоставляет:
    - Запись логов в файлы
    - Журнал операций для UI
    - Фильтрацию и экспорт
    """

    def __init__(self, logs_path: Path):
        """
        Инициализировать хранилище.

        Args:
            logs_path: Путь к папке logs/.
        """
        self.logs_path = logs_path
        self.app_log_path = logs_path / "app.log"
        self.tasks_log_path = logs_path / "tasks.log"
        self.operations_log_path = logs_path / "operations.jsonl"

        self._ensure_dirs()
        self._setup_logging()

        # Операции в памяти (для быстрого доступа UI)
        self._recent_operations: List[LogEntry] = []
        self._max_recent = 1000

    def _ensure_dirs(self) -> None:
        """Создать папку логов."""
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Настроить логирование."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        # App log
        app_handler = logging.FileHandler(self.app_log_path, encoding="utf-8")
        app_handler.setFormatter(logging.Formatter(log_format))

        # Tasks log
        tasks_handler = logging.FileHandler(self.tasks_log_path, encoding="utf-8")
        tasks_handler.setFormatter(logging.Formatter(log_format))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))

        # Настроить корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        self._add_handler_once(root_logger, app_handler)
        self._add_handler_once(root_logger, tasks_handler)
        self._add_handler_once(root_logger, console_handler)

    @staticmethod
    def _add_handler_once(root_logger: logging.Logger, handler: logging.Handler) -> None:
        """Добавить handler только если эквивалентный уже не подключен."""
        for existing in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and isinstance(existing, logging.FileHandler):
                if Path(existing.baseFilename).resolve() == Path(handler.baseFilename).resolve():
                    return

            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                if isinstance(existing, logging.StreamHandler) and not isinstance(existing, logging.FileHandler):
                    return

        root_logger.addHandler(handler)

    def log_operation(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        source: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """
        Записать операцию в журнал.

        Args:
            message: Сообщение.
            level: Уровень.
            source: Источник операции.
            data: Дополнительные данные.

        Returns:
            Запись журнала.
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
            data=data
        )

        # Записать в файл (JSONL формат)
        with open(self.operations_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

        # Добавить в память
        self._recent_operations.append(entry)
        if len(self._recent_operations) > self._max_recent:
            self._recent_operations.pop(0)

        # Записать в стандартный лог
        log_method = getattr(logging.getLogger("Operations"), level.value)
        log_method(f"[{source}] {message}")

        return entry

    def get_operations(
        self,
        limit: int = 100,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[LogEntry]:
        """
        Получить записи журнала.

        Args:
            limit: Ограничить количество.
            level: Фильтр по уровню.
            source: Фильтр по источнику.
            start_time: Начать с времени.
            end_time: Закончить временем.

        Returns:
            Список записей журнала.
        """
        result = []

        for entry in reversed(self._recent_operations):
            # Применить фильтры
            if level and entry.level != level:
                continue
            if source and entry.source != source:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            result.append(entry)

            if len(result) >= limit:
                break

        return result

    def export_operations(
        self,
        output_path: Path,
        fmt: str = "json"
    ) -> Path:
        """
        Экспортировать журнал.

        Args:
            output_path: Путь для экспорта.
            fmt: Формат (json, csv).

        Returns:
            Путь к файлу экспорта.
        """
        if fmt == "json":
            data = [entry.to_dict() for entry in self._recent_operations]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif fmt == "csv":
            import pandas as pd
            data = [entry.to_dict() for entry in self._recent_operations]
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        self.log_operation(
            f"Exported operations log to {output_path}",
            source="LogStore"
        )
        return output_path

    def get_error_count(self) -> int:
        """Получить количество ошибок."""
        return sum(
            1 for entry in self._recent_operations
            if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL)
        )

    def get_warning_count(self) -> int:
        """Получить количество предупреждений."""
        return sum(
            1 for entry in self._recent_operations
            if entry.level == LogLevel.WARNING
        )

    def clear_operations(self) -> None:
        """Очистить журнал операций."""
        self._recent_operations.clear()
        # Пересоздать файл
        if self.operations_log_path.exists():
            self.operations_log_path.unlink()
        self.log_operation("Operations log cleared", source="LogStore")

    def get_app_log_path(self) -> Path:
        """Получить путь к логу приложения."""
        return self.app_log_path

    def get_tasks_log_path(self) -> Path:
        """Получить путь к логу задач."""
        return self.tasks_log_path
