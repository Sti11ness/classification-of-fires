# src/fire_es_desktop/viewmodels/batch_predict_viewmodel.py
"""
BatchPredictViewModel — ViewModel для экрана пакетного прогноза.

Согласно spec_second.md раздел 8.2 (MVVM):
- View → ViewModel → UseCase
- Состояние экрана пакетного прогноза
- Команды для UI
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..use_cases import (
    BatchPredictExportUseCase,
    UseCaseResult,
    UseCaseStatus
)

logger = logging.getLogger("BatchPredictViewModel")


@dataclass
class BatchPredictState:
    """Состояние экрана пакетного прогноза."""
    # Входные данные
    input_file_path: Optional[Path] = None
    input_source: str = "excel"  # "excel" или "database"
    output_format: str = "excel"  # "excel" или "csv"
    feature_set: str = "online_tactical"
    top_k: int = 3
    use_bootstrap: bool = True
    n_bootstrap: int = 30

    # Результат
    is_completed: bool = False
    predictions_count: int = 0
    input_count: int = 0
    output_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    # Статус
    is_loading: bool = False
    progress_percent: int = 0
    status_message: str = ""
    error_message: Optional[str] = None


class BatchPredictViewModel:
    """
    ViewModel для экрана пакетного прогноза.

    Управляет состоянием экрана и вызывает Use Cases.
    """

    def __init__(
        self,
        models_path: Path,
        reports_path: Path
    ):
        """
        Инициализировать ViewModel.

        Args:
            models_path: Путь к папке моделей.
            reports_path: Путь к папке отчётов.
        """
        self.models_path = models_path
        self.reports_path = reports_path

        self.use_case = BatchPredictExportUseCase(models_path, reports_path)

        self.state = BatchPredictState()

        # Callbacks для UI
        self.on_state_changed: Optional[Callable[[], None]] = None
        self.on_complete: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[int, str], None]] = None

    def notify_state_changed(self) -> None:
        """Уведомить об изменении состояния."""
        if self.on_state_changed:
            self.on_state_changed()

    def set_input_file(self, path: Path) -> None:
        """
        Установить входной файл.

        Args:
            path: Путь к файлу.
        """
        self.state.input_file_path = path
        self.state.is_completed = False
        self.state.error_message = None
        self.notify_state_changed()

    def set_input_source(self, source: str) -> None:
        """
        Установить источник данных.

        Args:
            source: "excel" или "database".
        """
        self.state.input_source = source
        self.notify_state_changed()

    def set_output_format(self, fmt: str) -> None:
        """
        Установить формат экспорта.

        Args:
            fmt: "excel" или "csv".
        """
        self.state.output_format = fmt
        self.notify_state_changed()

    def set_feature_set(self, feature_set: str) -> None:
        """
        Установить набор признаков.

        Args:
            feature_set: Набор признаков.
        """
        self.state.feature_set = feature_set
        self.notify_state_changed()

    def set_top_k(self, top_k: int) -> None:
        """
        Установить количество вариантов прогноза.

        Args:
            top_k: Количество вариантов.
        """
        self.state.top_k = top_k
        self.notify_state_changed()

    def set_use_bootstrap(self, use: bool) -> None:
        """
        Установить использование бутстрапа.

        Args:
            use: Использовать ли бутстрап.
        """
        self.state.use_bootstrap = use
        self.notify_state_changed()

    def set_n_bootstrap(self, n: int) -> None:
        """
        Установить количество бутстрап-выборок.

        Args:
            n: Количество выборок.
        """
        self.state.n_bootstrap = n
        self.notify_state_changed()

    def reset(self) -> None:
        """Сбросить состояние."""
        self.state = BatchPredictState()
        self.notify_state_changed()

    def predict(self) -> None:
        """
        Выполнить пакетный прогноз.

        Асинхронный вызов через TaskRunner должен быть снаружи.
        """
        self.state.is_loading = True
        self.state.is_completed = False
        self.state.error_message = None
        self.state.progress_percent = 0
        self.notify_state_changed()

        try:
            # Выполнить UseCase
            result = self.use_case.execute(
                input_source=self.state.input_source,
                input_path=self.state.input_file_path,
                output_format=self.state.output_format,
                feature_set=self.state.feature_set,
                top_k=self.state.top_k,
                use_bootstrap=self.state.use_bootstrap,
                n_bootstrap=self.state.n_bootstrap
            )

            if result.success:
                # Обновить состояние
                data = result.data
                self.state.is_completed = True
                self.state.predictions_count = data.get("predictions_count", 0)
                self.state.input_count = data.get("input_count", 0)
                self.state.output_path = Path(data.get("output_path", ""))
                self.state.warnings = data.get("warnings", [])
                self.state.status_message = "Пакетный прогноз завершён"

                # Callback
                if self.on_complete:
                    self.on_complete(data)
            else:
                self.state.error_message = result.message
                if self.on_error:
                    self.on_error(result.message)

        except Exception as e:
            logger.error(f"Batch predict failed: {e}", exc_info=True)
            self.state.error_message = f"Ошибка: {str(e)}"
            if self.on_error:
                self.on_error(self.state.error_message)

        finally:
            self.state.is_loading = False
            self.state.progress_percent = 100
            self.notify_state_changed()

    def validate_input(self) -> tuple[bool, List[str]]:
        """
        Проверить валидность входных данных.

        Returns:
            (валидно, список ошибок).
        """
        errors = []

        if not self.state.input_file_path:
            errors.append("Не выбран входной файл")

        if self.state.input_file_path and not self.state.input_file_path.exists():
            errors.append("Файл не найден")

        return len(errors) == 0, errors
