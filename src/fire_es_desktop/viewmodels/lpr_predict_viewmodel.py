# src/fire_es_desktop/viewmodels/lpr_predict_viewmodel.py
"""
LPRPredictViewModel — ViewModel для экрана прогноза ЛПР.

Согласно spec_second.md раздел 8.2 (MVVM):
- View → ViewModel → UseCase
- Состояние экрана прогноза
- Команды для UI
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..use_cases import (
    PredictUseCase,
    SaveDecisionUseCase,
    UseCaseResult,
    UseCaseStatus
)

logger = logging.getLogger("LPRPredictViewModel")


@dataclass
class RankPrediction:
    """Прогноз ранга."""
    rank: str
    rank_name: str
    probability: float


@dataclass
class PredictState:
    """Состояние экрана прогноза."""
    # Входные данные (разделы 1-2)
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Результат прогноза
    is_predicted: bool = False
    top_k_ranks: List[RankPrediction] = field(default_factory=list)
    confidence: float = 0.0
    entropy: float = 0.0
    induced_rank: Optional[str] = None
    model_info: Dict[str, str] = field(default_factory=dict)

    # Решение ЛПР
    selected_rank: Optional[str] = None
    decision_comment: str = ""

    # Статус
    is_loading: bool = False
    error_message: Optional[str] = None
    status_message: str = ""


class LPRPredictViewModel:
    """
    ViewModel для экрана прогноза ЛПР.

    Управляет состоянием экрана и вызывает Use Cases.
    """

    def __init__(
        self,
        models_path: Path,
        db_path: Path
    ):
        """
        Инициализировать ViewModel.

        Args:
            models_path: Путь к папке моделей.
            db_path: Путь к БД.
        """
        self.models_path = models_path
        self.db_path = db_path

        self.predict_use_case = PredictUseCase(models_path)
        self.save_decision_use_case = SaveDecisionUseCase(db_path)

        self.state = PredictState()
        self._normative_table: Optional[pd.DataFrame] = None

        # Callbacks для UI
        self.on_state_changed: Optional[Callable[[], None]] = None
        self.on_prediction_complete: Optional[Callable[[Dict], None]] = None
        self.on_decision_saved: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def notify_state_changed(self) -> None:
        """Уведомить об изменении состояния."""
        if self.on_state_changed:
            self.on_state_changed()

    def set_input_data(self, key: str, value: Any) -> None:
        """
        Установить входное значение.

        Args:
            key: Ключ параметра.
            value: Значение.
        """
        self.state.input_data[key] = value
        self.state.is_predicted = False  # Сбросить прогноз при изменении входа
        self.state.error_message = None
        self.notify_state_changed()

    def get_input_data(self) -> Dict[str, Any]:
        """Получить входные данные."""
        return self.state.input_data.copy()

    def clear_input_data(self) -> None:
        """Очистить входные данные."""
        self.state.input_data.clear()
        self.state.is_predicted = False
        self.state.top_k_ranks.clear()
        self.state.selected_rank = None
        self.state.decision_comment = ""
        self.state.error_message = None
        self.notify_state_changed()

    def predict(self) -> None:
        """
        Выполнить прогноз.

        Асинхронный вызов через TaskRunner должен быть снаружи.
        """
        self.state.is_loading = True
        self.state.error_message = None
        self.notify_state_changed()

        try:
            # Выполнить прогноз
            result = self.predict_use_case.execute(
                input_data=self.state.input_data,
                top_k=3
            )

            if result.success:
                # Обновить состояние
                data = result.data
                self.state.is_predicted = True
                self.state.top_k_ranks = [
                    RankPrediction(
                        rank=r["rank"],
                        rank_name=r["rank_name"],
                        probability=r["probability"]
                    )
                    for r in data.get("top_k_ranks", [])
                ]
                self.state.confidence = data.get("confidence", 0.0)
                self.state.entropy = data.get("entropy", 0.0)
                self.state.induced_rank = data.get("induced_rank_p50")
                self.state.model_info = {
                    "model_id": data.get("model_id", ""),
                    "model_name": data.get("model_name", "")
                }
                self.state.status_message = "Прогноз выполнен успешно"

                # Callback
                if self.on_prediction_complete:
                    self.on_prediction_complete(data)
            else:
                self.state.error_message = result.message
                if self.on_error:
                    self.on_error(result.message)

        except Exception as e:
            logger.error(f"Predict failed: {e}", exc_info=True)
            self.state.error_message = f"Ошибка прогноза: {str(e)}"
            if self.on_error:
                self.on_error(self.state.error_message)

        finally:
            self.state.is_loading = False
            self.notify_state_changed()

    def set_selected_rank(self, rank: str) -> None:
        """
        Установить выбранный ранг решения.

        Args:
            rank: Выбранный ранг.
        """
        self.state.selected_rank = rank
        self.notify_state_changed()

    def set_decision_comment(self, comment: str) -> None:
        """
        Установить комментарий к решению.

        Args:
            comment: Комментарий.
        """
        self.state.decision_comment = comment
        self.notify_state_changed()

    def save_decision(self) -> None:
        """
        Сохранить решение ЛПР.

        Асинхронный вызов через TaskRunner должен быть снаружи.
        """
        if not self.state.is_predicted:
            self.state.error_message = "Сначала выполните прогноз"
            self.notify_state_changed()
            return

        if not self.state.selected_rank:
            self.state.error_message = "Выберите решение"
            self.notify_state_changed()
            return

        self.state.is_loading = True
        self.state.error_message = None
        self.notify_state_changed()

        try:
            # Сформировать данные прогноза для сохранения
            prediction_data = {
                "top_k_ranks": [
                    {
                        "rank": r.rank,
                        "probability": r.probability
                    }
                    for r in self.state.top_k_ranks
                ],
                "confidence": self.state.confidence,
                "entropy": self.state.entropy,
                "induced_rank": self.state.induced_rank,
                "model_id": self.state.model_info.get("model_id", ""),
                "model_name": self.state.model_info.get("model_name", "")
            }

            # Сохранить решение
            result = self.save_decision_use_case.execute(
                input_data=self.state.input_data,
                prediction_data=prediction_data,
                decision_rank=self.state.selected_rank,
                decision_comment=self.state.decision_comment
            )

            if result.success:
                self.state.status_message = "Решение сохранено"
                data = result.data
                if self.on_decision_saved:
                    self.on_decision_saved(data.get("decision_id", ""))
            else:
                self.state.error_message = result.message
                if self.on_error:
                    self.on_error(result.message)

        except Exception as e:
            logger.error(f"Save decision failed: {e}", exc_info=True)
            self.state.error_message = f"Ошибка сохранения: {str(e)}"
            if self.on_error:
                self.on_error(self.state.error_message)

        finally:
            self.state.is_loading = False
            self.notify_state_changed()

    def get_normative_table(self) -> pd.DataFrame:
        """
        Получить нормативную таблицу.

        Returns:
            DataFrame с нормативами.
        """
        if self._normative_table is None:
            self._normative_table = self.predict_use_case.get_normative_table()
        return self._normative_table

    def get_prediction_chart_data(self) -> Dict[str, Any]:
        """
        Получить данные для диаграммы вероятностей.

        Returns:
            Данные для графика.
        """
        if not self.state.is_predicted:
            return {"ranks": [], "probabilities": []}

        ranks = [r.rank_name for r in self.state.top_k_ranks]
        probs = [r.probability * 100 for r in self.state.top_k_ranks]

        return {
            "ranks": ranks,
            "probabilities": probs,
            "confidence": self.state.confidence * 100
        }

    def validate_input(self) -> tuple[bool, List[str]]:
        """
        Проверить валидность входных данных.

        Returns:
            (валидно, список ошибок).
        """
        errors = []

        # Пример валидации (зависит от конкретных полей)
        if not self.state.input_data:
            errors.append("Нет входных данных")

        # Проверка обязательных полей
        required_fields = []  # Определить требуемые поля
        for field in required_fields:
            if field not in self.state.input_data:
                errors.append(f"Отсутствует поле: {field}")

        return len(errors) == 0, errors
