"""
LPRDecisionHistoryViewModel — ViewModel для истории решений ЛПР.

Предоставляет:
- загрузку списка решений
- выбор карточки решения
- редактирование decision_rank и comment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..infra import DbRepository

logger = logging.getLogger("LPRDecisionHistoryViewModel")


RANK_LABEL_TO_VALUE = {
    "1": 1.0,
    "1-бис": 1.5,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
}
RANK_VALUE_TO_LABEL = {value: key for key, value in RANK_LABEL_TO_VALUE.items()}


@dataclass
class LPRDecisionHistoryState:
    """Состояние истории решений ЛПР."""

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    selected_decision_id: Optional[int] = None
    selected_detail: Optional[Dict[str, Any]] = None
    is_loading: bool = False
    error_message: Optional[str] = None
    status_message: str = ""


class LPRDecisionHistoryViewModel:
    """ViewModel для страницы истории решений ЛПР."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_repo = DbRepository(db_path)
        self.state = LPRDecisionHistoryState()

        self.on_state_changed: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    def close(self) -> None:
        """Закрыть соединение с репозиторием."""
        self.db_repo.close()

    def notify_state_changed(self) -> None:
        """Уведомить UI об изменении состояния."""
        if self.on_state_changed:
            self.on_state_changed()

    def load_decisions(self) -> None:
        """Загрузить список решений ЛПР."""
        self.state.is_loading = True
        self.state.error_message = None
        self.notify_state_changed()

        try:
            summaries = self.db_repo.get_lpr_decision_summaries()
            self.state.decisions = [self._decorate_summary(item) for item in summaries]
            self.state.status_message = (
                f"Загружено решений: {len(self.state.decisions)}"
                if self.state.decisions
                else "Сохраненных решений пока нет"
            )

            if self.state.selected_decision_id is not None:
                existing_ids = {item["decision_id"] for item in self.state.decisions}
                if self.state.selected_decision_id in existing_ids:
                    self.select_decision(self.state.selected_decision_id)
                else:
                    self.state.selected_decision_id = None
                    self.state.selected_detail = None
            elif self.state.decisions:
                self.select_decision(self.state.decisions[0]["decision_id"])
        except Exception as exc:
            logger.error("Load LPR decision history failed: %s", exc, exc_info=True)
            self.state.error_message = f"Ошибка загрузки истории: {exc}"
            self.state.status_message = "Ошибка загрузки"
            if self.on_error:
                self.on_error(self.state.error_message)
        finally:
            self.state.is_loading = False
            self.notify_state_changed()

    def select_decision(self, decision_id: int) -> None:
        """Выбрать решение и загрузить его детальную карточку."""
        self.state.selected_decision_id = decision_id
        self.state.error_message = None

        try:
            detail = self.db_repo.get_lpr_decision_detail(decision_id)
            self.state.selected_detail = self._decorate_detail(detail) if detail else None
            if detail:
                self.state.status_message = f"Открыто решение #{decision_id}"
            else:
                self.state.status_message = "Решение не найдено"
        except Exception as exc:
            logger.error("Select LPR decision failed: %s", exc, exc_info=True)
            self.state.error_message = f"Ошибка загрузки карточки: {exc}"
            self.state.status_message = "Ошибка загрузки карточки"
            if self.on_error:
                self.on_error(self.state.error_message)
        finally:
            self.notify_state_changed()

    def update_selected_decision(self, decision_rank_label: str, comment: str) -> bool:
        """Обновить редактируемую часть выбранного решения."""
        if self.state.selected_decision_id is None:
            self.state.error_message = "Не выбрано решение для редактирования"
            self.state.status_message = "Нет выбранного решения"
            self.notify_state_changed()
            return False

        rank_value = self.parse_rank_label(decision_rank_label)
        if rank_value is None:
            self.state.error_message = f"Некорректный ранг: {decision_rank_label}"
            self.state.status_message = "Ошибка валидации ранга"
            self.notify_state_changed()
            return False

        try:
            updated = self.db_repo.update_lpr_decision(
                self.state.selected_decision_id,
                decision_rank=rank_value,
                comment=comment,
            )
            if not updated:
                self.state.error_message = "Не удалось сохранить изменения"
                self.state.status_message = "Ошибка сохранения"
                self.notify_state_changed()
                return False

            selected_id = self.state.selected_decision_id
            self.load_decisions()
            self.select_decision(selected_id)
            self.state.status_message = f"Изменения решения #{selected_id} сохранены"
            self.notify_state_changed()
            return True
        except Exception as exc:
            logger.error("Update LPR decision failed: %s", exc, exc_info=True)
            self.state.error_message = f"Ошибка сохранения: {exc}"
            self.state.status_message = "Ошибка сохранения"
            if self.on_error:
                self.on_error(self.state.error_message)
            self.notify_state_changed()
            return False

    @staticmethod
    def parse_rank_label(value: Any) -> Optional[float]:
        """Преобразовать label ранга из UI в числовое значение БД."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return RANK_LABEL_TO_VALUE.get(str(value).strip())

    @staticmethod
    def format_rank_value(value: Any) -> str:
        """Форматировать числовой ранг для UI."""
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return RANK_VALUE_TO_LABEL.get(numeric, str(int(numeric)) if numeric.is_integer() else str(numeric))

    def _decorate_summary(self, item: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(item)
        enriched["decision_rank_label"] = self.format_rank_value(item.get("decision_rank"))
        enriched["predicted_rank_label"] = self.format_rank_value(item.get("predicted_rank"))
        return enriched

    def _decorate_detail(self, item: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(item)
        enriched["decision_rank_label"] = self.format_rank_value(item.get("decision_rank"))
        enriched["predicted_rank_label"] = self.format_rank_value(item.get("predicted_rank"))
        return enriched
