# src/fire_es_desktop/use_cases/save_decision_use_case.py
"""
SaveDecisionUseCase — сохранение решения ЛПР.

Согласно spec_first.md раздел 5.2 и spec_second.md раздел 4.1:
- Запись решения ЛПР в БД
- Фиксация связки "вход → прогноз → решение"
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

logger = logging.getLogger("SaveDecisionUseCase")


class SaveDecisionUseCase(BaseUseCase):
    """
    Сценарий сохранения решения ЛПР.

    Шаги:
    1. Валидация данных
    2. Запись входного кейса в fires
    3. Запись решения в lpr_decisions
    """

    _RANK_MAP = {
        "1": 1.0,
        "1-бис": 1.5,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
    }

    def __init__(self, db_path: Path):
        super().__init__(
            name="SaveDecision",
            description="Сохранение решения ЛПР",
        )
        self.db_path = db_path

    def execute(
        self,
        input_data: Dict[str, Any],
        prediction_data: Dict[str, Any],
        decision_rank: str,
        decision_comment: str = "",
        user_id: str = "LPR",
    ) -> UseCaseResult:
        """
        Сохранить решение ЛПР.

        Args:
            input_data: Входные данные (разделы 1-2).
            prediction_data: Данные прогноза.
            decision_rank: Выбранный ранг решения.
            decision_comment: Комментарий к решению.
            user_id: ID пользователя.

        Returns:
            Результат сохранения.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings = []

        try:
            # Шаг 1: Валидация
            self.report_progress(1, 3, "Валидация данных")
            self.check_cancelled()

            if not input_data:
                return UseCaseResult(
                    success=False,
                    message="Нет входных данных",
                    warnings=warnings,
                )

            if not prediction_data:
                return UseCaseResult(
                    success=False,
                    message="Нет данных прогноза",
                    warnings=warnings,
                )

            if not decision_rank:
                return UseCaseResult(
                    success=False,
                    message="Не выбрано решение",
                    warnings=warnings,
                )

            decision_rank_value = self._parse_rank(decision_rank)
            if decision_rank_value is None:
                return UseCaseResult(
                    success=False,
                    message=f"Неизвестный формат ранга: {decision_rank}",
                    warnings=warnings,
                )

            # Шаг 2: Подготовка и запись в БД
            self.report_progress(2, 3, "Запись решения в БД")
            self.check_cancelled()

            from fire_es.db import DatabaseManager, Fire, LPRDecision

            db = DatabaseManager(str(self.db_path))
            db.create_tables()

            top_k = prediction_data.get("top_k_ranks", [])
            predicted_rank = None
            if top_k:
                predicted_rank = self._parse_rank(top_k[0].get("rank"))

            now = datetime.utcnow()
            fire_payload = {
                "row_id": None,
                "source_sheet": "LPR_MANUAL_INPUT",
                "source_period": None,
                "region_code": None,
                "region_text": None,
                "fire_date": now,
                "year": now.year,
                "month": now.month,
                "building_floors": self._to_int(input_data.get("building_floors")),
                "fire_floor": self._to_int(input_data.get("fire_floor")),
                "distance_to_station": self._to_float(input_data.get("distance_to_station")),
                "fatalities": self._to_int(input_data.get("fatalities")),
                "injuries": self._to_int(input_data.get("injuries")),
                "direct_damage": self._to_float(input_data.get("direct_damage")),
                "rank_tz": predicted_rank,
                "created_at": now,
                "updated_at": now,
            }

            session = db.get_session()
            try:
                fire = Fire(**fire_payload)
                session.add(fire)
                session.flush()

                decision = LPRDecision(
                    fire_id=fire.id,
                    decision_rank=decision_rank_value,
                    decision_resources={
                        "selected_rank": decision_rank,
                        "user_id": user_id,
                    },
                    predicted_rank=predicted_rank,
                    predicted_probabilities=top_k,
                    comment=decision_comment,
                    save_to_db=True,
                    created_at=now,
                )
                session.add(decision)
                session.commit()

                decision_id = decision.id
                fire_id = fire.id
                logger.info("Saved LPR decision: decision_id=%s, fire_id=%s", decision_id, fire_id)
            finally:
                session.close()
                db.close()

            # Шаг 3: Завершение
            self.report_progress(3, 3, "Решение сохранено")

            return UseCaseResult(
                success=True,
                message="Решение сохранено",
                data={
                    "decision_id": decision_id,
                    "fire_id": fire_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "decision_rank": decision_rank_value,
                },
                warnings=warnings,
            )

        except Exception as e:
            logger.error("Save decision failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED

            return UseCaseResult(
                success=False,
                message=f"Ошибка сохранения: {str(e)}",
                error=str(e),
                warnings=warnings,
            )

    def _parse_rank(self, value: Any) -> Optional[float]:
        """Преобразовать ранг в числовое значение."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        value_str = str(value).strip()
        if value_str in self._RANK_MAP:
            return self._RANK_MAP[value_str]
        try:
            return float(value_str)
        except ValueError:
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        """Безопасно привести к int."""
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Безопасно привести к float."""
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None
