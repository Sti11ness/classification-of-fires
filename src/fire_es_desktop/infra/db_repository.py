# src/fire_es_desktop\infra\db_repository.py
"""
DbRepository — репозиторий для работы с базой данных.

Согласно spec_second.md раздел 8.1:
- Infrastructure слой: доступ к SQLite
- UI не должен содержать SQLAlchemy-логику
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import text

# Импорт из domain слоя
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.db import DatabaseManager, Fire, LPRDecision, Normative, Model

logger = logging.getLogger("DbRepository")


class DbRepository:
    """
    Репозиторий для работы с базой данных Fire ES.

    Предоставляет удобный интерфейс для CRUD операций,
    скрывая детали реализации SQLAlchemy.
    """

    def __init__(self, db_path: Path):
        """
        Инициализировать репозиторий.

        Args:
            db_path: Путь к SQLite базе данных.
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(str(db_path))
        self.db_manager.create_tables()
        self.engine = self.db_manager.engine
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(f"DbRepository.{db_path.name}")

    def get_session(self) -> Session:
        """Получить сессию базы данных."""
        return self.SessionLocal()

    # ========================================================================
    # Fires (пожары)
    # ========================================================================

    def get_fires_count(self) -> int:
        """Получить количество записей о пожарах."""
        with self.get_session() as session:
            return session.query(Fire).count()

    def get_all_fires(self, limit: Optional[int] = None) -> List[Fire]:
        """
        Получить все пожары.

        Args:
            limit: Ограничить количество записей.

        Returns:
            Список записей Fire.
        """
        with self.get_session() as session:
            query = session.query(Fire)
            if limit:
                query = query.limit(limit)
            return query.all()

    def get_fire_by_id(self, fire_id: int) -> Optional[Fire]:
        """Получить пожар по ID."""
        with self.get_session() as session:
            return session.query(Fire).filter(Fire.id == fire_id).first()

    def add_fire(self, fire_data: Dict[str, Any]) -> int:
        """
        Добавить пожар в базу.

        Args:
            fire_data: Данные пожара.

        Returns:
            ID созданной записи.
        """
        with self.get_session() as session:
            fire = Fire(**fire_data)
            session.add(fire)
            session.commit()
            session.refresh(fire)
            self.logger.debug(f"Added fire: {fire.id}")
            return fire.id

    def add_fires_batch(self, fires_data: List[Dict[str, Any]]) -> int:
        """
        Добавить много пожаров за раз (batch insert).

        Args:
            fires_data: Список данных пожаров.

        Returns:
            Количество добавленных записей.
        """
        with self.get_session() as session:
            fires = [Fire(**data) for data in fires_data]
            session.bulk_save_objects(fires)
            session.commit()
            self.logger.info(f"Batch added {len(fires)} fires")
            return len(fires)

    def update_fire(self, fire_id: int, updates: Dict[str, Any]) -> bool:
        """
        Обновить пожар.

        Args:
            fire_id: ID пожара.
            updates: Поля для обновления.

        Returns:
            True если обновлено.
        """
        with self.get_session() as session:
            fire = session.query(Fire).filter(Fire.id == fire_id).first()
            if not fire:
                return False

            for key, value in updates.items():
                if hasattr(fire, key):
                    setattr(fire, key, value)

            session.commit()
            self.logger.debug(f"Updated fire: {fire_id}")
            return True

    def delete_fire(self, fire_id: int) -> bool:
        """Удалить пожар."""
        with self.get_session() as session:
            fire = session.query(Fire).filter(Fire.id == fire_id).first()
            if not fire:
                return False

            session.delete(fire)
            session.commit()
            self.logger.debug(f"Deleted fire: {fire_id}")
            return True

    def get_fires_as_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Получить пожары как DataFrame.

        Args:
            limit: Ограничить количество записей.

        Returns:
            DataFrame с данными пожаров.
        """
        with self.get_session() as session:
            query = session.query(Fire)
            if limit:
                query = query.limit(limit)
            return pd.read_sql(query.statement, session.bind)

    def get_fires_with_rank_tz(self) -> pd.DataFrame:
        """Получить пожары с размеченным rank_tz."""
        with self.get_session() as session:
            df = pd.read_sql(
                session.query(Fire).filter(Fire.rank_tz.isnot(None)).statement,
                session.bind
            )
            return df

    # ========================================================================
    # LPR Decisions (решения ЛПР)
    # ========================================================================

    def get_lpr_decisions_count(self) -> int:
        """Получить количество решений ЛПР."""
        with self.get_session() as session:
            return session.query(LPRDecision).count()

    def add_lpr_decision(self, decision_data: Dict[str, Any]) -> int:
        """
        Добавить решение ЛПР.

        Args:
            decision_data: Данные решения.

        Returns:
            ID созданной записи.
        """
        with self.get_session() as session:
            decision = LPRDecision(**decision_data)
            session.add(decision)
            session.commit()
            session.refresh(decision)
            self.logger.info(f"Added LPR decision: {decision.id}")
            return decision.id

    def get_lpr_decisions(self, limit: Optional[int] = None) -> List[LPRDecision]:
        """Получить решения ЛПР."""
        with self.get_session() as session:
            query = session.query(LPRDecision)
            if limit:
                query = query.limit(limit)
            return query.all()

    def get_lpr_decision_summaries(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Получить список решений ЛПР для UI-таблицы."""
        with self.get_session() as session:
            query = (
                session.query(LPRDecision, Fire)
                .outerjoin(Fire, LPRDecision.fire_id == Fire.id)
                .order_by(LPRDecision.created_at.desc(), LPRDecision.id.desc())
            )
            if limit:
                query = query.limit(limit)

            rows: List[Dict[str, Any]] = []
            for decision, fire in query.all():
                comment = decision.comment or ""
                rows.append(
                    {
                        "decision_id": decision.id,
                        "created_at": (
                            decision.created_at.isoformat()
                            if decision.created_at
                            else None
                        ),
                        "decision_rank": decision.decision_rank,
                        "predicted_rank": decision.predicted_rank,
                        "comment_preview": (
                            f"{comment[:77]}..." if len(comment) > 80 else comment
                        ),
                        "comment": comment,
                        "fire_id": decision.fire_id,
                        "fire_date": (
                            fire.fire_date.isoformat()
                            if fire and fire.fire_date
                            else None
                        ),
                        "source_sheet": fire.source_sheet if fire else None,
                    }
                )
            return rows

    def get_lpr_decision_detail(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """Получить полную карточку решения ЛПР с привязанным пожаром."""
        with self.get_session() as session:
            row = (
                session.query(LPRDecision, Fire)
                .outerjoin(Fire, LPRDecision.fire_id == Fire.id)
                .filter(LPRDecision.id == decision_id)
                .first()
            )
            if not row:
                return None

            decision, fire = row
            return {
                "decision_id": decision.id,
                "fire_id": decision.fire_id,
                "decision_rank": decision.decision_rank,
                "decision_resources": decision.decision_resources or {},
                "predicted_rank": decision.predicted_rank,
                "predicted_probabilities": decision.predicted_probabilities or [],
                "comment": decision.comment or "",
                "save_to_db": bool(decision.save_to_db),
                "created_at": (
                    decision.created_at.isoformat()
                    if decision.created_at
                    else None
                ),
                "fire_snapshot": fire.to_dict() if fire else None,
            }

    def update_lpr_decision(
        self,
        decision_id: int,
        *,
        decision_rank: Optional[float],
        comment: str,
    ) -> bool:
        """Обновить редактируемую часть решения ЛПР."""
        with self.get_session() as session:
            decision = (
                session.query(LPRDecision)
                .filter(LPRDecision.id == decision_id)
                .first()
            )
            if not decision:
                return False

            decision.decision_rank = decision_rank
            decision.comment = comment
            if decision.save_to_db:
                fire = session.query(Fire).filter(Fire.id == decision.fire_id).first()
                if fire:
                    fire.rank_tz = decision_rank
                    fire.rank_tz_vector = decision_rank
                    fire.rank_label_source = "lpr_decision"
                    fire.human_verified = True
                    fire.usable_for_training = True
            session.commit()
            self.logger.info("Updated LPR decision: %s", decision_id)
            return True

    def get_decisions_as_dataframe(self) -> pd.DataFrame:
        """Получить решения ЛПР как DataFrame."""
        with self.get_session() as session:
            return pd.read_sql(
                session.query(LPRDecision).statement,
                session.bind
            )

    # ========================================================================
    # Normatives (нормативы)
    # ========================================================================

    def get_normatives(self) -> List[Normative]:
        """Получить все нормативы."""
        with self.get_session() as session:
            return session.query(Normative).all()

    def get_normatives_as_dataframe(self) -> pd.DataFrame:
        """Получить нормативы как DataFrame."""
        with self.get_session() as session:
            return pd.read_sql(
                session.query(Normative).statement,
                session.bind
            )

    def add_normative(self, normative_data: Dict[str, Any]) -> int:
        """Добавить норматив."""
        with self.get_session() as session:
            normative = Normative(**normative_data)
            session.add(normative)
            session.commit()
            session.refresh(normative)
            return normative.id

    # ========================================================================
    # Models (модели)
    # ========================================================================

    def get_models(self) -> List[Model]:
        """Получить все модели."""
        with self.get_session() as session:
            return session.query(Model).all()

    def get_active_model(self) -> Optional[Model]:
        """Получить активную модель."""
        with self.get_session() as session:
            return session.query(Model).filter(Model.is_active == True).first()

    def add_model(self, model_data: Dict[str, Any]) -> int:
        """Добавить модель."""
        with self.get_session() as session:
            model = Model(**model_data)
            session.add(model)
            session.commit()
            session.refresh(model)
            self.logger.info(f"Added model: {model.id}")
            return model.id

    def set_active_model(self, model_id: int) -> bool:
        """
        Сделать модель активной.

        Args:
            model_id: ID модели.

        Returns:
            True если успешно.
        """
        with self.get_session() as session:
            # Снять активность со всех моделей
            session.query(Model).update({"is_active": False})

            # Активировать выбранную
            model = session.query(Model).filter(Model.id == model_id).first()
            if not model:
                return False

            model.is_active = True
            session.commit()
            self.logger.info(f"Set active model: {model_id}")
            return True

    def get_model_by_id(self, model_id: int) -> Optional[Model]:
        """Получить модель по ID."""
        with self.get_session() as session:
            return session.query(Model).filter(Model.id == model_id).first()

    # ========================================================================
    # Статистика
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику базы данных.

        Returns:
            Словарь со статистикой.
        """
        with self.get_session() as session:
            try:
                fires_count = int(
                    session.execute(text("SELECT COUNT(*) FROM fires_historical")).scalar() or 0
                )
            except Exception:
                fires_count = session.query(Fire).count()
            decisions_count = session.query(LPRDecision).count()
            models_count = session.query(Model).count()
            try:
                lpr_train_count = int(
                    session.execute(text("SELECT COUNT(*) FROM fires_lpr_train")).scalar() or 0
                )
            except Exception:
                lpr_train_count = 0
            try:
                synthetic_count = int(
                    session.execute(text("SELECT COUNT(*) FROM train_synthetic")).scalar() or 0
                )
            except Exception:
                synthetic_count = 0
            active_model = session.query(Model).filter(Model.is_active == True).first()

            return {
                "fires_count": fires_count,
                "lpr_decisions_count": decisions_count,
                "lpr_train_count": lpr_train_count,
                "synthetic_count": synthetic_count,
                "models_count": models_count,
                "active_model_id": active_model.id if active_model else None,
                "active_model_name": (
                    f"{active_model.model_type}:{active_model.version or active_model.id}"
                    if active_model else None
                ),
            }

    def close(self) -> None:
        """Закрыть соединение с базой."""
        self.engine.dispose()
        self.logger.info("Database connection closed")
