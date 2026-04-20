"""
AssignResearchSeverityLabelUseCase — explicit research-only severity labeling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import sessionmaker

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.db import DatabaseManager, Fire
from fire_es.rank_tz_contract import LABEL_SOURCE_RESEARCH_ONLY

logger = logging.getLogger("AssignResearchSeverityLabelUseCase")


class AssignResearchSeverityLabelUseCase(BaseUseCase):
    """Persist research-only severity labels without entering canonical rank flow."""

    def __init__(self, db_path: Path):
        super().__init__(
            name="AssignResearchSeverityLabel",
            description="Разметка исследовательской severity-метки",
        )
        self.db_path = db_path

    def execute(self) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        db = DatabaseManager(str(self.db_path))
        db.create_tables()
        try:
            df = pd.read_sql("SELECT id, rank_ref FROM fires", db.engine)
            if df.empty:
                return UseCaseResult(success=False, message="Нет данных в БД")
            Session = sessionmaker(bind=db.engine)
            with Session() as session:
                for _, row in df.iterrows():
                    session.query(Fire).filter(Fire.id == row["id"]).update(
                        {
                            "rank_label_source": LABEL_SOURCE_RESEARCH_ONLY
                            if pd.notna(row.get("rank_ref"))
                            else None,
                        }
                    )
                session.commit()
            return UseCaseResult(
                success=True,
                message="Research severity labels marked",
                data={"rows": int(len(df))},
            )
        except Exception as e:
            logger.error("Research severity assignment failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(success=False, message=f"Ошибка разметки: {e}", error=str(e))
        finally:
            db.close()
