# src/fire_es_desktop/use_cases/assign_rank_tz_use_case.py
"""
AssignRankTzUseCase — explicit canonical rank labeling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.db import DatabaseManager, Fire
from fire_es.equipment_parse import build_unparsed_equipment_report
from fire_es.ranking import assign_rank_tz
from fire_es.rank_tz_contract import (
    LABEL_SOURCE_HISTORICAL_VECTOR,
    LABEL_SOURCE_PROXY_BOOTSTRAP,
    SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY,
    SEMANTIC_TARGET_RANK_TZ_VECTOR,
)

logger = logging.getLogger("AssignRankTzUseCase")


class AssignRankTzUseCase(BaseUseCase):
    """Mass label assignment for canonical rank space."""

    def __init__(self, db_path: Path):
        super().__init__(
            name="AssignRankTz",
            description="Явная разметка ранга пожара по нормативам",
        )
        self.db_path = db_path

    def execute(
        self,
        *,
        target_definition: str = SEMANTIC_TARGET_RANK_TZ_VECTOR,
        batch_size: int = 500,
        override_human_verified: bool = False,
        override_existing_labels: bool = False,
        source_table: str = "fires",
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []
        allowed_source_tables = {"fires", "fires_historical", "fires_lpr_train"}
        if source_table not in allowed_source_tables:
            return UseCaseResult(
                success=False,
                message=f"Неподдерживаемый источник разметки: {source_table}",
                warnings=warnings,
            )

        try:
            db = DatabaseManager(str(self.db_path))
            db.create_tables()

            self.report_progress(1, 4, "Загрузка данных из БД")
            self.check_cancelled()

            engine = create_engine(f"sqlite:///{self.db_path}")
            df = pd.read_sql(f"SELECT * FROM {source_table}", engine)
            if df.empty:
                db.close()
                return UseCaseResult(success=False, message="Нет данных в БД", warnings=warnings)

            human_verified_mask = df.get("human_verified", pd.Series(False, index=df.index)).fillna(False).astype(bool)
            lpr_decision_mask = (
                df.get("rank_label_source", pd.Series(index=df.index, dtype=object)).fillna("").eq("lpr_decision")
            )
            existing_label_mask = (
                df.get("rank_tz", pd.Series(index=df.index, dtype=float)).notna()
                | df.get("rank_tz_vector", pd.Series(index=df.index, dtype=float)).notna()
                | df.get("rank_tz_count_proxy", pd.Series(index=df.index, dtype=float)).notna()
                | df.get("rank_label_source", pd.Series(index=df.index, dtype=object)).fillna("").ne("")
            )
            existing_label_only_mask = existing_label_mask & ~human_verified_mask & ~lpr_decision_mask
            protected_mask = human_verified_mask | lpr_decision_mask
            if not override_existing_labels:
                protected_mask = protected_mask | existing_label_only_mask
            human_verified_skipped_count = int(human_verified_mask.sum())
            lpr_decision_skipped_count = int(lpr_decision_mask.sum())
            existing_label_count = int(existing_label_only_mask.sum())
            existing_label_skipped_count = existing_label_count if not override_existing_labels else 0
            existing_label_recalculated_count = existing_label_count if override_existing_labels else 0
            overridden_count = 0
            if protected_mask.any() and not override_human_verified:
                candidate_df = df.loc[~protected_mask].copy()
            else:
                candidate_df = df.copy()
                if protected_mask.any() and override_human_verified:
                    overridden_count = int(protected_mask.sum())
                    warnings.append("Override enabled: human-verified labels may be recalculated")
            if existing_label_only_mask.any() and override_existing_labels:
                warnings.append("Override enabled: existing historical labels may be recalculated")

            if candidate_df.empty:
                db.close()
                self.report_progress(4, 4, "Новых строк для разметки нет")
                return UseCaseResult(
                    success=True,
                    message="Новых строк для разметки нет",
                    data={
                        "total_records": 0,
                        "updated_records": 0,
                        "assigned_records": 0,
                        "rank_distribution": {},
                        "null_count": 0,
                        "semantic_target": target_definition,
                        "human_verified_skipped_count": human_verified_skipped_count,
                        "lpr_decision_skipped_count": lpr_decision_skipped_count,
                        "existing_label_skipped_count": existing_label_skipped_count,
                        "existing_label_recalculated_count": existing_label_recalculated_count,
                        "overridden_count": overridden_count,
                        "top_unparsed_equipment_formats": [],
                    },
                    warnings=warnings,
                )

            self.report_progress(2, 4, "Расчет рангов")
            self.check_cancelled()

            if target_definition == SEMANTIC_TARGET_RANK_TZ_VECTOR:
                labeled = assign_rank_tz(candidate_df, target_definition="vector")
                label_column = "rank_tz_vector"
                label_source = LABEL_SOURCE_HISTORICAL_VECTOR
            elif target_definition == SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY:
                labeled = assign_rank_tz(candidate_df, target_definition="count_proxy")
                label_column = "rank_tz_count_proxy"
                label_source = LABEL_SOURCE_PROXY_BOOTSTRAP
                warnings.append("Используется auxiliary count proxy, а не canonical vector target")
            else:
                return UseCaseResult(
                    success=False,
                    message=f"Неподдерживаемый target_definition: {target_definition}",
                    warnings=warnings,
                )

            rank_distribution = labeled[label_column].value_counts(dropna=False).to_dict() if not labeled.empty else {}
            null_count = int(labeled[label_column].isna().sum()) if not labeled.empty else 0
            unparsed_equipment_report = (
                build_unparsed_equipment_report(candidate_df)
                if target_definition == SEMANTIC_TARGET_RANK_TZ_VECTOR and not candidate_df.empty and "equipment" in candidate_df.columns
                else pd.DataFrame()
            )

            self.report_progress(3, 4, "Запись рангов в БД")
            self.check_cancelled()

            Session = sessionmaker(bind=engine)
            updated_count = 0
            assigned_count = 0
            update_columns = [
                "rank_tz",
                "rank_distance",
                "rank_tz_vector",
                "rank_tz_count_proxy",
                "rank_label_source",
                "rank_normative_version",
                "rank_quality_flags",
                "usable_for_training",
            ]
            with Session() as session:
                for start in range(0, len(labeled), batch_size):
                    batch = labeled.iloc[start : start + batch_size]
                    for _, row in batch.iterrows():
                        payload = {column: row.get(column) for column in update_columns}
                        if target_definition == SEMANTIC_TARGET_RANK_TZ_VECTOR and pd.notna(row.get("rank_tz_vector")):
                            payload["rank_tz"] = row.get("rank_tz_vector")
                        if target_definition == SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY and pd.notna(row.get("rank_tz_count_proxy")):
                            payload["rank_tz"] = row.get("rank_tz_count_proxy")
                        payload["rank_label_source"] = label_source if payload.get("rank_tz") is not None else None
                        if payload.get("rank_tz") is not None:
                            assigned_count += 1
                        assignments = ", ".join(f"{column} = :{column}" for column in payload.keys())
                        statement = text(
                            f"UPDATE {source_table} SET {assignments} WHERE id = :id"
                        )
                        params = dict(payload)
                        params["id"] = row["id"]
                        session.execute(statement, params)
                    session.commit()
                    updated_count += len(batch)

            db.close()
            self.report_progress(4, 4, "Разметка завершена")
            return UseCaseResult(
                success=True,
                message=f"Размечено {updated_count} записей",
                data={
                    "total_records": len(labeled),
                    "updated_records": updated_count,
                    "assigned_records": assigned_count,
                    "rank_distribution": rank_distribution,
                    "null_count": null_count,
                    "semantic_target": target_definition,
                        "human_verified_skipped_count": human_verified_skipped_count,
                        "lpr_decision_skipped_count": lpr_decision_skipped_count,
                        "existing_label_skipped_count": existing_label_skipped_count,
                        "existing_label_recalculated_count": existing_label_recalculated_count,
                        "overridden_count": overridden_count,
                    "top_unparsed_equipment_formats": (
                        unparsed_equipment_report.to_dict(orient="records")
                        if not unparsed_equipment_report.empty
                        else []
                    ),
                },
                warnings=warnings,
            )
        except Exception as e:
            logger.error("Rank assignment failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(
                success=False,
                message=f"Ошибка разметки: {str(e)}",
                error=str(e),
                warnings=warnings,
            )
