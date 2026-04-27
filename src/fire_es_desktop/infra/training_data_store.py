"""
Training data store and assembler for analyst-side datasets.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from fire_es.db import (
    DatabaseManager,
    Fire,
    FIRES_HISTORICAL_TABLE,
    FIRES_LPR_TRAIN_TABLE,
    LPRDecision,
    TRAIN_SYNTHETIC_TABLE,
)

logger = logging.getLogger("TrainingDataStore")


class TrainingDataStore:
    """Storage adapter for historical, LPR-promoted, and synthetic training sources."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_manager = DatabaseManager(str(db_path))
        self.db_manager.create_tables()
        self.engine = self.db_manager.engine
        self.SessionLocal = sessionmaker(bind=self.engine)

    def close(self) -> None:
        self.db_manager.close()

    def get_source_counts(self) -> dict[str, int]:
        with self.engine.begin() as conn:
            historical_total = self._scalar(conn, "SELECT COUNT(*) FROM fires_historical")
            historical_ready = self._scalar(
                conn,
                "SELECT COUNT(*) FROM fires_historical WHERE rank_tz IS NOT NULL "
                "AND COALESCE(usable_for_training, 0) = 1 "
                "AND COALESCE(is_canonical_event_record, 1) = 1",
            )
            lpr_total = self._scalar(conn, "SELECT COUNT(*) FROM fires_lpr_train")
            lpr_ready = self._scalar(
                conn,
                "SELECT COUNT(*) FROM fires_lpr_train WHERE rank_tz IS NOT NULL "
                "AND COALESCE(usable_for_training, 0) = 1 "
                "AND COALESCE(train_enabled, 1) = 1",
            )
            synthetic_total = self._scalar(conn, "SELECT COUNT(*) FROM train_synthetic")
            new_lpr_candidates = self._scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM lpr_decisions d
                LEFT JOIN fires_lpr_train t ON t.source_decision_id = d.id
                WHERE COALESCE(d.save_to_db, 1) = 1
                  AND t.source_decision_id IS NULL
                """,
            )
        return {
            "historical_total": historical_total,
            "historical_ready": historical_ready,
            "lpr_total": lpr_total,
            "lpr_ready": lpr_ready,
            "synthetic_total": synthetic_total,
            "new_lpr_candidates": new_lpr_candidates,
        }

    def insert_historical_records(self, records: list[dict[str, Any]]) -> int:
        """Insert normalized historical records into the analyst historical source table."""
        if not records:
            return 0
        table_columns = {column.name for column in FIRES_HISTORICAL_TABLE.columns}
        prepared = []
        for record in records:
            clean_record = {}
            for key, value in record.items():
                if key == "id" or key not in table_columns:
                    continue
                clean_record[key] = value
            prepared.append(clean_record)
        if not prepared:
            return 0
        with self.engine.begin() as conn:
            conn.execute(FIRES_HISTORICAL_TABLE.insert(), prepared)
        return len(prepared)

    def sync_new_lpr_decisions(self, *, promoted_by: str = "analyst") -> dict[str, int]:
        """Promote analyst-approved LPR decisions into the training LPR source layer."""
        with self.SessionLocal() as session:
            existing_ids = {
                value
                for value, in session.execute(
                    text("SELECT source_decision_id FROM fires_lpr_train WHERE source_decision_id IS NOT NULL")
                ).all()
                if value is not None
            }
            rows = (
                session.query(LPRDecision, Fire)
                .join(Fire, LPRDecision.fire_id == Fire.id)
                .filter(LPRDecision.save_to_db == True)
                .order_by(LPRDecision.created_at.asc(), LPRDecision.id.asc())
                .all()
            )
            now = datetime.utcnow()
            prepared: list[dict[str, Any]] = []
            for decision, fire in rows:
                if decision.id in existing_ids:
                    continue
                payload = fire.to_dict()
                payload.pop("id", None)
                payload["source_decision_id"] = decision.id
                payload["promoted_at"] = now
                payload["promoted_by"] = promoted_by
                payload["train_enabled"] = True
                payload["rank_tz"] = decision.decision_rank
                payload["rank_tz_vector"] = decision.decision_rank
                payload["rank_label_source"] = "lpr_decision"
                payload["human_verified"] = True
                payload["usable_for_training"] = True
                payload["predicted_rank_at_decision"] = decision.predicted_rank
                prepared.append(payload)

            if prepared:
                session.execute(FIRES_LPR_TRAIN_TABLE.insert(), prepared)
                session.commit()
            else:
                session.rollback()

        counts = self.get_source_counts()
        return {
            "added": len(prepared),
            "new_lpr_candidates": counts["new_lpr_candidates"],
            "lpr_total": counts["lpr_total"],
            "lpr_ready": counts["lpr_ready"],
        }

    def assemble_real_training_dataset(
        self,
        *,
        include_historical: bool = True,
        include_lpr: bool = True,
        label_column: str = "rank_tz_vector",
        canonical_only: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Assemble the real training dataset from historical and promoted LPR sources."""
        frames: list[pd.DataFrame] = []
        counts = {
            "historical_selected_rows": 0,
            "lpr_selected_rows": 0,
        }

        if include_historical:
            historical_query = (
                "SELECT * FROM fires_historical "
                f"WHERE {label_column} IS NOT NULL "
                "AND COALESCE(usable_for_training, 0) = 1"
            )
            if canonical_only:
                historical_query += " AND COALESCE(is_canonical_event_record, 1) = 1"
            historical = pd.read_sql(historical_query, self.engine)
            historical["training_source"] = "historical"
            counts["historical_selected_rows"] = int(len(historical))
            frames.append(historical)

        if include_lpr:
            lpr_query = (
                "SELECT * FROM fires_lpr_train "
                f"WHERE {label_column} IS NOT NULL "
                "AND COALESCE(usable_for_training, 0) = 1 "
                "AND COALESCE(train_enabled, 1) = 1"
            )
            if canonical_only:
                lpr_query += " AND COALESCE(is_canonical_event_record, 1) = 1"
            lpr_df = pd.read_sql(lpr_query, self.engine)
            lpr_df["training_source"] = "lpr"
            counts["lpr_selected_rows"] = int(len(lpr_df))
            frames.append(lpr_df)

        if not frames:
            return pd.DataFrame(), counts

        combined = pd.concat(frames, ignore_index=True)
        counts["real_rows_total_before_split"] = int(len(combined))
        return combined, counts

    def save_synthetic_batch(
        self,
        *,
        train_run_id: str,
        generator_method: str,
        generator_params: dict[str, Any],
        base_source_scope: dict[str, Any],
        source_feature_set: str,
        semantic_target: str,
        feature_frame: pd.DataFrame,
        target_series: pd.Series,
        original_rows_count: int,
        class_to_rank_map: dict[int, float],
    ) -> int:
        """Persist only synthetic rows produced for the current train run."""
        if len(feature_frame) <= original_rows_count:
            return 0

        synthetic_frame = feature_frame.iloc[original_rows_count:].reset_index(drop=True)
        synthetic_target = target_series.iloc[original_rows_count:].reset_index(drop=True)
        if synthetic_frame.empty:
            return 0

        created_at = datetime.utcnow()
        prepared = []
        for idx, (_, row) in enumerate(synthetic_frame.iterrows()):
            target_class = int(synthetic_target.iloc[idx])
            prepared.append(
                {
                    "train_run_id": train_run_id,
                    "row_number": idx,
                    "generator_method": generator_method,
                    "generator_params": json.dumps(generator_params, ensure_ascii=False),
                    "base_source_scope": json.dumps(base_source_scope, ensure_ascii=False),
                    "created_at": created_at,
                    "source_feature_set": source_feature_set,
                    "semantic_target": semantic_target,
                    "is_synthetic": True,
                    "target_class": target_class,
                    "target_rank": class_to_rank_map.get(target_class),
                    "features_json": json.dumps(row.to_dict(), ensure_ascii=False),
                }
            )
        with self.engine.begin() as conn:
            conn.execute(TRAIN_SYNTHETIC_TABLE.insert(), prepared)
        return len(prepared)

    @staticmethod
    def _scalar(conn, query: str) -> int:
        row = conn.execute(text(query)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
