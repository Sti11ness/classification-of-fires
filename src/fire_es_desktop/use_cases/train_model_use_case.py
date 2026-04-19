# src/fire_es_desktop/use_cases/train_model_use_case.py
"""
TrainModelUseCase — training and artifact packaging for rank_tz models.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import text

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Import from domain layer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.db import DatabaseManager
from fire_es.rank_tz_contract import (
    CLASS_TO_RANK_MAP,
    OFFLINE_DEPLOYMENT_ROLE,
    PRODUCTION_DEPLOYMENT_ROLE,
    add_rank_tz_engineered_features,
    apply_preprocessor_artifact,
    build_preprocessor_artifact,
    class_list_to_rank_values,
    ensure_feature_frame,
    get_feature_set_spec,
    map_rank_series_to_classes,
)

logger = logging.getLogger("TrainModelUseCase")


class TrainModelUseCase(BaseUseCase):
    """Training flow for rank_tz models with a shared preprocessing contract."""

    def __init__(self, db_path: Path, models_path: Path):
        super().__init__(
            name="TrainModel",
            description="Обучение модели прогнозирования",
        )
        self.db_path = db_path
        self.models_path = models_path

    def execute(
        self,
        target: str = "rank_tz",
        model_type: str = "random_forest",
        feature_set: str = "online_tactical",
        custom_features: Optional[list[str]] = None,
        test_size: float = 0.25,
        class_weight: Optional[str] = "balanced",
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []

        if target != "rank_tz":
            return UseCaseResult(
                success=False,
                message="Desktop ML pipeline currently supports only rank_tz",
                warnings=warnings,
            )

        db = DatabaseManager(str(self.db_path))
        random_state = 42

        try:
            self.report_progress(1, 6, "Загрузка данных из БД")
            self.check_cancelled()

            df = pd.read_sql(f"SELECT * FROM fires WHERE {target} IS NOT NULL", db.engine)
            if df.empty:
                auto_labeled = self._auto_assign_rank_tz(db)
                if auto_labeled > 0:
                    warnings.append(
                        f"rank_tz отсутствовал, автоматически размечено {auto_labeled} записей"
                    )
                    df = pd.read_sql(f"SELECT * FROM fires WHERE {target} IS NOT NULL", db.engine)

            if df.empty:
                return UseCaseResult(
                    success=False,
                    message="Нет данных с целевой переменной rank_tz",
                    warnings=warnings,
                )

            spec = self._resolve_feature_spec(feature_set, custom_features)
            df = add_rank_tz_engineered_features(df, spec["feature_set"])

            self.report_progress(2, 6, "Подготовка признаков и target")
            self.check_cancelled()

            raw_X = ensure_feature_frame(df, spec["feature_order"])
            y = map_rank_series_to_classes(df[target])
            valid_mask = y.notna()
            if (~valid_mask).any():
                warnings.append(
                    f"Пропущено {int((~valid_mask).sum())} записей с неподдерживаемым rank_tz"
                )
            raw_X = raw_X.loc[valid_mask].copy()
            y = y.loc[valid_mask].astype(int)

            if raw_X.empty or y.empty:
                return UseCaseResult(
                    success=False,
                    message="Нет данных после подготовки rank_tz",
                    warnings=warnings,
                )

            if y.nunique() < 2:
                return UseCaseResult(
                    success=False,
                    message="Недостаточно классов для обучения (нужно минимум 2)",
                    warnings=warnings,
                )

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                raw_X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )

            preprocessor_artifact, X_train = build_preprocessor_artifact(
                X_train_raw,
                feature_order=spec["feature_order"],
                feature_set=spec["feature_set"],
                fill_strategy=spec["default_fill_strategy"],
                fill_value=spec["default_fill_value"],
                training_rows=len(raw_X),
                test_size=test_size,
                random_state=random_state,
            )
            X_test = apply_preprocessor_artifact(X_test_raw, preprocessor_artifact)

            self.report_progress(3, 6, f"Обучение модели ({model_type})")
            self.check_cancelled()

            model = self._fit_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                class_weight=class_weight,
                random_state=random_state,
            )

            y_pred = model.predict(X_test)
            metrics = self._build_metrics(model, y_train, y_test, y_pred)

            self.report_progress(4, 6, "Расчет importance и benchmark")
            self.check_cancelled()

            importance_df = self._build_feature_importance(model, X_test, y_test)
            missingness = (raw_X.isna().mean() * 100).round(4).sort_values(ascending=False)
            benchmark_payload = {
                "model_type": model.__class__.__name__,
                "feature_set": spec["feature_set"],
                "deployment_role": spec["deployment_role"],
                "offline_only": spec["offline_only"],
                "training_rows": int(len(raw_X)),
                "test_rows": int(len(X_test)),
                "metrics": metrics,
                "missing_pct_top20": missingness.head(20).to_dict(),
                "impurity_importance_top20": (
                    importance_df.sort_values("impurity_importance", ascending=False)
                    .head(20)
                    .set_index("feature")["impurity_importance"]
                    .round(6)
                    .to_dict()
                ),
                "permutation_importance_top20": (
                    importance_df.sort_values("permutation_importance", ascending=False)
                    .head(20)
                    .set_index("feature")["permutation_importance"]
                    .round(6)
                    .to_dict()
                ),
            }

            self.report_progress(5, 6, "Сохранение артефактов")
            self.check_cancelled()

            self.models_path.mkdir(parents=True, exist_ok=True)
            model_id = str(uuid.uuid4())[:8]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{target}_{timestamp}"

            model_path = self.models_path / f"model_{model_id}.joblib"
            metadata_path = self.models_path / f"model_{model_id}_meta.json"
            preprocessor_path = self.models_path / f"model_{model_id}_preprocessor.json"
            metrics_path = self.models_path / f"model_{model_id}_metrics.csv"
            importance_path = self.models_path / f"model_{model_id}_feature_importance.csv"
            benchmark_path = self.models_path / f"model_{model_id}_benchmark.json"

            joblib.dump(model, model_path)
            importance_df.to_csv(importance_path, index=False)
            pd.DataFrame([self._flatten_metrics(metrics)]).to_csv(metrics_path, index=False)
            with open(preprocessor_path, "w", encoding="utf-8") as f:
                json.dump(preprocessor_artifact, f, indent=2, ensure_ascii=False)
            with open(benchmark_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)

            metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "model_type": model_type,
                "target": target,
                "feature_set": spec["feature_set"],
                "features": spec["feature_order"],
                "input_schema": preprocessor_artifact["input_schema"],
                "fill_strategy": preprocessor_artifact["fill_strategy"],
                "fill_values": preprocessor_artifact["fill_values"],
                "allowed_missing": preprocessor_artifact["allowed_missing"],
                "class_mapping": preprocessor_artifact["class_mapping"],
                "training_schema_version": preprocessor_artifact["schema_version"],
                "deployment_role": spec["deployment_role"],
                "offline_only": spec["offline_only"],
                "metrics": metrics,
                "params": {
                    "test_size": test_size,
                    "class_weight": class_weight,
                    "random_state": random_state,
                },
                "dataset_info": {
                    "samples": int(len(raw_X)),
                    "features_count": int(len(spec["feature_order"])),
                    "classes": [str(value) for value in sorted(y.unique().tolist())],
                },
                "created_at": pd.Timestamp.now().isoformat(),
                "artifact_path": model_path.name,
                "metadata_path": metadata_path.name,
                "preprocessor_path": preprocessor_path.name,
                "metrics_path": metrics_path.name,
                "benchmark_path": benchmark_path.name,
                "feature_importance_path": importance_path.name,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.report_progress(6, 6, "Обучение завершено")

            registry_extra = {
                "artifact_path": model_path.name,
                "metadata_path": metadata_path.name,
                "preprocessor_path": preprocessor_path.name,
                "metrics_path": metrics_path.name,
                "benchmark_path": benchmark_path.name,
                "feature_importance_path": importance_path.name,
                "deployment_role": spec["deployment_role"],
                "offline_only": spec["offline_only"],
                "feature_set": spec["feature_set"],
                "input_schema": preprocessor_artifact["input_schema"],
                "fill_strategy": preprocessor_artifact["fill_strategy"],
                "fill_values": preprocessor_artifact["fill_values"],
                "allowed_missing": preprocessor_artifact["allowed_missing"],
                "class_mapping": preprocessor_artifact["class_mapping"],
                "training_schema_version": preprocessor_artifact["schema_version"],
            }

            return UseCaseResult(
                success=True,
                message=f"Модель обучена: {model_name}",
                data={
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "preprocessor_path": str(preprocessor_path),
                    "benchmark_path": str(benchmark_path),
                    "feature_importance_path": str(importance_path),
                    "metrics": metrics,
                    "feature_names": spec["feature_order"],
                    "samples": int(len(raw_X)),
                    "deployment_role": spec["deployment_role"],
                    "offline_only": spec["offline_only"],
                    "can_activate": not spec["offline_only"],
                    "registry_extra": registry_extra,
                },
                warnings=warnings,
            )

        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(
                success=False,
                message=f"Ошибка обучения: {str(e)}",
                error=str(e),
                warnings=warnings,
            )
        finally:
            db.close()

    def _resolve_feature_spec(
        self,
        feature_set: str,
        custom_features: Optional[list[str]],
    ) -> dict[str, Any]:
        if feature_set == "custom":
            if not custom_features:
                raise ValueError("Для custom feature_set требуется список признаков")
            return {
                "feature_set": "custom",
                "feature_order": custom_features,
                "deployment_role": OFFLINE_DEPLOYMENT_ROLE,
                "offline_only": True,
                "default_fill_strategy": "median",
                "default_fill_value": None,
            }
        return get_feature_set_spec(feature_set)

    def _fit_model(
        self,
        *,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: Optional[str],
        random_state: int,
    ) -> DecisionTreeClassifier | RandomForestClassifier:
        if model_type == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=random_state,
                class_weight=class_weight,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=random_state,
                class_weight=class_weight,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        model.fit(X_train, y_train)
        return model

    def _build_metrics(
        self,
        model: DecisionTreeClassifier | RandomForestClassifier,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        cm = confusion_matrix(y_test, y_pred, labels=sorted(model.classes_.tolist()))
        per_class_recall = {}
        for idx, class_id in enumerate(sorted(model.classes_.tolist())):
            total = cm[idx].sum()
            recall_value = float(cm[idx, idx] / total) if total else 0.0
            per_class_recall[str(CLASS_TO_RANK_MAP.get(class_id, class_id))] = recall_value

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "n_classes": int(len(model.classes_)),
            "classes": sorted(model.classes_.tolist()),
            "classes_rank_values": class_list_to_rank_values(sorted(model.classes_.tolist())),
            "confusion_matrix": cm.tolist(),
            "per_class_recall": per_class_recall,
            "model_type": model.__class__.__name__,
            "n_estimators": int(getattr(model, "n_estimators", 1)),
        }

    def _build_feature_importance(
        self,
        model: DecisionTreeClassifier | RandomForestClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        sample_size = min(400, len(X_test))
        if sample_size < len(X_test):
            sampled_indices = X_test.sample(n=sample_size, random_state=42).index
            X_eval = X_test.loc[sampled_indices]
            y_eval = y_test.loc[sampled_indices]
        else:
            X_eval = X_test
            y_eval = y_test

        impurity = pd.Series(
            getattr(model, "feature_importances_", np.zeros(X_test.shape[1])),
            index=X_test.columns,
        )
        permutation = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring="f1_macro",
            n_repeats=3,
            random_state=42,
            n_jobs=1,
        )
        permutation_series = pd.Series(permutation.importances_mean, index=X_test.columns)
        return (
            pd.DataFrame(
                {
                    "feature": X_test.columns.tolist(),
                    "impurity_importance": impurity.reindex(X_test.columns).tolist(),
                    "permutation_importance": permutation_series.reindex(X_test.columns).tolist(),
                }
            )
            .sort_values("permutation_importance", ascending=False)
            .reset_index(drop=True)
        )

    def _flatten_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                flattened[key] = json.dumps(value, ensure_ascii=False)
            else:
                flattened[key] = value
        return flattened

    def _auto_assign_rank_tz(self, db: DatabaseManager) -> int:
        counts = pd.read_sql(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN rank_tz IS NOT NULL THEN 1 ELSE 0 END) AS labeled "
            "FROM fires",
            db.engine,
        )
        if counts.empty:
            return 0

        labeled = int(counts.iloc[0].get("labeled") or 0)
        if labeled > 0:
            return 0

        df = pd.read_sql("SELECT id, equipment_count FROM fires", db.engine)
        if df.empty:
            return 0

        updates: list[dict[str, Any]] = []
        for row in df.itertuples(index=False):
            equipment_count = getattr(row, "equipment_count", None)
            if pd.isna(equipment_count) or equipment_count is None or equipment_count < 1:
                rank = 1.0
                distance = 0.0
            else:
                count = int(equipment_count)
                if count == 1:
                    rank, distance = 1.0, 0.0
                elif count == 2:
                    rank, distance = 1.5, 0.0
                elif count == 3:
                    rank, distance = 2.0, 0.0
                elif count == 4:
                    rank, distance = 3.0, 0.0
                elif count == 5:
                    rank, distance = 4.0, 0.0
                else:
                    rank, distance = 5.0, float((count - 5) * 0.1)

            updates.append(
                {
                    "id": int(row.id),
                    "rank_tz": float(rank),
                    "rank_distance": float(distance),
                }
            )

        with db.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE fires "
                    "SET rank_tz = :rank_tz, rank_distance = :rank_distance "
                    "WHERE id = :id"
                ),
                updates,
            )

        return len(updates)
