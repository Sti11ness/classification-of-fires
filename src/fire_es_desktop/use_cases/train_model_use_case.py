# src/fire_es_desktop/use_cases/train_model_use_case.py
"""
TrainModelUseCase — leakage-safe training and artifact packaging for rank_tz models.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.metrics import build_classification_metrics
from fire_es.model_selection import (
    SPLIT_PROTOCOL_GROUP_KFOLD,
    SPLIT_PROTOCOL_GROUP_SHUFFLE,
    SPLIT_PROTOCOL_ROW_RANDOM_LEGACY,
    SPLIT_PROTOCOL_TEMPORAL_HOLDOUT,
    split_dataset,
)
from fire_es.normatives import get_normative_hash, load_rank_resource_normatives
from fire_es.rank_tz_contract import (
    CLASS_TO_RANK_MAP,
    OFFLINE_DEPLOYMENT_ROLE,
    PRODUCTION_DEPLOYMENT_ROLE,
    AVAILABILITY_STAGE_DISPATCH,
    SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY,
    SEMANTIC_TARGET_RANK_TZ_VECTOR,
    get_feature_set_forbidden_violations,
    get_optional_lpr_fields,
    prepare_feature_payload,
    apply_preprocessor_artifact,
    build_preprocessor_artifact,
    class_list_to_rank_values,
    ensure_feature_frame,
    get_feature_set_spec,
    map_rank_series_to_classes,
)
from fire_es.db import DatabaseManager

logger = logging.getLogger("TrainModelUseCase")


LABEL_COLUMN_BY_TARGET = {
    SEMANTIC_TARGET_RANK_TZ_VECTOR: "rank_tz_vector",
    SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY: "rank_tz_count_proxy",
}


class TrainModelUseCase(BaseUseCase):
    """Training flow for canonical rank_tz models with a shared preprocessing contract."""

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
        feature_set: str = "dispatch_initial_safe",
        custom_features: Optional[list[str]] = None,
        test_size: float = 0.25,
        class_weight: Optional[str] = "balanced",
        *,
        semantic_target: str = SEMANTIC_TARGET_RANK_TZ_VECTOR,
        availability_stage: Optional[str] = None,
        split_protocol: str = SPLIT_PROTOCOL_GROUP_SHUFFLE,
        canonical_only: bool = True,
        metric_primary: str = "f1_macro",
        allow_proxy_target: bool = False,
        allow_legacy_random_split: bool = False,
        calibration_method: Optional[str] = None,
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

        if semantic_target == SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY and not allow_proxy_target:
            return UseCaseResult(
                success=False,
                message="Auxiliary count proxy target requires allow_proxy_target=True",
                warnings=warnings,
            )
        if split_protocol == SPLIT_PROTOCOL_ROW_RANDOM_LEGACY and not allow_legacy_random_split:
            return UseCaseResult(
                success=False,
                message="Legacy row-random split requires allow_legacy_random_split=True",
                warnings=warnings,
            )

        db = DatabaseManager(str(self.db_path))
        random_state = 42

        try:
            self.report_progress(1, 7, "Загрузка данных из БД")
            self.check_cancelled()

            label_column = LABEL_COLUMN_BY_TARGET.get(semantic_target)
            if not label_column:
                return UseCaseResult(
                    success=False,
                    message=f"Неподдерживаемый semantic_target: {semantic_target}",
                    warnings=warnings,
                )

            query = (
                "SELECT * FROM fires "
                f"WHERE {label_column} IS NOT NULL "
                "AND COALESCE(usable_for_training, 0) = 1"
            )
            if canonical_only:
                query += " AND COALESCE(is_canonical_event_record, 1) = 1"
            df = pd.read_sql(query, db.engine)
            if df.empty:
                return UseCaseResult(
                    success=False,
                    message="Нет обучающей целевой переменной. Сначала выполните AssignRankTzUseCase.",
                    warnings=warnings,
                )

            spec = self._resolve_feature_spec(feature_set, custom_features)
            availability_stage = availability_stage or spec.get("availability_stage", AVAILABILITY_STAGE_DISPATCH)
            forbidden_feature_violations = get_feature_set_forbidden_violations(
                spec["feature_order"],
                availability_stage=availability_stage,
            )
            forbidden_feature_check_passed = not forbidden_feature_violations
            if not forbidden_feature_check_passed and not spec["offline_only"]:
                return UseCaseResult(
                    success=False,
                    message=(
                        "Набор признаков содержит запрещенные поля для выбранной стадии: "
                        + ", ".join(forbidden_feature_violations)
                    ),
                    warnings=warnings,
                )
            df = prepare_feature_payload(
                df,
                feature_set=spec["feature_set"],
                availability_stage=availability_stage,
            )

            self.report_progress(2, 7, "Подготовка признаков и target")
            self.check_cancelled()

            raw_X = ensure_feature_frame(df, spec["feature_order"])
            y = map_rank_series_to_classes(df[label_column])
            valid_mask = y.notna()
            if (~valid_mask).any():
                warnings.append(
                    f"Пропущено {int((~valid_mask).sum())} записей с неподдерживаемым рангом"
                )
            df = df.loc[valid_mask].reset_index(drop=True)
            raw_X = raw_X.loc[valid_mask].reset_index(drop=True)
            y = y.loc[valid_mask].astype(int).reset_index(drop=True)

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

            self.report_progress(3, 7, "Leakage-safe split")
            self.check_cancelled()

            split_result = split_dataset(
                df,
                y=y,
                split_protocol=split_protocol,
                test_size=test_size,
                random_state=random_state,
            )
            split_meta = split_result.metadata
            if split_meta.get("event_overlap_rate", 0.0) != 0.0 and split_protocol != SPLIT_PROTOCOL_ROW_RANDOM_LEGACY:
                return UseCaseResult(
                    success=False,
                    message="Нечестный split: event_overlap_rate должен быть равен 0",
                    warnings=warnings,
                )

            X_train_raw = raw_X.iloc[split_result.train_indices].reset_index(drop=True)
            X_test_raw = raw_X.iloc[split_result.test_indices].reset_index(drop=True)
            y_train = y.iloc[split_result.train_indices].reset_index(drop=True)
            y_test = y.iloc[split_result.test_indices].reset_index(drop=True)
            train_rows = df.iloc[split_result.train_indices].reset_index(drop=True)
            test_rows = df.iloc[split_result.test_indices].reset_index(drop=True)

            label_source_policy = sorted(
                {str(value) for value in df["rank_label_source"].dropna().tolist()}
            ) or [None]

            preprocessor_artifact, X_train = build_preprocessor_artifact(
                X_train_raw,
                feature_order=spec["feature_order"],
                feature_set=spec["feature_set"],
                fill_strategy=spec["default_fill_strategy"],
                fill_value=spec["default_fill_value"],
                training_rows=len(raw_X),
                test_size=test_size,
                random_state=random_state,
                semantic_target=semantic_target,
                label_source_policy=[value for value in label_source_policy if value],
            )
            X_test = apply_preprocessor_artifact(X_test_raw, preprocessor_artifact)

            self.report_progress(4, 7, f"Обучение модели ({model_type})")
            self.check_cancelled()

            model, calibration_status = self._fit_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                class_weight=class_weight,
                random_state=random_state,
                calibration_method=calibration_method,
            )

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = build_classification_metrics(
                y_train=y_train,
                y_test=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                classes=np.array(sorted(model.classes_.tolist())),
                class_to_rank_map=CLASS_TO_RANK_MAP,
                split_metadata=split_meta,
            )
            metrics["model_type"] = model.__class__.__name__
            metrics["n_estimators"] = int(getattr(model, "n_estimators", 1))
            metrics["calibration_status"] = calibration_status
            metrics["metric_primary"] = metric_primary

            self.report_progress(5, 7, "Расчет importance и benchmark")
            self.check_cancelled()

            importance_df = self._build_feature_importance(model, X_test, y_test)
            tree_artifact_name = None
            if model_type == "decision_tree":
                tree_artifact_name = self._export_decision_tree(model, preprocessor_artifact["feature_names_out"])

            missingness = (raw_X.isna().mean() * 100).round(4).sort_values(ascending=False)
            normative_payload = load_rank_resource_normatives()
            class_distribution_train = y_train.value_counts(normalize=True).sort_index().to_dict()
            class_distribution_test = y_test.value_counts(normalize=True).sort_index().to_dict()
            benchmark_payload = {
                "model_type": model.__class__.__name__,
                "feature_set": spec["feature_set"],
                "availability_stage": availability_stage,
                "deployment_role": spec["deployment_role"],
                "offline_only": spec["offline_only"],
                "semantic_target": semantic_target,
                "label_source_policy": [value for value in label_source_policy if value],
                "training_rows": int(len(raw_X)),
                "test_rows": int(len(X_test)),
                "metrics": metrics,
                "split_protocol": split_protocol,
                "canonical_only": canonical_only,
                "duplicate_policy": "canonical_event_only" if canonical_only else "all_rows",
                "normative_version": normative_payload["normative_version"],
                "normative_hash": get_normative_hash(normative_payload),
                "forbidden_feature_check_passed": forbidden_feature_check_passed,
                "forbidden_feature_violations": forbidden_feature_violations,
                "missing_policy": "none_is_unknown",
                "optional_lpr_fields": get_optional_lpr_fields(spec["feature_set"]),
                "missing_pct_top20": missingness.head(20).to_dict(),
                "class_distribution_train": class_distribution_train,
                "class_distribution_test": class_distribution_test,
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

            self.report_progress(6, 7, "Сохранение артефактов")
            self.check_cancelled()

            self.models_path.mkdir(parents=True, exist_ok=True)
            model_id = str(uuid.uuid4())[:8]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{semantic_target}_{timestamp}"

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
                "semantic_target": semantic_target,
                "target_definition": semantic_target,
                "label_source_policy": [value for value in label_source_policy if value],
                "feature_set": spec["feature_set"],
                "availability_stage": availability_stage,
                "features": spec["feature_order"],
                "feature_names_out": preprocessor_artifact["feature_names_out"],
                "input_schema": preprocessor_artifact["input_schema"],
                "fill_strategy": preprocessor_artifact["fill_strategy"],
                "fill_values": preprocessor_artifact["fill_values"],
                "allowed_missing": preprocessor_artifact["allowed_missing"],
                "class_mapping": preprocessor_artifact["class_mapping"],
                "training_schema_version": preprocessor_artifact["schema_version"],
                "preprocessing_version": preprocessor_artifact.get("preprocessing_version", 2),
                "deployment_role": spec["deployment_role"],
                "offline_only": spec["offline_only"],
                "metrics": metrics,
                "metric_primary": metric_primary,
                "params": {
                    "test_size": test_size,
                    "class_weight": class_weight,
                    "random_state": random_state,
                    "split_protocol": split_protocol,
                    "canonical_only": canonical_only,
                    "allow_proxy_target": allow_proxy_target,
                    "allow_legacy_random_split": allow_legacy_random_split,
                    "calibration_method": calibration_method,
                },
                "dataset_info": {
                    "samples": int(len(raw_X)),
                    "features_count": int(len(spec["feature_order"])),
                    "classes": [str(value) for value in sorted(y.unique().tolist())],
                    "canonical_only": canonical_only,
                    "class_distribution_train": class_distribution_train,
                    "class_distribution_test": class_distribution_test,
                },
                "split_protocol": split_protocol,
                "event_id_column": split_meta.get("event_id_column"),
                "event_overlap_rate": split_meta.get("event_overlap_rate"),
                "train_event_count": split_meta.get("train_event_count"),
                "test_event_count": split_meta.get("test_event_count"),
                "train_date_min": split_meta.get("train_date_min"),
                "train_date_max": split_meta.get("train_date_max"),
                "test_date_min": split_meta.get("test_date_min"),
                "test_date_max": split_meta.get("test_date_max"),
                "duplicate_policy": "canonical_event_only" if canonical_only else "all_rows",
                "canonical_only": canonical_only,
                "normative_version": normative_payload["normative_version"],
                "normative_hash": get_normative_hash(normative_payload),
                "calibration_status": calibration_status,
                "forbidden_feature_check_passed": forbidden_feature_check_passed,
                "forbidden_feature_violations": forbidden_feature_violations,
                "missing_policy": "none_is_unknown",
                "optional_lpr_fields": get_optional_lpr_fields(spec["feature_set"]),
                "created_at": pd.Timestamp.now().isoformat(),
                "artifact_path": model_path.name,
                "metadata_path": metadata_path.name,
                "preprocessor_path": preprocessor_path.name,
                "metrics_path": metrics_path.name,
                "benchmark_path": benchmark_path.name,
                "feature_importance_path": importance_path.name,
            }
            if tree_artifact_name:
                metadata["tree_artifact_path"] = tree_artifact_name
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            self.report_progress(7, 7, "Обучение завершено")

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
                "availability_stage": availability_stage,
                "input_schema": preprocessor_artifact["input_schema"],
                "fill_strategy": preprocessor_artifact["fill_strategy"],
                "fill_values": preprocessor_artifact["fill_values"],
                "allowed_missing": preprocessor_artifact["allowed_missing"],
                "class_mapping": preprocessor_artifact["class_mapping"],
                "training_schema_version": preprocessor_artifact["schema_version"],
                "preprocessing_version": preprocessor_artifact.get("preprocessing_version", 2),
                "semantic_target": semantic_target,
                "target_definition": semantic_target,
                "label_source_policy": [value for value in label_source_policy if value],
                "split_protocol": split_protocol,
                "event_overlap_rate": split_meta.get("event_overlap_rate"),
                "metric_primary": metric_primary,
                "normative_version": normative_payload["normative_version"],
                "calibration_status": calibration_status,
                "forbidden_feature_check_passed": forbidden_feature_check_passed,
                "forbidden_feature_violations": forbidden_feature_violations,
                "missing_policy": "none_is_unknown",
                "optional_lpr_fields": get_optional_lpr_fields(spec["feature_set"]),
            }
            if tree_artifact_name:
                registry_extra["tree_artifact_path"] = tree_artifact_name

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
                    "feature_names_out": preprocessor_artifact["feature_names_out"],
                    "samples": int(len(raw_X)),
                    "deployment_role": spec["deployment_role"],
                    "offline_only": spec["offline_only"],
                    "can_activate": (not spec["offline_only"]) and semantic_target == SEMANTIC_TARGET_RANK_TZ_VECTOR,
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
                "availability_stage": "retrospective",
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
        calibration_method: Optional[str],
    ) -> tuple[Any, str]:
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
        elif model_type == "gradient_boosting":
            model = HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.05,
                max_iter=300,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        model.fit(X_train, y_train)
        calibration_status = "not_calibrated"
        if calibration_method in {"sigmoid", "isotonic"}:
            calibrated = CalibratedClassifierCV(model, method=calibration_method, cv=3)
            calibrated.fit(X_train, y_train)
            model = calibrated
            calibration_status = calibration_method
        return model, calibration_status

    def _build_feature_importance(
        self,
        model: Any,
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

        base_model = getattr(model, "base_estimator", model)
        impurity = pd.Series(
            getattr(base_model, "feature_importances_", np.zeros(X_test.shape[1])),
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

    def _export_decision_tree(self, model: Any, feature_names: list[str]) -> Optional[str]:
        """Save a human-readable decision tree artifact for analyst workflows."""
        if not isinstance(model, DecisionTreeClassifier):
            return None
        model_id = str(uuid.uuid4())[:8]
        tree_txt_path = self.models_path / f"model_{model_id}_tree.txt"
        tree_txt_path.write_text(
            export_text(model, feature_names=feature_names, decimals=3),
            encoding="utf-8",
        )
        try:
            import matplotlib.pyplot as plt
            from sklearn import tree as sklearn_tree

            fig = plt.figure(figsize=(20, 12))
            sklearn_tree.plot_tree(
                model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                fontsize=7,
            )
            tree_png_path = self.models_path / f"model_{model_id}_tree.png"
            fig.savefig(tree_png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return tree_png_path.name
        except Exception as exc:  # pragma: no cover - best effort artifact
            logger.warning("Failed to render tree png: %s", exc)
            return tree_txt_path.name
