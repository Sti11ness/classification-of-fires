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
import optuna
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SMOTENC, SVMSMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from .base_use_case import BaseUseCase, UseCaseCancelledError, UseCaseResult, UseCaseStatus

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.metrics import build_classification_metrics
from fire_es.model_selection import (
    SPLIT_PROTOCOL_GROUP_KFOLD,
    SPLIT_PROTOCOL_GROUP_SHUFFLE,
    SPLIT_PROTOCOL_ROW_RANDOM_LEGACY,
    SPLIT_PROTOCOL_SOURCE_HOLDOUT,
    SPLIT_PROTOCOL_TEMPORAL_HOLDOUT,
    split_dataset,
)
from fire_es.normatives import get_normative_hash, load_rank_resource_normatives
from fire_es.rank_tz_contract import (
    CLASS_TO_RANK_MAP,
    FIELD_SPECS,
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
from ..infra import TrainingDataStore

logger = logging.getLogger("TrainModelUseCase")


LABEL_COLUMN_BY_TARGET = {
    SEMANTIC_TARGET_RANK_TZ_VECTOR: "rank_tz_vector",
    SEMANTIC_TARGET_RANK_TZ_COUNT_PROXY: "rank_tz_count_proxy",
}

SUPPORTED_TUNING_METRICS = {
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
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
        include_historical: bool = True,
        include_lpr: bool = True,
        synthetic_method: Optional[str] = None,
        synthetic_k_neighbors: int = 5,
        synthetic_m_neighbors: int = 10,
        tuning_enabled: bool = False,
        tuning_trials: int = 50,
        synthetic_target_total_rows: Optional[int] = None,
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []

        if metric_primary not in SUPPORTED_TUNING_METRICS:
            return UseCaseResult(
                success=False,
                message=f"Неподдерживаемая метрика подбора: {metric_primary}",
                warnings=warnings,
            )

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
        training_store = TrainingDataStore(self.db_path)
        random_state = 42
        total_workflow_steps = 8 if not tuning_enabled else 9 + max(1, int(tuning_trials))

        try:
            self.report_progress(1, total_workflow_steps, "Загрузка данных из источников обучения")
            self.check_cancelled()

            label_column = LABEL_COLUMN_BY_TARGET.get(semantic_target)
            if not label_column:
                return UseCaseResult(
                    success=False,
                    message=f"Неподдерживаемый semantic_target: {semantic_target}",
                    warnings=warnings,
                )

            df, source_counts = training_store.assemble_real_training_dataset(
                include_historical=include_historical,
                include_lpr=include_lpr,
                label_column=label_column,
                canonical_only=canonical_only,
            )
            if df.empty:
                return UseCaseResult(
                    success=False,
                    message="Нет данных для обучения в источниках historical/LPR. Сначала импортируйте исторические данные и/или подтяните решения ЛПР.",
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

            self.report_progress(2, total_workflow_steps, "Подготовка признаков и target")
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

            self.report_progress(3, total_workflow_steps, "Разделение данных на обучение и проверку")
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
            train_rows_before_synthetic = int(len(X_train_raw))
            test_rows_real = int(len(X_test_raw))
            effective_synthetic_target_total_rows = synthetic_target_total_rows
            if synthetic_method:
                (
                    effective_synthetic_target_total_rows,
                    target_total_capped,
                    target_total_cap,
                ) = self._effective_synthetic_target_total_rows(
                    y_train=y_train,
                    requested_total=synthetic_target_total_rows,
                )
                if target_total_capped:
                    warnings.append(
                        "Запрошенный размер обучающей части после синтетики ограничен "
                        f"максимально достижимым балансом: {target_total_cap}"
                    )

            label_source_policy = sorted(
                {str(value) for value in df["rank_label_source"].dropna().tolist()}
            ) or [None]

            if tuning_enabled and split_protocol in {
                SPLIT_PROTOCOL_TEMPORAL_HOLDOUT,
                SPLIT_PROTOCOL_SOURCE_HOLDOUT,
            }:
                return UseCaseResult(
                    success=False,
                    message=(
                        "Автоподбор гиперпараметров пока поддерживается только для "
                        "событийных и legacy-случайных режимов разделения. "
                        "Для temporal/source holdout отключите автоподбор."
                    ),
                    warnings=warnings,
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
                semantic_target=semantic_target,
                label_source_policy=[value for value in label_source_policy if value],
            )
            X_test = apply_preprocessor_artifact(X_test_raw, preprocessor_artifact)

            self.report_progress(4, total_workflow_steps, "Преобразование train/test")
            self.check_cancelled()

            tuning_trials_completed = 0
            best_cv_score: Optional[float] = None
            best_params: dict[str, Any] = {}
            model_params_override: Optional[dict[str, Any]] = None
            class_weight_used = class_weight
            if tuning_enabled:
                self.check_cancelled()
                tuning_payload = self._run_optuna_search(
                    model_type=model_type,
                    X_train_raw=X_train_raw,
                    y_train=y_train,
                    train_rows=train_rows,
                    split_protocol=split_protocol,
                    spec=spec,
                    semantic_target=semantic_target,
                    metric_primary=metric_primary,
                    synthetic_method=synthetic_method,
                    synthetic_k_neighbors=synthetic_k_neighbors,
                    synthetic_m_neighbors=synthetic_m_neighbors,
                    synthetic_target_total_rows=effective_synthetic_target_total_rows,
                    tuning_trials=tuning_trials,
                    random_state=random_state,
                    progress_offset=4,
                    total_workflow_steps=total_workflow_steps,
                )
                tuning_trials_completed = int(tuning_payload["trials_completed"])
                best_cv_score = float(tuning_payload["best_cv_score"])
                best_params = dict(tuning_payload["best_params"])
                model_params_override = dict(tuning_payload["model_params"])
                class_weight_used = tuning_payload["class_weight"]

            synthetic_rows_added = 0
            synthetic_status = "disabled"
            synthetic_run_id = None
            resampled_class_distribution = y_train.value_counts(normalize=True).sort_index().to_dict()
            if synthetic_method:
                synthetic_step = 5 + max(1, int(tuning_trials)) if tuning_enabled else 5
                self.report_progress(synthetic_step, total_workflow_steps, f"Синтетическое расширение ({synthetic_method})")
                self.check_cancelled()
                (
                    X_train,
                    y_train,
                    synthetic_rows_added,
                    synthetic_status,
                    synthetic_run_id,
                ) = self._apply_synthetic_augmentation(
                    method=synthetic_method,
                    X_train=X_train,
                    y_train=y_train,
                    raw_feature_frame=X_train_raw,
                    spec=spec,
                    preprocessor_artifact=preprocessor_artifact,
                    semantic_target=semantic_target,
                    source_counts=source_counts,
                    training_store=training_store,
                    random_state=random_state,
                    k_neighbors=synthetic_k_neighbors,
                    m_neighbors=synthetic_m_neighbors,
                    synthetic_target_total_rows=effective_synthetic_target_total_rows,
                )
                resampled_class_distribution = y_train.value_counts(normalize=True).sort_index().to_dict()
            train_rows_after_synthetic = int(len(X_train))

            train_step = 6 + max(1, int(tuning_trials)) if tuning_enabled else 6
            self.report_progress(train_step, total_workflow_steps, f"Финальное обучение модели ({model_type})")
            self.check_cancelled()

            model, calibration_status = self._fit_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                class_weight=class_weight_used,
                random_state=random_state,
                calibration_method=calibration_method,
                model_params=model_params_override,
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

            benchmark_step = 7 + max(1, int(tuning_trials)) if tuning_enabled else 7
            self.report_progress(benchmark_step, total_workflow_steps, "Расчет importance и benchmark")
            self.check_cancelled()

            importance_df = self._build_feature_importance(model, X_test, y_test, metric_primary=metric_primary)
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
                "historical_selected_rows": int(source_counts.get("historical_selected_rows", 0)),
                "lpr_selected_rows": int(source_counts.get("lpr_selected_rows", 0)),
                "real_rows_total_before_split": int(source_counts.get("real_rows_total_before_split", 0)),
                "train_rows_before_synthetic": train_rows_before_synthetic,
                "train_rows_after_synthetic": train_rows_after_synthetic,
                "test_rows_real": test_rows_real,
                "synthetic_rows_added": int(synthetic_rows_added),
                "synthetic_method": synthetic_method,
                "synthetic_status": synthetic_status,
                "synthetic_target_total_rows": effective_synthetic_target_total_rows,
                "metrics": metrics,
                "split_protocol": split_protocol,
                "canonical_only": canonical_only,
                "duplicate_policy": "canonical_event_only" if canonical_only else "all_rows",
                "normative_version": normative_payload["normative_version"],
                "normative_hash": get_normative_hash(normative_payload),
                "tuning_enabled": tuning_enabled,
                "tuning_trials_requested": int(tuning_trials),
                "tuning_trials_completed": int(tuning_trials_completed),
                "tuning_metric": metric_primary,
                "best_cv_score": best_cv_score,
                "best_params": best_params,
                "forbidden_feature_check_passed": forbidden_feature_check_passed,
                "forbidden_feature_violations": forbidden_feature_violations,
                "missing_policy": "none_is_unknown",
                "optional_lpr_fields": get_optional_lpr_fields(spec["feature_set"]),
                "missing_pct_top20": missingness.head(20).to_dict(),
                "class_distribution_train": class_distribution_train,
                "class_distribution_test": class_distribution_test,
                "class_distribution_train_after_resampling": resampled_class_distribution,
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

            save_step = 8 + max(1, int(tuning_trials)) if tuning_enabled else 8
            self.report_progress(save_step, total_workflow_steps, "Сохранение артефактов")
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
                    "class_weight_used": class_weight_used,
                    "random_state": random_state,
                    "split_protocol": split_protocol,
                    "canonical_only": canonical_only,
                    "allow_proxy_target": allow_proxy_target,
                    "allow_legacy_random_split": allow_legacy_random_split,
                    "calibration_method": calibration_method,
                    "tuning_enabled": tuning_enabled,
                    "tuning_trials": int(tuning_trials),
                    "synthetic_target_total_rows": effective_synthetic_target_total_rows,
                    "best_params": best_params,
                    "best_cv_score": best_cv_score,
                },
                "dataset_info": {
                    "samples": int(len(raw_X)),
                    "features_count": int(len(spec["feature_order"])),
                    "classes": [str(value) for value in sorted(y.unique().tolist())],
                    "canonical_only": canonical_only,
                    "historical_selected_rows": int(source_counts.get("historical_selected_rows", 0)),
                    "lpr_selected_rows": int(source_counts.get("lpr_selected_rows", 0)),
                    "real_rows_total_before_split": int(source_counts.get("real_rows_total_before_split", 0)),
                    "train_rows_before_synthetic": train_rows_before_synthetic,
                    "train_rows_after_synthetic": train_rows_after_synthetic,
                    "test_rows_real": test_rows_real,
                    "synthetic_rows_added": int(synthetic_rows_added),
                    "synthetic_method": synthetic_method,
                    "synthetic_status": synthetic_status,
                    "synthetic_run_id": synthetic_run_id,
                    "synthetic_target_total_rows": effective_synthetic_target_total_rows,
                    "class_distribution_train": class_distribution_train,
                    "class_distribution_test": class_distribution_test,
                    "class_distribution_train_after_resampling": resampled_class_distribution,
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
                "tuning_enabled": tuning_enabled,
                "tuning_trials_requested": int(tuning_trials),
                "tuning_trials_completed": int(tuning_trials_completed),
                "tuning_metric": metric_primary,
                "best_cv_score": best_cv_score,
                "best_params": best_params,
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
                "synthetic_run_id": synthetic_run_id,
                "synthetic_rows_added": int(synthetic_rows_added),
                "synthetic_method": synthetic_method,
                "synthetic_status": synthetic_status,
            }
            if tree_artifact_name:
                metadata["tree_artifact_path"] = tree_artifact_name
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            done_step = 9 + max(1, int(tuning_trials)) if tuning_enabled else 8
            self.report_progress(done_step, total_workflow_steps, "Обучение завершено")

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
                "class_weight_used": class_weight_used,
                "label_source_policy": [value for value in label_source_policy if value],
                "split_protocol": split_protocol,
                "event_overlap_rate": split_meta.get("event_overlap_rate"),
                "metric_primary": metric_primary,
                "normative_version": normative_payload["normative_version"],
                "calibration_status": calibration_status,
                "tuning_enabled": tuning_enabled,
                "tuning_trials_requested": int(tuning_trials),
                "tuning_trials_completed": int(tuning_trials_completed),
                "tuning_metric": metric_primary,
                "best_cv_score": best_cv_score,
                "best_params": best_params,
                "forbidden_feature_check_passed": forbidden_feature_check_passed,
                "forbidden_feature_violations": forbidden_feature_violations,
                "missing_policy": "none_is_unknown",
                "optional_lpr_fields": get_optional_lpr_fields(spec["feature_set"]),
                "synthetic_rows_added": int(synthetic_rows_added),
                "synthetic_method": synthetic_method,
                "synthetic_status": synthetic_status,
                "synthetic_run_id": synthetic_run_id,
                "synthetic_target_total_rows": effective_synthetic_target_total_rows,
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
                    "historical_selected_rows": int(source_counts.get("historical_selected_rows", 0)),
                    "lpr_selected_rows": int(source_counts.get("lpr_selected_rows", 0)),
                    "real_rows_total_before_split": int(source_counts.get("real_rows_total_before_split", 0)),
                    "train_rows_before_synthetic": train_rows_before_synthetic,
                    "train_rows_after_synthetic": train_rows_after_synthetic,
                    "test_rows_real": test_rows_real,
                    "synthetic_rows_added": int(synthetic_rows_added),
                    "synthetic_method": synthetic_method,
                    "synthetic_status": synthetic_status,
                    "synthetic_target_total_rows": effective_synthetic_target_total_rows,
                    "tuning_enabled": tuning_enabled,
                    "tuning_trials_requested": int(tuning_trials),
                    "tuning_trials_completed": int(tuning_trials_completed),
                    "tuning_metric": metric_primary,
                    "best_cv_score": best_cv_score,
                    "best_params": best_params,
                    "deployment_role": spec["deployment_role"],
                    "offline_only": spec["offline_only"],
                    "can_activate": (not spec["offline_only"]) and semantic_target == SEMANTIC_TARGET_RANK_TZ_VECTOR,
                    "registry_extra": registry_extra,
                },
                warnings=warnings,
            )

        except UseCaseCancelledError:
            logger.info("Training cancelled by user")
            self.status = UseCaseStatus.CANCELLED
            return UseCaseResult(
                success=False,
                message="Обучение остановлено пользователем",
                error="cancelled",
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
            training_store.close()
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

    def _run_optuna_search(
        self,
        *,
        model_type: str,
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        train_rows: pd.DataFrame,
        split_protocol: str,
        spec: dict[str, Any],
        semantic_target: str,
        metric_primary: str,
        synthetic_method: Optional[str],
        synthetic_k_neighbors: int,
        synthetic_m_neighbors: int,
        synthetic_target_total_rows: Optional[int],
        tuning_trials: int,
        random_state: int,
        progress_offset: int,
        total_workflow_steps: int,
    ) -> dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_splits = self._build_tuning_splits(
            train_rows=train_rows,
            y_train=y_train,
            split_protocol=split_protocol,
            random_state=random_state,
        )
        if len(inner_splits) < 2:
            raise ValueError("Недостаточно данных для автоподбора гиперпараметров")

        completed_trials = 0
        total_trials = max(1, int(tuning_trials))

        def objective(trial: optuna.trial.Trial) -> float:
            self.check_cancelled()
            model_params, tuned_class_weight = self._suggest_model_params(
                trial=trial,
                model_type=model_type,
            )
            scores: list[float] = []
            for inner_train_idx, inner_valid_idx in inner_splits:
                self.check_cancelled()
                inner_train_raw = X_train_raw.iloc[inner_train_idx].reset_index(drop=True)
                inner_valid_raw = X_train_raw.iloc[inner_valid_idx].reset_index(drop=True)
                inner_y_train = y_train.iloc[inner_train_idx].reset_index(drop=True)
                inner_y_valid = y_train.iloc[inner_valid_idx].reset_index(drop=True)

                preprocessor_inner, X_inner_train = build_preprocessor_artifact(
                    inner_train_raw,
                    feature_order=spec["feature_order"],
                    feature_set=spec["feature_set"],
                    fill_strategy=spec["default_fill_strategy"],
                    fill_value=spec["default_fill_value"],
                    training_rows=len(inner_train_raw),
                    test_size=0.0,
                    random_state=random_state + trial.number,
                    semantic_target=semantic_target,
                    label_source_policy=["tuning"],
                )
                X_inner_valid = apply_preprocessor_artifact(inner_valid_raw, preprocessor_inner)

                scaled_target_total = self._scale_synthetic_target_total_rows(
                    requested_total=synthetic_target_total_rows,
                    outer_train_rows=len(X_train_raw),
                    inner_train_rows=len(inner_train_raw),
                )
                if synthetic_method:
                    (
                        X_inner_train,
                        inner_y_train,
                        _,
                        _,
                        _,
                    ) = self._apply_synthetic_augmentation(
                        method=synthetic_method,
                        X_train=X_inner_train,
                        y_train=inner_y_train,
                        raw_feature_frame=inner_train_raw,
                        spec=spec,
                        preprocessor_artifact=preprocessor_inner,
                        semantic_target=semantic_target,
                        source_counts={"scope": "optuna_inner"},
                        training_store=None,
                        random_state=random_state + trial.number,
                        k_neighbors=synthetic_k_neighbors,
                        m_neighbors=synthetic_m_neighbors,
                        synthetic_target_total_rows=scaled_target_total,
                        persist_audit=False,
                    )

                model, _ = self._fit_model(
                    model_type=model_type,
                    X_train=X_inner_train,
                    y_train=inner_y_train,
                    class_weight=tuned_class_weight,
                    random_state=random_state + trial.number,
                    calibration_method=None,
                    model_params=model_params,
                )
                y_pred = model.predict(X_inner_valid)
                scores.append(float(self._score_predictions(inner_y_valid, y_pred, metric_primary)))

            trial.set_user_attr("model_params", model_params)
            trial.set_user_attr("class_weight", tuned_class_weight)
            return float(np.mean(scores))

        def on_trial_complete(
            study: optuna.Study,
            frozen_trial: optuna.trial.FrozenTrial,
        ) -> None:
            nonlocal completed_trials
            completed_trials += 1
            self.check_cancelled()
            self.report_progress(
                progress_offset + completed_trials,
                total_workflow_steps,
                f"Автоподбор: попытка {completed_trials}/{total_trials}",
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=total_trials, callbacks=[on_trial_complete], show_progress_bar=False)

        best_trial = study.best_trial
        model_params = dict(best_trial.user_attrs.get("model_params", {}))
        best_class_weight = best_trial.user_attrs.get("class_weight")
        best_params = dict(best_trial.params)
        best_params["class_weight"] = best_class_weight
        return {
            "trials_completed": completed_trials,
            "best_cv_score": float(best_trial.value),
            "best_params": best_params,
            "model_params": model_params,
            "class_weight": best_class_weight,
        }

    def _build_tuning_splits(
        self,
        *,
        train_rows: pd.DataFrame,
        y_train: pd.Series,
        split_protocol: str,
        random_state: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if split_protocol in {SPLIT_PROTOCOL_GROUP_SHUFFLE, SPLIT_PROTOCOL_GROUP_KFOLD}:
            if "event_id" not in train_rows.columns:
                raise ValueError("Для автоподбора в событийном режиме требуется event_id")
            groups = train_rows["event_id"].fillna("__missing__")
            unique_groups = int(groups.nunique())
            if unique_groups < 2:
                raise ValueError("Недостаточно разных событий для автоподбора")
            n_splits = min(3, unique_groups)
            splitter = GroupKFold(n_splits=n_splits)
            return [
                (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
                for train_idx, valid_idx in splitter.split(train_rows, y_train, groups=groups)
            ]

        if split_protocol == SPLIT_PROTOCOL_ROW_RANDOM_LEGACY:
            class_counts = y_train.value_counts()
            if class_counts.empty:
                raise ValueError("Нет данных для автоподбора")
            min_class_count = int(class_counts.min())
            if min_class_count < 2:
                raise ValueError("Недостаточно наблюдений по классам для автоподбора")
            n_splits = min(3, min_class_count)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return [
                (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
                for train_idx, valid_idx in splitter.split(train_rows, y_train)
            ]

        raise ValueError(
            "Автоподбор пока поддерживается только для событийных и legacy-случайных режимов разделения"
        )

    def _suggest_model_params(
        self,
        *,
        trial: optuna.trial.Trial,
        model_type: str,
    ) -> tuple[dict[str, Any], Optional[str]]:
        if model_type == "decision_tree":
            params = {
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            }
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
            return params, class_weight

        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
            class_weight = trial.suggest_categorical(
                "class_weight",
                [None, "balanced", "balanced_subsample"],
            )
            return params, class_weight

        if model_type == "gradient_boosting":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
                "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 10, 12, 16]),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1.0, log=True),
            }
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
            return params, class_weight

        raise ValueError(f"Неизвестный тип модели для автоподбора: {model_type}")

    def _build_model(
        self,
        *,
        model_type: str,
        class_weight: Optional[str],
        random_state: int,
        model_params: Optional[dict[str, Any]] = None,
    ) -> Any:
        params = dict(model_params or {})
        if model_type == "decision_tree":
            defaults = {
                "criterion": "gini",
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": None,
            }
            defaults.update(params)
            return DecisionTreeClassifier(
                random_state=random_state,
                class_weight=class_weight,
                **defaults,
            )

        if model_type == "random_forest":
            defaults = {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "bootstrap": True,
            }
            defaults.update(params)
            return RandomForestClassifier(
                random_state=random_state,
                class_weight=class_weight,
                n_jobs=-1,
                **defaults,
            )

        if model_type == "gradient_boosting":
            defaults = {
                "max_depth": 8,
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
            }
            defaults.update(params)
            return HistGradientBoostingClassifier(
                random_state=random_state,
                class_weight=class_weight,
                **defaults,
            )

        raise ValueError(f"Неизвестный тип модели: {model_type}")

    def _fit_model(
        self,
        *,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: Optional[str],
        random_state: int,
        calibration_method: Optional[str],
        model_params: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        model = self._build_model(
            model_type=model_type,
            class_weight=class_weight,
            random_state=random_state,
            model_params=model_params,
        )
        model.fit(X_train, y_train)
        calibration_status = "not_calibrated"
        if calibration_method in {"sigmoid", "isotonic"}:
            calibrated = CalibratedClassifierCV(model, method=calibration_method, cv=3)
            calibrated.fit(X_train, y_train)
            model = calibrated
            calibration_status = calibration_method
        return model, calibration_status

    def _build_sampler(
        self,
        *,
        method_key: str,
        y_train: pd.Series,
        sampling_strategy: Any,
        random_state: int,
        k_neighbors: int,
        m_neighbors: int,
    ) -> tuple[Any, dict[str, Any]]:
        sampler_params: dict[str, Any] = {
            "random_state": random_state,
            "sampling_strategy": sampling_strategy,
        }
        safe_k = self._safe_neighbor_count(
            y_train=y_train,
            sampling_strategy=sampling_strategy,
            requested_neighbors=k_neighbors,
        )
        if method_key == "smote":
            sampler_params["k_neighbors"] = safe_k
            return SMOTE(**sampler_params), {"k_neighbors": safe_k}
        if method_key == "adasyn":
            sampler_params["n_neighbors"] = safe_k
            return ADASYN(**sampler_params), {"n_neighbors": safe_k}
        if method_key == "borderlinesmote":
            safe_m = max(2, min(int(m_neighbors), max(2, len(y_train) - 1)))
            sampler_params["k_neighbors"] = safe_k
            sampler_params["m_neighbors"] = safe_m
            return BorderlineSMOTE(**sampler_params), {"k_neighbors": safe_k, "m_neighbors": safe_m}
        if method_key == "svmsmote":
            safe_m = max(2, min(int(m_neighbors), max(2, len(y_train) - 1)))
            sampler_params["k_neighbors"] = safe_k
            sampler_params["m_neighbors"] = safe_m
            return SVMSMOTE(**sampler_params), {"k_neighbors": safe_k, "m_neighbors": safe_m}
        raise ValueError(f"Неизвестный метод синтетики: {method_key}")

    def _apply_smotenc_augmentation(
        self,
        *,
        raw_feature_frame: pd.DataFrame,
        y_train: pd.Series,
        preprocessor_artifact: dict[str, Any],
        sampling_strategy: Any,
        random_state: int,
        k_neighbors: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        feature_order = preprocessor_artifact["feature_order"]
        raw = raw_feature_frame[feature_order].copy()
        encoded = pd.DataFrame(index=raw.index)
        categorical_indices: list[int] = []
        inverse_maps: dict[str, list[str]] = {}

        for idx, column in enumerate(feature_order):
            kind = FIELD_SPECS.get(column, {}).get("kind", "numeric")
            if kind == "categorical":
                categorical_indices.append(idx)
                series = raw[column].map(lambda value: "__missing__" if pd.isna(value) else str(value))
                categories = sorted(series.unique().tolist(), key=str)
                mapping = {value: cat_idx for cat_idx, value in enumerate(categories)}
                inverse_maps[column] = categories
                encoded[column] = series.map(mapping).astype(int)
            else:
                fill_value = float(preprocessor_artifact["fill_values"].get(column, 0.0))
                encoded[column] = pd.to_numeric(raw[column], errors="coerce").fillna(fill_value).astype(float)

        if not categorical_indices:
            raise ValueError("SMOTENC недоступен: в выбранном наборе признаков нет категориальных полей.")

        safe_k = self._safe_neighbor_count(
            y_train=y_train,
            sampling_strategy=sampling_strategy,
            requested_neighbors=k_neighbors,
        )
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=safe_k,
        )
        X_resampled_encoded, y_resampled = sampler.fit_resample(encoded, y_train)
        if not isinstance(X_resampled_encoded, pd.DataFrame):
            X_resampled_encoded = pd.DataFrame(X_resampled_encoded, columns=feature_order)
        if not isinstance(y_resampled, pd.Series):
            y_resampled = pd.Series(y_resampled, name=y_train.name)

        decoded = pd.DataFrame(index=X_resampled_encoded.index)
        for column in feature_order:
            kind = FIELD_SPECS.get(column, {}).get("kind", "numeric")
            if kind == "categorical":
                categories = inverse_maps[column]
                codes = pd.to_numeric(X_resampled_encoded[column], errors="coerce").fillna(0).round().astype(int)
                codes = codes.clip(lower=0, upper=max(0, len(categories) - 1))
                decoded[column] = codes.map(lambda code: np.nan if categories[code] == "__missing__" else categories[code])
            else:
                decoded[column] = pd.to_numeric(X_resampled_encoded[column], errors="coerce")

        transformed = apply_preprocessor_artifact(decoded, preprocessor_artifact)
        return transformed.reset_index(drop=True), y_resampled.reset_index(drop=True)

    @staticmethod
    def _sampling_target_classes(y_train: pd.Series, sampling_strategy: Any) -> list[int]:
        class_counts = y_train.value_counts().sort_index()
        if isinstance(sampling_strategy, dict):
            return [int(cls) for cls in sampling_strategy.keys()]
        majority_count = int(class_counts.max()) if not class_counts.empty else 0
        return [int(cls) for cls, count in class_counts.items() if int(count) < majority_count]

    def _safe_neighbor_count(
        self,
        *,
        y_train: pd.Series,
        sampling_strategy: Any,
        requested_neighbors: int,
    ) -> int:
        class_counts = y_train.value_counts().sort_index()
        target_classes = self._sampling_target_classes(y_train, sampling_strategy)
        if not target_classes:
            raise ValueError("Синтетическое расширение не требуется: все классы уже выровнены.")
        min_class_count = min(int(class_counts.loc[cls]) for cls in target_classes if cls in class_counts.index)
        safe_neighbors = min(int(requested_neighbors), max(1, min_class_count - 1))
        if safe_neighbors < 1 or min_class_count < 2:
            raise ValueError(
                "Для выбранного метода синтетики слишком мало примеров редкого класса. "
                "Нужно как минимум 2 строки в каждом расширяемом классе."
            )
        return safe_neighbors

    @staticmethod
    def _humanize_sampler_error(method_key: str, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "expected n_neighbors <= n_samples" in lowered:
            return (
                "Для выбранного метода синтетики слишком мало примеров редкого класса. "
                "Уменьшите k соседей или выберите другой метод."
            )
        if method_key == "svmsmote" and "support vectors are considered as noise" in lowered:
            return (
                "SVM-SMOTE не подходит для текущего набора данных. "
                "Попробуйте SMOTE или BorderlineSMOTE."
            )
        if method_key == "adasyn" and "no samples will be generated" in lowered:
            return (
                "ADASYN не может сгенерировать дополнительные точки для текущего набора данных. "
                "Попробуйте SMOTE или BorderlineSMOTE."
            )
        return message

    def _apply_synthetic_augmentation(
        self,
        *,
        method: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        raw_feature_frame: pd.DataFrame,
        spec: dict[str, Any],
        preprocessor_artifact: dict[str, Any],
        semantic_target: str,
        source_counts: dict[str, Any],
        training_store: Optional[TrainingDataStore],
        random_state: int,
        k_neighbors: int,
        m_neighbors: int,
        synthetic_target_total_rows: Optional[int] = None,
        persist_audit: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, int, str, Optional[str]]:
        """Apply train-only synthetic augmentation and persist the synthetic tail for audit."""
        method_key = (method or "").lower().strip()
        if not method_key or method_key == "none":
            return X_train, y_train, 0, "disabled", None

        sampler_params: dict[str, Any] = {
            "random_state": random_state,
        }
        sampling_strategy = self._resolve_sampling_strategy(
            y_train=y_train,
            requested_total=synthetic_target_total_rows,
        )
        if sampling_strategy is None:
            return X_train, y_train, 0, "train_only", None
        sampler_params["sampling_strategy"] = sampling_strategy
        try:
            if method_key == "smotenc":
                X_resampled, y_resampled = self._apply_smotenc_augmentation(
                    raw_feature_frame=raw_feature_frame,
                    y_train=y_train,
                    preprocessor_artifact=preprocessor_artifact,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors,
                )
            else:
                sampler, adjusted_params = self._build_sampler(
                    method_key=method_key,
                    y_train=y_train,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                    k_neighbors=k_neighbors,
                    m_neighbors=m_neighbors,
                )
                sampler_params.update(adjusted_params)
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                if not isinstance(X_resampled, pd.DataFrame):
                    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                if not isinstance(y_resampled, pd.Series):
                    y_resampled = pd.Series(y_resampled, name=y_train.name)
        except Exception as exc:
            raise ValueError(self._humanize_sampler_error(method_key, exc)) from exc

        synthetic_rows_added = max(0, len(X_resampled) - len(X_train))
        train_run_id = str(uuid.uuid4())[:12]
        if synthetic_rows_added > 0 and persist_audit and training_store is not None:
            training_store.save_synthetic_batch(
                train_run_id=train_run_id,
                generator_method=method_key,
                generator_params=sampler_params,
                base_source_scope=source_counts,
                source_feature_set=spec["feature_set"],
                semantic_target=semantic_target,
                feature_frame=X_resampled.reset_index(drop=True),
                target_series=y_resampled.reset_index(drop=True),
                original_rows_count=len(X_train),
                class_to_rank_map=CLASS_TO_RANK_MAP,
            )

        return (
            X_resampled.reset_index(drop=True),
            y_resampled.reset_index(drop=True),
            synthetic_rows_added,
            "train_only",
            train_run_id if synthetic_rows_added > 0 and persist_audit else None,
        )

    def _resolve_sampling_strategy(
        self,
        *,
        y_train: pd.Series,
        requested_total: Optional[int],
    ) -> Optional[Any]:
        class_counts = y_train.value_counts().sort_index()
        if class_counts.empty:
            return None

        if requested_total is None or int(requested_total) <= 0:
            return "not majority"

        effective_total, _, _ = self._effective_synthetic_target_total_rows(
            y_train=y_train,
            requested_total=requested_total,
        )
        current_total = int(class_counts.sum())
        if effective_total is None or effective_total <= current_total:
            return None

        deficits = {
            int(cls): max(0, int(class_counts.max()) - int(count))
            for cls, count in class_counts.items()
        }
        total_capacity = sum(deficits.values())
        if total_capacity <= 0:
            return None

        extra_needed = min(effective_total - current_total, total_capacity)
        total_deficit = sum(deficits.values())
        allocated = {cls: 0 for cls in deficits}

        for cls, deficit in deficits.items():
            if deficit <= 0:
                continue
            share = int(np.floor(extra_needed * (deficit / total_deficit)))
            allocated[cls] = min(deficit, share)

        remainder = extra_needed - sum(allocated.values())
        if remainder > 0:
            ordered_classes = sorted(deficits, key=lambda cls: deficits[cls], reverse=True)
            while remainder > 0:
                progressed = False
                for cls in ordered_classes:
                    if allocated[cls] < deficits[cls]:
                        allocated[cls] += 1
                        remainder -= 1
                        progressed = True
                        if remainder == 0:
                            break
                if not progressed:
                    break

        target_counts = {
            int(cls): int(class_counts.loc[cls]) + allocated[int(cls)]
            for cls in class_counts.index
            if allocated[int(cls)] > 0
        }
        return target_counts or None

    @staticmethod
    def _effective_synthetic_target_total_rows(
        *,
        y_train: pd.Series,
        requested_total: Optional[int],
    ) -> tuple[Optional[int], bool, int]:
        class_counts = y_train.value_counts()
        if class_counts.empty:
            return requested_total, False, 0
        current_total = int(class_counts.sum())
        majority_count = int(class_counts.max())
        max_total = int(majority_count * len(class_counts))
        if requested_total is None or int(requested_total) <= 0:
            return requested_total, False, max_total
        requested_total = int(requested_total)
        effective_total = min(requested_total, max_total)
        return effective_total, effective_total < requested_total, max_total

    @staticmethod
    def _scale_synthetic_target_total_rows(
        *,
        requested_total: Optional[int],
        outer_train_rows: int,
        inner_train_rows: int,
    ) -> Optional[int]:
        if requested_total is None or requested_total <= 0 or outer_train_rows <= 0:
            return requested_total
        scale = float(inner_train_rows) / float(outer_train_rows)
        scaled = int(round(requested_total * scale))
        return max(inner_train_rows, scaled)

    def _build_feature_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        metric_primary: str,
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
            scoring=self._permutation_scoring(metric_primary),
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

    @staticmethod
    def _permutation_scoring(metric_primary: str) -> str:
        mapping = {
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "f1_weighted": "f1_weighted",
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
        }
        return mapping.get(metric_primary, "f1_macro")

    @staticmethod
    def _score_predictions(y_true: pd.Series, y_pred: np.ndarray, metric_primary: str) -> float:
        if metric_primary == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro"))
        if metric_primary == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted"))
        if metric_primary == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_primary == "precision_macro":
            return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_primary == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return float(f1_score(y_true, y_pred, average="macro"))

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
