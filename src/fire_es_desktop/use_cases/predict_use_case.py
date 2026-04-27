# src/fire_es_desktop/use_cases/predict_use_case.py
"""
PredictUseCase — production-safe rank_tz inference for LPR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Import from domain layer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.predict import RANK_NAMES, predict_with_confidence
from fire_es.rank_tz_contract import (
    AVAILABILITY_STAGE_RETROSPECTIVE,
    DEFAULT_LPR_FEATURE_SET,
    PRODUCTION_DEPLOYMENT_ROLE,
    SEMANTIC_TARGET_RANK_TZ_VECTOR,
    get_feature_set_spec,
    get_input_schema,
    prepare_feature_payload,
    validate_stage_input_requirements,
    apply_preprocessor_artifact,
)
from fire_es.normatives import get_normative_rank_table

logger = logging.getLogger("PredictUseCase")


class PredictUseCase(BaseUseCase):
    """Production prediction flow for rank_tz."""

    def __init__(self, models_path: Path):
        super().__init__(
            name="Predict",
            description="Прогноз ранга/ресурсов пожара",
        )
        self.models_path = models_path

    def execute(
        self,
        model_id: Optional[str] = None,
        input_data: Optional[dict[str, Any]] = None,
        top_k: int = 3,
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []

        try:
            self.report_progress(1, 3, "Загрузка рабочей модели")
            self.check_cancelled()

            if input_data is None:
                return UseCaseResult(
                    success=False,
                    message="Нет входных данных для прогноза",
                    warnings=warnings,
                )

            model_info, model_error_message = self._resolve_model_info(model_id=model_id)
            if not model_info:
                return UseCaseResult(
                    success=False,
                    message=model_error_message or "Рабочая модель для прогноза ЛПР не выбрана",
                    warnings=warnings,
                )

            artifact_path = self.models_path / model_info["artifact_path"]
            preprocessor_rel = model_info.get("preprocessor_path")
            if not preprocessor_rel:
                return UseCaseResult(
                    success=False,
                    message=(
                        "У выбранной модели отсутствует файл подготовки признаков. "
                        "Переобучите модель заново."
                    ),
                    warnings=warnings,
                )

            preprocessor_path = self.models_path / preprocessor_rel
            if not artifact_path.exists() or not preprocessor_path.exists():
                return UseCaseResult(
                    success=False,
                    message="Не найдены файлы рабочей модели",
                    warnings=warnings,
                )

            model = joblib.load(artifact_path)
            with open(preprocessor_path, "r", encoding="utf-8") as f:
                preprocessor_artifact = json.load(f)

            self.report_progress(2, 3, "Подготовка входных данных")
            self.check_cancelled()

            stage_errors = validate_stage_input_requirements(
                input_data,
                availability_stage=model_info.get("availability_stage"),
            )
            if stage_errors:
                return UseCaseResult(
                    success=False,
                    message=(
                        "Недостаточно данных для выбранной стадии: "
                        + ", ".join(stage_errors)
                    ),
                    warnings=warnings,
                )

            engineered = prepare_feature_payload(
                input_data,
                feature_set=model_info.get("feature_set", preprocessor_artifact.get("feature_set", DEFAULT_LPR_FEATURE_SET)),
                availability_stage=model_info.get("availability_stage"),
            )
            X = apply_preprocessor_artifact(engineered, preprocessor_artifact)

            self.report_progress(3, 3, "Прогнозирование")
            self.check_cancelled()

            prediction = predict_with_confidence(
                model,
                X,
                top_k=top_k,
                use_bootstrap=True,
                n_bootstrap=10,
            )

            pred_df = prediction.get("predictions")
            if pred_df is None or pred_df.empty:
                return UseCaseResult(
                    success=False,
                    message="Модель не вернула прогноз",
                    warnings=warnings,
                )

            pred_row = pred_df.iloc[0]
            top_k_ranks = []
            all_probabilities = {}
            for index in range(1, top_k + 1):
                rank_key = f"top{index}_rank"
                prob_key = f"top{index}_prob"
                if rank_key not in pred_row:
                    continue
                rank_value = pred_row.get(rank_key)
                probability = float(pred_row.get(prob_key, 0.0))
                all_probabilities[str(rank_value)] = probability
                top_k_ranks.append(
                    {
                        "rank": rank_value,
                        "rank_name": RANK_NAMES.get(rank_value, str(rank_value)),
                        "probability": probability,
                    }
                )

            confidence = float(pred_row.get("mean_prob_class", pred_row.get("top1_prob", 0.0)))
            entropy = float(pred_row.get("entropy", 0.0))
            induced_rank = pred_row.get("predicted_rank")

            return UseCaseResult(
                success=True,
                message="Прогноз выполнен",
                data={
                    "model_id": model_info["model_id"],
                    "model_name": model_info.get("name", ""),
                    "deployment_role": model_info.get("deployment_role"),
                    "feature_set": model_info.get("feature_set"),
                    "semantic_target": model_info.get("semantic_target"),
                    "label_source_policy": model_info.get("label_source_policy", []),
                    "availability_stage": model_info.get("availability_stage"),
                    "split_protocol": model_info.get("split_protocol"),
                    "event_overlap_rate": model_info.get("event_overlap_rate", 0.0),
                    "metric_primary": model_info.get("metric_primary"),
                    "normative_version": model_info.get("normative_version"),
                    "calibration_status": model_info.get("calibration_status", "not_calibrated"),
                    "input_schema": preprocessor_artifact.get("input_schema", []),
                    "input_data": input_data,
                    "top_k_ranks": top_k_ranks,
                    "confidence": confidence,
                    "entropy": entropy,
                    "induced_rank_p50": induced_rank,
                    "all_probabilities": all_probabilities,
                    "warnings": self._build_model_warnings(model_info, confidence, entropy),
                },
                warnings=warnings,
            )

        except Exception as e:
            logger.error("Prediction failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(
                success=False,
                message=f"Ошибка прогноза: {str(e)}",
                error=str(e),
                warnings=warnings,
            )

    def get_normative_table(self) -> pd.DataFrame:
        """Return a lightweight rank reference table for the UI."""
        table = get_normative_rank_table()
        return table.rename(
            columns={
                "label": "rank_name",
                "display_name": "rank_display_name",
                "min_equipment_count": "equipment_count",
            }
        )

    def get_input_contract(self, model_id: Optional[str] = None) -> dict[str, Any]:
        """Return the input schema contract for LPR UI construction."""
        model_info, model_error_message = self._resolve_model_info(model_id=model_id)
        if not model_info:
            spec = get_feature_set_spec(DEFAULT_LPR_FEATURE_SET)
            active_info = self._resolve_raw_active_model_info()
            if active_info and not self._is_model_safe(active_info):
                model_status = "Текущая активная модель не подходит для прогноза ЛПР"
                warnings = self._build_activation_reasons(active_info)
            else:
                model_status = model_error_message or "Рабочая модель для прогноза ЛПР не выбрана"
                warnings = ["Используется базовая форма ввода до выбора рабочей модели"]
            return {
                "model_id": None,
                "model_status": model_status,
                "feature_set": spec["feature_set"],
                "availability_stage": spec["availability_stage"],
                "deployment_role": spec["deployment_role"],
                "semantic_target": spec["semantic_target_default"],
                "input_schema": get_input_schema(spec["feature_set"]),
                "feature_order": spec["feature_order"],
                "warnings": warnings,
                "fallback": True,
            }
        input_schema = model_info.get("input_schema")
        if not input_schema:
            preprocessor_rel = model_info.get("preprocessor_path")
            if preprocessor_rel:
                with open(self.models_path / preprocessor_rel, "r", encoding="utf-8") as fh:
                    preprocessor = json.load(fh)
                input_schema = preprocessor.get("input_schema")
            else:
                input_schema = get_input_schema(model_info.get("feature_set", DEFAULT_LPR_FEATURE_SET))
        warnings = []
        if model_info.get("feature_set") == "online_tactical" or model_info.get("legacy_alias"):
            warnings.append("Активная модель использует устаревшую схему признаков")
        return {
            "model_id": model_info.get("model_id"),
            "model_status": "Для прогноза используется рабочая модель",
            "feature_set": model_info.get("feature_set"),
            "availability_stage": model_info.get("availability_stage"),
            "deployment_role": model_info.get("deployment_role"),
            "semantic_target": model_info.get("semantic_target"),
            "input_schema": input_schema,
            "feature_order": model_info.get("features", []),
            "split_protocol": model_info.get("split_protocol"),
            "normative_version": model_info.get("normative_version"),
            "warnings": warnings,
            "fallback": False,
        }

    def _resolve_model_info(self, model_id: Optional[str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        if model_id:
            model_info = registry.get_model_info(model_id)
            if not model_info:
                return None, "Выбранная модель не найдена"
            if registry.is_model_production_safe(model_info):
                return model_info, None
            reasons = registry.get_production_unsafe_reasons(model_info)
            return None, "Выбранная модель не подходит для прогноза ЛПР: " + "; ".join(reasons)

        active = registry.get_active_model_info()
        if active and registry.is_model_production_safe(active):
            return active, None
        if active:
            reasons = registry.get_production_unsafe_reasons(active)
            return None, "Текущая активная модель не подходит для прогноза ЛПР: " + "; ".join(reasons)
        return None, "Рабочая модель для прогноза ЛПР не выбрана"

    def _resolve_raw_active_model_info(self) -> Optional[dict[str, Any]]:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        return registry.get_active_model_info()

    def _is_model_safe(self, model_info: dict[str, Any]) -> bool:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        return registry.is_model_production_safe(model_info)

    def _build_activation_reasons(self, model_info: dict[str, Any]) -> list[str]:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        return registry.get_production_unsafe_reasons(model_info)

    def _build_model_warnings(
        self,
        model_info: dict[str, Any],
        confidence: float,
        entropy: float,
    ) -> list[str]:
        warnings: list[str] = []
        if model_info.get("semantic_target") != SEMANTIC_TARGET_RANK_TZ_VECTOR:
            warnings.append("Модель использует другой тип целевой разметки")
        if model_info.get("availability_stage") == AVAILABILITY_STAGE_RETROSPECTIVE:
            warnings.append("Модель относится к архивному режиму и не подходит для ЛПР")
        if model_info.get("feature_set") == "online_tactical" or model_info.get("legacy_alias"):
            warnings.append("Используется устаревшая схема признаков")
        if model_info.get("event_overlap_rate", 0.0) != 0.0:
            warnings.append("При проверке качества пересекались одни и те же события")
        if model_info.get("split_protocol") == "row_random_legacy":
            warnings.append("Использована устаревшая схема разделения выборки")
        if confidence < 0.5:
            warnings.append("Низкая уверенность прогноза")
        if entropy > 1.2:
            warnings.append("Высокая неопределенность прогноза")
        if model_info.get("calibration_status") in (None, "not_calibrated"):
            warnings.append("Вероятности не калиброваны")
        return warnings
