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
from fire_es.rank_tz_contract import PRODUCTION_DEPLOYMENT_ROLE, apply_preprocessor_artifact

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
            self.report_progress(1, 3, "Загрузка production-модели")
            self.check_cancelled()

            if input_data is None:
                return UseCaseResult(
                    success=False,
                    message="Нет входных данных для прогноза",
                    warnings=warnings,
                )

            model_info = self._resolve_model_info(model_id=model_id)
            if not model_info:
                return UseCaseResult(
                    success=False,
                    message="Нет активной production-safe модели rank_tz для ЛПР",
                    warnings=warnings,
                )

            artifact_path = self.models_path / model_info["artifact_path"]
            preprocessor_rel = model_info.get("preprocessor_path")
            if not preprocessor_rel:
                return UseCaseResult(
                    success=False,
                    message=(
                        "У активной модели отсутствует preprocessor artifact. "
                        "Переобучите модель через новый ML pipeline."
                    ),
                    warnings=warnings,
                )

            preprocessor_path = self.models_path / preprocessor_rel
            if not artifact_path.exists() or not preprocessor_path.exists():
                return UseCaseResult(
                    success=False,
                    message="Не найдены артефакты активной production модели",
                    warnings=warnings,
                )

            model = joblib.load(artifact_path)
            with open(preprocessor_path, "r", encoding="utf-8") as f:
                preprocessor_artifact = json.load(f)

            self.report_progress(2, 3, "Подготовка входных данных")
            self.check_cancelled()

            X = apply_preprocessor_artifact(input_data, preprocessor_artifact)

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
                    "input_schema": preprocessor_artifact.get("input_schema", []),
                    "input_data": input_data,
                    "top_k_ranks": top_k_ranks,
                    "confidence": confidence,
                    "entropy": entropy,
                    "induced_rank_p50": induced_rank,
                    "all_probabilities": all_probabilities,
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
        return pd.DataFrame(
            {
                "rank": ["1", "1-бис", "2", "3", "4", "5"],
                "rank_name": [
                    "Ранг 1",
                    "Ранг 1-бис",
                    "Ранг 2",
                    "Ранг 3",
                    "Ранг 4",
                    "Ранг 5",
                ],
                "equipment_count": [2, 2, 3, 5, 8, 12],
                "description": [
                    "Минимальный",
                    "Минимальный усиленный",
                    "Средний",
                    "Повышенный",
                    "Высокий",
                    "Максимальный",
                ],
            }
        )

    def _resolve_model_info(self, model_id: Optional[str]) -> Optional[dict[str, Any]]:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        if model_id:
            model_info = registry.get_model_info(model_id)
            if not model_info:
                return None
            if model_info.get("target") != "rank_tz":
                return None
            if model_info.get("deployment_role") != PRODUCTION_DEPLOYMENT_ROLE:
                return None
            if model_info.get("offline_only"):
                return None
            return model_info

        return registry.get_active_model_for_role(
            target="rank_tz",
            deployment_role=PRODUCTION_DEPLOYMENT_ROLE,
        )
