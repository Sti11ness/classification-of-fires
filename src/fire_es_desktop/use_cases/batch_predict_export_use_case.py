# src/fire_es_desktop/use_cases/batch_predict_export_use_case.py
"""
BatchPredictExportUseCase — batch rank_tz inference using the production contract.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
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
    PRODUCTION_DEPLOYMENT_ROLE,
    apply_preprocessor_artifact,
    prepare_feature_payload,
)

logger = logging.getLogger("BatchPredictExportUseCase")


class BatchPredictExportUseCase(BaseUseCase):
    """Batch rank_tz inference and export for the production-safe model."""

    def __init__(self, models_path: Path, reports_path: Path):
        super().__init__(
            name="BatchPredictExport",
            description="Пакетный прогноз и экспорт результатов",
        )
        self.models_path = models_path
        self.reports_path = reports_path

    def execute(
        self,
        input_source: str,
        input_path: Optional[Path] = None,
        output_format: str = "excel",
        feature_set: str = "online_tactical",
        top_k: int = 3,
        use_bootstrap: bool = True,
        n_bootstrap: int = 30,
        model_id: Optional[str] = None,
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []

        try:
            self.report_progress(1, 4, "Загрузка входных данных")
            self.check_cancelled()

            if input_source != "excel":
                return UseCaseResult(
                    success=False,
                    message="Для production pipeline поддержан только Excel input",
                    warnings=warnings,
                )
            if not input_path or not input_path.exists():
                return UseCaseResult(
                    success=False,
                    message="Входной Excel файл не найден",
                    warnings=warnings,
                )

            df_input = pd.read_excel(input_path)

            self.report_progress(2, 4, "Загрузка production-модели")
            self.check_cancelled()

            model_info = self._resolve_model_info(model_id=model_id)
            if not model_info:
                return UseCaseResult(
                    success=False,
                    message="Нет активной production-safe модели rank_tz для пакетного прогноза",
                    warnings=warnings,
                )

            artifact_path = self.models_path / model_info["artifact_path"]
            preprocessor_path = self.models_path / model_info["preprocessor_path"]
            if not artifact_path.exists() or not preprocessor_path.exists():
                return UseCaseResult(
                    success=False,
                    message="Не найдены артефакты активной production модели",
                    warnings=warnings,
                )

            model = joblib.load(artifact_path)
            with open(preprocessor_path, "r", encoding="utf-8") as f:
                preprocessor_artifact = json.load(f)

            missing_input_columns = [
                field["name"]
                for field in preprocessor_artifact.get("input_schema", [])
                if field["name"] not in df_input.columns
            ]
            if missing_input_columns:
                warnings.append(
                    "Во входном файле отсутствуют колонки, они будут заполнены train-time imputer: "
                    + ", ".join(missing_input_columns)
                )

            self.report_progress(3, 4, f"Прогноз ({len(df_input)} записей)")
            self.check_cancelled()

            engineered = prepare_feature_payload(
                df_input,
                feature_set=model_info.get("feature_set", preprocessor_artifact.get("feature_set", feature_set)),
                availability_stage=model_info.get("availability_stage"),
            )
            X = apply_preprocessor_artifact(engineered, preprocessor_artifact)

            rows: list[dict[str, Any]] = []
            for index in range(len(X)):
                self.check_cancelled()
                if (index + 1) % 100 == 0 or (index + 1) == len(X):
                    self.report_progress(3, 4, f"Прогноз: {index + 1}/{len(X)}")

                pred_result = predict_with_confidence(
                    model,
                    X.iloc[[index]],
                    top_k=top_k,
                    use_bootstrap=use_bootstrap,
                    n_bootstrap=n_bootstrap,
                )
                pred_df = pred_result.get("predictions")
                if pred_df is None or pred_df.empty:
                    raise RuntimeError(f"Пустой результат модели для строки {index}")

                pred_row = pred_df.iloc[0].to_dict()
                row_result = {
                    "row_index": index,
                    "predicted_rank": pred_row.get("predicted_rank"),
                    "predicted_rank_name": RANK_NAMES.get(
                        pred_row.get("predicted_rank"),
                        str(pred_row.get("predicted_rank")),
                    ),
                    "model_id": model_info["model_id"],
                    "model_name": model_info.get("name", ""),
                    "deployment_role": model_info.get("deployment_role"),
                    "semantic_target": model_info.get("semantic_target"),
                    "availability_stage": model_info.get("availability_stage"),
                    "split_protocol": model_info.get("split_protocol"),
                    "event_overlap_rate": model_info.get("event_overlap_rate", 0.0),
                }
                for k in range(1, top_k + 1):
                    rank_key = f"top{k}_rank"
                    if rank_key not in pred_row:
                        continue
                    row_result[rank_key] = pred_row.get(rank_key)
                    row_result[f"top{k}_rank_name"] = pred_row.get(
                        f"top{k}_rank_name",
                        str(pred_row.get(rank_key)),
                    )
                    row_result[f"top{k}_prob"] = pred_row.get(f"top{k}_prob", 0.0)

                if use_bootstrap:
                    row_result["confidence"] = pred_row.get("mean_prob_class", 0.0)
                    row_result["uncertainty"] = pred_row.get("std_prob_class", 0.0)

                rows.append(row_result)

            df_predictions = pd.DataFrame(rows)
            df_result = pd.concat([df_input.reset_index(drop=True), df_predictions], axis=1)

            self.report_progress(4, 4, "Экспорт результатов")
            self.check_cancelled()

            output_dir = self.reports_path / "tables"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_format == "excel":
                output_path = output_dir / f"batch_prediction_{timestamp}.xlsx"
                df_result.to_excel(output_path, index=False, engine="openpyxl")
            elif output_format == "csv":
                output_path = output_dir / f"batch_prediction_{timestamp}.csv"
                df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
            else:
                return UseCaseResult(
                    success=False,
                    message=f"Неизвестный формат экспорта: {output_format}",
                    warnings=warnings,
                )

            self.status = UseCaseStatus.COMPLETED
            return UseCaseResult(
                success=True,
                message=f"Пакетный прогноз завершён: {len(rows)} записей",
                data={
                    "output_path": str(output_path),
                    "predictions_count": len(rows),
                    "input_count": len(df_input),
                    "warnings": warnings,
                    "model_info": {
                        "model_id": model_info["model_id"],
                        "model_name": model_info.get("name", ""),
                        "deployment_role": model_info.get("deployment_role"),
                    },
                },
                warnings=warnings,
            )

        except Exception as e:
            logger.error("Batch predict failed: %s", e, exc_info=True)
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(
                success=False,
                message=f"Ошибка пакетного прогноза: {str(e)}",
                error=str(e),
                warnings=warnings,
            )

    def _resolve_model_info(self, model_id: Optional[str]) -> Optional[dict[str, Any]]:
        from ..infra import ModelRegistry

        registry = ModelRegistry(self.models_path)
        if model_id:
            model_info = registry.get_model_info(model_id)
            if not model_info:
                return None
            if not registry.is_model_production_safe(model_info):
                return None
            if not model_info.get("preprocessor_path"):
                return None
            return model_info

        active = registry.get_active_model_info()
        if active and registry.is_model_production_safe(active):
            return active
        return None
