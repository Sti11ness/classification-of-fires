# src/fire_es_desktop/use_cases/batch_predict_export_use_case.py
"""
BatchPredictExportUseCase — пакетный прогноз и экспорт результатов.

Согласно spec_first.md раздел 8.2:
- вход: Excel/выборка из БД
- выход: Excel/CSV с прогнозами + доверительность + версия модели
- сохранение в reports/tables
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.model_train import FEATURE_SETS
from fire_es.predict import RANK_NAMES, predict_with_confidence

logger = logging.getLogger("BatchPredictExportUseCase")


class BatchPredictExportUseCase(BaseUseCase):
    """
    Сценарий пакетного прогноза и экспорта.

    Шаги:
    1. Загрузка входных данных (Excel/БД)
    2. Загрузка активной модели
    3. Прогноз по всем записям
    4. Экспорт результатов (Excel/CSV)
    """

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
        """
        Выполнить пакетный прогноз и экспорт.

        Args:
            input_source: Источник данных ("excel" или "database").
            input_path: Путь к Excel файлу (если input_source="excel").
            output_format: Формат экспорта ("excel" или "csv").
            feature_set: Набор признаков для модели.
            top_k: Количество вариантов прогноза.
            use_bootstrap: Использовать ли бутстрап.
            n_bootstrap: Количество бутстрап-выборок.
            model_id: ID модели (None = активная модель).

        Returns:
            Результат выполнения.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: List[str] = []

        try:
            import joblib

            # Шаг 1: Загрузка входных данных
            self.report_progress(1, 4, "Загрузка входных данных")
            self.check_cancelled()

            if input_source == "excel":
                if not input_path:
                    return UseCaseResult(
                        success=False,
                        message="Не указан путь к Excel файлу",
                        warnings=warnings,
                    )
                if not input_path.exists():
                    return UseCaseResult(
                        success=False,
                        message=f"Файл не найден: {input_path}",
                        warnings=warnings,
                    )
                df_input = pd.read_excel(input_path)
                logger.info("Загружено %s записей из Excel", len(df_input))
            elif input_source == "database":
                return UseCaseResult(
                    success=False,
                    message="Загрузка из БД пока не реализована для пакетного прогноза",
                    warnings=warnings,
                )
            else:
                return UseCaseResult(
                    success=False,
                    message=f"Неизвестный источник: {input_source}",
                    warnings=warnings,
                )

            # Шаг 2: Загрузка модели
            self.report_progress(2, 4, "Загрузка модели")
            self.check_cancelled()

            model_path, metadata = self._resolve_model(model_id=model_id, warnings=warnings)
            if model_path is None or not model_path.exists():
                return UseCaseResult(
                    success=False,
                    message="Модель не найдена",
                    warnings=warnings,
                )

            model = joblib.load(model_path)
            model_id_value = model_path.stem.replace("model_", "", 1)
            model_info = {
                "model_id": model_id_value,
                "model_name": metadata.get("model_name", model_path.stem),
                "model_path": str(model_path),
                "model_type": type(model).__name__,
            }

            # Шаг 3: Подготовка и прогноз
            self.report_progress(3, 4, f"Прогноз ({len(df_input)} записей)")
            self.check_cancelled()

            required_features = metadata.get("features") or FEATURE_SETS.get(feature_set, [])
            if not required_features:
                return UseCaseResult(
                    success=False,
                    message=(
                        f"Не удалось определить признаки для feature_set='{feature_set}'. "
                        "Проверьте метаданные модели."
                    ),
                    warnings=warnings,
                )

            missing_cols = [c for c in required_features if c not in df_input.columns]
            if missing_cols:
                warnings.append(f"Отсутствуют колонки: {missing_cols}")
                for col in missing_cols:
                    df_input[col] = np.nan

            X = df_input[required_features].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            if feature_set in {"online_dispatch", "online_early", "online_tactical"}:
                X = X.fillna(-1)
            else:
                X = X.fillna(0)

            predictions: List[Dict[str, Any]] = []
            n_records = len(X)
            for i in range(n_records):
                self.check_cancelled()
                if (i + 1) % 100 == 0 or (i + 1) == n_records:
                    self.report_progress(3, 4, f"Прогноз: {i + 1}/{n_records}")

                X_row = X.iloc[[i]]
                try:
                    pred_result = predict_with_confidence(
                        model,
                        X_row,
                        top_k=top_k,
                        use_bootstrap=use_bootstrap,
                        n_bootstrap=n_bootstrap,
                    )
                    pred_df = pred_result.get("predictions")
                    if pred_df is None or pred_df.empty:
                        raise RuntimeError("пустой ответ модели")
                    pred_data = pred_df.iloc[0].to_dict()

                    row_result: Dict[str, Any] = {
                        "row_index": i,
                        "predicted_rank": pred_data.get("predicted_rank"),
                        "predicted_rank_name": RANK_NAMES.get(
                            pred_data.get("predicted_rank"),
                            str(pred_data.get("predicted_rank")),
                        ),
                    }

                    for k in range(1, top_k + 1):
                        rank_key = f"top{k}_rank"
                        if rank_key not in pred_data:
                            continue
                        row_result[rank_key] = pred_data.get(rank_key)
                        row_result[f"top{k}_rank_name"] = pred_data.get(
                            f"top{k}_rank_name",
                            str(pred_data.get(rank_key)),
                        )
                        row_result[f"top{k}_prob"] = pred_data.get(f"top{k}_prob", 0.0)

                    if use_bootstrap:
                        row_result["confidence"] = pred_data.get("mean_prob_class", 0.0)
                        row_result["uncertainty"] = pred_data.get("std_prob_class", 0.0)

                    row_result.update(model_info)
                    predictions.append(row_result)
                except Exception as e:
                    err_text = f"Ошибка прогноза для строки {i}: {e}"
                    logger.error(err_text)
                    warnings.append(err_text)
                    predictions.append({"row_index": i, "error": str(e), **model_info})

            df_predictions = pd.DataFrame(predictions)
            df_result = pd.concat([df_input.reset_index(drop=True), df_predictions], axis=1)

            # Шаг 4: Экспорт
            self.report_progress(4, 4, "Экспорт результатов")
            self.check_cancelled()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.reports_path / "tables"
            output_dir.mkdir(parents=True, exist_ok=True)

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
                message=f"Пакетный прогноз завершён: {len(predictions)} записей",
                data={
                    "output_path": str(output_path),
                    "predictions_count": len(predictions),
                    "input_count": n_records,
                    "warnings": warnings,
                    "model_info": model_info,
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

    def _resolve_model(
        self,
        model_id: Optional[str],
        warnings: List[str],
    ) -> Tuple[Optional[Path], Dict[str, Any]]:
        """Найти модель и метаданные (active -> latest fallback)."""
        model_path: Optional[Path] = None
        metadata: Dict[str, Any] = {}

        if model_id:
            candidate = self.models_path / f"model_{model_id}.joblib"
            if candidate.exists():
                model_path = candidate
        else:
            try:
                from ..infra import ModelRegistry

                registry = ModelRegistry(self.models_path)
                active = registry.get_active_model_info()
                if active:
                    artifact_rel = active.get("artifact_path")
                    if artifact_rel:
                        candidate = self.models_path / artifact_rel
                    else:
                        candidate = self.models_path / f"model_{active['model_id']}.joblib"
                    if candidate.exists():
                        model_path = candidate
            except Exception as e:
                warnings.append(f"Не удалось прочитать registry моделей: {e}")

            if model_path is None:
                model_files = sorted(
                    self.models_path.glob("model_*.joblib"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if model_files:
                    model_path = model_files[0]

        if model_path:
            metadata_path = model_path.with_name(f"{model_path.stem}_meta.json")
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception as e:
                    warnings.append(f"Не удалось загрузить метаданные модели: {e}")

        return model_path, metadata
