# src/fire_es_desktop/use_cases/predict_use_case.py
"""
PredictUseCase — прогноз ранга/ресурсов для ЛПР.

Согласно spec_first.md раздел 5.2 и spec_second.md раздел 11.6:
- Прогноз ранга как распределение вероятностей (Top-K)
- Прогноз ресурсов сценариями (P20/P50/P80) с доверительностью
- Нормативная таблица
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.predict import (
    predict_with_confidence,
    RANK_NAMES
)

logger = logging.getLogger("PredictUseCase")


class PredictUseCase(BaseUseCase):
    """
    Сценарий прогнозирования для ЛПР.

    Шаги:
    1. Загрузка активной модели
    2. Подготовка входных данных
    3. Прогноз с доверительностью
    4. Формирование результата
    """

    def __init__(self, models_path: Path):
        super().__init__(
            name="Predict",
            description="Прогноз ранга/ресурсов пожара"
        )
        self.models_path = models_path

    def execute(
        self,
        model_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> UseCaseResult:
        """
        Выполнить прогноз.

        Args:
            model_id: ID модели (None = активная модель).
            input_data: Входные данные (разделы 1-2).
            top_k: Количество вариантов прогноза.

        Returns:
            Результат прогноза.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings = []

        try:
            # Шаг 1: Загрузка модели
            self.report_progress(1, 3, "Загрузка модели")
            self.check_cancelled()

            import joblib

            if model_id:
                model_path = self.models_path / f"model_{model_id}.joblib"
                metadata_path = self.models_path / f"model_{model_id}_meta.json"
            else:
                # Найти последнюю модель
                model_files = sorted(
                    self.models_path.glob("model_*.joblib"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if not model_files:
                    return UseCaseResult(
                        success=False,
                        message="Нет обученных моделей",
                        warnings=warnings
                    )
                model_path = model_files[0]
                model_id = model_path.stem.replace("model_", "")
                metadata_path = self.models_path / f"model_{model_id}_meta.json"

            if not model_path.exists():
                return UseCaseResult(
                    success=False,
                    message=f"Модель не найдена: {model_id}",
                    warnings=warnings
                )

            # Загрузить модель и метаданные
            model = joblib.load(model_path)

            import json
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            logger.info(f"Loaded model: {model_id}")

            # Шаг 2: Подготовка данных
            self.report_progress(2, 3, "Подготовка данных")
            self.check_cancelled()

            if input_data is None:
                return UseCaseResult(
                    success=False,
                    message="Нет входных данных для прогноза",
                    warnings=warnings
                )

            # Получить список признаков модели
            feature_names = metadata.get("features", [])
            if not feature_names:
                feature_names = list(input_data.keys())
                warnings.append(
                    "В метаданных модели не найден список признаков, "
                    "используются входные поля формы"
                )

            # Создать DataFrame с признаками
            input_df = pd.DataFrame([input_data])

            # Добавить недостающие признаки (заполнить нулями)
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Оставить только нужные признаки
            X = input_df[feature_names]

            logger.info(f"Input features: {len(feature_names)}")

            # Шаг 3: Прогноз
            self.report_progress(3, 3, "Прогнозирование")
            self.check_cancelled()

            # Прогноз с доверительностью
            prediction = predict_with_confidence(
                model,
                X,
                top_k=top_k,
                use_bootstrap=True,
                n_bootstrap=10  # Бутстрап для доверительности
            )

            pred_df = prediction.get("predictions")
            if pred_df is None or pred_df.empty:
                return UseCaseResult(
                    success=False,
                    message="Модель не вернула прогноз",
                    warnings=warnings
                )
            pred_row = pred_df.iloc[0]

            # Top-K вероятностей рангов из формата predict_with_confidence
            top_k_ranks = []
            rank_probs = {}
            for idx in range(1, top_k + 1):
                rank_key = f"top{idx}_rank"
                prob_key = f"top{idx}_prob"
                if rank_key not in pred_row:
                    continue
                rank_value = pred_row.get(rank_key)
                prob_value = float(pred_row.get(prob_key, 0.0))
                rank_probs[str(rank_value)] = prob_value
                top_k_ranks.append(
                    {
                        "rank": rank_value,
                        "rank_name": RANK_NAMES.get(rank_value, str(rank_value)),
                        "probability": prob_value,
                    }
                )

            if not top_k_ranks and "predicted_rank" in pred_row:
                rank_value = pred_row.get("predicted_rank")
                prob_value = float(pred_row.get("top1_prob", 0.0))
                top_k_ranks = [
                    {
                        "rank": rank_value,
                        "rank_name": RANK_NAMES.get(rank_value, str(rank_value)),
                        "probability": prob_value,
                    }
                ]
                rank_probs[str(rank_value)] = prob_value

            # Доверительность
            confidence = float(
                pred_row.get("mean_prob_class", pred_row.get("top1_prob", 0.0))
            )
            entropy = float(pred_row.get("entropy", 0.0))

            # Индуцированный ранг (если есть прогноз ресурсов)
            induced_rank = pred_row.get("induced_rank_p50")
            if induced_rank is None and top_k_ranks:
                induced_rank = top_k_ranks[0]["rank"]

            # Сформировать результат
            result = {
                "model_id": model_id,
                "model_name": metadata.get("model_name", ""),
                "input_data": input_data,
                "top_k_ranks": top_k_ranks,
                "confidence": confidence,
                "entropy": entropy,
                "induced_rank_p50": induced_rank,
                "all_probabilities": rank_probs
            }

            logger.info(f"Prediction complete. Top rank: {top_k_ranks[0] if top_k_ranks else 'N/A'}")

            return UseCaseResult(
                success=True,
                message="Прогноз выполнен",
                data=result,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            self.status = UseCaseStatus.FAILED

            return UseCaseResult(
                success=False,
                message=f"Ошибка прогноза: {str(e)}",
                error=str(e),
                warnings=warnings
            )

    def get_normative_table(self) -> pd.DataFrame:
        """
        Получить нормативную таблицу рангов.

        Returns:
            DataFrame с нормативами.
        """
        # Упрощённая нормативная таблица
        # В реальности должна загружаться из БД/файла
        return pd.DataFrame({
            'rank': ['1', '1-бис', '2', '3', '4', '5'],
            'rank_name': [
                'Ранг 1', 'Ранг 1-бис', 'Ранг 2',
                'Ранг 3', 'Ранг 4', 'Ранг 5'
            ],
            'equipment_count': [2, 2, 3, 5, 8, 12],
            'description': [
                'Минимальный',
                'Минимальный усиленный',
                'Средний',
                'Повышенный',
                'Высокий',
                'Максимальный'
            ]
        })
