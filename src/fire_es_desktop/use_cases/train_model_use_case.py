# src/fire_es_desktop/use_cases/train_model_use_case.py
"""
TrainModelUseCase — обучение модели прогнозирования ранга.

Согласно spec_first.md раздел 5.1 и spec_second.md раздел 11.5:
- Выбор цели (ранг или показатель)
- Выбор признаков
- Обучение дерева/леса
- Оценка качества
- Сохранение артефактов и регистрация модели
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.db import DatabaseManager
from fire_es.model_train import train_classifier, train_random_forest_classifier, visualize_tree
from fire_es.predict import RANK_CLASSES
from fire_es.ranking import calculate_rank_by_count

logger = logging.getLogger("TrainModelUseCase")


class TrainModelUseCase(BaseUseCase):
    """
    Сценарий обучения модели.

    Шаги:
    1. Загрузка данных из БД
    2. Подготовка признаков
    3. Обучение модели
    4. Сохранение артефактов
    5. Завершение
    """

    _DEFAULT_FEATURES: Dict[str, List[str]] = {
        "basic": [
            "building_floors",
            "fire_floor",
            "distance_to_station",
            "fatalities",
            "injuries",
            "direct_damage",
        ],
        "extended": [
            "region_code",
            "settlement_type_code",
            "fire_protection_code",
            "enterprise_type_code",
            "building_floors",
            "fire_floor",
            "fire_resistance_code",
            "source_item_code",
            "distance_to_station",
            "fatalities",
            "injuries",
            "direct_damage",
            "t_detect_min",
            "t_report_min",
            "t_arrival_min",
            "t_first_hose_min",
            "t_contained_min",
            "t_extinguished_min",
        ],
    }

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
        feature_set: str = "basic",
        custom_features: Optional[List[str]] = None,
        test_size: float = 0.2,
        class_weight: str = "balanced",
    ) -> UseCaseResult:
        """
        Выполнить обучение модели.

        Args:
            target: Целевая переменная.
            model_type: Тип модели ('decision_tree' или 'random_forest').
            feature_set: Набор признаков ('basic', 'extended', 'custom').
            custom_features: Пользовательский список признаков.
            test_size: Доля тестовой выборки.
            class_weight: Балансировка классов.

        Returns:
            Результат обучения.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: List[str] = []

        db = DatabaseManager(str(self.db_path))

        try:
            # Шаг 1: Загрузка данных
            self.report_progress(1, 5, "Загрузка данных из БД")
            self.check_cancelled()

            query = f"SELECT * FROM fires WHERE {target} IS NOT NULL"
            df = pd.read_sql(query, db.engine)

            if df.empty and target == "rank_tz":
                auto_labeled = self._auto_assign_rank_tz(db)
                if auto_labeled > 0:
                    warnings.append(
                        f"rank_tz отсутствовал, автоматически размечено {auto_labeled} записей"
                    )
                    df = pd.read_sql(query, db.engine)

            if df.empty:
                return UseCaseResult(
                    success=False,
                    message=f"Нет данных с целевой переменной '{target}'",
                    warnings=warnings,
                )

            logger.info("Loaded %s records for training", len(df))

            # Шаг 2: Подготовка признаков
            self.report_progress(2, 5, "Подготовка признаков")
            self.check_cancelled()

            features = self._resolve_features(feature_set, custom_features)
            missing = [f for f in features if f not in df.columns]
            if missing:
                warnings.append(f"Отсутствуют признаки: {missing}")
                features = [f for f in features if f in df.columns]

            if not features:
                return UseCaseResult(
                    success=False,
                    message="Нет признаков для обучения",
                    warnings=warnings,
                )

            X = df[features].copy()
            y = df[target].copy()

            mask = y.notna()
            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

            if target == "rank_tz":
                rank_to_class = {float(v): int(k) for k, v in RANK_CLASSES.items()}
                mapped = y.map(lambda v: rank_to_class.get(float(v)) if v is not None else None)
                valid_mask = mapped.notna()
                dropped = int((~valid_mask).sum())
                if dropped > 0:
                    warnings.append(
                        f"Пропущено {dropped} записей с неподдерживаемыми значениями rank_tz"
                    )
                X = X.loc[valid_mask].copy()
                y = mapped.loc[valid_mask].astype(int)

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            if X.empty or len(y) == 0:
                return UseCaseResult(
                    success=False,
                    message="Нет данных после подготовки",
                    warnings=warnings,
                )

            if y.nunique() < 2:
                return UseCaseResult(
                    success=False,
                    message="Недостаточно классов для обучения (нужно минимум 2)",
                    warnings=warnings,
                )

            feature_names = X.columns.tolist()
            logger.info("Features: %s, Samples: %s", len(feature_names), len(y))

            # Шаг 3: Обучение модели
            self.report_progress(3, 5, f"Обучение модели ({model_type})")
            self.check_cancelled()

            if model_type == "decision_tree":
                train_result = train_classifier(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                )
            elif model_type == "random_forest":
                train_result = train_random_forest_classifier(
                    X,
                    y,
                    test_size=test_size,
                    class_weight=class_weight,
                    random_state=42,
                )
            else:
                return UseCaseResult(
                    success=False,
                    message=f"Неизвестный тип модели: {model_type}",
                )

            model = train_result["model"]
            metrics = train_result["metrics"]
            logger.info("Training complete. Metrics keys: %s", list(metrics.keys()))

            # Шаг 4: Сохранение артефактов
            self.report_progress(4, 5, "Сохранение артефактов")
            self.check_cancelled()

            self.models_path.mkdir(parents=True, exist_ok=True)

            model_id = str(uuid.uuid4())[:8]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{target}_{timestamp}"

            model_path = self.models_path / f"model_{model_id}.joblib"
            joblib.dump(model, model_path)

            metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "model_type": model_type,
                "target": target,
                "features": feature_names,
                "feature_set": feature_set,
                "metrics": metrics,
                "params": {
                    "test_size": test_size,
                    "class_weight": class_weight,
                    "random_state": 42,
                },
                "dataset_info": {
                    "samples": int(len(y)),
                    "features_count": int(len(feature_names)),
                    "classes": [str(v) for v in pd.Series(y).dropna().unique().tolist()],
                },
                "created_at": pd.Timestamp.now().isoformat(),
                "model_file": f"model_{model_id}.joblib",
            }

            metadata_path = self.models_path / f"model_{model_id}_meta.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            metrics_df = pd.DataFrame([metrics])
            metrics_path = self.models_path / f"model_{model_id}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)

            if model_type == "decision_tree":
                try:
                    fig_path = self.models_path / f"model_{model_id}_tree.png"
                    class_names = [str(c) for c in metrics.get("classes", [])]
                    visualize_tree(
                        model=model,
                        feature_names=feature_names,
                        class_names=class_names,
                        output_path=str(fig_path),
                    )
                except Exception as e:
                    warnings.append(f"Не удалось сохранить визуализацию: {e}")

            logger.info("Saved model artifacts: %s", model_id)

            # Шаг 5: Завершение
            self.report_progress(5, 5, "Обучение завершено")

            return UseCaseResult(
                success=True,
                message=f"Модель обучена: {model_name}",
                data={
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "metrics": metrics,
                    "feature_names": feature_names,
                    "samples": int(len(y)),
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

    def _resolve_features(
        self, feature_set: str, custom_features: Optional[List[str]]
    ) -> List[str]:
        """Сформировать финальный список признаков для обучения."""
        if feature_set == "custom":
            return list(custom_features or [])
        return self._DEFAULT_FEATURES.get(feature_set, self._DEFAULT_FEATURES["basic"]).copy()

    def _auto_assign_rank_tz(self, db: DatabaseManager) -> int:
        """
        Автоматически заполнить rank_tz по equipment_count, если поле пустое.

        Returns:
            Количество обновленных записей.
        """
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

        updates: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False):
            rank, distance = calculate_rank_by_count(getattr(row, "equipment_count", None))
            updates.append(
                {
                    "id": int(getattr(row, "id")),
                    "rank_tz": float(rank) if rank is not None else None,
                    "rank_distance": float(distance) if distance is not None else None,
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
