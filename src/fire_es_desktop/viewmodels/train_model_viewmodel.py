# src/fire_es_desktop/viewmodels/train_model_viewmodel.py
"""
TrainModelViewModel — ViewModel для экрана обучения модели.

Согласно spec_second.md раздел 11.5:
- Выбор цели и признаков
- Обучение дерева/леса
- Оценка качества
- Активация модели
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from ..use_cases import TrainModelUseCase, UseCaseResult
from ..infra import ModelRegistry

logger = logging.getLogger("TrainModelViewModel")


class TrainModelViewModel:
    """
    ViewModel для экрана обучения.

    Управляет процессом обучения и регистрации моделей.
    """

    def __init__(
        self,
        db_path: Path,
        models_path: Path
    ):
        """
        Инициализировать ViewModel.

        Args:
            db_path: Путь к БД.
            models_path: Путь к папке моделей.
        """
        self.db_path = db_path
        self.models_path = models_path

        self.train_use_case = TrainModelUseCase(db_path, models_path)
        self.model_registry = ModelRegistry(models_path)

        # Состояние
        self.is_training = False
        self.training_result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None

        # Параметры обучения
        self.target = "rank_tz"
        self.model_type = "random_forest"  # 'decision_tree' или 'random_forest'
        self.feature_set = "basic"  # 'basic', 'extended', 'custom'
        self.custom_features: List[str] = []
        self.test_size = 0.2
        self.class_weight = "balanced"

        # Callbacks
        self.on_training_started: Optional[Callable[[], None]] = None
        self.on_training_complete: Optional[Callable[[Dict], None]] = None
        self.on_training_failed: Optional[Callable[[str], None]] = None
        self.on_progress: Optional[Callable[[int, int, str], None]] = None

    def set_target(self, target: str) -> None:
        """Установить целевую переменную."""
        self.target = target

    def set_model_type(self, model_type: str) -> None:
        """Установить тип модели."""
        self.model_type = model_type

    def set_feature_set(self, feature_set: str) -> None:
        """Установить набор признаков."""
        self.feature_set = feature_set

    def set_custom_features(self, features: List[str]) -> None:
        """Установить пользовательские признаки."""
        self.custom_features = features

    def set_test_size(self, test_size: float) -> None:
        """Установить долю тестовой выборки."""
        self.test_size = max(0.1, min(0.5, test_size))

    def set_class_weight(self, class_weight: str) -> None:
        """Установить балансировку классов."""
        self.class_weight = class_weight

    def train(self) -> None:
        """
        Запустить обучение.

        Асинхронный вызов через TaskRunner должен быть снаружи.
        """
        self.is_training = True
        self.training_result = None
        self.error_message = None

        if self.on_training_started:
            self.on_training_started()

        # Установить callback прогресса
        def on_progress(current, total, description):
            if self.on_progress:
                self.on_progress(current, total, description)

        self.train_use_case.set_progress_callback(on_progress)

        try:
            # Выполнить обучение
            result = self.train_use_case.execute(
                target=self.target,
                model_type=self.model_type,
                feature_set=self.feature_set,
                custom_features=self.custom_features if self.feature_set == "custom" else None,
                test_size=self.test_size,
                class_weight=self.class_weight
            )

            if result.success:
                self.training_result = result.data

                # Зарегистрировать модель
                self.model_registry.register_model(
                    model_id=result.data["model_id"],
                    name=result.data["model_name"],
                    model_type=self.model_type,
                    target=self.target,
                    features=result.data["feature_names"],
                    metrics=result.data["metrics"],
                    params={
                        "test_size": self.test_size,
                        "class_weight": self.class_weight
                    },
                    dataset_info={
                        "samples": result.data["samples"],
                        "features_count": len(result.data["feature_names"])
                    }
                )

                if self.on_training_complete:
                    self.on_training_complete(result.data)
            else:
                self.error_message = result.message
                if self.on_training_failed:
                    self.on_training_failed(result.message)

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.error_message = f"Ошибка обучения: {str(e)}"
            if self.on_training_failed:
                self.on_training_failed(self.error_message)

        finally:
            self.is_training = False

    def get_available_targets(self) -> List[str]:
        """Получить доступные целевые переменные."""
        return ["rank_tz", "equipment_count", "nozzle_count"]

    def get_available_model_types(self) -> List[Dict[str, str]]:
        """Получить доступные типы моделей."""
        return [
            {"value": "random_forest", "label": "Случайный лес (рекомендуется)"},
            {"value": "decision_tree", "label": "Дерево решений"}
        ]

    def get_available_feature_sets(self) -> List[Dict[str, str]]:
        """Получить доступные наборы признаков."""
        return [
            {"value": "basic", "label": "Базовый"},
            {"value": "extended", "label": "Расширенный"},
            {"value": "custom", "label": "Пользовательский"}
        ]

    def set_model_active(self, model_id: str) -> bool:
        """
        Сделать модель активной.

        Args:
            model_id: ID модели.

        Returns:
            True если успешно.
        """
        success = self.model_registry.set_active_model(model_id)
        if success:
            logger.info(f"Set active model: {model_id}")
        return success

    def get_registered_models(self) -> List[Dict[str, Any]]:
        """Получить список зарегистрированных моделей."""
        return self.model_registry.list_models()

    def get_active_model_id(self) -> Optional[str]:
        """Получить ID активной модели."""
        return self.model_registry.get_active_model_id()
