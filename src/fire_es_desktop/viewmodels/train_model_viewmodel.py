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
from ..infra import ModelRegistry, TrainingDataStore
from fire_es.rank_tz_contract import FIELD_SPECS, get_feature_set_spec

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
        self.training_store = TrainingDataStore(db_path)

        # Состояние
        self.is_training = False
        self.training_result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None

        # Параметры обучения
        self.target = "rank_tz"
        self.model_type = "random_forest"  # 'decision_tree' или 'random_forest'
        self.feature_set = "dispatch_initial_safe"
        self.custom_features: List[str] = []
        self.test_size = 0.25
        self.class_weight = "balanced"
        self.include_historical = True
        self.include_lpr = True
        self.synthetic_method: Optional[str] = None
        self.synthetic_k_neighbors = 5
        self.synthetic_m_neighbors = 10
        self.synthetic_target_total_rows: Optional[int] = None
        self.tuning_enabled = False
        self.tuning_trials = 50
        self.metric_primary = "f1_macro"

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

    def set_include_historical(self, include_historical: bool) -> None:
        self.include_historical = include_historical

    def set_include_lpr(self, include_lpr: bool) -> None:
        self.include_lpr = include_lpr

    def set_synthetic_method(self, synthetic_method: Optional[str]) -> None:
        self.synthetic_method = synthetic_method

    def set_synthetic_k_neighbors(self, k_neighbors: int) -> None:
        self.synthetic_k_neighbors = max(2, int(k_neighbors))

    def set_synthetic_m_neighbors(self, m_neighbors: int) -> None:
        self.synthetic_m_neighbors = max(2, int(m_neighbors))

    def set_synthetic_target_total_rows(self, total_rows: Optional[int]) -> None:
        self.synthetic_target_total_rows = None if total_rows in (None, 0) else max(1, int(total_rows))

    def set_tuning_enabled(self, tuning_enabled: bool) -> None:
        self.tuning_enabled = bool(tuning_enabled)

    def set_tuning_trials(self, tuning_trials: int) -> None:
        self.tuning_trials = max(1, int(tuning_trials))

    def set_metric_primary(self, metric_primary: str) -> None:
        self.metric_primary = metric_primary

    def cancel_training(self) -> None:
        self.train_use_case.cancel()

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
                class_weight=self.class_weight,
                include_historical=self.include_historical,
                include_lpr=self.include_lpr,
                synthetic_method=self.synthetic_method,
                synthetic_k_neighbors=self.synthetic_k_neighbors,
                synthetic_m_neighbors=self.synthetic_m_neighbors,
                synthetic_target_total_rows=self.synthetic_target_total_rows,
                tuning_enabled=self.tuning_enabled,
                tuning_trials=self.tuning_trials,
                metric_primary=self.metric_primary,
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
                        "class_weight": self.class_weight,
                        "tuning_enabled": self.tuning_enabled,
                        "tuning_trials": self.tuning_trials,
                        "synthetic_target_total_rows": self.synthetic_target_total_rows,
                        "metric_primary": self.metric_primary,
                    },
                    dataset_info={
                        "samples": result.data["samples"],
                        "features_count": len(result.data["feature_names"])
                    },
                    extra=result.data.get("registry_extra"),
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
        return ["rank_tz"]

    def get_available_model_types(self) -> List[Dict[str, str]]:
        """Получить доступные типы моделей."""
        return [
            {"value": "random_forest", "label": "Случайный лес (рекомендуется)"},
            {"value": "decision_tree", "label": "Дерево решений"},
            {"value": "gradient_boosting", "label": "HistGradientBoostingClassifier"},
        ]

    def get_available_feature_sets(self) -> List[Dict[str, str]]:
        """Получить доступные наборы признаков."""
        return [
            {"value": "dispatch_initial_safe", "label": get_feature_set_spec("dispatch_initial_safe")["label"]},
            {"value": "arrival_update_safe", "label": get_feature_set_spec("arrival_update_safe")["label"]},
            {"value": "first_hose_update_safe", "label": get_feature_set_spec("first_hose_update_safe")["label"]},
            {"value": "online_tactical", "label": get_feature_set_spec("online_tactical")["label"]},
            {"value": "retrospective_benchmark", "label": get_feature_set_spec("retrospective_benchmark")["label"]},
            {"value": "enhanced_tactical", "label": get_feature_set_spec("enhanced_tactical")["label"]},
            {"value": "custom", "label": "Custom (offline only)"},
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

    def get_training_source_counts(self) -> Dict[str, int]:
        return self.training_store.get_source_counts()

    def sync_new_lpr_decisions(self, *, promoted_by: str = "analyst") -> Dict[str, int]:
        return self.training_store.sync_new_lpr_decisions(promoted_by=promoted_by)

    def get_available_synthetic_methods(self) -> List[Dict[str, Any]]:
        feature_spec = get_feature_set_spec(self.feature_set)
        has_categorical = any(
            FIELD_SPECS.get(name, {}).get("kind") == "categorical"
            for name in feature_spec["feature_order"]
        )
        return [
            {"value": None, "label": "Без синтетики", "enabled": True, "reason": ""},
            {"value": "smote", "label": "SMOTE", "enabled": True, "reason": ""},
            {"value": "adasyn", "label": "ADASYN", "enabled": True, "reason": ""},
            {"value": "borderlinesmote", "label": "BorderlineSMOTE", "enabled": True, "reason": ""},
            {"value": "svmsmote", "label": "SVMSMOTE", "enabled": True, "reason": ""},
            {
                "value": "smotenc",
                "label": "SMOTENC",
                "enabled": has_categorical,
                "reason": "" if has_categorical else "В выбранном наборе признаков нет категориальных полей.",
            },
        ]

    def close(self) -> None:
        self.training_store.close()
