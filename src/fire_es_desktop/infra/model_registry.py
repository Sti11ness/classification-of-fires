# src/fire_es_desktop/infra/model_registry.py
"""
ModelRegistry — реестр моделей и управление активной моделью.

Согласно spec_second.md раздел 14:
- Управление версиями моделей
- Выбор активной модели
- Метаданные и трассируемость
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .artifact_store import ArtifactStore


class ModelRegistry:
    """
    Реестр моделей ML.

    Хранит информацию о всех обученных моделях,
    управляет активной моделью для прогнозирования.
    """

    def __init__(self, models_path: Path):
        """
        Инициализировать реестр.

        Args:
            models_path: Путь к папке models/.
        """
        self.models_path = models_path
        self.registry_file = models_path / "registry.json"
        self.logger = logging.getLogger(f"ModelRegistry.{models_path.name}")

        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """Создать или загрузить реестр."""
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": [],
                "active_model_id": None,
                "last_updated": None
            }
            self._save_registry()

    def _save_registry(self) -> None:
        """Сохранить реестр."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
        self.logger.debug("Registry saved")

    def register_model(
        self,
        model_id: str,
        name: str,
        model_type: str,
        target: str,
        features: List[str],
        metrics: Dict[str, float],
        params: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Зарегистрировать модель.

        Args:
            model_id: Уникальный ID модели.
            name: Отображаемое имя.
            model_type: Тип модели (DecisionTree, RandomForest).
            target: Целевая переменная.
            features: Список признаков.
            metrics: Метрики качества.
            params: Параметры модели.
            dataset_info: Информация о датасете.
        """
        model_entry = {
            "model_id": model_id,
            "name": name,
            "model_type": model_type,
            "target": target,
            "features": features,
            "metrics": metrics,
            "params": params,
            "dataset_info": dataset_info or {},
            "created_at": datetime.now().isoformat(),
            "is_active": False,
            "artifact_path": f"model_{model_id}.joblib",
            "metadata_path": f"model_{model_id}_meta.json"
        }

        self.registry["models"].append(model_entry)
        self._save_registry()
        self.logger.info(f"Registered model: {model_id}")

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о модели.

        Args:
            model_id: ID модели.

        Returns:
            Информация о модели или None.
        """
        self._ensure_registry()
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """Получить список всех моделей."""
        self._ensure_registry()
        return self.registry["models"].copy()

    def get_active_model_id(self) -> Optional[str]:
        """Получить ID активной модели."""
        self._ensure_registry()
        return self.registry["active_model_id"]

    def set_active_model(self, model_id: str) -> bool:
        """
        Сделать модель активной.

        Args:
            model_id: ID модели.

        Returns:
            True если успешно.
        """
        # Проверить существование
        model_info = self.get_model_info(model_id)
        if not model_info:
            self.logger.error(f"Model not found: {model_id}")
            return False

        # Снять активность со всех
        for model in self.registry["models"]:
            model["is_active"] = False

        # Активировать выбранную
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                model["is_active"] = True
                break

        self.registry["active_model_id"] = model_id
        self._save_registry()
        self.logger.info(f"Set active model: {model_id}")
        return True

    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """Получить информацию об активной модели."""
        self._ensure_registry()
        active_id = self.get_active_model_id()
        if not active_id:
            return None
        return self.get_model_info(active_id)

    def delete_model(self, model_id: str) -> None:
        """
        Удалить модель из реестра.

        Args:
            model_id: ID модели.
        """
        self.registry["models"] = [
            m for m in self.registry["models"]
            if m["model_id"] != model_id
        ]

        # Если удалили активную, сбросить
        if self.registry["active_model_id"] == model_id:
            self.registry["active_model_id"] = None

        self._save_registry()
        self.logger.info(f"Deleted model from registry: {model_id}")

    def get_model_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Получить метрики модели."""
        self._ensure_registry()
        model_info = self.get_model_info(model_id)
        return model_info["metrics"] if model_info else None

    def compare_models(
        self,
        model_ids: List[str],
        metric_name: str = "macro_f1"
    ) -> List[Dict[str, Any]]:
        """
        Сравнить модели по метрике.

        Args:
            model_ids: Список ID моделей.
            metric_name: Имя метрики для сравнения.

        Returns:
            Список моделей с метриками, отсортированный.
        """
        self._ensure_registry()
        comparison = []
        for model_id in model_ids:
            model_info = self.get_model_info(model_id)
            if model_info:
                comparison.append({
                    "model_id": model_id,
                    "name": model_info["name"],
                    "metric_value": model_info["metrics"].get(metric_name, 0),
                    "created_at": model_info["created_at"]
                })

        # Сортировка по убыванию метрики
        comparison.sort(key=lambda x: x["metric_value"], reverse=True)
        return comparison
