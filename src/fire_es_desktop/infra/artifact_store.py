# src/fire_es_desktop/infra/artifact_store.py
"""
ArtifactStore — хранилище артефактов (модели, графики, таблицы).

Согласно spec_second.md раздел 14:
- reports/models/ — модели и метаданные
- reports/figs/ — графики
- reports/tables/ — таблицы/выгрузки
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

import joblib
import pandas as pd


class ArtifactStore:
    """
    Хранилище артефактов.

    Управляет сохранением и загрузкой:
    - Моделей ML
    - Графиков
    - Таблиц и отчётов
    - Метаданных
    """

    def __init__(self, reports_path: Path):
        """
        Инициализировать хранилище.

        Args:
            reports_path: Путь к папке reports/.
        """
        self.reports_path = reports_path
        self.logger = logging.getLogger(f"ArtifactStore.{reports_path.name}")

        # Создать структуру папок
        self.models_path = reports_path / "models"
        self.figs_path = reports_path / "figs"
        self.tables_path = reports_path / "tables"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Создать необходимые папки."""
        for path in [self.models_path, self.figs_path, self.tables_path]:
            path.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Модели
    # ========================================================================

    def save_model(
        self,
        model: Any,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Сохранить модель и метаданные.

        Args:
            model: Объект модели.
            model_id: ID модели.
            metadata: Метаданные модели.

        Returns:
            Путь к файлу модели.
        """
        # Сохранить модель
        model_path = self.models_path / f"model_{model_id}.joblib"
        joblib.dump(model, model_path)
        self.logger.info(f"Saved model: {model_path}")

        # Сохранить метаданные
        metadata_path = self.models_path / f"model_{model_id}_meta.json"
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["model_file"] = model_path.name

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved model metadata: {metadata_path}")

        return model_path

    def load_model(self, model_id: str) -> tuple[Any, Dict[str, Any]]:
        """
        Загрузить модель и метаданные.

        Args:
            model_id: ID модели.

        Returns:
            Кортеж (модель, метаданные).
        """
        model_path = self.models_path / f"model_{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        metadata_path = self.models_path / f"model_{model_id}_meta.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        self.logger.info(f"Loaded model: {model_id}")
        return model, metadata

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Получить метаданные модели."""
        metadata_path = self.models_path / f"model_{model_id}_meta.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_models(self) -> List[str]:
        """Получить список ID моделей."""
        models = []
        for path in self.models_path.glob("model_*.joblib"):
            # Извлечь ID из имени файла
            model_id = path.stem.replace("model_", "")
            models.append(model_id)
        return models

    def delete_model(self, model_id: str) -> None:
        """Удалить модель и метаданные."""
        model_path = self.models_path / f"model_{model_id}.joblib"
        metadata_path = self.models_path / f"model_{model_id}_meta.json"

        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

        self.logger.info(f"Deleted model: {model_id}")

    # ========================================================================
    # Графики
    # ========================================================================

    def save_figure(
        self,
        fig: Any,
        name: str,
        category: str = "",
        fmt: str = "png"
    ) -> Path:
        """
        Сохранить график.

        Args:
            fig: Объект графика (plotly/matplotlib).
            name: Имя файла.
            category: Категория (подпапка).
            fmt: Формат файла.

        Returns:
            Путь к файлу графика.
        """
        if category:
            category_path = self.figs_path / category
            category_path.mkdir(exist_ok=True)
            save_path = category_path / f"{name}.{fmt}"
        else:
            save_path = self.figs_path / f"{name}.{fmt}"

        # Plotly figure
        if hasattr(fig, "write_image"):
            fig.write_image(save_path)
        # Matplotlib figure
        elif hasattr(fig, "savefig"):
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            raise ValueError("Unsupported figure type")

        self.logger.info(f"Saved figure: {save_path}")
        return save_path

    # ========================================================================
    # Таблицы
    # ========================================================================

    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        category: str = "",
        fmt: str = "csv"
    ) -> Path:
        """
        Сохранить таблицу.

        Args:
            df: DataFrame.
            name: Имя файла.
            category: Категория (подпапка).
            fmt: Формат файла (csv, xlsx).

        Returns:
            Путь к файлу таблицы.
        """
        if category:
            category_path = self.tables_path / category
            category_path.mkdir(exist_ok=True)
            save_path = category_path / f"{name}.{fmt}"
        else:
            save_path = self.tables_path / f"{name}.{fmt}"

        if fmt == "csv":
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
        elif fmt == "xlsx":
            df.to_excel(save_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        self.logger.info(f"Saved table: {save_path}")
        return save_path

    def load_table(
        self,
        name: str,
        category: str = "",
        fmt: str = "csv"
    ) -> pd.DataFrame:
        """
        Загрузить таблицу.

        Args:
            name: Имя файла.
            category: Категория.
            fmt: Формат файла.

        Returns:
            DataFrame.
        """
        if category:
            load_path = self.tables_path / category / f"{name}.{fmt}"
        else:
            load_path = self.tables_path / f"{name}.{fmt}"

        if not load_path.exists():
            raise FileNotFoundError(f"Table not found: {load_path}")

        if fmt == "csv":
            return pd.read_csv(load_path, encoding="utf-8-sig")
        elif fmt == "xlsx":
            return pd.read_excel(load_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # ========================================================================
    # Отчёты
    # ========================================================================

    def save_report(
        self,
        content: str,
        name: str,
        category: str = ""
    ) -> Path:
        """
        Сохранить текстовый отчёт.

        Args:
            content: Содержимое отчёта.
            name: Имя файла.
            category: Категория.

        Returns:
            Путь к файлу отчёта.
        """
        if category:
            category_path = self.tables_path / category
            category_path.mkdir(exist_ok=True)
            save_path = category_path / f"{name}.md"
        else:
            save_path = self.tables_path / f"{name}.md"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.logger.info(f"Saved report: {save_path}")
        return save_path

    def get_artifact_path(self, artifact_type: str, name: str) -> Path:
        """
        Получить путь к артефакту.

        Args:
            artifact_type: Тип артефакта (model, fig, table).
            name: Имя артефакта.

        Returns:
            Путь к артефакту.
        """
        if artifact_type == "model":
            return self.models_path / f"model_{name}.joblib"
        elif artifact_type == "fig":
            return self.figs_path / f"{name}.png"
        elif artifact_type == "table":
            return self.tables_path / f"{name}.csv"
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
