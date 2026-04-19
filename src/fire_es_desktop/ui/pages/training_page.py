# src/fire_es_desktop/ui/pages/training_page.py
"""
TrainingPage — страница обучения модели.

Согласно spec_second.md раздел 11.5:
- Выбор цели и признаков
- Обучение дерева/леса
- Оценка качества
- Сохранение и активация модели
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QComboBox, QCheckBox, QProgressBar, QMessageBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Optional, Dict, Any

import json
import sqlite3


class TrainWorker(QThread):
    """Рабочий поток для обучения."""

    progress = Signal(int, int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        """Выполнить обучение."""
        def on_progress(current, total, description):
            self.progress.emit(current, total, description)

        self.viewmodel.on_progress = on_progress

        try:
            self.viewmodel.train()

            if self.viewmodel.error_message:
                self.error.emit(self.viewmodel.error_message)
            else:
                self.complete.emit(self.viewmodel.training_result or {})
        except Exception as e:
            self.error.emit(str(e))


class TrainingPage(QWidget):
    """Страница обучения."""

    def __init__(self):
        super().__init__()
        self.viewmodel = None
        self.worker = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Заголовок
        title = QLabel("Обучение модели")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)

        source_group = QGroupBox("Источник данных")
        source_layout = QGridLayout(source_group)
        source_layout.setSpacing(10)

        source_layout.addWidget(QLabel("Источник обучения:"), 0, 0)
        self.source_kind_value = QLabel("Workspace database (fires)")
        source_layout.addWidget(self.source_kind_value, 0, 1)

        source_layout.addWidget(QLabel("База данных:"), 1, 0)
        self.source_db_value = QLabel("Workspace не открыт")
        self.source_db_value.setWordWrap(True)
        source_layout.addWidget(self.source_db_value, 1, 1)

        source_layout.addWidget(QLabel("Строк для обучения:"), 2, 0)
        self.source_rows_value = QLabel("0")
        source_layout.addWidget(self.source_rows_value, 2, 1)

        self.source_hint = QLabel(
            "Обучение запускается не из Excel-файла, а из таблицы fires текущего workspace."
        )
        self.source_hint.setWordWrap(True)
        self.source_hint.setStyleSheet("font-size: 12px; color: #f0f0f0;")
        source_layout.addWidget(self.source_hint, 3, 0, 1, 2)

        layout.addWidget(source_group)

        # Параметры обучения
        params_group = QGroupBox("Параметры обучения")
        params_layout = QGridLayout(params_group)
        params_layout.setSpacing(10)

        # Цель
        params_layout.addWidget(QLabel("Целевая переменная:"), 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["rank_tz"])
        params_layout.addWidget(self.target_combo, 0, 1)

        # Тип модели
        params_layout.addWidget(QLabel("Тип модели:"), 1, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "random_forest",
            "decision_tree"
        ])
        params_layout.addWidget(self.model_type_combo, 1, 1)

        # Набор признаков
        params_layout.addWidget(QLabel("Набор признаков:"), 2, 0)
        self.feature_set_combo = QComboBox()
        self.feature_set_combo.addItem("online_tactical (production)", "online_tactical")
        self.feature_set_combo.addItem("extended (offline benchmark)", "extended")
        self.feature_set_combo.addItem("enhanced_tactical (offline experiment)", "enhanced_tactical")
        self.feature_set_combo.addItem("custom (offline only)", "custom")
        params_layout.addWidget(self.feature_set_combo, 2, 1)

        self.feature_mode_hint = QLabel("")
        self.feature_mode_hint.setWordWrap(True)
        self.feature_mode_hint.setStyleSheet("font-size: 12px; color: #f0f0f0;")
        params_layout.addWidget(self.feature_mode_hint, 2, 2, 3, 1)

        # Test size
        params_layout.addWidget(QLabel("Доля тестовой выборки:"), 3, 0)
        self.test_size_combo = QComboBox()
        self.test_size_combo.addItems(["0.2", "0.25", "0.3"])
        params_layout.addWidget(self.test_size_combo, 3, 1)

        # Class weight
        params_layout.addWidget(QLabel("Балансировка классов:"), 4, 0)
        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItems(["balanced", "None"])
        params_layout.addWidget(self.class_weight_combo, 4, 1)

        layout.addWidget(params_group)

        # Прогресс и кнопки
        control_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar, 1)

        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.setFixedHeight(40)
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.train_btn)

        layout.addLayout(control_layout)

        # Результаты
        results_group = QGroupBox("Результаты обучения")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(200)
        results_layout.addWidget(self.results_text)

        # Кнопка активации
        self.activate_btn = QPushButton("Сделать модель активной")
        self.activate_btn.setEnabled(False)
        self.activate_btn.setFixedHeight(35)
        results_layout.addWidget(self.activate_btn)

        layout.addWidget(results_group)

        layout.addStretch()

        self._connect_signals()
        self._update_feature_mode_hint()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.train_btn.clicked.connect(self._on_train)
        self.activate_btn.clicked.connect(self._on_activate)
        self.feature_set_combo.currentIndexChanged.connect(self._update_feature_mode_hint)

    def set_paths(self, db_path: Optional[Path],
                  models_path: Optional[Path]) -> None:
        """Установить пути."""
        if db_path and models_path:
            from ...viewmodels import TrainModelViewModel
            self.viewmodel = TrainModelViewModel(db_path, models_path)
            self.train_btn.setEnabled(True)
            self._update_source_info(db_path)
        else:
            self.viewmodel = None
            self.train_btn.setEnabled(False)
            self.source_db_value.setText("Workspace не открыт")
            self.source_rows_value.setText("0")

    def _on_train(self) -> None:
        """Запустить обучение."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
            return

        # Установить параметры
        self.viewmodel.set_target(self.target_combo.currentText())
        self.viewmodel.set_model_type(self.model_type_combo.currentText())
        self.viewmodel.set_feature_set(self.feature_set_combo.currentData())
        self.viewmodel.set_test_size(
            float(self.test_size_combo.currentText())
        )
        cw = self.class_weight_combo.currentText()
        self.viewmodel.set_class_weight(
            "balanced" if cw == "balanced" else None
        )

        # Запустить в фоне
        self.worker = TrainWorker(self.viewmodel)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.results_text.append("Обучение модели...")

        self.worker.start()

    def _on_progress(self, current: int, total: int, description: str) -> None:
        """Обновление прогресса."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
        self.results_text.append(f"Шаг: {description or '...'}")

    def _on_complete(self, result: Dict[str, Any]) -> None:
        """Обучение завершено."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.activate_btn.setEnabled(bool(result.get("can_activate", False)))
        self._current_model_id = result.get("model_id")

        # Показать метрики
        metrics = result.get("metrics", {})
        metrics_text = "\n".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )

        self.results_text.clear()
        self.results_text.append("=== Обучение завершено ===\n")
        self.results_text.append(f"Модель: {result.get('model_name', '')}\n")
        self.results_text.append(f"ID: {result.get('model_id', '')}\n")
        if self.viewmodel:
            self.results_text.append(f"Источник: {self.viewmodel.db_path}\n")
        self.results_text.append(f"Выборка: {result.get('samples', 0)} записей\n")
        self.results_text.append(f"Признаков: {len(result.get('feature_names', []))}\n\n")
        self.results_text.append("Метрики:\n")
        self.results_text.append(metrics_text)
        self.results_text.append(
            f"\nРоль deployment: {result.get('deployment_role', 'unknown')}"
        )
        if result.get("offline_only"):
            self.results_text.append(
                "\nℹ Модель сохранена как offline-only benchmark."
                "\nОна использует исследовательский набор признаков и не совпадает с production-вводом экрана ЛПР,"
                "\nпоэтому ее активация для оперативного прогноза заблокирована."
            )
        else:
            self.results_text.append(
                "\n✓ Модель использует production-safe набор признаков,"
                "\nсовместима с экраном ЛПР и может быть активирована."
            )

    def _on_error(self, message: str) -> None:
        """Ошибка обучения."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results_text.append(f"\n✗ Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка обучения", message)

    def _on_activate(self) -> None:
        """Активировать модель."""
        if not self.viewmodel or not hasattr(self, "_current_model_id"):
            return

        success = self.viewmodel.set_model_active(self._current_model_id)
        if success:
            QMessageBox.information(
                self, "Активация",
                "Модель активирована"
            )
        else:
            QMessageBox.critical(
                self, "Ошибка",
                "Не удалось активировать модель. Для rank_tz разрешены только production-safe модели."
            )

    def _update_feature_mode_hint(self) -> None:
        """Пояснить текущий режим набора признаков."""
        feature_set = self.feature_set_combo.currentData()
        hints = {
            "online_tactical": (
                "Production: обучается на признаках, которые реально вводятся на экране ЛПР. "
                "Такую модель можно активировать для рабочего прогноза."
            ),
            "extended": (
                "Offline benchmark: использует расширенный набор признаков для сравнения качества. "
                "Сохраняется для анализа, но не активируется для ЛПР."
            ),
            "enhanced_tactical": (
                "Offline experiment: исследовательский режим с engineered features. "
                "Нужен для экспериментов и сравнения, не для активации в ЛПР."
            ),
            "custom": (
                "Offline only: произвольный набор признаков без гарантии совместимости с экраном ЛПР. "
                "Для рабочего прогноза не активируется."
            ),
        }
        self.feature_mode_hint.setText(hints.get(feature_set, ""))

    def _update_source_info(self, db_path: Path) -> None:
        """Показать, откуда именно берутся данные для обучения."""
        self.source_db_value.setText(str(db_path))
        labeled_rows = 0
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT COUNT(*) FROM fires WHERE rank_tz IS NOT NULL"
            ).fetchone()
            labeled_rows = int(row[0]) if row and row[0] is not None else 0
        except Exception:
            labeled_rows = 0
        finally:
            try:
                conn.close()
            except Exception:
                pass
        self.source_rows_value.setText(str(labeled_rows))
