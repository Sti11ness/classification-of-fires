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

        # Параметры обучения
        params_group = QGroupBox("Параметры обучения")
        params_layout = QGridLayout(params_group)
        params_layout.setSpacing(10)

        # Цель
        params_layout.addWidget(QLabel("Целевая переменная:"), 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["rank_tz", "equipment_count", "nozzle_count"])
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
        self.feature_set_combo.addItems(["basic", "extended", "custom"])
        params_layout.addWidget(self.feature_set_combo, 2, 1)

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

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.train_btn.clicked.connect(self._on_train)
        self.activate_btn.clicked.connect(self._on_activate)

    def set_paths(self, db_path: Optional[Path],
                  models_path: Optional[Path]) -> None:
        """Установить пути."""
        if db_path and models_path:
            from ...viewmodels import TrainModelViewModel
            self.viewmodel = TrainModelViewModel(db_path, models_path)
            self.train_btn.setEnabled(True)
        else:
            self.viewmodel = None
            self.train_btn.setEnabled(False)

    def _on_train(self) -> None:
        """Запустить обучение."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
            return

        # Установить параметры
        self.viewmodel.set_target(self.target_combo.currentText())
        self.viewmodel.set_model_type(self.model_type_combo.currentText())
        self.viewmodel.set_feature_set(self.feature_set_combo.currentText())
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
        self.activate_btn.setEnabled(True)
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
        self.results_text.append(f"Выборка: {result.get('samples', 0)} записей\n")
        self.results_text.append(f"Признаков: {len(result.get('feature_names', []))}\n\n")
        self.results_text.append("Метрики:\n")
        self.results_text.append(metrics_text)

        self.results_text.append("\n✓ Модель сохранена и зарегистрирована")

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
                "Не удалось активировать модель"
            )
