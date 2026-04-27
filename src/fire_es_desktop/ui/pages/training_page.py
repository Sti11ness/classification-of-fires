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
    QTableWidget, QTableWidgetItem, QTextEdit, QGridLayout, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Optional, Dict, Any

import json

from ..theme import (
    configure_text_panel,
    create_hint,
    create_page_header,
    create_scrollable_page,
    create_status_label,
    ResponsiveFormWidget,
    style_button,
    style_label,
)
from ...use_cases.assign_rank_tz_use_case import AssignRankTzUseCase


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


class AssignRankWorker(QThread):
    """Рабочий поток для разметки рангов."""

    progress = Signal(int, int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        db_path: Path,
        *,
        override_existing_labels: bool = False,
        source_table: str = "fires_historical",
    ):
        super().__init__()
        self.db_path = db_path
        self.override_existing_labels = override_existing_labels
        self.source_table = source_table

    def run(self):
        try:
            use_case = AssignRankTzUseCase(self.db_path)
            use_case.set_progress_callback(
                lambda current, total, description: self.progress.emit(current, total, description)
            )
            result = use_case.execute(
                override_existing_labels=self.override_existing_labels,
                source_table=self.source_table,
            )
            if result.success:
                self.complete.emit(result.data or {})
            else:
                self.error.emit(result.message)
        except Exception as e:
            self.error.emit(str(e))


class LPRSyncWorker(QThread):
    """Рабочий поток для подтягивания новых решений ЛПР в training source."""

    progress = Signal(int, int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        try:
            self.progress.emit(1, 3, "Поиск новых решений ЛПР")
            result = self.viewmodel.sync_new_lpr_decisions(promoted_by="analyst")
            self.progress.emit(2, 3, "Обновление analyst-источника")
            self.progress.emit(3, 3, "Подтягивание завершено")
            self.complete.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainingPage(QWidget):
    """Страница обучения."""

    def __init__(self):
        super().__init__()
        self.viewmodel = None
        self.worker = None
        self.assign_worker = None
        self.sync_worker = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        _, _, _, layout = create_scrollable_page(self)

        # Заголовок
        layout.addWidget(
            create_page_header(
                "Обучение модели",
                "Аналитический цикл: выбор источника, параметров, запуск обучения, просмотр метрик и выбор рабочей модели.",
            )
        )

        source_group = QGroupBox("Подготовка данных")
        source_layout = QVBoxLayout(source_group)
        source_form = ResponsiveFormWidget(compact_breakpoint=880)
        source_layout.addWidget(source_form)

        self.source_kind_value = QLabel("Виртуальная сборка historical + LPR + synthetic")
        style_label(self.source_kind_value, "value", word_wrap=True)
        source_form.add_row("Источник обучения:", self.source_kind_value)

        self.source_db_value = QLabel("Рабочее пространство не открыто")
        self.source_db_value.setWordWrap(True)
        style_label(self.source_db_value, "value", word_wrap=True)
        source_form.add_row("База данных:", self.source_db_value)

        self.historical_rows_value = QLabel("0")
        style_label(self.historical_rows_value, "value", word_wrap=True)
        source_form.add_row("Исторических записей:", self.historical_rows_value)

        self.historical_ready_value = QLabel("0")
        style_label(self.historical_ready_value, "value", word_wrap=True)
        source_form.add_row("Исторических готово к обучению:", self.historical_ready_value)

        self.lpr_rows_value = QLabel("0")
        style_label(self.lpr_rows_value, "value", word_wrap=True)
        source_form.add_row("Записей ЛПР в обучающем слое:", self.lpr_rows_value)

        self.new_lpr_value = QLabel("0")
        style_label(self.new_lpr_value, "value", word_wrap=True)
        source_form.add_row("Новых решений ЛПР:", self.new_lpr_value)

        self.synthetic_rows_value = QLabel("0")
        style_label(self.synthetic_rows_value, "value", word_wrap=True)
        source_form.add_row("Синтетических строк (аудит):", self.synthetic_rows_value)

        self.label_status_value = QLabel("Разметка рангов не выполнялась")
        self.label_status_value.setWordWrap(True)
        style_label(self.label_status_value, "value", word_wrap=True)
        source_form.add_row("Состояние разметки:", self.label_status_value)

        self.include_historical_checkbox = QCheckBox("Исторические данные")
        self.include_historical_checkbox.setChecked(True)
        self.include_lpr_checkbox = QCheckBox("Данные ЛПР")
        self.include_lpr_checkbox.setChecked(True)
        source_flags_widget = QWidget()
        source_flags_layout = QHBoxLayout()
        source_flags_widget.setLayout(source_flags_layout)
        source_flags_layout.setContentsMargins(0, 0, 0, 0)
        source_flags_layout.setSpacing(10)
        source_flags_layout.addWidget(self.include_historical_checkbox)
        source_flags_layout.addWidget(self.include_lpr_checkbox)
        source_flags_layout.addStretch()
        source_form.add_row("Источники для fit:", source_flags_widget)

        self.assign_mode_combo = QComboBox()
        self.assign_mode_combo.addItem("Только новые неразмеченные записи", "new_only")
        self.assign_mode_combo.addItem("Пересчитать только авторазметку", "relabel_auto")
        source_form.add_row("Режим разметки:", self.assign_mode_combo)

        self.assign_rank_btn = QPushButton("Разметить ранги")
        style_button(self.assign_rank_btn, "ghost")
        source_form.add_full_width(self.assign_rank_btn)

        self.sync_lpr_btn = QPushButton("Загрузить новые решения ЛПР")
        style_button(self.sync_lpr_btn, "ghost")
        source_form.add_full_width(self.sync_lpr_btn)

        self.source_hint = QLabel(
            "После импорта нужно разметить ранги пожаров по нормативам. "
            "По умолчанию размечаются только новые строки без меток. "
            "Режим перерасчета трогает только автоматическую разметку и не затрагивает human/LPR-решения."
        )
        style_label(self.source_hint, "section-hint", word_wrap=True)
        source_form.add_full_width(self.source_hint)

        layout.addWidget(source_group)

        # Параметры обучения
        params_group = QGroupBox("Параметры обучения")
        params_layout = QVBoxLayout(params_group)
        params_form = ResponsiveFormWidget(compact_breakpoint=880)
        params_layout.addWidget(params_form)

        # Цель
        self.target_combo = QComboBox()
        self.target_combo.addItems(["Ранг пожара (rank_tz)"])
        params_form.add_row("Целевая переменная:", self.target_combo)

        # Тип модели
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Случайный лес", "random_forest")
        self.model_type_combo.addItem("Дерево решений", "decision_tree")
        self.model_type_combo.addItem("Градиентный бустинг", "gradient_boosting")
        params_form.add_row("Тип модели:", self.model_type_combo)

        # Набор признаков
        self.feature_set_combo = QComboBox()
        self.feature_set_combo.addItem("До прибытия подразделения (рабочий набор)", "dispatch_initial_safe")
        self.feature_set_combo.addItem("После прибытия подразделения", "arrival_update_safe")
        self.feature_set_combo.addItem("После подачи первого ствола", "first_hose_update_safe")
        self.feature_set_combo.addItem("Устаревший оперативный набор", "online_tactical")
        self.feature_set_combo.addItem("Архивный сравнительный набор", "retrospective_benchmark")
        self.feature_set_combo.addItem("Расширенный исследовательский набор", "enhanced_tactical")
        self.feature_set_combo.addItem("Пользовательский набор (не для ЛПР)", "custom")
        params_form.add_row("Набор признаков:", self.feature_set_combo)

        self.feature_mode_hint = QLabel("")
        self.feature_mode_hint.setWordWrap(True)
        style_label(self.feature_mode_hint, "section-hint", word_wrap=True)
        params_form.add_full_width(self.feature_mode_hint)

        # Test size
        self.test_size_combo = QComboBox()
        self.test_size_combo.addItems(["0.2", "0.25", "0.3"])
        params_form.add_row("Доля тестовой выборки:", self.test_size_combo)

        # Class weight
        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItem("Автоматическая балансировка", "balanced")
        self.class_weight_combo.addItem("Без балансировки", None)
        params_form.add_row("Балансировка классов:", self.class_weight_combo)

        self.tuning_checkbox = QCheckBox("Автоподбор гиперпараметров")
        self.tuning_checkbox.setChecked(False)
        params_form.add_full_width(self.tuning_checkbox)

        self.tuning_trials_combo = QComboBox()
        self.tuning_trials_combo.addItems(["20", "50", "100"])
        self.tuning_trials_combo.setCurrentText("50")
        params_form.add_row("Бюджет поиска:", self.tuning_trials_combo)

        self.metric_primary_combo = QComboBox()
        self.metric_primary_combo.addItem("F1 macro", "f1_macro")
        self.metric_primary_combo.addItem("F1 micro", "f1_micro")
        self.metric_primary_combo.addItem("F1 weighted", "f1_weighted")
        self.metric_primary_combo.addItem("Accuracy", "accuracy")
        self.metric_primary_combo.addItem("Precision macro", "precision_macro")
        self.metric_primary_combo.addItem("Recall macro", "recall_macro")
        params_form.add_row("Целевая метрика подбора:", self.metric_primary_combo)

        self.synthetic_method_label = QLabel("Синтетическое расширение:")
        style_label(self.synthetic_method_label, "metric", word_wrap=True)
        self.synthetic_method_combo = QComboBox()
        params_form.add_row(self.synthetic_method_label, self.synthetic_method_combo)

        self.synthetic_k_label = QLabel("k соседей:")
        style_label(self.synthetic_k_label, "metric", word_wrap=True)
        self.synthetic_k_combo = QComboBox()
        self.synthetic_k_combo.addItems(["1", "2", "3", "5", "7", "9"])
        self.synthetic_k_combo.setCurrentText("5")
        params_form.add_row(self.synthetic_k_label, self.synthetic_k_combo)

        self.synthetic_m_label = QLabel("m соседей:")
        style_label(self.synthetic_m_label, "metric", word_wrap=True)
        self.synthetic_m_combo = QComboBox()
        self.synthetic_m_combo.addItems(["5", "10", "15"])
        self.synthetic_m_combo.setCurrentText("10")
        params_form.add_row(self.synthetic_m_label, self.synthetic_m_combo)

        self.synthetic_target_total_label = QLabel("Размер train после синтетики:")
        style_label(self.synthetic_target_total_label, "metric", word_wrap=True)
        self.synthetic_target_total_spin = QSpinBox()
        self.synthetic_target_total_spin.setRange(0, 1000000)
        self.synthetic_target_total_spin.setValue(0)
        self.synthetic_target_total_spin.setSpecialValueText("Авто")
        params_form.add_row(self.synthetic_target_total_label, self.synthetic_target_total_spin)

        layout.addWidget(params_group)

        # Прогресс и кнопки
        control_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar, 1)

        self.train_btn = QPushButton("Обучить модель")
        style_button(self.train_btn, "primary", large=True)
        control_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Остановить обучение")
        style_button(self.stop_btn, "ghost")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        layout.addLayout(control_layout)

        # Результаты
        results_group = QGroupBox("Результаты обучения")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        configure_text_panel(self.results_text, min_height=260)
        self.results_text.setMaximumHeight(340)
        results_layout.addWidget(self.results_text)

        # Кнопка активации
        self.activate_btn = QPushButton("Сделать модель активной")
        self.activate_btn.setEnabled(False)
        style_button(self.activate_btn, "success")
        results_layout.addWidget(self.activate_btn)

        layout.addWidget(results_group)

        self.status_label = create_status_label()
        self.status_label.setText("Выберите набор признаков, модель и параметры проверки, затем запустите обучение.")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self._connect_signals()
        self._update_feature_mode_hint()
        self._update_tuning_controls()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.train_btn.clicked.connect(self._on_train)
        self.activate_btn.clicked.connect(self._on_activate)
        self.assign_rank_btn.clicked.connect(self._on_assign_rank)
        self.sync_lpr_btn.clicked.connect(self._on_sync_lpr)
        self.feature_set_combo.currentIndexChanged.connect(self._update_feature_mode_hint)
        self.synthetic_method_combo.currentIndexChanged.connect(self._update_synthetic_controls)
        self.tuning_checkbox.toggled.connect(self._update_tuning_controls)
        self.stop_btn.clicked.connect(self._on_stop_training)

    def set_paths(self, db_path: Optional[Path],
                  models_path: Optional[Path]) -> None:
        """Установить пути."""
        if db_path and models_path:
            from ...viewmodels import TrainModelViewModel
            if self.viewmodel is not None:
                self.viewmodel.close()
            self.viewmodel = TrainModelViewModel(db_path, models_path)
            self.train_btn.setEnabled(True)
            self.assign_rank_btn.setEnabled(True)
            self.sync_lpr_btn.setEnabled(True)
            self._populate_synthetic_methods()
            self._update_source_info(db_path)
        else:
            if self.viewmodel is not None:
                self.viewmodel.close()
            self.viewmodel = None
            self.train_btn.setEnabled(False)
            self.assign_rank_btn.setEnabled(False)
            self.sync_lpr_btn.setEnabled(False)
            self.source_db_value.setText("Рабочее пространство не открыто")
            self.historical_rows_value.setText("0")
            self.historical_ready_value.setText("0")
            self.lpr_rows_value.setText("0")
            self.new_lpr_value.setText("0")
            self.synthetic_rows_value.setText("0")
            self.label_status_value.setText("Разметка недоступна")

    def _on_train(self) -> None:
        """Запустить обучение."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
            return
        if not self.include_historical_checkbox.isChecked() and not self.include_lpr_checkbox.isChecked():
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один источник данных для обучения")
            return

        # Установить параметры
        self.viewmodel.set_target("rank_tz")
        self.viewmodel.set_model_type(self.model_type_combo.currentData())
        self.viewmodel.set_feature_set(self.feature_set_combo.currentData())
        self.viewmodel.set_test_size(
            float(self.test_size_combo.currentText())
        )
        self.viewmodel.set_class_weight(self.class_weight_combo.currentData())
        self.viewmodel.set_include_historical(self.include_historical_checkbox.isChecked())
        self.viewmodel.set_include_lpr(self.include_lpr_checkbox.isChecked())
        self.viewmodel.set_synthetic_method(self.synthetic_method_combo.currentData())
        self.viewmodel.set_synthetic_k_neighbors(int(self.synthetic_k_combo.currentText()))
        self.viewmodel.set_synthetic_m_neighbors(int(self.synthetic_m_combo.currentText()))
        self.viewmodel.set_synthetic_target_total_rows(self.synthetic_target_total_spin.value())
        self.viewmodel.set_tuning_enabled(self.tuning_checkbox.isChecked())
        self.viewmodel.set_tuning_trials(int(self.tuning_trials_combo.currentText()))
        self.viewmodel.set_metric_primary(self.metric_primary_combo.currentData())

        # Запустить в фоне
        self.worker = TrainWorker(self.viewmodel)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)

        self.train_btn.setEnabled(False)
        self.assign_rank_btn.setEnabled(False)
        self.sync_lpr_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.results_text.append("Обучение модели...")
        self.status_label.setText("Подготовка и запуск обучения…")

        self.worker.start()

    def _on_progress(self, current: int, total: int, description: str) -> None:
        """Обновление прогресса."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
        self.results_text.append(f"Шаг: {description or '...'}")
        self.status_label.setText(description or "Выполнение шага обучения…")

    def _on_stop_training(self) -> None:
        if not self.viewmodel or not self.viewmodel.is_training:
            return
        self.viewmodel.cancel_training()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Остановка обучения после завершения текущего шага…")
        self.results_text.append("Запрошена остановка обучения…")

    def _on_assign_progress(self, current: int, total: int, description: str) -> None:
        """Показать прогресс разметки рангов."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
        self.status_label.setText(description or "Выполняется разметка рангов…")
        self.results_text.append(f"Разметка: {description or '...'}")

    def _on_complete(self, result: Dict[str, Any]) -> None:
        """Обучение завершено."""
        self.train_btn.setEnabled(True)
        self.assign_rank_btn.setEnabled(True)
        self.sync_lpr_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.activate_btn.setEnabled(bool(result.get("can_activate", False)))
        self._current_model_id = result.get("model_id")
        if self.viewmodel:
            self._update_source_info(self.viewmodel.db_path)

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
        self.results_text.append(
            "Источники: historical={historical}, lpr={lpr}, synthetic={synthetic}\n".format(
                historical=result.get("historical_selected_rows", 0),
                lpr=result.get("lpr_selected_rows", 0),
                synthetic=result.get("synthetic_rows_added", 0),
            )
        )
        self.results_text.append(
            "Train/Test: до синтетики train={train_rows}, после синтетики train={train_after}, test={test_rows}\n".format(
                train_rows=result.get("train_rows_before_synthetic", 0),
                train_after=result.get("train_rows_after_synthetic", 0),
                test_rows=result.get("test_rows_real", 0),
            )
        )
        self.results_text.append(f"Признаков: {len(result.get('feature_names', []))}\n\n")
        self.results_text.append("Метрики:\n")
        self.results_text.append(metrics_text)
        if result.get("synthetic_method"):
            self.results_text.append(
                f"\nСинтетическое расширение: {result.get('synthetic_method')} "
                f"(добавлено строк: {result.get('synthetic_rows_added', 0)}, "
                f"цель train: {result.get('synthetic_target_total_rows', 0) or 'авто'})"
            )
        if result.get("tuning_enabled"):
            self.results_text.append(
                "\nАвтоподбор гиперпараметров: включен"
                f"\nПопыток: {result.get('tuning_trials_completed', 0)} / {result.get('tuning_trials_requested', 0)}"
                f"\nЦелевая метрика: {result.get('tuning_metric', 'f1_macro')}"
                f"\nЛучшая внутренняя оценка: {result.get('best_cv_score', 0):.4f}"
                f"\nЛучшие параметры: {json.dumps(result.get('best_params', {}), ensure_ascii=False)}"
            )
        else:
            self.results_text.append("\nАвтоподбор гиперпараметров: выключен")
        self.results_text.append(
            f"\nРоль модели: {result.get('deployment_role', 'unknown')}"
        )
        if result.get("offline_only"):
            self.results_text.append(
                "\nℹ Модель сохранена как исследовательская."
                "\nОна использует исследовательский набор признаков и не совпадает с production-вводом экрана ЛПР,"
                "\nпоэтому ее активация для оперативного прогноза заблокирована."
            )
            self.status_label.setText("Обучение завершено. Модель сохранена как исследовательская.")
        else:
            self.results_text.append(
                "\n✓ Модель использует рабочий набор признаков,"
                "\nсовместима с экраном ЛПР и может быть активирована."
            )
            self.status_label.setText("Обучение завершено. Модель готова к активации в рабочем контуре.")

    def _on_error(self, message: str) -> None:
        """Ошибка обучения."""
        self.train_btn.setEnabled(True)
        self.assign_rank_btn.setEnabled(True)
        self.sync_lpr_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.results_text.append(f"\n✗ Ошибка: {message}")
        user_message = message
        lowered = message.lower()
        if "остановлено пользователем" in lowered:
            self.status_label.setText("Обучение остановлено пользователем.")
            self.results_text.append("Обучение остановлено пользователем.")
            QMessageBox.information(self, "Остановлено", "Обучение остановлено пользователем.")
            return
        if "assignranktzusecase" in lowered or "нет обучающей целевой переменной" in lowered:
            user_message = (
                "Данные еще не готовы к обучению.\n\n"
                "Сначала выполните шаг «Разметить ранги», чтобы система рассчитала обучающие метки."
            )
            self.status_label.setText("Сначала разметьте ранги, затем запускайте обучение.")
        else:
            self.status_label.setText("Ошибка обучения. Проверьте параметры и журнал выполнения.")
        QMessageBox.critical(self, "Ошибка обучения", user_message)

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
            self.status_label.setText("Модель успешно активирована.")
        else:
            QMessageBox.critical(
                    self, "Ошибка",
                    "Не удалось сделать модель рабочей для ЛПР."
                )

    def _update_feature_mode_hint(self) -> None:
        """Пояснить текущий режим набора признаков."""
        feature_set = self.feature_set_combo.currentData()
        hints = {
            "dispatch_initial_safe": (
                "Рабочий набор для первичного прогноза до прибытия подразделения. "
                "Именно его стоит использовать для модели ЛПР."
            ),
            "arrival_update_safe": (
                "Набор для сценария после прибытия подразделения. "
                "Пока это не основной рабочий экран ЛПР."
            ),
            "first_hose_update_safe": (
                "Набор для сценария после подачи первого ствола. "
                "Нужен для специальных сценариев, не для базового экрана ЛПР."
            ),
            "online_tactical": (
                "Устаревший оперативный набор признаков. "
                "Его не стоит выбирать для нового рабочего контура."
            ),
            "enhanced_tactical": (
                "Исследовательский режим с дополнительными расчетными признаками. "
                "Подходит для экспериментов и сравнения, не для рабочего экрана ЛПР."
            ),
            "retrospective_benchmark": (
                "Архивный сравнительный набор для анализа качества на исторических данных. "
                "Не используется на рабочем экране ЛПР."
            ),
            "custom": (
                "Пользовательский набор без гарантии совместимости с экраном ЛПР. "
                "Для рабочего прогноза не активируется."
            ),
        }
        self.feature_mode_hint.setText(hints.get(feature_set, ""))
        self._populate_synthetic_methods()

    def _update_source_info(self, db_path: Path) -> None:
        """Показать, откуда именно берутся данные для обучения."""
        self.source_db_value.setText(str(db_path))
        counts = self.viewmodel.get_training_source_counts() if self.viewmodel else {}
        historical_total = int(counts.get("historical_total", 0))
        historical_ready = int(counts.get("historical_ready", 0))
        lpr_total = int(counts.get("lpr_total", 0))
        new_lpr = int(counts.get("new_lpr_candidates", 0))
        synthetic_total = int(counts.get("synthetic_total", 0))
        self.historical_rows_value.setText(str(historical_total))
        self.historical_ready_value.setText(str(historical_ready))
        self.lpr_rows_value.setText(str(lpr_total))
        self.new_lpr_value.setText(str(new_lpr))
        self.synthetic_rows_value.setText(str(synthetic_total))
        if historical_ready > 0 or lpr_total > 0:
            self.label_status_value.setText("Источники обучения подготовлены. Можно обучать модель.")
        elif historical_total > 0:
            self.label_status_value.setText("Исторические данные загружены, но ранги еще не размечены.")
        else:
            self.label_status_value.setText("Источники обучения пока пусты.")

    def _populate_synthetic_methods(self) -> None:
        if not self.viewmodel:
            self.synthetic_method_combo.clear()
            self.synthetic_method_combo.addItem("Без синтетики", None)
            self._update_synthetic_controls()
            return
        current_value = self.synthetic_method_combo.currentData()
        self.synthetic_method_combo.clear()
        for item in self.viewmodel.get_available_synthetic_methods():
            self.synthetic_method_combo.addItem(item["label"], item["value"])
            index = self.synthetic_method_combo.count() - 1
            model_item = self.synthetic_method_combo.model().item(index)
            if model_item is not None and not item.get("enabled", True):
                model_item.setEnabled(False)
                reason = item.get("reason", "")
                if reason:
                    model_item.setToolTip(reason)
        index = self.synthetic_method_combo.findData(current_value)
        if index >= 0:
            model_item = self.synthetic_method_combo.model().item(index)
            if model_item is not None and model_item.isEnabled():
                self.synthetic_method_combo.setCurrentIndex(index)
            else:
                self.synthetic_method_combo.setCurrentIndex(0)
        else:
            self.synthetic_method_combo.setCurrentIndex(0)
        self._update_synthetic_controls()

    def _update_synthetic_controls(self) -> None:
        method = self.synthetic_method_combo.currentData()
        enabled = method not in (None, "", "none")
        self.synthetic_k_label.setVisible(enabled)
        self.synthetic_k_combo.setEnabled(enabled)
        self.synthetic_k_combo.setVisible(enabled)
        show_m = enabled and method in {"borderlinesmote", "svmsmote"}
        self.synthetic_m_label.setVisible(show_m)
        self.synthetic_m_combo.setEnabled(enabled and method in {"borderlinesmote", "svmsmote"})
        self.synthetic_m_combo.setVisible(show_m)
        self.synthetic_target_total_label.setVisible(enabled)
        self.synthetic_target_total_spin.setEnabled(enabled)
        self.synthetic_target_total_spin.setVisible(enabled)

    def _update_tuning_controls(self) -> None:
        tuning_enabled = self.tuning_checkbox.isChecked()
        self.class_weight_combo.setEnabled(not tuning_enabled)
        self.tuning_trials_combo.setEnabled(tuning_enabled)
        self.metric_primary_combo.setEnabled(tuning_enabled)

    def _on_sync_lpr(self) -> None:
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
            return
        self.sync_worker = LPRSyncWorker(self.viewmodel)
        self.sync_worker.progress.connect(self._on_sync_progress)
        self.sync_worker.complete.connect(self._on_sync_complete)
        self.sync_worker.error.connect(self._on_sync_error)
        self.sync_lpr_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.assign_rank_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Проверка и подтягивание новых решений ЛПР...")
        self.results_text.append("Запущена загрузка новых решений ЛПР...")
        self.sync_worker.start()

    def _on_sync_progress(self, current: int, total: int, description: str) -> None:
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(description or "Подтягивание решений ЛПР...")

    def _on_sync_complete(self, data: Dict[str, Any]) -> None:
        self.sync_lpr_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.assign_rank_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.viewmodel:
            self._update_source_info(self.viewmodel.db_path)
        added = int(data.get("added", 0))
        total = int(data.get("lpr_total", 0))
        self.status_label.setText(f"Подтягивание ЛПР завершено: добавлено {added}, всего в слое ЛПР {total}.")
        self.results_text.append(
            f"\nЛПР-данные обновлены.\nДобавлено новых записей: {added}\nВсего в analyst-слое ЛПР: {total}"
        )

    def _on_sync_error(self, message: str) -> None:
        self.sync_lpr_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.assign_rank_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка подтягивания решений ЛПР.")
        QMessageBox.critical(self, "Ошибка ЛПР-синхронизации", message)

    def _on_assign_rank(self) -> None:
        """Запустить явную разметку рангов перед обучением."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
            return

        mode = self.assign_mode_combo.currentData()
        override_existing_labels = mode == "relabel_auto"
        if override_existing_labels:
            confirmation = QMessageBox.question(
                self,
                "Пересчитать авторазметку",
                "Будет пересчитана только автоматическая разметка.\n\n"
                "Human-verified строки и решения ЛПР останутся защищены.\n"
                "Продолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmation != QMessageBox.Yes:
                return

        self.assign_worker = AssignRankWorker(
            self.viewmodel.db_path,
            override_existing_labels=override_existing_labels,
            source_table="fires_historical",
        )
        self.assign_worker.progress.connect(self._on_assign_progress)
        self.assign_worker.complete.connect(self._on_assign_complete)
        self.assign_worker.error.connect(self._on_assign_error)
        self.assign_rank_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.sync_lpr_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Выполняется разметка рангов...")
        self.results_text.clear()
        self.results_text.append("Запущена разметка рангов...")
        self.assign_worker.start()

    def _on_assign_complete(self, data: Dict[str, Any]) -> None:
        self.assign_rank_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.sync_lpr_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.viewmodel:
            self._update_source_info(self.viewmodel.db_path)
        updated = data.get("assigned_records", data.get("updated_records", 0))
        null_count = data.get("null_count", 0)
        skipped_existing = data.get("existing_label_skipped_count", 0)
        recalculated_existing = data.get("existing_label_recalculated_count", 0)
        skipped_human = data.get("human_verified_skipped_count", 0)
        skipped_lpr = data.get("lpr_decision_skipped_count", 0)
        ready_now = self.historical_ready_value.text()
        if updated == 0 and (skipped_existing or skipped_human or skipped_lpr):
            self.status_label.setText("Новых строк для разметки нет. Уже размеченные данные защищены от перезаписи.")
        else:
            self.status_label.setText(
                f"Разметка завершена: рассчитано рангов {updated}, готово к обучению {ready_now}."
            )
        lines = [
            "",
            "Разметка завершена.",
            f"Рассчитано рангов: {updated}",
            f"Без рассчитанного ранга: {null_count}",
            f"Готово к обучению: {ready_now}",
        ]
        if skipped_existing:
            lines.append(f"Уже размечённых строк защищено от перезаписи: {skipped_existing}")
        if recalculated_existing:
            lines.append(f"Автоматически размечённых строк пересчитано: {recalculated_existing}")
        if skipped_human:
            lines.append(f"Human-verified строк пропущено: {skipped_human}")
        if skipped_lpr:
            lines.append(f"Строк с решениями ЛПР пропущено: {skipped_lpr}")
        self.results_text.append("\n".join(lines))

        info_lines = [
            "Разметка завершена.",
            "",
            f"Рассчитано рангов: {updated}",
            f"Без рассчитанного ранга: {null_count}",
            f"Готово к обучению: {ready_now}",
        ]
        if skipped_existing:
            info_lines.append(f"Уже размечённых строк защищено от перезаписи: {skipped_existing}")
        if recalculated_existing:
            info_lines.append(f"Автоматически размечённых строк пересчитано: {recalculated_existing}")
        if skipped_human:
            info_lines.append(f"Human-verified строк пропущено: {skipped_human}")
        if skipped_lpr:
            info_lines.append(f"Строк с решениями ЛПР пропущено: {skipped_lpr}")
        QMessageBox.information(
            self,
            "Разметка рангов",
            "\n".join(info_lines),
        )

    def _on_assign_error(self, message: str) -> None:
        self.assign_rank_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.sync_lpr_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка разметки рангов.")
        self.results_text.append(f"\nОшибка разметки: {message}")
        QMessageBox.critical(self, "Ошибка разметки", message)
