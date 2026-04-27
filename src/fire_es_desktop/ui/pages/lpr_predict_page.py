# src/fire_es_desktop/ui/pages/lpr_predict_page.py
from __future__ import annotations

"""
LPRPredictPage — экран прогноза для ЛПР.

Согласно spec_second.md раздел 11.6 и рис. 1:
- Слева: барчарт вероятностей по рангам
- Справа: нормативная таблица
- Снизу: блок решения ЛПР (dropdown + кнопки)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QComboBox, QTextEdit, QSplitter, QMessageBox, QProgressBar,
    QAbstractSpinBox,
    QTableWidget, QTableWidgetItem, QGridLayout, QToolButton, QSizePolicy,
    QBoxLayout,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent, QValidator
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from fire_es.normatives import get_normative_rank_table
from fire_es.rank_tz_contract import DEFAULT_LPR_FEATURE_SET, get_input_schema
from ..theme import (
    configure_form_layout,
    configure_grid_layout,
    configure_table,
    configure_text_panel,
    create_hint,
    create_page_header,
    create_scrollable_page,
    create_status_label,
    style_button,
    style_label,
)


class NullableSpinBox(QSpinBox):
    """QSpinBox with placeholder-style null state."""

    def __init__(
        self,
        null_value: int = -1,
        *,
        input_min: int = 0,
        allow_negative_input: bool = False,
    ):
        super().__init__()
        self._null_value = null_value
        self._input_min = input_min
        self._allow_negative_input = allow_negative_input
        self._display_suffix = ""
        self._is_null = True
        self._syncing_null_state = False

        self.setSpecialValueText("")
        self.lineEdit().setPlaceholderText("Не указано")
        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.editingFinished.connect(self._on_editing_finished)
        self.valueChanged.connect(self._on_value_changed)
        self.set_nullable_value(None)

    def set_nullable_value(self, value: Optional[int]) -> None:
        """Set a nullable value; None keeps the field visually empty."""
        self._syncing_null_state = True
        if value is None:
            self._is_null = True
            self.setSuffix("")
            baseline = self._null_value
            if baseline < self.minimum() or baseline > self.maximum():
                baseline = self._input_min
            super().setValue(int(baseline))
        else:
            self._is_null = False
            self.setSuffix(self._display_suffix)
            super().setValue(int(value))
        self._syncing_null_state = False

    def nullable_value(self) -> Optional[int]:
        """Return None when the field is visually empty."""
        return None if self._is_null else int(super().value())

    def _on_text_edited(self, text: str) -> None:
        self._is_null = text.strip() == ""

    def _on_editing_finished(self) -> None:
        if self.lineEdit().text().strip() == "":
            self.set_nullable_value(None)
        else:
            self.interpretText()
            self._is_null = False
            self.setSuffix(self._display_suffix)

    def _on_value_changed(self, _value: int) -> None:
        if not self._syncing_null_state and self.lineEdit().text().strip():
            self._is_null = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._is_null and self._should_prepare_for_typing(event):
            self._prime_for_direct_input(0)
        super().keyPressEvent(event)

    def _should_prepare_for_typing(self, event: QKeyEvent) -> bool:
        text = event.text()
        return bool(text) and (text.isdigit() or (text == "-" and self._allow_negative_input))

    def _prime_for_direct_input(self, seed_value: int) -> None:
        self._syncing_null_state = True
        self.setSuffix(self._display_suffix)
        super().setValue(seed_value)
        self.lineEdit().selectAll()
        self._syncing_null_state = False
        self._is_null = False

    def stepBy(self, steps: int) -> None:
        if self._is_null:
            self._is_null = False
            self.setSuffix(self._display_suffix)
            super().setValue(self._input_min)
        if self._allow_negative_input:
            super().stepBy(steps)
            return
        target = max(self._input_min, int(super().value() + steps * self.singleStep()))
        super().setValue(target)

    def wheelEvent(self, event) -> None:
        event.ignore()

    def validate(self, text: str, pos: int):
        if not self._allow_negative_input and text.strip().startswith("-"):
            return QValidator.Invalid, text, pos
        return super().validate(text, pos)

    def textFromValue(self, value: int) -> str:
        if self._is_null:
            return ""
        return super().textFromValue(value)

    def set_display_suffix(self, suffix: str) -> None:
        """Configure a suffix that is hidden while the field is null."""
        self._display_suffix = suffix
        if not self._is_null:
            self.setSuffix(suffix)


class NullableDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with placeholder-style null state."""

    def __init__(
        self,
        null_value: float = -1.0,
        *,
        input_min: float = 0.0,
        allow_negative_input: bool = False,
    ):
        super().__init__()
        self._null_value = null_value
        self._input_min = input_min
        self._allow_negative_input = allow_negative_input
        self._display_suffix = ""
        self._is_null = True
        self._syncing_null_state = False

        self.setSpecialValueText("")
        self.lineEdit().setPlaceholderText("Не указано")
        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.editingFinished.connect(self._on_editing_finished)
        self.valueChanged.connect(self._on_value_changed)
        self.set_nullable_value(None)

    def set_nullable_value(self, value: Optional[float]) -> None:
        """Set a nullable value; None keeps the field visually empty."""
        self._syncing_null_state = True
        if value is None:
            self._is_null = True
            self.setSuffix("")
            baseline = self._null_value
            if baseline < self.minimum() or baseline > self.maximum():
                baseline = self._input_min
            super().setValue(float(baseline))
        else:
            self._is_null = False
            self.setSuffix(self._display_suffix)
            super().setValue(float(value))
        self._syncing_null_state = False

    def nullable_value(self) -> Optional[float]:
        """Return None when the field is visually empty."""
        return None if self._is_null else float(super().value())

    def _on_text_edited(self, text: str) -> None:
        self._is_null = text.strip() == ""

    def _on_editing_finished(self) -> None:
        if self.lineEdit().text().strip() == "":
            self.set_nullable_value(None)
        else:
            self.interpretText()
            self._is_null = False
            self.setSuffix(self._display_suffix)

    def _on_value_changed(self, _value: float) -> None:
        if not self._syncing_null_state and self.lineEdit().text().strip():
            self._is_null = False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._is_null and self._should_prepare_for_typing(event):
            self._prime_for_direct_input(0.0)
        super().keyPressEvent(event)

    def _should_prepare_for_typing(self, event: QKeyEvent) -> bool:
        text = event.text()
        allowed_prefix = {"-", ".", ","} if self._allow_negative_input else {".", ","}
        return bool(text) and (text.isdigit() or text in allowed_prefix)

    def _prime_for_direct_input(self, seed_value: float) -> None:
        self._syncing_null_state = True
        self.setSuffix(self._display_suffix)
        super().setValue(seed_value)
        self.lineEdit().selectAll()
        self._syncing_null_state = False
        self._is_null = False

    def stepBy(self, steps: int) -> None:
        if self._is_null:
            self._is_null = False
            self.setSuffix(self._display_suffix)
            super().setValue(self._input_min)
        if self._allow_negative_input:
            super().stepBy(steps)
            return
        target = max(self._input_min, float(super().value() + steps * self.singleStep()))
        super().setValue(target)

    def wheelEvent(self, event) -> None:
        event.ignore()

    def validate(self, text: str, pos: int):
        if not self._allow_negative_input and text.strip().startswith("-"):
            return QValidator.Invalid, text, pos
        return super().validate(text, pos)

    def textFromValue(self, value: float) -> str:
        if self._is_null:
            return ""
        return super().textFromValue(value)

    def set_display_suffix(self, suffix: str) -> None:
        """Configure a suffix that is hidden while the field is null."""
        self._display_suffix = suffix
        if not self._is_null:
            self.setSuffix(suffix)


class PredictWorker(QThread):
    """Рабочий поток для прогноза."""

    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        """Выполнить прогноз."""
        try:
            self.viewmodel.predict()

            if self.viewmodel.state.error_message:
                self.error.emit(self.viewmodel.state.error_message)
            else:
                self.complete.emit(self.viewmodel.get_prediction_chart_data())
        except Exception as e:
            self.error.emit(str(e))


class SaveDecisionWorker(QThread):
    """Рабочий поток для сохранения решения."""

    complete = Signal(str)
    error = Signal(str)

    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        """Сохранить решение."""
        try:
            self.viewmodel.save_decision()

            if self.viewmodel.state.error_message:
                self.error.emit(self.viewmodel.state.error_message)
            else:
                self.complete.emit("Решение сохранено")
        except Exception as e:
            self.error.emit(str(e))


class LPRPredictPage(QWidget):
    """
    Страница прогноза для ЛПР.

    Компонвка по рис. 1:
    - Верх: ввод параметров (разделы 1-2)
    - Слева: диаграмма вероятностей
    - Справа: нормативная таблица
    - Низ: решение ЛПР
    """

    def __init__(self):
        super().__init__()
        self.viewmodel = None
        self.predict_worker = None
        self.save_worker = None
        self.input_schema = get_input_schema(DEFAULT_LPR_FEATURE_SET)
        self.input_widgets: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self.input_grid_layout: Optional[QGridLayout] = None
        self._current_input_columns = 2
        self.input_group: Optional[QGroupBox] = None
        self.model_state_group: Optional[QGroupBox] = None
        self.top_section_layout: Optional[QBoxLayout] = None
        self.results_section_layout: Optional[QBoxLayout] = None
        self.top_section_widget: Optional[QWidget] = None
        self.results_section_widget: Optional[QWidget] = None
        self.normative_group: Optional[QGroupBox] = None
        self.input_hint_label: Optional[QLabel] = None
        self._top_compact_mode: Optional[bool] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        _, _, _, main_layout = create_scrollable_page(self)

        # Заголовок
        main_layout.addWidget(
            create_page_header(
                "Прогноз ранга пожара",
                "Оперативный сценарий до прибытия подразделения: введите данные пожара, посмотрите наиболее вероятные варианты и сохраните решение.",
            )
        )

        scenario_label = QLabel("Прогноз до прибытия подразделения")
        scenario_label.setAlignment(Qt.AlignCenter)
        style_label(scenario_label, "metric", word_wrap=False)
        main_layout.addWidget(scenario_label)

        # Зона 1: Ввод параметров (разделы 1-2)
        input_group = QGroupBox("Исходные данные для прогноза")
        input_layout = QGridLayout(input_group)
        configure_grid_layout(input_layout)
        self.input_grid_layout = input_layout
        self.input_group = input_group

        model_state_group = QGroupBox("Состояние модели")
        model_state_layout = QVBoxLayout(model_state_group)
        self.model_state_group = model_state_group
        self.model_state_label = QLabel("Рабочая модель для прогноза ЛПР не выбрана")
        style_label(self.model_state_label, "value", word_wrap=True)
        self.model_state_label.setMinimumHeight(72)
        model_state_layout.addWidget(self.model_state_label)
        self.model_state_hint = create_hint(
            "Если рабочая модель не выбрана, можно заполнить базовую форму и сохранить решение вручную."
        )
        model_state_layout.addWidget(self.model_state_hint)

        self.service_toggle_btn = QToolButton()
        self.service_toggle_btn.setText("Показать служебные сведения")
        self.service_toggle_btn.setCheckable(True)
        self.service_toggle_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.service_toggle_btn.setCursor(Qt.PointingHandCursor)
        model_state_layout.addWidget(self.service_toggle_btn, alignment=Qt.AlignLeft)

        contract_group = QGroupBox("Служебные сведения")
        contract_group.setVisible(False)
        contract_layout = QFormLayout(contract_group)
        configure_form_layout(contract_layout)
        self.contract_feature_set = QLabel("dispatch_initial_safe")
        self.contract_stage = QLabel("dispatch_initial")
        self.contract_target = QLabel("rank_tz_vector")
        self.contract_split = QLabel("n/a")
        self.contract_normative_version = QLabel("n/a")
        self.contract_warnings = QLabel("")
        self.contract_warnings.setWordWrap(True)
        style_label(self.contract_feature_set, "value", word_wrap=True)
        style_label(self.contract_stage, "value", word_wrap=True)
        style_label(self.contract_target, "value", word_wrap=True)
        style_label(self.contract_split, "value", word_wrap=True)
        style_label(self.contract_normative_version, "value", word_wrap=True)
        style_label(self.contract_warnings, "section-hint", word_wrap=True)
        contract_layout.addRow("Набор признаков:", self.contract_feature_set)
        contract_layout.addRow("Стадия данных:", self.contract_stage)
        contract_layout.addRow("Тип разметки:", self.contract_target)
        contract_layout.addRow("Проверка качества:", self.contract_split)
        contract_layout.addRow("Версия нормативов:", self.contract_normative_version)
        contract_layout.addRow("Предупреждения:", self.contract_warnings)
        self.service_group = contract_group
        model_state_layout.addWidget(self.service_group)
        self.service_toggle_btn.hide()
        self.service_group.hide()

        self._set_input_schema(self.input_schema)

        input_group.setMinimumHeight(280)
        model_state_group.setMinimumHeight(210)
        model_state_group.setMinimumWidth(250)
        model_state_group.setMaximumWidth(360)
        model_state_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        input_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        top_section = QWidget()
        self.top_section_widget = top_section
        self.top_section_layout = QBoxLayout(QBoxLayout.LeftToRight, top_section)
        self.top_section_layout.setContentsMargins(0, 0, 0, 0)
        self.top_section_layout.setSpacing(12)
        self.top_section_layout.addWidget(input_group, 1)
        self.top_section_layout.addWidget(model_state_group, 0)
        top_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        main_layout.addWidget(top_section)

        # Кнопка прогноза
        self.predict_btn = QPushButton("Построить прогноз")
        style_button(self.predict_btn, "primary", large=True)
        main_layout.addWidget(self.predict_btn)

        # Разделитель
        results_section = QWidget()
        self.results_section_widget = results_section
        self.results_section_layout = QBoxLayout(QBoxLayout.LeftToRight, results_section)
        self.results_section_layout.setContentsMargins(0, 0, 0, 0)
        self.results_section_layout.setSpacing(12)
        results_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Зона 2: Диаграмма вероятностей (слева)
        chart_group = QGroupBox("Наиболее вероятные варианты")
        chart_layout = QVBoxLayout(chart_group)

        self.chart_placeholder = QLabel(
            "Нажмите «Прогнозировать» для получения результата"
        )
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setWordWrap(True)
        self.chart_placeholder.setStyleSheet(self._chart_empty_style())
        chart_layout.addWidget(self.chart_placeholder)

        # Индикатор уверенности
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet(self._confidence_label_style())
        chart_layout.addWidget(self.confidence_label)

        chart_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.results_section_layout.addWidget(chart_group, 1)

        # Зона 3: Нормативная таблица (справа)
        normative_group = QGroupBox("Нормативная таблица рангов")
        self.normative_group = normative_group
        normative_layout = QVBoxLayout(normative_group)

        self.normative_table = self._create_normative_table()
        normative_layout.addWidget(self.normative_table)
        normative_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.results_section_layout.addWidget(normative_group, 1)

        main_layout.addWidget(results_section)
        self.normative_hint = create_hint(
            "Таблица показывает справочное соответствие ранга и минимально требуемых ресурсов."
        )
        main_layout.addWidget(self.normative_hint)

        # Зона 4: Решение ЛПР (внизу)
        decision_group = QGroupBox("Решение ЛПР о ранге")
        decision_layout = QVBoxLayout(decision_group)

        # Выбор ранга
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Выбранный ранг:"))

        self.rank_combo = QComboBox()
        self.rank_combo.addItems([
            "Не выбрано",
            "1", "1-бис", "2", "3", "4", "5"
        ])
        self.rank_combo.setMinimumWidth(160)
        self.rank_combo.setMaximumWidth(220)
        select_layout.addWidget(self.rank_combo)
        select_layout.addStretch()

        decision_layout.addLayout(select_layout)

        # Комментарий
        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Комментарий к решению (необязательно)")
        configure_text_panel(self.comment_edit, min_height=120)
        self.comment_edit.setMaximumHeight(150)
        decision_layout.addWidget(self.comment_edit)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.save_btn = QPushButton("Сохранить решение")
        self.save_btn.setEnabled(False)
        style_button(self.save_btn, "success")
        buttons_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Отмена")
        style_button(self.cancel_btn, "ghost")
        buttons_layout.addWidget(self.cancel_btn)

        buttons_layout.addStretch()

        decision_layout.addLayout(buttons_layout)

        main_layout.addWidget(decision_group)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Статус
        self.status_label = create_status_label()
        self.status_label.setText("Если значение неизвестно, оставьте поле пустым. Ноль не используется вместо неизвестного значения.")
        main_layout.addWidget(self.status_label)

        self.service_toggle_btn.toggled.connect(self._toggle_service_details)
        self._connect_signals()
        self._update_responsive_layout()

    def _create_normative_table(self) -> QTableWidget:
        """Создать нормативную таблицу."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Ранг", "Название", "Техника (ед.)", "Состав ресурсов", "Версия"
        ])
        configure_table(table, min_height=200, stretch_last=True)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        return table

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.predict_btn.clicked.connect(self._on_predict)
        self.save_btn.clicked.connect(self._on_save)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.rank_combo.currentTextChanged.connect(self._on_rank_changed)

    def set_paths(self, db_path: Optional[Path],
                  models_path: Optional[Path]) -> None:
        """Установить пути."""
        if db_path and models_path:
            from ...viewmodels import LPRPredictViewModel
            self.viewmodel = LPRPredictViewModel(models_path, db_path)
            self.predict_btn.setEnabled(True)
            self._apply_input_contract(self.viewmodel.get_input_contract())
            self._populate_normative_table(self.viewmodel.get_normative_table())
        else:
            self.viewmodel = None
            self.predict_btn.setEnabled(False)
            self._apply_input_contract(
                {
                    "model_status": "Рабочая модель для прогноза ЛПР не выбрана",
                    "feature_set": DEFAULT_LPR_FEATURE_SET,
                    "availability_stage": "dispatch_initial",
                    "semantic_target": "rank_tz_vector",
                    "input_schema": get_input_schema(DEFAULT_LPR_FEATURE_SET),
                    "split_protocol": "n/a",
                    "normative_version": "n/a",
                    "warnings": [],
                }
            )
            self._populate_normative_table(
                get_normative_rank_table().rename(
                    columns={
                        "label": "rank_name",
                        "display_name": "rank_display_name",
                        "min_equipment_count": "equipment_count",
                    }
                )
            )

    def _on_predict(self) -> None:
        """Выполнить прогноз."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
            return

        # Собрать входные данные согласно production contract.
        input_data = self._collect_input_data()

        # Установить в ViewModel
        for key, value in input_data.items():
            self.viewmodel.set_input_data(key, value)

        # Запустить в фоне
        self.predict_worker = PredictWorker(self.viewmodel)
        self.predict_worker.complete.connect(self._on_predict_complete)
        self.predict_worker.error.connect(self._on_predict_error)

        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Выполняется прогноз...")

        self.predict_worker.start()

    def _on_predict_complete(self, chart_data: Dict) -> None:
        """Прогноз завершён."""
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Прогноз выполнен")
        self._sync_contract_from_viewmodel()

        # Обновить диаграмму (заглушка — в реальности использовать график)
        ranks = chart_data.get("ranks", [])
        probs = chart_data.get("probabilities", [])
        confidence = chart_data.get("confidence", 0)

        if ranks:
            chart_text = "Наиболее вероятные варианты:\n\n"
            for rank, prob in zip(ranks, probs):
                chart_text += f"{rank}: {prob:.1f}%\n"
            self.chart_placeholder.setText(chart_text)
            self.chart_placeholder.setStyleSheet(self._chart_result_style())

        self.confidence_label.setText(
            f"Уверенность топ-1 прогноза: {confidence:.1f}%"
        )

        # Разблокировать сохранение
        self.save_btn.setEnabled(True)

    def _on_predict_error(self, message: str) -> None:
        """Ошибка прогноза."""
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка прогноза")
        self.chart_placeholder.setText("Ошибка прогноза")
        self.confidence_label.setText("")
        QMessageBox.critical(self, "Ошибка прогноза", message)

    def _on_rank_changed(self, rank: str) -> None:
        """Изменён выбранный ранг."""
        if self.viewmodel and rank != "Не выбрано":
            self.viewmodel.set_selected_rank(rank)

    def _on_save(self) -> None:
        """Сохранить решение."""
        if not self.viewmodel:
            return

        if self.viewmodel.state.selected_rank is None:
            QMessageBox.warning(
                self, "Предупреждение",
                "Выберите ранг решения"
            )
            return

        self.viewmodel.set_decision_comment(self.comment_edit.toPlainText())

        # Запустить в фоне
        self.save_worker = SaveDecisionWorker(self.viewmodel)
        self.save_worker.complete.connect(self._on_save_complete)
        self.save_worker.error.connect(self._on_save_error)

        self.save_btn.setEnabled(False)
        self.status_label.setText("Сохранение решения...")

        self.save_worker.start()

    def _on_save_complete(self, message: str) -> None:
        """Решение сохранено."""
        self.save_btn.setEnabled(True)
        self.status_label.setText(message)
        QMessageBox.information(self, "Сохранено", message)

    def _on_save_error(self, message: str) -> None:
        """Ошибка сохранения."""
        self.save_btn.setEnabled(True)
        self.status_label.setText("Ошибка сохранения")
        QMessageBox.critical(self, "Ошибка", message)

    def _on_cancel(self) -> None:
        """Отмена."""
        if self.viewmodel:
            self.viewmodel.clear_input_data()

        for field in self.input_schema:
            widget = self.input_widgets[field["name"]]
            widget.set_nullable_value(None)
        self.rank_combo.setCurrentIndex(0)
        self.comment_edit.clear()
        self.chart_placeholder.setText(
            "Нажмите «Прогнозировать» для получения результата"
        )
        self.chart_placeholder.setStyleSheet(self._chart_empty_style())
        self.confidence_label.setText("")
        self.status_label.setText("Если значение неизвестно, оставьте поле пустым. Ноль не используется вместо неизвестного значения.")
        self.save_btn.setEnabled(False)

    def _chart_empty_style(self) -> str:
        """Style for the empty prediction area before any result is shown."""
        return (
            "QLabel {"
            " background: #0d1217;"
            " color: #9fb0bc;"
            " padding: 16px;"
            " border-radius: 10px;"
            " border: 1px solid #28333d;"
            " font-size: 14px;"
            " }"
        )

    def _chart_result_style(self) -> str:
        """Style for prediction output with readable contrast."""
        return (
            "QLabel {"
            " background: #162129;"
            " color: #eff5f8;"
            " padding: 16px;"
            " border-radius: 10px;"
            " border: 1px solid #2d6f73;"
            " font-size: 16px;"
            " font-weight: 600;"
            " }"
        )

    def _confidence_label_style(self) -> str:
        """Style for confidence text under the probability block."""
        return "font-size: 13px; font-weight: bold; color: #d3eef0;"

    def _build_input_widgets(self, layout: QGridLayout) -> None:
        """Создать более компактную сетку ввода для экрана ЛПР."""
        column_count = self._input_column_count()
        for column in range(4):
            layout.setColumnMinimumWidth(column, 0)
            layout.setColumnStretch(column, 0)
        if column_count == 1:
            layout.setColumnStretch(0, 1)
        else:
            layout.setColumnMinimumWidth(0, 170)
            layout.setColumnMinimumWidth(1, 220)
            layout.setColumnMinimumWidth(2, 170)
            layout.setColumnMinimumWidth(3, 220)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(2, 1)
        for index, field in enumerate(self.input_schema):
            widget = self._create_widget_for_field(field)
            self.input_widgets[field["name"]] = widget
            label = QLabel(field["label"])
            style_label(label, "metric", word_wrap=False)
            label.setMinimumWidth(150)
            if column_count == 1:
                row = index * 2
                layout.addWidget(label, row, 0, 1, 4)
                layout.addWidget(widget, row + 1, 0, 1, 4)
            else:
                row = index // 2
                label_column = (index % 2) * 2
                layout.addWidget(label, row, label_column)
                layout.addWidget(widget, row, label_column + 1)

        input_hint = QLabel(
            "Если значение неизвестно, оставьте поле пустым. Ноль не используется вместо неизвестного значения."
        )
        style_label(input_hint, "section-hint", word_wrap=True)
        self.input_hint_label = input_hint
        hint_row = (len(self.input_schema) * 2 + 1) if column_count == 1 else ((len(self.input_schema) + 1) // 2 + 1)
        layout.addWidget(input_hint, hint_row, 0, 1, 4 if column_count > 1 else 2)
        self._refresh_dynamic_heights()

    def _set_input_schema(self, schema: list[dict[str, Any]]) -> None:
        """Replace the current input schema and rebuild the widgets."""
        self.input_schema = schema
        self.input_widgets = {}
        self.input_hint_label = None
        if self.input_grid_layout is None:
            return
        while self.input_grid_layout.count():
            item = self.input_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._build_input_widgets(self.input_grid_layout)

    def _apply_input_contract(self, contract: Dict[str, Any]) -> None:
        """Apply active-model input contract to the page."""
        self.model_state_label.setText(contract.get("model_status", "Рабочая модель для прогноза ЛПР не выбрана"))
        self.contract_feature_set.setText(str(contract.get("feature_set", DEFAULT_LPR_FEATURE_SET)))
        self.contract_stage.setText(self._format_stage_value(contract.get("availability_stage", "dispatch_initial")))
        self.contract_target.setText(self._format_target_value(contract.get("semantic_target", "rank_tz_vector")))
        self.contract_split.setText(self._format_split_value(contract.get("split_protocol", "n/a")))
        self.contract_normative_version.setText(str(contract.get("normative_version", "n/a")))
        warnings = contract.get("warnings") or []
        self.contract_warnings.setText("; ".join(warnings) if warnings else "Нет")
        schema = contract.get("input_schema") or get_input_schema(DEFAULT_LPR_FEATURE_SET)
        self._set_input_schema(schema)

    def _populate_normative_table(self, normative_df: pd.DataFrame) -> None:
        """Render normative rows from the shared loader output."""
        self.normative_table.setUpdatesEnabled(False)
        try:
            self.normative_table.setRowCount(0)
            if normative_df is None or normative_df.empty:
                self._refresh_dynamic_heights()
                return
            self.normative_table.setRowCount(len(normative_df))
            for row_index, (_, row) in enumerate(normative_df.iterrows()):
                values = [
                    str(row.get("rank_name", "")),
                    str(row.get("rank_display_name", row.get("rank_name", ""))),
                    str(row.get("equipment_count", "")),
                    str(row.get("resource_vector", "")),
                    self.contract_normative_version.text(),
                ]
                for col_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.normative_table.setItem(row_index, col_index, item)
            self.normative_table.resizeRowsToContents()
            header_height = self.normative_table.horizontalHeader().height()
            rows_height = sum(self.normative_table.rowHeight(i) for i in range(len(normative_df)))
            scrollbar_height = self.normative_table.horizontalScrollBar().sizeHint().height()
            table_height = header_height + rows_height + scrollbar_height + self.normative_table.frameWidth() * 2 + 12
            self.normative_table.setMinimumHeight(table_height)
            self.normative_table.setMaximumHeight(table_height)
            self._refresh_dynamic_heights()
        finally:
            self.normative_table.setUpdatesEnabled(True)

    def _sync_contract_from_viewmodel(self) -> None:
        """Refresh read-only contract labels from current viewmodel state."""
        if not self.viewmodel:
            return
        model_info = self.viewmodel.state.model_info
        if model_info:
            self.model_state_label.setText("Для прогноза используется рабочая модель")
            self.contract_feature_set.setText(model_info.get("feature_set", self.contract_feature_set.text()))
            self.contract_stage.setText(self._format_stage_value(model_info.get("availability_stage", self.contract_stage.text())))
            self.contract_target.setText(self._format_target_value(model_info.get("semantic_target", self.contract_target.text())))
            self.contract_split.setText(self._format_split_value(model_info.get("split_protocol", self.contract_split.text())))
            self.contract_normative_version.setText(
                model_info.get("normative_version", self.contract_normative_version.text())
            )
        warnings = self.viewmodel.state.model_warnings
        self.contract_warnings.setText("; ".join(warnings) if warnings else "Нет")

    def _toggle_service_details(self, checked: bool) -> None:
        self.service_group.setVisible(checked)
        self.service_toggle_btn.setText(
            "Скрыть служебные сведения" if checked else "Показать служебные сведения"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self) -> None:
        target_columns = self._input_column_count()
        compact_top = self.width() < 1700 or target_columns == 1
        compact_results = self.width() < 1280
        top_mode_changed = compact_top != self._top_compact_mode
        self._top_compact_mode = compact_top
        if self.top_section_layout is not None:
            self.top_section_layout.setDirection(QBoxLayout.TopToBottom if compact_top else QBoxLayout.LeftToRight)
        if self.results_section_layout is not None:
            self.results_section_layout.setDirection(QBoxLayout.TopToBottom if compact_results else QBoxLayout.LeftToRight)
        if self.model_state_group is not None:
            if compact_top:
                self.model_state_group.setMinimumWidth(0)
                self.model_state_group.setMaximumWidth(16777215)
                self.model_state_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            else:
                self.model_state_group.setMinimumWidth(250)
                self.model_state_group.setMaximumWidth(360)
                self.model_state_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        if target_columns != self._current_input_columns or top_mode_changed:
            self._current_input_columns = target_columns
            self._set_input_schema(self.input_schema)
        else:
            self._refresh_dynamic_heights()

    def _input_column_count(self) -> int:
        available_width = self.input_group.width() if self.input_group is not None else 0
        if available_width <= 0:
            available_width = self.width()
        return 1 if available_width < 1380 else 2

    def _refresh_dynamic_heights(self) -> None:
        if self.input_group is not None:
            self.input_group.layout().activate()
            self.input_group.setMinimumHeight(max(280, self.input_group.sizeHint().height()))
        if self.model_state_group is not None:
            self.model_state_group.layout().activate()
            self.model_state_group.setMinimumHeight(max(210, self.model_state_group.sizeHint().height()))
        if self.top_section_widget is not None and self.top_section_layout is not None:
            spacing = self.top_section_layout.spacing()
            if self.top_section_layout.direction() == QBoxLayout.TopToBottom:
                top_height = self.input_group.minimumHeight() + self.model_state_group.minimumHeight() + spacing
            else:
                top_height = max(self.input_group.minimumHeight(), self.model_state_group.minimumHeight())
            self.top_section_widget.setMinimumHeight(top_height)
        if self.results_section_widget is not None and self.results_section_layout is not None:
            chart_widget = self.results_section_layout.itemAt(0).widget() if self.results_section_layout.count() > 0 else None
            chart_height = chart_widget.sizeHint().height() if chart_widget is not None else 0
            normative_height = self.normative_group.sizeHint().height() if self.normative_group is not None else 0
            spacing = self.results_section_layout.spacing()
            if self.results_section_layout.direction() == QBoxLayout.TopToBottom:
                results_height = chart_height + normative_height + spacing
            else:
                results_height = max(chart_height, normative_height)
            self.results_section_widget.setMinimumHeight(results_height)

    @staticmethod
    def _format_stage_value(value: Any) -> str:
        mapping = {
            "dispatch_initial": "до прибытия подразделения",
            "arrival_update": "после прибытия подразделения",
            "first_hose_update": "после подачи первого ствола",
            "retrospective": "архивный режим",
        }
        return mapping.get(str(value), str(value))

    @staticmethod
    def _format_target_value(value: Any) -> str:
        mapping = {
            "rank_tz_vector": "по нормативному вектору",
            "rank_tz_count_proxy": "по количеству техники",
        }
        return mapping.get(str(value), str(value))

    @staticmethod
    def _format_split_value(value: Any) -> str:
        mapping = {
            "group_shuffle": "групповое разделение без пересечения событий",
            "group_kfold": "групповая перекрестная проверка",
            "temporal_holdout": "разделение по времени",
            "row_random_legacy": "устаревшее случайное разделение",
            "n/a": "—",
            "": "—",
            None: "—",
        }
        return mapping.get(value, str(value))

    def _create_widget_for_field(
        self,
        field: Dict[str, Any],
    ) -> NullableSpinBox | NullableDoubleSpinBox:
        """Создать виджет для одного поля input schema."""
        field_min = field.get("min", 0)
        field_max = field.get("max", 1000000)
        null_sentinel = field.get("null_sentinel", -1 if field.get("type") != "float" else -1.0)
        allow_negative_input = field.get("allow_negative_input", field_min < 0)
        effective_min = min(field_min, null_sentinel)

        if field.get("type") == "float":
            widget = NullableDoubleSpinBox(
                null_sentinel,
                input_min=float(field_min),
                allow_negative_input=allow_negative_input,
            )
            widget.setDecimals(field.get("decimals", 2))
            widget.setRange(float(effective_min), float(field_max))
        else:
            widget = NullableSpinBox(
                int(null_sentinel),
                input_min=int(field_min),
                allow_negative_input=allow_negative_input,
            )
            widget.setRange(int(effective_min), int(field_max))
        widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
        widget.setAccelerated(True)
        widget.setKeyboardTracking(False)
        widget.setFrame(True)
        widget.setMinimumWidth(180)
        widget.setMaximumWidth(220)
        widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        widget.set_display_suffix(field.get("suffix", ""))
        widget.setAlignment(Qt.AlignLeft)
        widget.setMinimumHeight(34)
        line_edit = widget.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("Не указано")
            line_edit.setClearButtonEnabled(False)
        return widget

    def _collect_input_data(self) -> Dict[str, Any]:
        """Собрать input payload, сохраняя unknown как None."""
        payload: Dict[str, Any] = {}
        for field in self.input_schema:
            widget = self.input_widgets[field["name"]]
            payload[field["name"]] = widget.nullable_value()
        return payload
