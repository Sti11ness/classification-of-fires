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
    QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent, QValidator
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from fire_es.normatives import get_normative_rank_table
from fire_es.rank_tz_contract import DEFAULT_LPR_FEATURE_SET, get_input_schema


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
        self.input_form_layout: Optional[QFormLayout] = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Заголовок
        title = QLabel("Прогноз ранга пожара")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Зона 1: Ввод параметров (разделы 1-2)
        input_group = QGroupBox("Входные параметры пожара (разделы 1-2)")
        input_layout = QFormLayout(input_group)
        input_layout.setSpacing(10)
        self.input_form_layout = input_layout

        main_layout.addWidget(input_group)

        contract_group = QGroupBox("Активный контракт модели")
        contract_layout = QFormLayout(contract_group)
        contract_layout.setSpacing(8)
        self.contract_feature_set = QLabel("dispatch_initial_safe")
        self.contract_stage = QLabel("dispatch_initial")
        self.contract_target = QLabel("rank_tz_vector")
        self.contract_split = QLabel("n/a")
        self.contract_normative_version = QLabel("n/a")
        self.contract_warnings = QLabel("")
        self.contract_warnings.setWordWrap(True)
        self.contract_warnings.setStyleSheet("font-size: 12px; color: #ffd27f;")
        contract_layout.addRow("Feature set:", self.contract_feature_set)
        contract_layout.addRow("Stage:", self.contract_stage)
        contract_layout.addRow("Semantic target:", self.contract_target)
        contract_layout.addRow("Split protocol:", self.contract_split)
        contract_layout.addRow("Normative version:", self.contract_normative_version)
        contract_layout.addRow("Warnings:", self.contract_warnings)
        main_layout.addWidget(contract_group)

        self._set_input_schema(self.input_schema)

        # Кнопка прогноза
        self.predict_btn = QPushButton("Прогнозировать")
        self.predict_btn.setFixedHeight(45)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2085d9;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        main_layout.addWidget(self.predict_btn)

        # Разделитель
        splitter = QSplitter(Qt.Horizontal)

        # Зона 2: Диаграмма вероятностей (слева)
        chart_group = QGroupBox("Прогноз: вероятности рангов")
        chart_layout = QVBoxLayout(chart_group)

        self.chart_placeholder = QLabel(
            "Нажмите «Прогнозировать» для получения результата"
        )
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setStyleSheet(self._chart_empty_style())
        chart_layout.addWidget(self.chart_placeholder)

        # Индикатор уверенности
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet(self._confidence_label_style())
        chart_layout.addWidget(self.confidence_label)

        splitter.addWidget(chart_group)

        # Зона 3: Нормативная таблица (справа)
        normative_group = QGroupBox("Нормативная таблица рангов")
        normative_layout = QVBoxLayout(normative_group)

        self.normative_table = self._create_normative_table()
        normative_layout.addWidget(self.normative_table)

        splitter.addWidget(normative_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

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
        self.rank_combo.setFixedWidth(200)
        select_layout.addWidget(self.rank_combo)
        select_layout.addStretch()

        decision_layout.addLayout(select_layout)

        # Комментарий
        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Комментарий к решению (необязательно)")
        self.comment_edit.setMaximumHeight(80)
        decision_layout.addWidget(self.comment_edit)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.save_btn = QPushButton("Сохранить решение")
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedHeight(40)
        self.save_btn.setStyleSheet("""
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
        buttons_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #9e9e9e;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 4px;
            }
        """)
        buttons_layout.addWidget(self.cancel_btn)

        buttons_layout.addStretch()

        decision_layout.addLayout(buttons_layout)

        main_layout.addWidget(decision_group)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Статус
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: white;")
        main_layout.addWidget(self.status_label)

        self._connect_signals()

    def _create_normative_table(self) -> QTableWidget:
        """Создать нормативную таблицу."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Ранг", "Название", "Техника (ед.)", "Ресурсный вектор", "Версия"
        ])
        table.horizontalHeader().setStretchLastSection(True)
        table.setFixedHeight(200)

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
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
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
        self.status_label.setText("Прогнозирование...")

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
            chart_text = "Топ-K вариантов:\n\n"
            for rank, prob in zip(ranks, probs):
                chart_text += f"{rank}: {prob:.1f}%\n"
            self.chart_placeholder.setText(chart_text)
            self.chart_placeholder.setStyleSheet(self._chart_result_style())

        self.confidence_label.setText(
            f"Уверенность: {confidence:.1f}%"
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
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        QMessageBox.information(self, "Сохранено", message)

    def _on_save_error(self, message: str) -> None:
        """Ошибка сохранения."""
        self.save_btn.setEnabled(True)
        self.status_label.setText("Ошибка сохранения")
        self.status_label.setStyleSheet("color: white;")
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
        self.status_label.setText("")
        self.save_btn.setEnabled(False)

    def _chart_empty_style(self) -> str:
        """Style for the empty prediction area before any result is shown."""
        return (
            "QLabel {"
            " background: #f2f2f2;"
            " color: #444444;"
            " padding: 20px;"
            " border-radius: 4px;"
            " border: 1px solid #b8b8b8;"
            " font-size: 14px;"
            " }"
        )

    def _chart_result_style(self) -> str:
        """Style for prediction output with readable contrast."""
        return (
            "QLabel {"
            " background: #ececec;"
            " color: #1e1e1e;"
            " padding: 20px;"
            " border-radius: 4px;"
            " border: 1px solid #b8b8b8;"
            " font-size: 20px;"
            " font-weight: 600;"
            " }"
        )

    def _confidence_label_style(self) -> str:
        """Style for confidence text under the probability block."""
        return "font-size: 14px; font-weight: bold; color: #ffffff;"

    def _build_input_widgets(self, layout: QFormLayout) -> None:
        """Создать поля ввода из production input schema."""
        for field in self.input_schema:
            widget = self._create_widget_for_field(field)
            self.input_widgets[field["name"]] = widget
            layout.addRow(f"{field['label']}:", widget)
        input_hint = QLabel(
            "Поля редактируются напрямую с клавиатуры. Значение «Не указано» оставляет поле пустым для модели."
        )
        input_hint.setWordWrap(True)
        input_hint.setStyleSheet("font-size: 12px; color: #f0f0f0;")
        layout.addRow("", input_hint)

    def _set_input_schema(self, schema: list[dict[str, Any]]) -> None:
        """Replace the current input schema and rebuild the widgets."""
        self.input_schema = schema
        self.input_widgets = {}
        if self.input_form_layout is None:
            return
        while self.input_form_layout.rowCount():
            self.input_form_layout.removeRow(0)
        self._build_input_widgets(self.input_form_layout)

    def _apply_input_contract(self, contract: Dict[str, Any]) -> None:
        """Apply active-model input contract to the page."""
        self.contract_feature_set.setText(str(contract.get("feature_set", DEFAULT_LPR_FEATURE_SET)))
        self.contract_stage.setText(str(contract.get("availability_stage", "dispatch_initial")))
        self.contract_target.setText(str(contract.get("semantic_target", "rank_tz_vector")))
        self.contract_split.setText(str(contract.get("split_protocol", "n/a")))
        self.contract_normative_version.setText(str(contract.get("normative_version", "n/a")))
        warnings = contract.get("warnings") or []
        self.contract_warnings.setText("; ".join(warnings) if warnings else "Нет")
        schema = contract.get("input_schema") or get_input_schema(DEFAULT_LPR_FEATURE_SET)
        self._set_input_schema(schema)

    def _populate_normative_table(self, normative_df: pd.DataFrame) -> None:
        """Render normative rows from the shared loader output."""
        self.normative_table.setRowCount(0)
        if normative_df is None or normative_df.empty:
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

    def _sync_contract_from_viewmodel(self) -> None:
        """Refresh read-only contract labels from current viewmodel state."""
        if not self.viewmodel:
            return
        model_info = self.viewmodel.state.model_info
        if model_info:
            self.contract_feature_set.setText(model_info.get("feature_set", self.contract_feature_set.text()))
            self.contract_stage.setText(model_info.get("availability_stage", self.contract_stage.text()))
            self.contract_target.setText(model_info.get("semantic_target", self.contract_target.text()))
            self.contract_split.setText(model_info.get("split_protocol", self.contract_split.text()))
            self.contract_normative_version.setText(
                model_info.get("normative_version", self.contract_normative_version.text())
            )
        warnings = self.viewmodel.state.model_warnings
        self.contract_warnings.setText("; ".join(warnings) if warnings else "Нет")

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
        widget.setMinimumWidth(220)
        widget.set_display_suffix(field.get("suffix", ""))
        widget.setAlignment(Qt.AlignLeft)
        widget.setStyleSheet(
            """
            QSpinBox, QDoubleSpinBox {
                padding: 6px 10px;
                font-size: 14px;
            }
            """
        )
        line_edit = widget.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("Не указано")
            line_edit.setClearButtonEnabled(False)
        if field.get("optional_for_lpr", False):
            tooltip = (
                "Если значение неизвестно, оставьте поле пустым. "
                "Не вводите 0 вместо неизвестного значения."
            )
            widget.setToolTip(tooltip)
            if line_edit is not None:
                line_edit.setToolTip(tooltip)
        return widget

    def _collect_input_data(self) -> Dict[str, Any]:
        """Собрать input payload, сохраняя unknown как None."""
        payload: Dict[str, Any] = {}
        for field in self.input_schema:
            widget = self.input_widgets[field["name"]]
            payload[field["name"]] = widget.nullable_value()
        return payload
