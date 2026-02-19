# src/fire_es_desktop/ui/pages/lpr_predict_page.py
"""
LPRPredictPage — экран прогноза для ЛПР.

Согласно spec_second.md раздел 11.6 и рис. 1:
- Слева: барчарт вероятностей по рангам
- Справа: нормативная таблица
- Снизу: блок решения ЛПР (dropdown + кнопки)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox, QTextEdit, QFrame, QSplitter, QMessageBox, QProgressBar,
    QTableWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


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

        # Пример полей (должны соответствовать признакам модели)
        self.building_floors_spin = QSpinBox()
        self.building_floors_spin.setRange(1, 100)
        self.building_floors_spin.setValue(1)
        input_layout.addRow("Этажность здания:", self.building_floors_spin)

        self.fire_floor_spin = QSpinBox()
        self.fire_floor_spin.setRange(1, 100)
        self.fire_floor_spin.setValue(1)
        input_layout.addRow("Этаж пожара:", self.fire_floor_spin)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0, 100)
        self.distance_spin.setValue(1.0)
        self.distance_spin.setSuffix(" км")
        input_layout.addRow("Расстояние до станции:", self.distance_spin)

        self.fatalities_spin = QSpinBox()
        self.fatalities_spin.setRange(0, 100)
        self.fatalities_spin.setValue(0)
        input_layout.addRow("Погибшие:", self.fatalities_spin)

        self.injuries_spin = QSpinBox()
        self.injuries_spin.setRange(0, 500)
        self.injuries_spin.setValue(0)
        input_layout.addRow("Пострадавшие:", self.injuries_spin)

        self.damage_spin = QDoubleSpinBox()
        self.damage_spin.setRange(0, 1000000)
        self.damage_spin.setValue(0)
        self.damage_spin.setSuffix(" руб.")
        input_layout.addRow("Прямой ущерб:", self.damage_spin)

        main_layout.addWidget(input_group)

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
        self.chart_placeholder.setStyleSheet(
            "QLabel { background: #f5f5f5; padding: 20px; border-radius: 4px; }"
        )
        chart_layout.addWidget(self.chart_placeholder)

        # Индикатор уверенности
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 14px; font-weight: bold;")
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
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels([
            "Ранг", "Название", "Техника (ед.)", "Описание"
        ])

        # Данные (упрощённые)
        data = [
            ("1", "Ранг 1", "2", "Минимальный"),
            ("1-бис", "Ранг 1-бис", "2", "Минимальный усиленный"),
            ("2", "Ранг 2", "3", "Средний"),
            ("3", "Ранг 3", "5", "Повышенный"),
            ("4", "Ранг 4", "8", "Высокий"),
            ("5", "Ранг 5", "12", "Максимальный")
        ]

        table.setRowCount(len(data))
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, j, item)

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
        else:
            self.viewmodel = None
            self.predict_btn.setEnabled(False)

    def _on_predict(self) -> None:
        """Выполнить прогноз."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
            return

        # Собрать входные данные
        input_data = {
            "building_floors": self.building_floors_spin.value(),
            "fire_floor": self.fire_floor_spin.value(),
            "distance_to_station": self.distance_spin.value(),
            "fatalities": self.fatalities_spin.value(),
            "injuries": self.injuries_spin.value(),
            "direct_damage": self.damage_spin.value()
        }

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

        # Обновить диаграмму (заглушка — в реальности использовать график)
        ranks = chart_data.get("ranks", [])
        probs = chart_data.get("probabilities", [])
        confidence = chart_data.get("confidence", 0)

        if ranks:
            chart_text = "Топ-K вариантов:\n\n"
            for rank, prob in zip(ranks, probs):
                chart_text += f"{rank}: {prob:.1f}%\n"
            self.chart_placeholder.setText(chart_text)
            self.chart_placeholder.setStyleSheet(
                "QLabel { background: #e3f2fd; padding: 20px; "
                "border-radius: 4px; font-size: 14px; }"
            )

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

        self.rank_combo.setCurrentIndex(0)
        self.comment_edit.clear()
        self.chart_placeholder.setText(
            "Нажмите «Прогнозировать» для получения результата"
        )
        self.chart_placeholder.setStyleSheet(
            "QLabel { background: #f5f5f5; padding: 20px; border-radius: 4px; }"
        )
        self.confidence_label.setText("")
        self.status_label.setText("")
        self.save_btn.setEnabled(False)
