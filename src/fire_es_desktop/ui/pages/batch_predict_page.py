# src/fire_es_desktop/ui/pages/batch_predict_page.py
"""
BatchPredictPage — экран пакетного прогноза.

Согласно spec_first.md раздел 8.2:
- загрузить файл с множеством кейсов
- прогнать активную модель по всем строкам
- сохранить на выход Excel/CSV
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QFileDialog, QProgressBar,
    QMessageBox, QComboBox, QSpinBox, QCheckBox, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from pathlib import Path
from typing import Optional, Dict, Any

import logging

logger = logging.getLogger("BatchPredictPage")


class BatchPredictWorker(QThread):
    """Рабочий поток для пакетного прогноза."""

    progress = Signal(int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self):
        """Выполнить пакетный прогноз."""
        try:
            # Подключиться к прогрессу
            def on_progress(percent: int, message: str):
                self.progress.emit(percent, message)

            self.viewmodel.on_progress = on_progress
            self.viewmodel.predict()

            if self.viewmodel.state.error_message:
                self.error.emit(self.viewmodel.state.error_message)
            else:
                self.complete.emit({
                    "predictions_count": self.viewmodel.state.predictions_count,
                    "output_path": str(self.viewmodel.state.output_path)
                })
        except Exception as e:
            self.error.emit(str(e))


class BatchPredictPage(QWidget):
    """
    Страница пакетного прогноза.
    """

    def __init__(self):
        super().__init__()
        self.viewmodel = None
        self.worker = None

        self._init_ui()

    def _init_ui(self) -> None:
        """Инициализировать UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Заголовок
        title = QLabel("Пакетный прогноз")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        main_layout.addWidget(title)

        # Зона 1: Выбор файла
        file_group = QGroupBox("Входные данные")
        file_layout = QFormLayout(file_group)

        # Путь к файлу
        file_path_layout = QHBoxLayout()
        self.file_path_edit = QLabel("Не выбран")
        self.file_path_edit.setWordWrap(True)
        file_path_layout.addWidget(self.file_path_edit)

        select_btn = QPushButton("Выбрать файл")
        select_btn.setFixedWidth(120)
        select_btn.clicked.connect(self._on_select_file)
        file_path_layout.addWidget(select_btn)

        file_layout.addRow("Excel файл:", file_path_layout)

        # Источник данных
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Excel файл", "База данных (недоступно)"])
        self.source_combo.setCurrentIndex(0)
        self.source_combo.setEnabled(True)  # Только Excel пока
        file_layout.addRow("Источник данных:", self.source_combo)

        main_layout.addWidget(file_group)

        # Зона 2: Параметры прогноза
        params_group = QGroupBox("Параметры прогноза")
        params_layout = QFormLayout(params_group)

        # Формат экспорта
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel (.xlsx)", "CSV (.csv)"])
        params_layout.addRow("Формат экспорта:", self.format_combo)

        # Набор признаков
        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "online_tactical (13 признаков)",
            "online_early (12 признаков)",
            "online_dispatch (9 признаков)"
        ])
        params_layout.addRow("Набор признаков:", self.feature_combo)

        # Top-K
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5)
        self.top_k_spin.setValue(3)
        params_layout.addRow("Количество вариантов (Top-K):", self.top_k_spin)

        # Бутстрап
        self.bootstrap_check = QCheckBox("Использовать бутстрап")
        self.bootstrap_check.setChecked(True)
        params_layout.addRow(self.bootstrap_check)

        # N бутстрап
        self.n_bootstrap_spin = QSpinBox()
        self.n_bootstrap_spin.setRange(10, 100)
        self.n_bootstrap_spin.setValue(30)
        params_layout.addRow("Количество бутстрап-выборок:", self.n_bootstrap_spin)

        main_layout.addWidget(params_group)

        # Кнопка запуска
        self.start_btn = QPushButton("Запустить пакетный прогноз")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setStyleSheet("""
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
        main_layout.addWidget(self.start_btn)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)

        # Статус
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px;")
        main_layout.addWidget(self.status_label)

        # Результат
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel("Нет результатов")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)

        # Предупреждения
        self.warnings_edit = QTextEdit()
        self.warnings_edit.setPlaceholderText("Предупреждения (если есть)")
        self.warnings_edit.setMaximumHeight(100)
        self.warnings_edit.setReadOnly(True)
        result_layout.addWidget(self.warnings_edit)

        main_layout.addWidget(result_group)

        self._connect_signals()

    def _connect_signals(self) -> None:
        """Подключить сигналы."""
        self.start_btn.clicked.connect(self._on_start)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

    def set_paths(self, db_path: Optional[Path],
                  models_path: Optional[Path],
                  reports_path: Optional[Path]) -> None:
        """Установить пути."""
        if models_path and reports_path:
            from ...viewmodels import BatchPredictViewModel
            self.viewmodel = BatchPredictViewModel(models_path, reports_path)
            self.start_btn.setEnabled(True)
        else:
            self.viewmodel = None
            self.start_btn.setEnabled(False)

    def _on_select_file(self) -> None:
        """Выбрать файл."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите Excel файл",
            "",
            "Excel файлы (*.xlsx);;Все файлы (*)"
        )

        if file_path:
            path = Path(file_path)
            self.file_path_edit.setText(str(path))
            
            if self.viewmodel:
                self.viewmodel.set_input_file(path)

    def _on_source_changed(self, index: int) -> None:
        """Изменён источник данных."""
        if self.viewmodel:
            source = "excel" if index == 0 else "database"
            self.viewmodel.set_input_source(source)

    def _on_start(self) -> None:
        """Запустить пакетный прогноз."""
        if not self.viewmodel:
            QMessageBox.warning(self, "Предупреждение", "Workspace не открыт")
            return

        # Валидация
        valid, errors = self.viewmodel.validate_input()
        if not valid:
            QMessageBox.warning(
                self, "Предупреждение",
                "\n".join(errors)
            )
            return

        # Установить параметры
        self.viewmodel.set_output_format(
            "excel" if self.format_combo.currentIndex() == 0 else "csv"
        )
        
        feature_map = {
            0: "online_tactical",
            1: "online_early",
            2: "online_dispatch"
        }
        self.viewmodel.set_feature_set(
            feature_map.get(self.feature_combo.currentIndex(), "online_tactical")
        )
        
        self.viewmodel.set_top_k(self.top_k_spin.value())
        self.viewmodel.set_use_bootstrap(self.bootstrap_check.isChecked())
        self.viewmodel.set_n_bootstrap(self.n_bootstrap_spin.value())

        # Запустить в фоне
        self.worker = BatchPredictWorker(self.viewmodel)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск...")

        self.worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        """Прогресс выполнения."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_complete(self, result: Dict) -> None:
        """Прогноз завершён."""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Завершено")

        # Показать результат
        self.result_label.setText(
            f"✅ Прогноз выполнен\n"
            f"Записей: {result['predictions_count']}\n"
            f"Файл: {result['output_path']}"
        )
        self.result_label.setStyleSheet("color: white; font-weight: bold;")

        # Предупреждения
        if self.viewmodel and self.viewmodel.state.warnings:
            self.warnings_edit.setText(
                "\n".join(self.viewmodel.state.warnings[:10])  # Первые 10
            )
        else:
            self.warnings_edit.clear()

        QMessageBox.information(
            self,
            "Завершено",
            f"Пакетный прогноз завершён!\n\n"
            f"Файл сохранён:\n{result['output_path']}"
        )

    def _on_error(self, message: str) -> None:
        """Ошибка."""
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка")
        self.result_label.setText("Ошибка выполнения")
        self.result_label.setStyleSheet("color: white;")
        
        QMessageBox.critical(self, "Ошибка", message)
