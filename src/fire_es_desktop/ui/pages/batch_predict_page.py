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
    QGroupBox, QFileDialog, QProgressBar,
    QMessageBox, QComboBox, QSpinBox, QCheckBox, QTextEdit,
    QBoxLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Optional, Dict, Any

import logging

logger = logging.getLogger("BatchPredictPage")

from ..theme import (
    configure_text_panel,
    create_hint,
    create_page_header,
    create_static_page,
    create_status_label,
    ResponsiveFormWidget,
    style_button,
    style_label,
)


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


class ResponsiveInlineWidget(QWidget):
    """Inline widget that switches between horizontal and vertical layouts."""

    def __init__(self, *, compact_breakpoint: int = 1100):
        super().__init__()
        self._compact_breakpoint = compact_breakpoint
        self._compact_mode: Optional[bool] = None
        self._layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)

    def addWidget(self, widget: QWidget, stretch: int = 0) -> None:  # noqa: N802 - Qt naming
        self._layout.addWidget(widget, stretch)

    def setCompactBreakpoint(self, width: int) -> None:  # noqa: N802 - Qt naming
        self._compact_breakpoint = width
        self._update_layout(force=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_layout()

    def _update_layout(self, *, force: bool = False) -> None:
        compact = self.width() < self._compact_breakpoint if self.width() > 0 else False
        if not force and compact == self._compact_mode:
            return
        self._compact_mode = compact
        self._layout.setDirection(QBoxLayout.TopToBottom if compact else QBoxLayout.LeftToRight)
        self._layout.setAlignment(Qt.AlignTop)


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
        main_layout = create_static_page(self)

        # Заголовок
        main_layout.addWidget(
            create_page_header(
                "Пакетный прогноз",
                "Массовый прогон активной модели по Excel-файлу и экспорт результатов в таблицу.",
            )
        )

        # Зона 1: Выбор файла
        file_group = QGroupBox("Входные данные")
        file_layout = QVBoxLayout(file_group)
        self.file_form = ResponsiveFormWidget(compact_breakpoint=1260)
        file_layout.addWidget(self.file_form)

        # Путь к файлу
        self.file_path_widget = ResponsiveInlineWidget(compact_breakpoint=920)
        self.file_path_edit = QLabel("Не выбран")
        self.file_path_edit.setWordWrap(True)
        self.file_path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        style_label(self.file_path_edit, "muted", word_wrap=True)
        self.file_path_widget.addWidget(self.file_path_edit, 1)

        select_btn = QPushButton("Выбрать файл")
        style_button(select_btn, "primary")
        select_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        select_btn.clicked.connect(self._on_select_file)
        self.file_path_widget.addWidget(select_btn)

        self.file_form.add_row("Excel-файл:", self.file_path_widget)

        # Источник данных
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Excel файл", "База данных (недоступно)"])
        self.source_combo.setCurrentIndex(0)
        self.source_combo.setEnabled(True)  # Только Excel пока
        self.source_combo.setMinimumWidth(0)
        self.file_form.add_row("Источник данных:", self.source_combo)

        main_layout.addWidget(file_group)

        # Зона 2: Параметры прогноза
        params_group = QGroupBox("Параметры прогноза")
        params_layout = QVBoxLayout(params_group)
        self.params_form = ResponsiveFormWidget(compact_breakpoint=1260)
        params_layout.addWidget(self.params_form)

        # Формат экспорта
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Excel (.xlsx)", "CSV (.csv)"])
        self.format_combo.setMinimumWidth(0)
        self.params_form.add_row("Формат экспорта:", self.format_combo)

        # Набор признаков
        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "online_tactical (служебный режим)"
        ])
        self.feature_combo.setEnabled(False)
        self.feature_combo.setMinimumWidth(0)
        self.params_form.add_row("Набор признаков:", self.feature_combo)

        # Top-K
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 5)
        self.top_k_spin.setValue(3)
        self.top_k_spin.setMinimumWidth(0)
        self.params_form.add_row("Количество вероятных вариантов:", self.top_k_spin, full_width=True)

        # Бутстрап
        self.bootstrap_check = QCheckBox("Использовать бутстрап")
        self.bootstrap_check.setChecked(True)
        self.params_form.add_full_width(self.bootstrap_check)

        # N бутстрап
        self.n_bootstrap_spin = QSpinBox()
        self.n_bootstrap_spin.setRange(10, 100)
        self.n_bootstrap_spin.setValue(30)
        self.n_bootstrap_spin.setMinimumWidth(0)
        self.params_form.add_row("Количество бутстрап-выборок:", self.n_bootstrap_spin, full_width=True)

        main_layout.addWidget(params_group)

        # Кнопка запуска
        self.start_btn = QPushButton("Запустить пакетный прогноз")
        style_button(self.start_btn, "primary", large=True)
        main_layout.addWidget(self.start_btn)

        # Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)

        # Статус
        self.status_label = create_status_label()
        self.status_label.setText("Выберите файл и проверьте параметры пакетного прогона.")
        main_layout.addWidget(self.status_label)

        # Результат
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel("Нет результатов")
        self.result_label.setWordWrap(True)
        style_label(self.result_label, "muted", word_wrap=True)
        result_layout.addWidget(self.result_label)

        # Предупреждения
        self.warnings_edit = QTextEdit()
        self.warnings_edit.setPlaceholderText("Предупреждения (если есть)")
        self.warnings_edit.setReadOnly(True)
        configure_text_panel(self.warnings_edit, min_height=120)
        result_layout.addWidget(self.warnings_edit)

        result_layout.addWidget(
            create_hint(
                "Batch-режим ориентирован на аналитика и не заменяет оперативный экран ЛПР."
            )
        )

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
            QMessageBox.warning(self, "Предупреждение", "Рабочее пространство не открыто")
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
        
        self.viewmodel.set_feature_set("online_tactical")
        
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
        style_label(self.result_label, "metric", word_wrap=True)

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
        style_label(self.result_label, "metric", word_wrap=True)
        
        QMessageBox.critical(self, "Ошибка", message)
