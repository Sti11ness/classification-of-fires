# src/fire_es_desktop/ui/pages/digital_twin_page.py
"""
DigitalTwinPage — analyst-only digital twin and synthetic experiment screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...viewmodels import DigitalTwinViewModel
from ..theme import (
    ResponsiveFormWidget,
    configure_table,
    configure_text_panel,
    create_hint,
    create_page_header,
    create_scrollable_page,
    create_status_label,
    style_button,
    style_label,
)


class DigitalTwinWorker(QThread):
    """Worker thread for long-running digital twin experiments."""

    progress = Signal(int, int, str)
    complete = Signal(dict)
    error = Signal(str)

    def __init__(self, viewmodel: DigitalTwinViewModel):
        super().__init__()
        self.viewmodel = viewmodel

    def run(self) -> None:
        def on_progress(current: int, total: int, description: str) -> None:
            self.progress.emit(current, total, description)

        self.viewmodel.on_progress = on_progress
        try:
            self.viewmodel.run()
            if self.viewmodel.error_message:
                self.error.emit(self.viewmodel.error_message)
            else:
                self.complete.emit(self.viewmodel.result or {})
        except Exception as exc:
            self.error.emit(str(exc))


class DigitalTwinPage(QWidget):
    """Digital twin page for analyst research workflows."""

    def __init__(self):
        super().__init__()
        self.viewmodel: DigitalTwinViewModel | None = None
        self.worker: DigitalTwinWorker | None = None
        self.reports_path: Path | None = None
        self.input_path: Path | None = Path("clean_df_enhanced.csv")
        self.last_result: dict[str, Any] | None = None

        self._mode_checks: dict[str, QCheckBox] = {}
        self._feature_checks: dict[str, QCheckBox] = {}
        self._model_checks: dict[str, QCheckBox] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        _, _, _, layout = create_scrollable_page(self)
        layout.addWidget(
            create_page_header(
                "Цифровой двойник среды",
                "Research-only контур: профиль исходной выборки, генерация ИВД, искажения и сравнение моделей на реальном holdout.",
            )
        )
        layout.addWidget(
            create_hint(
                "ЦДС не используется в оперативном режиме ЛПР. Рабочая production-модель остается на раннем наборе признаков."
            )
        )

        status_group = QGroupBox("Статус последнего запуска")
        status_layout = QHBoxLayout(status_group)
        self.source_metric = self._metric_label("Источник", "не выбран")
        self.real_rows_metric = self._metric_label("Real-test", "0")
        self.synthetic_metric = self._metric_label("ИВД", "0")
        self.best_f1_metric = self._metric_label("Best real F1", "-")
        self.duplicates_metric = self._metric_label("Дубли", "-")
        for widget in [
            self.source_metric,
            self.real_rows_metric,
            self.synthetic_metric,
            self.best_f1_metric,
            self.duplicates_metric,
        ]:
            status_layout.addWidget(widget)
        layout.addWidget(status_group)

        tabs = QTabWidget()
        tabs.addTab(self._build_source_tab(), "1. Источник")
        tabs.addTab(self._build_profile_tab(), "2. Профиль")
        tabs.addTab(self._build_generation_tab(), "3. Генерация ИВД")
        tabs.addTab(self._build_distortion_tab(), "4. Искажения")
        tabs.addTab(self._build_experiment_tab(), "5. Эксперимент")
        tabs.addTab(self._build_reports_tab(), "6. Отчеты")
        layout.addWidget(tabs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = create_status_label()
        self.status_label.setText("Откройте workspace или выберите CSV-файл для эксперимента ЦДС.")
        layout.addWidget(self.status_label)

    def _metric_label(self, title: str, value: str) -> QWidget:
        wrapper = QWidget()
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)
        title_label = QLabel(title)
        style_label(title_label, "muted", word_wrap=False)
        value_label = QLabel(value)
        value_label.setObjectName(f"digital_twin_metric_{title}")
        style_label(value_label, "metric", word_wrap=True)
        inner.addWidget(title_label)
        inner.addWidget(value_label)
        wrapper.value_label = value_label  # type: ignore[attr-defined]
        return wrapper

    def _build_source_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        source_group = QGroupBox("Источник данных")
        source_layout = QVBoxLayout(source_group)
        form = ResponsiveFormWidget(compact_breakpoint=920)
        source_layout.addWidget(form)

        path_row = QWidget()
        row_layout = QHBoxLayout(path_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.input_path_label = QLabel(str(self.input_path) if self.input_path else "не выбран")
        self.input_path_label.setWordWrap(True)
        style_label(self.input_path_label, "value", word_wrap=True)
        browse_btn = QPushButton("Выбрать CSV")
        style_button(browse_btn, "primary")
        browse_btn.clicked.connect(self._on_browse_input)
        row_layout.addWidget(self.input_path_label, 1)
        row_layout.addWidget(browse_btn)
        form.add_row("CSV-файл:", path_row, full_width=True)

        self.target_edit = QLineEdit("rank_tz")
        form.add_row("Target:", self.target_edit)
        layout.addWidget(source_group)

        self.source_preview = QTextEdit()
        configure_text_panel(self.source_preview, min_height=180)
        self.source_preview.setReadOnly(True)
        layout.addWidget(self.source_preview)

        preview_btn = QPushButton("Проверить источник")
        style_button(preview_btn, "secondary")
        preview_btn.clicked.connect(self._on_preview_source)
        layout.addWidget(preview_btn)
        layout.addStretch()
        return tab

    def _build_profile_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.profile_text = QTextEdit()
        self.profile_text.setReadOnly(True)
        configure_text_panel(self.profile_text, min_height=320)
        self.profile_text.setPlainText(
            "Профиль ЦДС строится при запуске эксперимента и сохраняется в generation_profile.json."
        )
        layout.addWidget(self.profile_text)
        return tab

    def _build_generation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Параметры генерации ИВД")
        group_layout = QVBoxLayout(group)
        form = ResponsiveFormWidget(compact_breakpoint=920)
        group_layout.addWidget(form)

        self.synthetic_rows_spin = self._spin(100, 1_000_000, 100_000)
        self.train_rows_spin = self._spin(100, 1_000_000, 97_000)
        self.validation_rows_spin = self._spin(100, 100_000, 3_000)
        self.seed_spin = self._spin(0, 999_999, 42)
        form.add_row("Всего synthetic rows:", self.synthetic_rows_spin)
        form.add_row("Train rows:", self.train_rows_spin)
        form.add_row("Validation rows:", self.validation_rows_spin)
        form.add_row("Seed:", self.seed_spin)

        self.noise_spin = self._double_spin(0.0, 1.0, 0.08, 0.01)
        self.smoothing_spin = self._double_spin(0.01, 10.0, 0.5, 0.05)
        self.global_mix_spin = self._double_spin(0.0, 1.0, 0.15, 0.05)
        form.add_row("Robust numeric noise:", self.noise_spin)
        form.add_row("Dirichlet smoothing:", self.smoothing_spin)
        form.add_row("Global/rank mix:", self.global_mix_spin)

        layout.addWidget(group)
        layout.addWidget(
            create_hint(
                "Генератор сохраняет распределение ранга, но искажает числовые и категориальные признаки, чтобы не копировать реальные строки."
            )
        )
        layout.addStretch()
        return tab

    def _build_distortion_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Искажения и сценарии")
        group_layout = QVBoxLayout(group)
        form = ResponsiveFormWidget(compact_breakpoint=920)
        group_layout.addWidget(form)
        self.missing_spin = self._double_spin(0.0, 0.5, 0.0, 0.01)
        form.add_row("Дополнительные пропуски:", self.missing_spin)
        self.distortion_hint = QLabel(
            "Сценарий distortion_study использует повышенный шум, большую долю global-mix и дополнительные пропуски."
        )
        self.distortion_hint.setWordWrap(True)
        style_label(self.distortion_hint, "muted", word_wrap=True)
        group_layout.addWidget(self.distortion_hint)
        layout.addWidget(group)
        layout.addStretch()
        return tab

    def _build_experiment_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        controls.addWidget(self._checkbox_group("Режимы", {
            "baseline_real": True,
            "synthetic_only": True,
            "real_plus_synthetic": False,
            "distortion_study": False,
        }, self._mode_checks))
        controls.addWidget(self._checkbox_group("Модели", {
            "decision_tree": True,
            "random_forest": True,
            "hist_gradient_boosting": True,
        }, self._model_checks))
        layout.addLayout(controls)

        feature_defaults = {
            "features_10_dispatch": True,
            "features_13_arrival": True,
            "features_15_first_hose": True,
            "features_19_retrospective_time": True,
            "features_35_enhanced_retrospective": True,
            "features_50_wide_no_target": True,
            "features_all_known_numeric": True,
        }
        layout.addWidget(self._checkbox_group("Наборы признаков", feature_defaults, self._feature_checks))

        self.start_btn = QPushButton("Запустить эксперимент ЦДС")
        style_button(self.start_btn, "primary", large=True)
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)
        layout.addStretch()
        return tab

    def _build_reports_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels([
            "mode",
            "feature_set",
            "model",
            "dataset",
            "f1_macro",
            "accuracy",
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        configure_table(self.results_table)
        layout.addWidget(self.results_table)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        configure_text_panel(self.result_text, min_height=160)
        layout.addWidget(self.result_text)

        actions = QHBoxLayout()
        self.open_folder_btn = QPushButton("Открыть папку запуска")
        self.open_report_btn = QPushButton("Открыть REPORT.md")
        style_button(self.open_folder_btn, "secondary")
        style_button(self.open_report_btn, "secondary")
        self.open_folder_btn.clicked.connect(self._open_run_folder)
        self.open_report_btn.clicked.connect(self._open_report)
        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.open_report_btn)
        layout.addLayout(actions)
        return tab

    def _checkbox_group(
        self,
        title: str,
        values: dict[str, bool],
        target: dict[str, QCheckBox],
    ) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        for value, checked in values.items():
            checkbox = QCheckBox(value)
            checkbox.setChecked(checked)
            target[value] = checkbox
            layout.addWidget(checkbox)
        return group

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(100)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def set_paths(self, db_path: Path | None, reports_path: Path | None) -> None:
        del db_path
        self.reports_path = reports_path
        if reports_path:
            self.viewmodel = DigitalTwinViewModel(reports_path)
            if self.input_path:
                self.viewmodel.set_input_path(self.input_path)
            self.status_label.setText(f"Reports: {reports_path}")
        else:
            self.viewmodel = None
            self.status_label.setText("Рабочее пространство не открыто.")

    def _selected(self, checks: dict[str, QCheckBox]) -> list[str]:
        return [value for value, checkbox in checks.items() if checkbox.isChecked()]

    def _on_browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "", "CSV Files (*.csv)")
        if path:
            self.input_path = Path(path)
            self.input_path_label.setText(str(self.input_path))
            if self.viewmodel:
                self.viewmodel.set_input_path(self.input_path)
            self._update_metric(self.source_metric, self.input_path.name)

    def _on_preview_source(self) -> None:
        if not self.input_path:
            QMessageBox.warning(self, "Источник", "CSV-файл не выбран")
            return
        try:
            df = pd.read_csv(self.input_path, nrows=5000)
            target = self.target_edit.text().strip() or "rank_tz"
            lines = [
                f"Файл: {self.input_path}",
                f"Показано строк для проверки: {len(df)}",
                f"Колонок: {len(df.columns)}",
                f"Target `{target}` найден: {'да' if target in df.columns else 'нет'}",
            ]
            if "source_sheet" in df.columns:
                lines.append("source_sheet:")
                for key, value in df["source_sheet"].value_counts().head(5).items():
                    lines.append(f"  - {key}: {value}")
            if target in df.columns:
                lines.append("rank distribution:")
                for key, value in df[target].value_counts().sort_index().items():
                    lines.append(f"  - {key}: {value}")
            self.source_preview.setPlainText("\n".join(lines))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка чтения", str(exc))

    def _on_start(self) -> None:
        if not self.viewmodel:
            QMessageBox.warning(self, "ЦДС", "Рабочее пространство не открыто")
            return
        if not self.input_path:
            QMessageBox.warning(self, "ЦДС", "CSV-файл не выбран")
            return
        self.viewmodel.set_input_path(self.input_path)
        self.viewmodel.target = self.target_edit.text().strip() or "rank_tz"
        self.viewmodel.set_counts(
            synthetic_rows=self.synthetic_rows_spin.value(),
            train_rows=self.train_rows_spin.value(),
            validation_rows=self.validation_rows_spin.value(),
        )
        self.viewmodel.set_generation_params(
            seed=self.seed_spin.value(),
            numeric_noise_scale=self.noise_spin.value(),
            categorical_smoothing=self.smoothing_spin.value(),
            global_mix=self.global_mix_spin.value(),
            extra_missing_rate=self.missing_spin.value(),
        )
        self.viewmodel.set_selection(
            modes=self._selected(self._mode_checks),
            feature_sets=self._selected(self._feature_checks),
            models=self._selected(self._model_checks),
        )

        self.start_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Запуск эксперимента ЦДС...")
        self.worker = DigitalTwinWorker(self.viewmodel)
        self.worker.progress.connect(self._on_progress)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, current: int, total: int, description: str) -> None:
        percent = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(description)

    def _on_complete(self, result: dict) -> None:
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.last_result = result
        self.status_label.setText("Эксперимент ЦДС завершен")
        self._render_result(result)
        QMessageBox.information(self, "ЦДС", "Эксперимент завершен. Отчет сохранен.")

    def _on_error(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ошибка эксперимента ЦДС")
        QMessageBox.critical(self, "Ошибка ЦДС", message)

    def _render_result(self, result: dict[str, Any]) -> None:
        self._update_metric(self.source_metric, self.input_path.name if self.input_path else "-")
        self._update_metric(self.real_rows_metric, str(result.get("real_test_rows", 0)))
        self._update_metric(
            self.synthetic_metric,
            f"{result.get('synthetic_train_rows', 0)} / {result.get('synthetic_validation_rows', 0)}",
        )
        best = result.get("best_real_f1_macro")
        self._update_metric(self.best_f1_metric, "-" if best is None else f"{best:.4f}")
        duplicate_rates = result.get("duplicate_rates_by_feature_set", {})
        max_dup = max(duplicate_rates.values()) if duplicate_rates else None
        self._update_metric(self.duplicates_metric, "-" if max_dup is None else f"{max_dup:.4f}")
        self.result_text.setPlainText(
            "\n".join(
                [
                    f"Папка запуска: {result.get('run_dir')}",
                    f"Отчет: {result.get('report_path')}",
                    f"Метрики: {result.get('metrics_path')}",
                    f"Удалено дублей real-test: {result.get('duplicates_removed')}",
                    f"Строк метрик: {result.get('metrics_rows')}",
                ]
            )
        )
        self._load_metrics_table(result.get("metrics_path"))
        self._load_profile_preview(result.get("profile_path"))

    def _update_metric(self, wrapper: QWidget, value: str) -> None:
        label = getattr(wrapper, "value_label", None)
        if label:
            label.setText(value)

    def _load_metrics_table(self, metrics_path: Any) -> None:
        self.results_table.setRowCount(0)
        if not metrics_path:
            return
        path = Path(metrics_path)
        if not path.exists():
            return
        df = pd.read_csv(path)
        if "eval_dataset" in df.columns:
            df = df.loc[df["eval_dataset"] == "real_test"].copy()
        if "f1_macro" in df.columns:
            df = df.sort_values("f1_macro", ascending=False, na_position="last").head(20)
        columns = ["mode", "feature_set", "model", "eval_dataset", "f1_macro", "accuracy"]
        self.results_table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, column in enumerate(columns):
                value = row.get(column, "")
                if isinstance(value, float):
                    text = "" if pd.isna(value) else f"{value:.4f}"
                else:
                    text = "" if pd.isna(value) else str(value)
                self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(text))

    def _load_profile_preview(self, profile_path: Any) -> None:
        if not profile_path:
            return
        path = Path(profile_path)
        if not path.exists():
            return
        self.profile_text.setPlainText(path.read_text(encoding="utf-8")[:6000])

    def _open_run_folder(self) -> None:
        if self.last_result and self.last_result.get("run_dir"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_result["run_dir"]))))

    def _open_report(self) -> None:
        if self.last_result and self.last_result.get("report_path"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_result["report_path"]))))
