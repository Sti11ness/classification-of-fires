"""
LPRDecisionHistoryPage — страница истории решений ЛПР.

Показывает:
- список сохраненных решений
- карточку выбранного решения
- редактирование ранга и комментария
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from fire_es.rank_tz_contract import DEFAULT_LPR_FEATURE_SET, get_input_schema

from ...viewmodels import LPRDecisionHistoryViewModel
from ..theme import (
    configure_form_layout,
    configure_table,
    configure_text_panel,
    create_page_header,
    create_scrollable_page,
    create_status_label,
    style_button,
    style_label,
)


class LPRDecisionHistoryPage(QWidget):
    """Страница истории решений ЛПР."""

    EDITABLE_RANKS = ["1", "1-бис", "2", "3", "4", "5"]

    def __init__(self):
        super().__init__()
        self.viewmodel: Optional[LPRDecisionHistoryViewModel] = None
        self.db_path: Optional[Path] = None
        self.input_schema = get_input_schema(DEFAULT_LPR_FEATURE_SET)
        self._all_decisions: list[Dict[str, Any]] = []

        self._init_ui()
        self._connect_signals()
        self._render_empty_workspace_state()

    def _init_ui(self) -> None:
        _, _, _, layout = create_scrollable_page(self)

        layout.addWidget(
            create_page_header(
                "История решений ЛПР",
                "Просмотр сохраненных решений, сравнение с прогнозом модели и правка решения или комментария.",
            )
        )

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.refresh_btn = QPushButton("Обновить")
        style_button(self.refresh_btn, "ghost")
        controls_layout.addWidget(self.refresh_btn)

        self.save_btn = QPushButton("Сохранить изменения")
        self.save_btn.setEnabled(False)
        style_button(self.save_btn, "success")
        controls_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Сбросить")
        self.reset_btn.setEnabled(False)
        style_button(self.reset_btn, "ghost")
        controls_layout.addWidget(self.reset_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        filters_group = QGroupBox("Фильтры и сортировка")
        filters_layout = QGridLayout(filters_group)
        filters_layout.setSpacing(10)

        filters_layout.addWidget(QLabel("ID:"), 0, 0)
        self.id_filter_edit = QLineEdit()
        self.id_filter_edit.setPlaceholderText("например, 15")
        filters_layout.addWidget(self.id_filter_edit, 0, 1)

        filters_layout.addWidget(QLabel("Время:"), 0, 2)
        self.time_filter_edit = QLineEdit()
        self.time_filter_edit.setPlaceholderText("2026-03-30")
        filters_layout.addWidget(self.time_filter_edit, 0, 3)

        filters_layout.addWidget(QLabel("Решение:"), 1, 0)
        self.decision_rank_filter = QComboBox()
        self.decision_rank_filter.addItems(["Все", *self.EDITABLE_RANKS])
        filters_layout.addWidget(self.decision_rank_filter, 1, 1)

        filters_layout.addWidget(QLabel("Прогноз:"), 1, 2)
        self.predicted_rank_filter = QComboBox()
        self.predicted_rank_filter.addItems(["Все", *self.EDITABLE_RANKS])
        filters_layout.addWidget(self.predicted_rank_filter, 1, 3)

        filters_layout.addWidget(QLabel("Комментарий:"), 2, 0)
        self.comment_filter_edit = QLineEdit()
        self.comment_filter_edit.setPlaceholderText("поиск по тексту")
        filters_layout.addWidget(self.comment_filter_edit, 2, 1)

        filters_layout.addWidget(QLabel("Fire ID:"), 2, 2)
        self.fire_id_filter_edit = QLineEdit()
        self.fire_id_filter_edit.setPlaceholderText("например, 3232")
        filters_layout.addWidget(self.fire_id_filter_edit, 2, 3)

        self.clear_filters_btn = QPushButton("Сбросить фильтры")
        filters_layout.addWidget(self.clear_filters_btn, 3, 0, 1, 2)

        sort_hint = QLabel("Сортировка: нажмите на заголовок столбца, чтобы изменить порядок строк")
        style_label(sort_hint, "section-hint", word_wrap=True)
        filters_layout.addWidget(sort_hint, 3, 2, 1, 2)

        layout.addWidget(filters_group)

        self.status_label = create_status_label()
        layout.addWidget(self.status_label)

        table_group = QGroupBox("Сохраненные решения")
        table_layout = QVBoxLayout(table_group)

        self.decisions_table = QTableWidget()
        self.decisions_table.setColumnCount(6)
        self.decisions_table.setHorizontalHeaderLabels(
            ["ID", "Время", "Решение", "Прогноз", "Комментарий", "Fire ID"]
        )
        self.decisions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.decisions_table.setSelectionMode(QTableWidget.SingleSelection)
        self.decisions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        configure_table(self.decisions_table, min_height=320, sortable=True)
        header = self.decisions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.decisions_table.setSortingEnabled(True)
        table_layout.addWidget(self.decisions_table)

        layout.addWidget(table_group)

        meta_group = QGroupBox("Карточка решения")
        details_layout = QVBoxLayout(meta_group)

        meta_layout = QGridLayout()
        meta_layout.addWidget(QLabel("ID решения:"), 0, 0)
        self.decision_id_value = QLabel("—")
        style_label(self.decision_id_value, "value", word_wrap=True)
        meta_layout.addWidget(self.decision_id_value, 0, 1)

        meta_layout.addWidget(QLabel("Дата сохранения:"), 0, 2)
        self.created_at_value = QLabel("—")
        style_label(self.created_at_value, "value", word_wrap=True)
        meta_layout.addWidget(self.created_at_value, 0, 3)

        meta_layout.addWidget(QLabel("Fire ID:"), 1, 0)
        self.fire_id_value = QLabel("—")
        style_label(self.fire_id_value, "value", word_wrap=True)
        meta_layout.addWidget(self.fire_id_value, 1, 1)

        meta_layout.addWidget(QLabel("Прогноз модели:"), 1, 2)
        self.predicted_rank_value = QLabel("—")
        style_label(self.predicted_rank_value, "value", word_wrap=True)
        meta_layout.addWidget(self.predicted_rank_value, 1, 3)

        details_layout.addLayout(meta_layout)
        layout.addWidget(meta_group)

        edit_group = QGroupBox("Редактируемая часть решения")
        edit_layout = QFormLayout(edit_group)
        configure_form_layout(edit_layout)

        self.decision_rank_combo = QComboBox()
        self.decision_rank_combo.addItems(self.EDITABLE_RANKS)
        edit_layout.addRow("Решение ЛПР:", self.decision_rank_combo)

        self.comment_edit = QTextEdit()
        configure_text_panel(self.comment_edit, min_height=120)
        self.comment_edit.setMaximumHeight(150)
        self.comment_edit.setPlaceholderText("Комментарий к решению")
        edit_layout.addRow("Комментарий:", self.comment_edit)

        layout.addWidget(edit_group)

        snapshot_splitter = QSplitter(Qt.Horizontal)
        snapshot_splitter.setChildrenCollapsible(False)

        probabilities_group = QGroupBox("Вероятности прогноза")
        probabilities_layout = QVBoxLayout(probabilities_group)
        self.probabilities_text = QTextEdit()
        self.probabilities_text.setReadOnly(True)
        configure_text_panel(self.probabilities_text, min_height=200)
        self.probabilities_text.setMaximumHeight(220)
        probabilities_layout.addWidget(self.probabilities_text)
        snapshot_splitter.addWidget(probabilities_group)

        input_group = QGroupBox("Входные данные пожара")
        input_layout = QVBoxLayout(input_group)
        self.input_snapshot_text = QTextEdit()
        self.input_snapshot_text.setReadOnly(True)
        configure_text_panel(self.input_snapshot_text, min_height=200)
        self.input_snapshot_text.setMaximumHeight(220)
        input_layout.addWidget(self.input_snapshot_text)
        snapshot_splitter.addWidget(input_group)

        snapshot_splitter.setStretchFactor(0, 1)
        snapshot_splitter.setStretchFactor(1, 1)
        layout.addWidget(snapshot_splitter)

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.save_btn.clicked.connect(self._on_save)
        self.reset_btn.clicked.connect(self._on_reset)
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        self.decisions_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.id_filter_edit.textChanged.connect(self._apply_filters)
        self.time_filter_edit.textChanged.connect(self._apply_filters)
        self.decision_rank_filter.currentTextChanged.connect(self._apply_filters)
        self.predicted_rank_filter.currentTextChanged.connect(self._apply_filters)
        self.comment_filter_edit.textChanged.connect(self._apply_filters)
        self.fire_id_filter_edit.textChanged.connect(self._apply_filters)

    def set_db_path(self, db_path: Optional[Path]) -> None:
        """Подключить страницу к текущему workspace DB."""
        if self.viewmodel is not None:
            self.viewmodel.close()
            self.viewmodel = None

        self.db_path = db_path
        if db_path and db_path.exists():
            self.viewmodel = LPRDecisionHistoryViewModel(db_path)
            self.viewmodel.on_state_changed = self._render_state
            self.viewmodel.on_error = self._show_error
            self.viewmodel.load_decisions()
        else:
            self._render_empty_workspace_state()

    def _render_state(self) -> None:
        """Обновить UI из текущего состояния ViewModel."""
        if not self.viewmodel:
            self._render_empty_workspace_state()
            return

        state = self.viewmodel.state
        self.status_label.setText(state.status_message or "")
        self._all_decisions = list(state.decisions)
        self._apply_filters(state.selected_decision_id)

        if state.selected_detail:
            self._render_detail(state.selected_detail)
        elif state.decisions:
            self._clear_detail("Выберите запись, чтобы открыть карточку решения")
        else:
            self._clear_detail("Сохраненных решений пока нет")

    def _render_empty_workspace_state(self) -> None:
        self.status_label.setText("Рабочее пространство не открыто")
        self._all_decisions = []
        self.decisions_table.setRowCount(0)
        self._clear_detail("Откройте рабочее пространство, чтобы просмотреть историю решений")

    def _populate_table(
        self,
        decisions: list[Dict[str, Any]],
        selected_decision_id: Optional[int],
    ) -> None:
        sort_column = self.decisions_table.horizontalHeader().sortIndicatorSection()
        sort_order = self.decisions_table.horizontalHeader().sortIndicatorOrder()
        self.decisions_table.setSortingEnabled(False)
        with QSignalBlocker(self.decisions_table):
            self.decisions_table.setRowCount(0)
            for decision in decisions:
                row = self.decisions_table.rowCount()
                self.decisions_table.insertRow(row)

                values = [
                    str(decision.get("decision_id", "")),
                    self._format_timestamp(decision.get("created_at")),
                    decision.get("decision_rank_label", "—"),
                    decision.get("predicted_rank_label", "—"),
                    decision.get("comment_preview", ""),
                    "" if decision.get("fire_id") is None else str(decision.get("fire_id")),
                ]
                sort_keys = [
                    int(decision.get("decision_id", 0) or 0),
                    decision.get("created_at") or "",
                    self._rank_sort_value(decision.get("decision_rank")),
                    self._rank_sort_value(decision.get("predicted_rank")),
                    (decision.get("comment_preview") or "").lower(),
                    int(decision.get("fire_id", 0) or 0),
                ]
                for column, value in enumerate(values):
                    item = SortableTableWidgetItem(value)
                    item.setData(Qt.UserRole, decision.get("decision_id"))
                    item.set_sort_key(sort_keys[column])
                    self.decisions_table.setItem(row, column, item)

                if selected_decision_id == decision.get("decision_id"):
                    self.decisions_table.selectRow(row)

        self.decisions_table.setSortingEnabled(True)
        if self.decisions_table.rowCount() > 0:
            if sort_column >= 0:
                self.decisions_table.sortItems(sort_column, sort_order)
            else:
                self.decisions_table.sortItems(1, Qt.DescendingOrder)

    def _render_detail(self, detail: Dict[str, Any]) -> None:
        self.decision_id_value.setText(str(detail.get("decision_id", "—")))
        self.created_at_value.setText(self._format_timestamp(detail.get("created_at")))
        self.fire_id_value.setText(str(detail.get("fire_id", "—")))
        self.predicted_rank_value.setText(detail.get("predicted_rank_label", "—"))

        rank_label = detail.get("decision_rank_label", "—")
        combo_index = self.decision_rank_combo.findText(rank_label)
        if combo_index >= 0:
            self.decision_rank_combo.setCurrentIndex(combo_index)
        else:
            self.decision_rank_combo.setCurrentIndex(0)
        self.comment_edit.setPlainText(detail.get("comment", ""))
        self.probabilities_text.setPlainText(
            self._format_probabilities(detail.get("predicted_probabilities"))
        )
        self.input_snapshot_text.setPlainText(
            self._format_fire_snapshot(detail.get("fire_snapshot"))
        )
        self.save_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

    def _clear_detail(self, message: str) -> None:
        self.decision_id_value.setText("—")
        self.created_at_value.setText("—")
        self.fire_id_value.setText("—")
        self.predicted_rank_value.setText("—")
        self.decision_rank_combo.setCurrentIndex(0)
        self.comment_edit.setPlainText("")
        self.probabilities_text.setPlainText(message)
        self.input_snapshot_text.setPlainText(message)
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

    def _on_refresh(self) -> None:
        if not self.viewmodel:
            QMessageBox.warning(self, "История решений", "Рабочее пространство не открыто")
            return
        self.viewmodel.load_decisions()

    def _on_table_selection_changed(self) -> None:
        if not self.viewmodel:
            return
        selected_items = self.decisions_table.selectedItems()
        if not selected_items:
            self._clear_detail("Выберите запись, чтобы открыть карточку решения")
            return

        decision_id = selected_items[0].data(Qt.UserRole)
        if decision_id is not None:
            self.viewmodel.select_decision(int(decision_id))

    def _on_save(self) -> None:
        if not self.viewmodel or self.viewmodel.state.selected_decision_id is None:
            QMessageBox.warning(self, "История решений", "Сначала выберите решение")
            return

        success = self.viewmodel.update_selected_decision(
            self.decision_rank_combo.currentText(),
            self.comment_edit.toPlainText(),
        )
        if success:
            QMessageBox.information(self, "История решений", "Изменения сохранены")

    def _on_reset(self) -> None:
        if not self.viewmodel or not self.viewmodel.state.selected_detail:
            return
        self._render_detail(self.viewmodel.state.selected_detail)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "История решений", message)

    def _apply_filters(self, selected_decision_id: Optional[int] = None) -> None:
        filtered = []

        id_filter = self.id_filter_edit.text().strip()
        time_filter = self.time_filter_edit.text().strip().lower()
        decision_rank_filter = self.decision_rank_filter.currentText()
        predicted_rank_filter = self.predicted_rank_filter.currentText()
        comment_filter = self.comment_filter_edit.text().strip().lower()
        fire_id_filter = self.fire_id_filter_edit.text().strip()

        for decision in self._all_decisions:
            if id_filter and id_filter not in str(decision.get("decision_id", "")):
                continue
            if time_filter and time_filter not in self._format_timestamp(decision.get("created_at")).lower():
                continue
            if (
                decision_rank_filter != "Все"
                and decision.get("decision_rank_label") != decision_rank_filter
            ):
                continue
            if (
                predicted_rank_filter != "Все"
                and decision.get("predicted_rank_label") != predicted_rank_filter
            ):
                continue
            if comment_filter and comment_filter not in (decision.get("comment", "") or "").lower():
                continue
            if fire_id_filter and fire_id_filter not in str(decision.get("fire_id", "")):
                continue
            filtered.append(decision)

        if selected_decision_id is None and self.viewmodel:
            selected_decision_id = self.viewmodel.state.selected_decision_id

        visible_ids = {item["decision_id"] for item in filtered}
        if selected_decision_id not in visible_ids:
            selected_decision_id = None

        self._populate_table(filtered, selected_decision_id)
        if not filtered:
            self._clear_detail("Нет записей, подходящих под текущие фильтры")

    def _clear_filters(self) -> None:
        with QSignalBlocker(self.id_filter_edit):
            self.id_filter_edit.clear()
        with QSignalBlocker(self.time_filter_edit):
            self.time_filter_edit.clear()
        with QSignalBlocker(self.decision_rank_filter):
            self.decision_rank_filter.setCurrentIndex(0)
        with QSignalBlocker(self.predicted_rank_filter):
            self.predicted_rank_filter.setCurrentIndex(0)
        with QSignalBlocker(self.comment_filter_edit):
            self.comment_filter_edit.clear()
        with QSignalBlocker(self.fire_id_filter_edit):
            self.fire_id_filter_edit.clear()
        self._apply_filters()

    def _format_probabilities(self, payload: Any) -> str:
        if not payload:
            return "Вероятности прогноза отсутствуют"

        lines = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rank = item.get("rank", "—")
                    probability = item.get("probability")
                    if isinstance(probability, (float, int)):
                        probability_value = float(probability)
                        if probability_value <= 1:
                            probability_value *= 100
                        probability_text = f"{probability_value:.1f}%"
                    else:
                        probability_text = str(probability)
                    lines.append(f"{rank}: {probability_text}")
                else:
                    lines.append(str(item))
        else:
            lines.append(str(payload))
        return "\n".join(lines)

    def _format_fire_snapshot(self, snapshot: Any) -> str:
        if not isinstance(snapshot, dict):
            return "Связанный кейс пожара не найден"

        lines = []
        for meta_field, label in [
            ("source_sheet", "Источник"),
            ("fire_date", "Дата пожара"),
            ("year", "Год"),
            ("month", "Месяц"),
        ]:
            value = snapshot.get(meta_field)
            if value not in (None, ""):
                lines.append(f"{label}: {value}")

        for field in self.input_schema:
            value = snapshot.get(field["name"])
            if value in (None, ""):
                value_text = "Не указано"
            else:
                suffix = field.get("suffix", "")
                value_text = f"{value}{suffix}"
            lines.append(f"{field['label']}: {value_text}")

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if not value:
            return "—"
        return str(value).replace("T", " ")[:19]

    @staticmethod
    def _rank_sort_value(value: Any) -> float:
        if value is None:
            return float("inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")


class SortableTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem with stable numeric/date-aware sort key."""

    def __init__(self, text: str):
        super().__init__(text)
        self._sort_key: Any = text

    def set_sort_key(self, value: Any) -> None:
        self._sort_key = value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, SortableTableWidgetItem):
            return self._sort_key < other._sort_key
        return super().__lt__(other)
