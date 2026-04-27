"""
Shared desktop UI theme helpers for Fire ES.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

FONT_FAMILY = "Segoe UI"
PAGE_MARGIN = 24
PAGE_SPACING = 18
CARD_SPACING = 12
CONTROL_HEIGHT = 40
CONTROL_HEIGHT_LARGE = 46
TEXT_PANEL_MIN_HEIGHT = 160
TABLE_MIN_HEIGHT = 260
ROW_HEIGHT = 32

APP_STYLE_SHEET = """
QMainWindow {
    background-color: #10151b;
    font-family: "Segoe UI";
    font-size: 13px;
}

QWidget#PageRoot,
QWidget#PageContent {
    background-color: transparent;
    color: #e6ebf0;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #0d1217;
    width: 12px;
    margin: 2px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #42606b;
    min-height: 32px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #56808e;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: none;
    border: none;
}

QScrollBar:horizontal {
    background: #0d1217;
    height: 12px;
    margin: 2px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #42606b;
    min-width: 32px;
    border-radius: 6px;
}

QListWidget#NavigationList {
    background-color: #0f161d;
    border-right: 1px solid #1f2b33;
    color: #e6ebf0;
    padding: 10px 8px;
}

QListWidget#NavigationList::item {
    min-height: 42px;
    margin: 2px 0;
    padding: 10px 12px;
    border-radius: 12px;
}

QListWidget#NavigationList::item:selected {
    background-color: #4f9798;
    color: #f5fbfc;
    font-weight: 700;
}

QListWidget#NavigationList::item:hover {
    background-color: #1a242d;
}

QFrame#ContextPanel {
    background-color: #10161c;
    border-left: 1px solid #1f2b33;
    color: #e6ebf0;
}

QStackedWidget#PagesStack {
    background: #10151b;
}

QLabel[role="page-title"] {
    font-size: 28px;
    font-weight: 700;
    color: #f7fbfe;
}

QLabel[role="page-subtitle"] {
    font-size: 13px;
    color: #a8b4bf;
}

QLabel[role="section-hint"] {
    font-size: 12px;
    color: #96a5b0;
}

QLabel[role="status"] {
    padding: 10px 12px;
    border-radius: 10px;
    background: #162028;
    border: 1px solid #2a3944;
    color: #e6ebf0;
}

QLabel[role="muted"] {
    color: #b4bec7;
}

QLabel[role="metric"] {
    font-size: 14px;
    font-weight: 600;
    color: #f1f6fa;
}

QLabel[role="value"] {
    padding: 2px 0;
    color: #e6ebf0;
    background: transparent;
}

QLabel[role="ok"] {
    color: #bce8d6;
    font-weight: 600;
}

QLabel[role="problem"] {
    color: #ffd1d1;
    font-weight: 600;
}

QGroupBox {
    margin-top: 12px;
    padding: 14px 14px 12px 14px;
    border: 1px solid #2a3640;
    border-radius: 16px;
    background-color: #172029;
    font-weight: 600;
    color: #eef4f7;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #92d9d5;
    background: #10151b;
}

QLineEdit,
QTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTableWidget,
QListWidget {
    background-color: #0e1318;
    color: #e6ebf0;
    border: 1px solid #2a3843;
    border-radius: 12px;
    selection-background-color: #4f9798;
    selection-color: #f6fbfc;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {
    min-height: 38px;
    padding: 6px 10px;
}

QTextEdit {
    padding: 10px 12px;
}

QLineEdit:focus,
QTextEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QTableWidget:focus,
QListWidget:focus {
    border: 1px solid #8ed6cf;
}

QComboBox::drop-down {
    border: none;
    width: 28px;
}

QComboBox QAbstractItemView {
    background: #121921;
    color: #e6ebf0;
    border: 1px solid #2a3843;
    selection-background-color: #4f9798;
}

QPushButton {
    min-height: 38px;
    padding: 8px 14px;
    border-radius: 12px;
    border: 1px solid #355062;
    background-color: #264051;
    color: #f4fbfc;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #315163;
}

QPushButton:pressed {
    background-color: #213847;
}

QPushButton:disabled {
    background-color: #141a20;
    color: #6c7984;
    border-color: #222d36;
}

QPushButton[role="primary"] {
    background-color: #5daeb0;
    border-color: #7cc6c8;
    color: #f7fcfc;
}

QPushButton[role="primary"]:hover {
    background-color: #74bbbc;
}

QPushButton[role="success"] {
    background-color: #69b58b;
    border-color: #88c7a4;
    color: #f7fcfc;
}

QPushButton[role="success"]:hover {
    background-color: #7ac398;
}

QPushButton[role="danger"] {
    background-color: #b96c77;
    border-color: #cf8791;
    color: #fdf6f7;
}

QPushButton[role="danger"]:hover {
    background-color: #c57b85;
}

QPushButton[role="ghost"] {
    background-color: #1a232c;
    border-color: #324653;
    color: #e2eef4;
}

QPushButton[role="ghost"]:hover {
    background-color: #22303b;
}

QTableWidget {
    gridline-color: #202a33;
    alternate-background-color: #141b22;
}

QHeaderView::section {
    background-color: #1b2630;
    color: #eef4f7;
    border: 1px solid #28333d;
    padding: 8px 10px;
    font-weight: 700;
}

QProgressBar {
    min-height: 18px;
    border: 1px solid #28333d;
    border-radius: 9px;
    background: #0e1318;
    color: transparent;
}

QProgressBar::chunk {
    background: #5daeb0;
    border-radius: 8px;
}

QStatusBar {
    background-color: #0f141a;
    color: #a8b4bf;
    border-top: 1px solid #1f2b33;
}

QSplitter::handle {
    background: #162129;
}
"""


def repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_role(widget: QWidget, role: str) -> None:
    widget.setProperty("role", role)
    repolish(widget)


def style_button(button: QPushButton, role: str = "secondary", *, large: bool = False) -> None:
    set_role(button, role)
    button.setCursor(Qt.PointingHandCursor)
    button.setMinimumHeight(CONTROL_HEIGHT_LARGE if large else CONTROL_HEIGHT)
    button.setMinimumWidth(132)
    button.setMaximumWidth(360)
    button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


def style_label(label: QLabel, role: str = "muted", *, word_wrap: bool = True) -> None:
    set_role(label, role)
    label.setWordWrap(word_wrap)


def create_page_header(title: str, subtitle: Optional[str] = None) -> QWidget:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    title_label = QLabel(title)
    style_label(title_label, "page-title", word_wrap=False)
    layout.addWidget(title_label)

    if subtitle:
        subtitle_label = QLabel(subtitle)
        style_label(subtitle_label, "page-subtitle", word_wrap=True)
        layout.addWidget(subtitle_label)

    return wrapper


def create_scrollable_page(page: QWidget) -> tuple[QVBoxLayout, QScrollArea, QWidget, QVBoxLayout]:
    page.setObjectName("PageRoot")
    root_layout = QVBoxLayout(page)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(0)

    scroll = QScrollArea()
    scroll.setObjectName("PageScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    container = QWidget()
    container.setObjectName("PageContent")
    content_layout = QVBoxLayout(container)
    content_layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    content_layout.setSpacing(PAGE_SPACING)

    scroll.setWidget(container)
    root_layout.addWidget(scroll)
    return root_layout, scroll, container, content_layout


def create_static_page(page: QWidget) -> QVBoxLayout:
    page.setObjectName("PageRoot")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    layout.setSpacing(PAGE_SPACING)
    return layout


class ResponsiveFormWidget(QWidget):
    """Form container that switches between desktop and compact layouts."""

    def __init__(self, *, compact_breakpoint: int = 760):
        super().__init__()
        self._compact_breakpoint = compact_breakpoint
        self._compact_mode: Optional[bool] = None
        self._rows: list[dict[str, Any]] = []
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(CARD_SPACING)
        self._layout.setVerticalSpacing(CARD_SPACING)

    def add_row(
        self,
        label: Union[str, QLabel],
        field: QWidget,
        *,
        full_width: bool = False,
        label_role: str = "metric",
    ) -> None:
        if isinstance(label, str):
            label_widget = QLabel(label)
            style_label(label_widget, label_role, word_wrap=True)
        else:
            label_widget = label
        self._rows.append(
            {
                "kind": "pair",
                "label": label_widget,
                "field": field,
                "full_width": full_width,
            }
        )
        self._rebuild_layout(force=True)

    def add_full_width(self, widget: QWidget) -> None:
        self._rows.append({"kind": "full", "widget": widget})
        self._rebuild_layout(force=True)

    def set_compact_breakpoint(self, width: int) -> None:
        self._compact_breakpoint = width
        self._rebuild_layout(force=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_layout()

    def _rebuild_layout(self, *, force: bool = False) -> None:
        compact = self.width() < self._compact_breakpoint if self.width() > 0 else False
        if not force and compact == self._compact_mode:
            return
        self._compact_mode = compact

        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self)

        row = 0
        self._layout.setColumnStretch(0, 0)
        self._layout.setColumnStretch(1, 1)
        for item in self._rows:
            if item["kind"] == "full":
                self._layout.addWidget(item["widget"], row, 0, 1, 2)
                row += 1
                continue

            label = item["label"]
            field = item["field"]
            if compact or item.get("full_width"):
                self._layout.addWidget(label, row, 0, 1, 2)
                row += 1
                self._layout.addWidget(field, row, 0, 1, 2)
                row += 1
            else:
                self._layout.addWidget(label, row, 0)
                self._layout.addWidget(field, row, 1)
                row += 1

        self._layout.setRowStretch(row, 1)


def configure_form_layout(layout: QFormLayout) -> None:
    layout.setSpacing(CARD_SPACING)
    layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setFormAlignment(Qt.AlignTop)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    layout.setRowWrapPolicy(QFormLayout.WrapLongRows)


def configure_grid_layout(layout: QGridLayout) -> None:
    layout.setHorizontalSpacing(CARD_SPACING)
    layout.setVerticalSpacing(CARD_SPACING)


def configure_table(
    table: QTableWidget,
    *,
    min_height: int = TABLE_MIN_HEIGHT,
    alternating: bool = True,
    stretch_last: bool = False,
    sortable: bool = False,
) -> None:
    table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
    table.setAlternatingRowColors(alternating)
    table.setSortingEnabled(sortable)
    table.setMinimumHeight(min_height)
    table.setWordWrap(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    table.setShowGrid(True)
    table.setCornerButtonEnabled(False)
    table.setTextElideMode(Qt.ElideRight)
    header = table.horizontalHeader()
    header.setHighlightSections(False)
    header.setMinimumSectionSize(96)
    header.setStretchLastSection(stretch_last)


def configure_text_panel(panel: QTextEdit, *, min_height: int = TEXT_PANEL_MIN_HEIGHT) -> None:
    panel.setMinimumHeight(min_height)
    panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


def create_hint(text: str) -> QLabel:
    label = QLabel(text)
    style_label(label, "section-hint", word_wrap=True)
    return label


def create_status_label() -> QLabel:
    label = QLabel("")
    label.setAlignment(Qt.AlignCenter)
    style_label(label, "status", word_wrap=True)
    return label


def make_field_caption(text: str) -> QLabel:
    label = QLabel(text)
    style_label(label, "muted", word_wrap=True)
    return label


def widen_actions(buttons: Iterable[QPushButton], *, large: bool = False) -> None:
    for button in buttons:
        button.setMinimumHeight(CONTROL_HEIGHT_LARGE if large else CONTROL_HEIGHT)
        button.setMinimumWidth(144)


def soften_group(group: QGroupBox) -> None:
    group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
