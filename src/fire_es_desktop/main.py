# src/fire_es_desktop/main.py
"""
Fire ES Desktop — главное приложение.

Запуск:
  python -m fire_es_desktop.main --role analyst
  python -m fire_es_desktop.main --role lpr

Сборка exe:
  pyinstaller build/pyinstaller.spec
"""

import sys
import argparse
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("FireES.Desktop")


def _apply_text_outline(window):
    """Добавить визуальную чёрную обводку белому тексту через shadow-эффект."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QAbstractButton,
        QGraphicsDropShadowEffect,
        QGroupBox,
        QLabel,
        QWidget,
    )

    text_widgets = (QLabel, QAbstractButton, QGroupBox)

    for widget in window.findChildren(QWidget):
        if not isinstance(widget, text_widgets):
            continue
        effect = QGraphicsDropShadowEffect(widget)
        effect.setOffset(0, 0)
        effect.setBlurRadius(2)
        effect.setColor(QColor(0, 0, 0))
        widget.setGraphicsEffect(effect)


def main():
    """Точка входа приложения."""
    parser = argparse.ArgumentParser(
        description="Fire ES Desktop — Экспертная система классификации пожаров"
    )
    parser.add_argument(
        "--role",
        choices=["analyst", "lpr"],
        default="analyst",
        help="Роль пользователя (analyst или lpr)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Путь к Workspace (необязательно)"
    )

    args = parser.parse_args()

    logger.info(f"Starting Fire ES Desktop, role: {args.role}")

    # Импорт PySide6
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
    except ImportError as e:
        logger.error(f"PySide6 not installed: {e}")
        print("Ошибка: PySide6 не установлен.")
        print("Установите: pip install PySide6")
        sys.exit(1)

    # Создать приложение
    app = QApplication(sys.argv)
    app.setApplicationName("Fire ES Desktop")
    app.setOrganizationName("Fire ES")

    # Настроить шрифт
    font = QFont("Arial", 10)
    app.setFont(font)

    # Настроить стиль
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #666666;
            color: white;
        }
        QStackedWidget, QStatusBar {
            background-color: #5f5f5f;
            color: white;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid black;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: #707070;
            color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: white;
        }
        QLabel {
            color: white;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #5b5b5b;
            color: white;
            border: 1px solid black;
            selection-background-color: #8a8a8a;
        }
        QTableView, QTableWidget, QListWidget {
            background-color: #5f5f5f;
            color: white;
            border: 1px solid black;
            gridline-color: black;
        }
        QHeaderView::section {
            background-color: #4f4f4f;
            color: white;
            border: 1px solid black;
        }
        QPushButton {
            background-color: #7a7a7a;
            color: white;
            border: 1px solid black;
            border-radius: 4px;
            padding: 6px 10px;
        }
        QPushButton:hover {
            background-color: #8a8a8a;
        }
        QPushButton:disabled {
            background-color: #555555;
            color: white;
        }
    """)

    # Создать главное окно
    from fire_es_desktop.ui import MainWindow
    window = MainWindow(role=args.role)
    _apply_text_outline(window)
    window.show()

    logger.info("Application started")

    # Запуск цикла
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
