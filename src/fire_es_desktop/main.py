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
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Настроить стиль
    from fire_es_desktop.ui.theme import APP_STYLE_SHEET

    app.setStyleSheet(APP_STYLE_SHEET)

    # Создать главное окно
    from fire_es_desktop.ui import MainWindow
    window = MainWindow(role=args.role)
    window.show()

    logger.info("Application started")

    # Запуск цикла
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
