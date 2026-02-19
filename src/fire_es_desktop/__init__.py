# src/fire_es_desktop/__init__.py
"""
Fire ES Desktop — экспертная система для классификации пожаров и выбора ресурсов тушения.

Desktop приложение на PySide6 для работы в режимах:
- Аналитик (полный функционал)
- ЛПР (прогноз и сохранение решений)

Архитектура соответствует требованиям из spec_first.md и spec_second.md:
- Слои: Presentation, Application (Use Cases), Domain, Infrastructure
- Workspace как контракт
- TaskRunner для фоновых задач
- MVVM паттерн (View → ViewModel → UseCase)
- Чёткое разделение ролей: analyst / lpr

Запуск:
    python -m fire_es_desktop.main --role analyst
    python -m fire_es_desktop.main --role lpr

Сборка exe:
    pyinstaller build/pyinstaller.spec
"""

__version__ = "0.2.0"
__author__ = "Fire ES Team"
