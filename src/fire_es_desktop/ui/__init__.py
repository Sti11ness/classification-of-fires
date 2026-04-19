# src/fire_es_desktop/ui/__init__.py
"""
UI — пользовательский интерфейс Fire ES Desktop.

Согласно spec_second.md раздел 8.2:
- Presentation слой (UI)
- MVVM паттерн (View → ViewModel → UseCase)
- PySide6 для desktop приложения
"""

from .main_window import MainWindow
from .pages import (
    ProjectPage,
    ImportPage,
    TrainingPage,
    LPRPredictPage,
    LPRDecisionHistoryPage,
    ModelsPage,
    LogPage
)

__all__ = [
    "MainWindow",
    "ProjectPage",
    "ImportPage",
    "TrainingPage",
    "LPRPredictPage",
    "LPRDecisionHistoryPage",
    "ModelsPage",
    "LogPage",
]
