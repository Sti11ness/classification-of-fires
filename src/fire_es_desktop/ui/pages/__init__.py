"""Страницы UI для Fire ES Desktop."""

from .project_page import ProjectPage
from .import_page import ImportPage
from .training_page import TrainingPage
from .lpr_predict_page import LPRPredictPage
from .models_page import ModelsPage
from .log_page import LogPage
from .batch_predict_page import BatchPredictPage

__all__ = [
    "ProjectPage",
    "ImportPage",
    "TrainingPage",
    "LPRPredictPage",
    "ModelsPage",
    "LogPage",
    "BatchPredictPage",
]
