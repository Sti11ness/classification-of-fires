"""ViewModels для Fire ES Desktop."""

from .project_viewmodel import ProjectViewModel
from .import_data_viewmodel import ImportDataViewModel
from .train_model_viewmodel import TrainModelViewModel
from .lpr_predict_viewmodel import LPRPredictViewModel
from .batch_predict_viewmodel import BatchPredictViewModel

# Backward compatibility alias
TrainingViewModel = TrainModelViewModel

__all__ = [
    "ProjectViewModel",
    "ImportDataViewModel",
    "TrainModelViewModel",
    "TrainingViewModel",
    "LPRPredictViewModel",
    "BatchPredictViewModel",
]
