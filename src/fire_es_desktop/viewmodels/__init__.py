"""ViewModels для Fire ES Desktop."""

from .project_viewmodel import ProjectViewModel
from .import_data_viewmodel import ImportDataViewModel
from .train_model_viewmodel import TrainModelViewModel
from .lpr_predict_viewmodel import LPRPredictViewModel
from .lpr_decision_history_viewmodel import LPRDecisionHistoryViewModel
from .batch_predict_viewmodel import BatchPredictViewModel
from .digital_twin_viewmodel import DigitalTwinViewModel

# Backward compatibility alias
TrainingViewModel = TrainModelViewModel

__all__ = [
    "ProjectViewModel",
    "ImportDataViewModel",
    "TrainModelViewModel",
    "TrainingViewModel",
    "LPRPredictViewModel",
    "LPRDecisionHistoryViewModel",
    "BatchPredictViewModel",
    "DigitalTwinViewModel",
]
