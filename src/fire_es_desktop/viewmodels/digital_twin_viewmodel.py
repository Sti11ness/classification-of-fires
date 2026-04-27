"""
ViewModel for the analyst digital twin page.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from fire_es.research.synthetic_rank_experiment import (
    DEFAULT_FEATURE_SET_NAMES,
    DEFAULT_MODEL_NAMES,
    DEFAULT_MODES,
)

from ..use_cases import DigitalTwinExperimentUseCase

logger = logging.getLogger("DigitalTwinViewModel")


class DigitalTwinViewModel:
    """Store digital twin settings and run the experiment use case."""

    def __init__(self, reports_path: Path):
        self.reports_path = reports_path
        self.use_case = DigitalTwinExperimentUseCase(reports_path)

        self.input_path = Path("clean_df_enhanced.csv")
        self.target = "rank_tz"
        self.synthetic_rows = 100_000
        self.train_rows = 97_000
        self.synthetic_validation_rows = 3_000
        self.seed = 42
        self.modes = DEFAULT_MODES.copy()
        self.feature_sets = DEFAULT_FEATURE_SET_NAMES.copy()
        self.models = DEFAULT_MODEL_NAMES.copy()
        self.numeric_noise_scale = 0.08
        self.categorical_smoothing = 0.5
        self.global_mix = 0.15
        self.extra_missing_rate = 0.0

        self.result: dict[str, Any] | None = None
        self.error_message: str | None = None
        self.is_running = False

        self.on_started: Callable[[], None] | None = None
        self.on_complete: Callable[[dict[str, Any]], None] | None = None
        self.on_failed: Callable[[str], None] | None = None
        self.on_progress: Callable[[int, int, str], None] | None = None

    def set_input_path(self, path: Path) -> None:
        self.input_path = path

    def set_counts(self, *, synthetic_rows: int, train_rows: int, validation_rows: int) -> None:
        self.synthetic_rows = max(1, int(synthetic_rows))
        self.train_rows = max(1, int(train_rows))
        self.synthetic_validation_rows = max(1, int(validation_rows))
        required_total = self.train_rows + self.synthetic_validation_rows
        self.synthetic_rows = max(self.synthetic_rows, required_total)

    def set_generation_params(
        self,
        *,
        seed: int,
        numeric_noise_scale: float,
        categorical_smoothing: float,
        global_mix: float,
        extra_missing_rate: float,
    ) -> None:
        self.seed = int(seed)
        self.numeric_noise_scale = max(0.0, float(numeric_noise_scale))
        self.categorical_smoothing = max(0.01, float(categorical_smoothing))
        self.global_mix = min(1.0, max(0.0, float(global_mix)))
        self.extra_missing_rate = min(1.0, max(0.0, float(extra_missing_rate)))

    def set_selection(
        self,
        *,
        modes: list[str],
        feature_sets: list[str],
        models: list[str],
    ) -> None:
        self.modes = modes or DEFAULT_MODES.copy()
        self.feature_sets = feature_sets or DEFAULT_FEATURE_SET_NAMES.copy()
        self.models = models or DEFAULT_MODEL_NAMES.copy()

    def cancel(self) -> None:
        self.use_case.cancel()

    def run(self) -> None:
        self.is_running = True
        self.result = None
        self.error_message = None
        if self.on_started:
            self.on_started()

        def on_progress(current: int, total: int, description: str) -> None:
            if self.on_progress:
                self.on_progress(current, total, description)

        self.use_case.set_progress_callback(on_progress)
        try:
            result = self.use_case.execute(
                input_path=self.input_path,
                target=self.target,
                synthetic_rows=self.synthetic_rows,
                train_rows=self.train_rows,
                synthetic_validation_rows=self.synthetic_validation_rows,
                seed=self.seed,
                modes=self.modes,
                feature_sets=self.feature_sets,
                models=self.models,
                numeric_noise_scale=self.numeric_noise_scale,
                categorical_smoothing=self.categorical_smoothing,
                global_mix=self.global_mix,
                extra_missing_rate=self.extra_missing_rate,
            )
            if result.success:
                self.result = result.data or {}
                if self.on_complete:
                    self.on_complete(self.result)
            else:
                self.error_message = result.message
                if self.on_failed:
                    self.on_failed(result.message)
        except Exception as exc:
            logger.error("Digital twin experiment failed: %s", exc, exc_info=True)
            self.error_message = str(exc)
            if self.on_failed:
                self.on_failed(str(exc))
        finally:
            self.is_running = False
