"""
Digital twin experiment use case for analyst-only desktop workflows.
"""

from __future__ import annotations

from pathlib import Path

from fire_es.research.synthetic_rank_experiment import (
    DEFAULT_FEATURE_SET_NAMES,
    DEFAULT_MODEL_NAMES,
    SyntheticRankExperimentConfig,
    run_synthetic_rank_experiment,
)

from .base_use_case import BaseUseCase, UseCaseCancelledError, UseCaseResult, UseCaseStatus


class DigitalTwinExperimentUseCase(BaseUseCase):
    """Run the research-only synthetic rank experiment."""

    def __init__(self, reports_path: Path):
        super().__init__(
            name="DigitalTwinExperiment",
            description="ЦДС: генерация ИВД и сравнение моделей",
        )
        self.reports_path = reports_path

    def execute(
        self,
        *,
        input_path: Path,
        target: str = "rank_tz",
        synthetic_rows: int = 100_000,
        train_rows: int = 97_000,
        synthetic_validation_rows: int = 3_000,
        seed: int = 42,
        modes: list[str] | None = None,
        feature_sets: list[str] | None = None,
        models: list[str] | None = None,
        numeric_noise_scale: float = 0.08,
        categorical_smoothing: float = 0.5,
        global_mix: float = 0.15,
        extra_missing_rate: float = 0.0,
    ) -> UseCaseResult:
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings: list[str] = []

        try:
            self.report_progress(1, 5, "Проверка параметров ЦДС")
            self.check_cancelled()
            if not input_path.exists():
                return UseCaseResult(
                    success=False,
                    message=f"Файл данных не найден: {input_path}",
                    warnings=warnings,
                )

            output_dir = self.reports_path / "synthetic_rank_experiment"
            self.report_progress(2, 5, "Построение профиля и генерация ИВД")
            self.check_cancelled()

            config = SyntheticRankExperimentConfig(
                input_path=input_path,
                output_dir=output_dir,
                target=target,
                synthetic_rows=synthetic_rows,
                train_rows=train_rows,
                synthetic_validation_rows=synthetic_validation_rows,
                seed=seed,
                modes=modes or ["baseline_real", "synthetic_only"],
                feature_sets=feature_sets or DEFAULT_FEATURE_SET_NAMES.copy(),
                models=models or DEFAULT_MODEL_NAMES.copy(),
                numeric_noise_scale=numeric_noise_scale,
                categorical_smoothing=categorical_smoothing,
                global_mix=global_mix,
                extra_missing_rate=extra_missing_rate,
            )
            self.report_progress(3, 5, "Обучение моделей и расчет метрик")
            self.check_cancelled()
            result = run_synthetic_rank_experiment(config)
            self.report_progress(4, 5, "Сохранение отчетов")
            self.check_cancelled()
            self.report_progress(5, 5, "Эксперимент ЦДС завершен")
            self.status = UseCaseStatus.COMPLETED
            return UseCaseResult(
                success=True,
                message="Эксперимент ЦДС завершен",
                data=result,
                warnings=warnings,
            )
        except UseCaseCancelledError:
            self.status = UseCaseStatus.CANCELLED
            return UseCaseResult(
                success=False,
                message="Эксперимент ЦДС отменен",
                warnings=warnings,
            )
        except Exception as exc:
            self.status = UseCaseStatus.FAILED
            return UseCaseResult(
                success=False,
                message=f"Ошибка эксперимента ЦДС: {exc}",
                error=str(exc),
                warnings=warnings,
            )
