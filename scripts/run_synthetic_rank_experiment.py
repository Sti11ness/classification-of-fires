from __future__ import annotations

import argparse
from pathlib import Path

from fire_es.research.synthetic_rank_experiment import (
    DEFAULT_FEATURE_SET_NAMES,
    DEFAULT_MODEL_NAMES,
    DEFAULT_MODES,
    SyntheticRankExperimentConfig,
    run_synthetic_rank_experiment,
)


def _csv_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run digital-twin synthetic rank_tz experiment.",
    )
    parser.add_argument("--input", type=Path, default=Path("clean_df_enhanced.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/synthetic_rank_experiment"))
    parser.add_argument("--target", default="rank_tz")
    parser.add_argument("--synthetic-rows", type=int, default=100_000)
    parser.add_argument("--train-rows", type=int, default=97_000)
    parser.add_argument("--synthetic-validation-rows", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SET_NAMES))
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_NAMES))
    parser.add_argument("--numeric-noise-scale", type=float, default=0.08)
    parser.add_argument("--categorical-smoothing", type=float, default=0.5)
    parser.add_argument("--global-mix", type=float, default=0.15)
    parser.add_argument("--extra-missing-rate", type=float, default=0.0)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    config = SyntheticRankExperimentConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        synthetic_rows=args.synthetic_rows,
        train_rows=args.train_rows,
        synthetic_validation_rows=args.synthetic_validation_rows,
        seed=args.seed,
        modes=_csv_list(args.modes, DEFAULT_MODES),
        feature_sets=_csv_list(args.feature_sets, DEFAULT_FEATURE_SET_NAMES),
        models=_csv_list(args.models, DEFAULT_MODEL_NAMES),
        numeric_noise_scale=args.numeric_noise_scale,
        categorical_smoothing=args.categorical_smoothing,
        global_mix=args.global_mix,
        extra_missing_rate=args.extra_missing_rate,
        timestamp=args.timestamp,
    )
    result = run_synthetic_rank_experiment(config)
    print(f"REPORT: {result['report_path']}")
    print(f"METRICS: {result['metrics_path']}")
    print(f"RUN_DIR: {result['run_dir']}")


if __name__ == "__main__":
    main()
