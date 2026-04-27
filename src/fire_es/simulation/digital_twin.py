"""
Digital twin profile and rank-conditional synthetic data generator.

The legacy public function ``generate_synthetic_fire_dataset`` is kept for
backward compatibility. The research contour should use
``generate_rank_conditional_synthetic_dataset`` because it preserves the target
distribution and distorts feature values instead of copying source rows as-is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fire_es.rank_tz_contract import FIELD_SPECS

from .distortions import apply_missingness, apply_numeric_noise

CANONICAL_SOURCE_SHEET = "БД-1...2000--2020 (1+2)"
SYNTHETIC_SOURCE = "digital_twin_rank_conditional"

TIME_ORDER_COLUMNS = [
    "t_detect_min",
    "t_report_min",
    "t_arrival_min",
    "t_first_hose_min",
    "t_contained_min",
    "t_extinguished_min",
]

DERIVED_COLUMNS = {
    "day_of_week",
    "quarter",
    "season",
    "is_weekend",
    "delta_detect_to_report",
    "delta_report_to_arrival",
    "delta_arrival_to_hose",
    "delta_hose_to_contained",
    "delta_contained_to_extinguished",
    "total_response_time",
    "fire_floor_ratio",
    "is_high_rise",
    "is_super_high_rise",
    "basement_fire",
    "top_floor_fire",
    "is_far",
    "is_very_far",
    "distance_category",
    "risk_category_missing",
    "fpo_class_missing",
    "risk_category_code_missing",
    "fpo_class_code_missing",
    "building_floors_bin",
    "equipment_count_bin",
    "floors_x_distance",
    "weekend_x_detect",
    "high_rise_x_floors",
}


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "missing": int(series.isna().sum()),
            "mean": None,
            "std": None,
            "median": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "iqr": None,
            "mad": None,
        }
    median = numeric.median()
    q05, q25, q75, q95 = numeric.quantile([0.05, 0.25, 0.75, 0.95]).tolist()
    mad = (numeric - median).abs().median()
    return {
        "count": int(numeric.count()),
        "missing": int(series.isna().sum()),
        "mean": _safe_float(numeric.mean()),
        "std": _safe_float(numeric.std(ddof=0)),
        "median": _safe_float(median),
        "q05": _safe_float(q05),
        "q25": _safe_float(q25),
        "q75": _safe_float(q75),
        "q95": _safe_float(q95),
        "iqr": _safe_float(q75 - q25),
        "mad": _safe_float(mad),
        "min": _safe_float(numeric.min()),
        "max": _safe_float(numeric.max()),
    }


def _value_counts(series: pd.Series, *, top_n: int = 30) -> dict[str, int]:
    counts = series.astype("object").where(series.notna(), "__missing__").value_counts(dropna=False)
    return {str(_json_scalar(key)): int(value) for key, value in counts.head(top_n).items()}


def _rank_distribution(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    return {
        str(_json_scalar(key)): {
            "count": int(value),
            "share": float(value / total) if total else 0.0,
        }
        for key, value in counts.items()
    }


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Infer columns that should be sampled as categorical values."""
    categorical: set[str] = set(df.select_dtypes(include=["object", "category", "bool"]).columns)
    for column in df.columns:
        if column in DERIVED_COLUMNS:
            continue
        if column.endswith("_code") or column.endswith("_invalid") or column.startswith("flag_"):
            categorical.add(column)
            continue
        spec = FIELD_SPECS.get(column, {})
        if spec.get("kind") == "categorical":
            categorical.add(column)
            continue
        if pd.api.types.is_integer_dtype(df[column]) and df[column].nunique(dropna=True) <= 50:
            categorical.add(column)
    return sorted(categorical)


def build_statistical_profile(
    df: pd.DataFrame,
    *,
    target_column: str = "rank_tz",
    canonical_source_sheet: str = CANONICAL_SOURCE_SHEET,
    max_correlation_columns: int = 50,
) -> dict[str, Any]:
    """Build a statistical profile used by the digital twin.

    The returned dictionary deliberately includes the legacy keys used by older
    tests: ``row_count``, ``numeric_summary``, ``categorical_summary`` and
    ``missingness``.
    """
    numeric = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(exclude=["number"])
    profile: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "canonical_source_sheet": canonical_source_sheet,
        "canonical_rows": int((df.get("source_sheet") == canonical_source_sheet).sum())
        if "source_sheet" in df.columns
        else None,
        "numeric_summary": {
            column: _numeric_summary(numeric[column])
            for column in numeric.columns
        },
        "categorical_summary": {
            column: _value_counts(categorical[column])
            for column in categorical.columns
        },
        "missingness": (df.isna().mean() * 100).round(4).to_dict(),
    }

    if target_column in df.columns:
        target = df[target_column]
        profile["target_column"] = target_column
        profile["rank_distribution"] = _rank_distribution(target.dropna())
        rank_conditional_numeric: dict[str, Any] = {}
        rank_conditional_categorical: dict[str, Any] = {}
        for rank_value, rank_df in df.dropna(subset=[target_column]).groupby(target_column, dropna=False):
            key = str(_json_scalar(rank_value))
            rank_conditional_numeric[key] = {
                column: _numeric_summary(rank_df[column])
                for column in numeric.columns
            }
            rank_conditional_categorical[key] = {
                column: _value_counts(rank_df[column])
                for column in categorical.columns
            }
        profile["rank_conditional_numeric"] = rank_conditional_numeric
        profile["rank_conditional_categorical"] = rank_conditional_categorical

    corr_columns = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() >= 3 and numeric[column].nunique(dropna=True) > 1
    ][:max_correlation_columns]
    if len(corr_columns) >= 2:
        corr = numeric[corr_columns].corr(numeric_only=True).fillna(0.0).round(4)
        profile["numeric_correlations"] = corr.to_dict()
    else:
        profile["numeric_correlations"] = {}

    if target_column in df.columns:
        categorical_rank_links: dict[str, Any] = {}
        for column in infer_categorical_columns(df):
            if column == target_column or column not in df.columns:
                continue
            series = df[column].astype("object").where(df[column].notna(), "__missing__")
            top_values = series.value_counts().head(20).index
            subset = df.loc[series.isin(top_values), [target_column]].copy()
            subset[column] = series.loc[subset.index]
            if subset.empty:
                continue
            table = pd.crosstab(subset[column], subset[target_column], normalize="index")
            categorical_rank_links[column] = table.round(4).to_dict()
        profile["categorical_rank_links"] = categorical_rank_links

    return profile


def _robust_scale(series: pd.Series, fallback: float = 1.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return fallback
    q25, q75 = numeric.quantile([0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349) if q75 != q25 else 0.0
    median = numeric.median()
    mad_scale = float((numeric - median).abs().median() * 1.4826)
    std_scale = float(numeric.std(ddof=0))
    for candidate in (iqr_scale, mad_scale, std_scale):
        if np.isfinite(candidate) and candidate > 0:
            return candidate
    return fallback


def _is_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool(np.all(np.isclose(numeric, np.round(numeric))))


def _sample_values(
    rng: np.random.Generator,
    rank_series: pd.Series,
    global_series: pd.Series,
    *,
    size: int,
    smoothing: float,
    global_mix: float,
) -> np.ndarray:
    rank_values = rank_series.astype("object").where(rank_series.notna(), "__missing__")
    global_values = global_series.astype("object").where(global_series.notna(), "__missing__")
    values = pd.Index(rank_values.drop_duplicates().tolist() + global_values.drop_duplicates().tolist()).drop_duplicates()
    if len(values) == 0:
        return np.full(size, np.nan, dtype=object)

    rank_counts = rank_values.value_counts().reindex(values, fill_value=0).astype(float)
    global_counts = global_values.value_counts().reindex(values, fill_value=0).astype(float)
    rank_probs = (rank_counts + smoothing) / float(rank_counts.sum() + smoothing * len(values))
    global_probs = (global_counts + smoothing) / float(global_counts.sum() + smoothing * len(values))
    mixed_probs = (1.0 - global_mix) * rank_probs.to_numpy() + global_mix * global_probs.to_numpy()
    mixed_probs = mixed_probs / mixed_probs.sum()
    alpha = np.maximum(mixed_probs * max(float(rank_counts.sum()), 1.0), smoothing)
    sampled_probs = rng.dirichlet(alpha)
    sampled = rng.choice(values.to_numpy(dtype=object), size=size, replace=True, p=sampled_probs)
    return np.where(sampled == "__missing__", np.nan, sampled)


def _enforce_floor_constraints(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"building_floors", "fire_floor"}.issubset(result.columns):
        building = pd.to_numeric(result["building_floors"], errors="coerce")
        fire_floor = pd.to_numeric(result["fire_floor"], errors="coerce")
        building = building.clip(lower=0).round()
        fire_floor = fire_floor.round()
        over_mask = fire_floor.notna() & building.notna() & (fire_floor > building)
        fire_floor.loc[over_mask] = building.loc[over_mask]
        result["building_floors"] = building
        result["fire_floor"] = fire_floor
    return result


def _enforce_time_constraints(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    existing = [column for column in TIME_ORDER_COLUMNS if column in result.columns]
    previous: pd.Series | None = None
    for column in existing:
        current = pd.to_numeric(result[column], errors="coerce")
        current = current.mask(current < 0, 0)
        if previous is not None:
            mask = current.notna() & previous.notna() & (current < previous)
            current.loc[mask] = previous.loc[mask]
            current = current.round()
            previous = current.combine_first(previous)
        else:
            current = current.round()
            previous = current
        result[column] = current
    return result


def recompute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute feature-engineering columns after synthetic distortion."""
    result = df.copy()

    if "fire_date" in result.columns:
        dates = pd.to_datetime(result["fire_date"], errors="coerce")
        result["day_of_week"] = dates.dt.dayofweek
        result["month"] = dates.dt.month if "month" in result.columns else dates.dt.month
        result["quarter"] = dates.dt.quarter
        result["season"] = dates.dt.month.map(
            lambda month: np.nan
            if pd.isna(month)
            else 1
            if month in (12, 1, 2)
            else 2
            if month in (3, 4, 5)
            else 3
            if month in (6, 7, 8)
            else 4
        )
        result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(float)
    elif "month" in result.columns:
        month = pd.to_numeric(result["month"], errors="coerce")
        result["quarter"] = np.ceil(month / 3).clip(lower=1, upper=4)
        result["season"] = month.map(
            lambda value: np.nan
            if pd.isna(value)
            else 1
            if value in (12, 1, 2)
            else 2
            if value in (3, 4, 5)
            else 3
            if value in (6, 7, 8)
            else 4
        )

    pairs = [
        ("t_detect_min", "t_report_min", "delta_detect_to_report"),
        ("t_report_min", "t_arrival_min", "delta_report_to_arrival"),
        ("t_arrival_min", "t_first_hose_min", "delta_arrival_to_hose"),
        ("t_first_hose_min", "t_contained_min", "delta_hose_to_contained"),
        ("t_contained_min", "t_extinguished_min", "delta_contained_to_extinguished"),
    ]
    for left, right, target in pairs:
        if {left, right}.issubset(result.columns):
            result[target] = pd.to_numeric(result[right], errors="coerce") - pd.to_numeric(
                result[left],
                errors="coerce",
            )

    if {"t_detect_min", "t_extinguished_min"}.issubset(result.columns):
        result["total_response_time"] = pd.to_numeric(result["t_extinguished_min"], errors="coerce") - pd.to_numeric(
            result["t_detect_min"],
            errors="coerce",
        )

    if {"building_floors", "fire_floor"}.issubset(result.columns):
        floors = pd.to_numeric(result["building_floors"], errors="coerce")
        fire_floor = pd.to_numeric(result["fire_floor"], errors="coerce")
        result["fire_floor_ratio"] = np.where(floors > 0, fire_floor / floors, np.nan)
        result["is_high_rise"] = (floors >= 10).astype(float)
        result["is_super_high_rise"] = (floors >= 25).astype(float)
        result["basement_fire"] = (fire_floor < 0).astype(float)
        result["top_floor_fire"] = ((floors > 0) & (fire_floor == floors)).astype(float)
        result["building_floors_bin"] = pd.cut(
            floors,
            bins=[-np.inf, 1, 5, 9, 16, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

    if "distance_to_station" in result.columns:
        distance = pd.to_numeric(result["distance_to_station"], errors="coerce")
        result["is_far"] = (distance >= 10).astype(float)
        result["is_very_far"] = (distance >= 25).astype(float)
        result["distance_category"] = pd.cut(
            distance,
            bins=[-np.inf, 3, 10, 25, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)

    if {"building_floors", "distance_to_station"}.issubset(result.columns):
        result["floors_x_distance"] = pd.to_numeric(result["building_floors"], errors="coerce") * pd.to_numeric(
            result["distance_to_station"],
            errors="coerce",
        )
    if {"is_weekend", "t_detect_min"}.issubset(result.columns):
        result["weekend_x_detect"] = pd.to_numeric(result["is_weekend"], errors="coerce") * pd.to_numeric(
            result["t_detect_min"],
            errors="coerce",
        )
    if {"is_high_rise", "building_floors"}.issubset(result.columns):
        result["high_rise_x_floors"] = pd.to_numeric(result["is_high_rise"], errors="coerce") * pd.to_numeric(
            result["building_floors"],
            errors="coerce",
        )

    for source, target in [
        ("risk_category_code", "risk_category_missing"),
        ("fpo_class_code", "fpo_class_missing"),
        ("risk_category_code", "risk_category_code_missing"),
        ("fpo_class_code", "fpo_class_code_missing"),
    ]:
        if source in result.columns:
            result[target] = result[source].isna().astype(float)

    if "equipment_count" in result.columns:
        equipment_count = pd.to_numeric(result["equipment_count"], errors="coerce")
        result["equipment_count_bin"] = pd.cut(
            equipment_count,
            bins=[-np.inf, 1, 2, 4, 8, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

    return result


def generate_rank_conditional_synthetic_dataset(
    df: pd.DataFrame,
    *,
    n_rows: int,
    target_column: str = "rank_tz",
    random_state: int = 42,
    numeric_noise_scale: float = 0.08,
    categorical_smoothing: float = 0.5,
    global_mix: float = 0.15,
    extra_missing_rate: float = 0.0,
    distortion_scenario: str = "baseline",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate rank-conditional distorted synthetic rows."""
    if df.empty:
        raise ValueError("Cannot build a synthetic dataset from an empty dataframe")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column for rank-conditional generation: {target_column}")

    base = df.dropna(subset=[target_column]).reset_index(drop=True)
    if base.empty:
        raise ValueError(f"Target column {target_column} has no non-empty values")

    rng = np.random.default_rng(random_state)
    profile = build_statistical_profile(base, target_column=target_column)
    n_rows = int(n_rows)

    rank_counts = base[target_column].value_counts().sort_index()
    rank_values = rank_counts.index.to_numpy()
    rank_probabilities = (rank_counts.to_numpy(dtype=float) + categorical_smoothing)
    rank_probabilities = rank_probabilities / rank_probabilities.sum()
    target_ranks = rng.choice(rank_values, size=n_rows, replace=True, p=rank_probabilities)

    sampled_indices = np.empty(n_rows, dtype=int)
    for rank_value in rank_values:
        mask = target_ranks == rank_value
        rank_indices = base.index[base[target_column] == rank_value].to_numpy()
        sampled_indices[mask] = rng.choice(rank_indices, size=int(mask.sum()), replace=True)

    synthetic = base.iloc[sampled_indices].reset_index(drop=True).copy()
    synthetic[target_column] = target_ranks

    categorical_columns = set(infer_categorical_columns(base))
    categorical_columns.discard(target_column)
    numeric_columns = [
        column
        for column in base.select_dtypes(include=["number"]).columns
        if column != target_column and column not in DERIVED_COLUMNS
    ]
    non_numeric_columns = [
        column
        for column in base.columns
        if column not in numeric_columns and column != target_column and column not in DERIVED_COLUMNS
    ]

    for column in sorted(categorical_columns.intersection(base.columns)):
        for rank_value in rank_values:
            mask = synthetic[target_column] == rank_value
            if not mask.any():
                continue
            synthetic.loc[mask, column] = _sample_values(
                rng,
                base.loc[base[target_column] == rank_value, column],
                base[column],
                size=int(mask.sum()),
                smoothing=categorical_smoothing,
                global_mix=global_mix,
            )

    for column in numeric_columns:
        if column in categorical_columns:
            continue
        source = pd.to_numeric(base[column], errors="coerce")
        generated = pd.to_numeric(synthetic[column], errors="coerce")
        global_median = source.median() if source.notna().any() else 0.0
        q05 = source.quantile(0.05) if source.notna().any() else global_median
        q95 = source.quantile(0.95) if source.notna().any() else global_median
        for rank_value in rank_values:
            mask = synthetic[target_column] == rank_value
            if not mask.any():
                continue
            rank_source = pd.to_numeric(base.loc[base[target_column] == rank_value, column], errors="coerce")
            rank_median = rank_source.median() if rank_source.notna().any() else global_median
            values = generated.loc[mask].fillna(rank_median)
            sigma = _robust_scale(rank_source, fallback=_robust_scale(source, fallback=1.0))
            noise = rng.normal(0.0, sigma * float(numeric_noise_scale), size=int(mask.sum()))
            generated.loc[mask] = values.to_numpy(dtype=float) + noise
        generated = generated.clip(lower=q05, upper=q95)
        if _is_integer_like(base[column]) or FIELD_SPECS.get(column, {}).get("type") == "int":
            generated = generated.round()
        synthetic[column] = generated

    for column in non_numeric_columns:
        if column in categorical_columns or column in {"source_sheet", "source_period"}:
            continue
        if column not in synthetic.columns or column in {"fire_date"}:
            continue
        # Text fields are not model inputs, but sampling them avoids direct row copies.
        if synthetic[column].dtype == object:
            for rank_value in rank_values:
                mask = synthetic[target_column] == rank_value
                if not mask.any():
                    continue
                synthetic.loc[mask, column] = _sample_values(
                    rng,
                    base.loc[base[target_column] == rank_value, column],
                    base[column],
                    size=int(mask.sum()),
                    smoothing=max(categorical_smoothing, 1.0),
                    global_mix=max(global_mix, 0.35),
                )

    synthetic = _enforce_floor_constraints(synthetic)
    synthetic = _enforce_time_constraints(synthetic)
    synthetic = recompute_derived_features(synthetic)

    if extra_missing_rate > 0:
        protected = {
            target_column,
            "synthetic_id",
            "synthetic_source",
            "synthetic_seed",
            "synthetic_base_rank",
            "synthetic_distortion_scenario",
        }
        mutable_columns = [column for column in synthetic.columns if column not in protected]
        missing_mask = rng.random((len(synthetic), len(mutable_columns))) < float(extra_missing_rate)
        synthetic.loc[:, mutable_columns] = synthetic[mutable_columns].mask(missing_mask)
        synthetic = _enforce_floor_constraints(_enforce_time_constraints(synthetic))
        synthetic = recompute_derived_features(synthetic)

    if "row_id" in synthetic.columns:
        synthetic["row_id"] = np.arange(1, n_rows + 1)
    synthetic["source_sheet"] = SYNTHETIC_SOURCE
    synthetic["source_period"] = "synthetic" if "source_period" in synthetic.columns else "synthetic"
    synthetic["synthetic_id"] = [f"syn-{random_state}-{idx:07d}" for idx in range(n_rows)]
    synthetic["synthetic_source"] = SYNTHETIC_SOURCE
    synthetic["synthetic_seed"] = int(random_state)
    synthetic["synthetic_base_rank"] = target_ranks
    synthetic["synthetic_distortion_scenario"] = distortion_scenario

    profile["synthetic_generation"] = {
        "rows": n_rows,
        "target_column": target_column,
        "random_state": int(random_state),
        "numeric_noise_scale": float(numeric_noise_scale),
        "categorical_smoothing": float(categorical_smoothing),
        "global_mix": float(global_mix),
        "extra_missing_rate": float(extra_missing_rate),
        "distortion_scenario": distortion_scenario,
    }
    return synthetic.reset_index(drop=True), profile


def generate_synthetic_fire_dataset(
    df: pd.DataFrame,
    *,
    n_rows: int,
    missing_rate: float = 0.0,
    noise_scale: float = 0.0,
    random_state: int = 42,
    target_column: str = "rank_tz",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic dataset and its profile.

    If ``target_column`` exists, the rank-conditional digital twin generator is
    used. Otherwise the legacy bootstrap-style behavior is retained.
    """
    if df.empty:
        raise ValueError("Cannot build a synthetic dataset from an empty dataframe")
    if target_column in df.columns:
        return generate_rank_conditional_synthetic_dataset(
            df,
            n_rows=n_rows,
            target_column=target_column,
            random_state=random_state,
            numeric_noise_scale=noise_scale if noise_scale > 0 else 0.08,
            extra_missing_rate=missing_rate,
        )

    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(0, len(df), size=int(n_rows))
    synthetic = df.iloc[sample_indices].reset_index(drop=True).copy()
    numeric_columns = synthetic.select_dtypes(include=["number"]).columns.tolist()
    if missing_rate > 0:
        synthetic = apply_missingness(synthetic, missing_rate=missing_rate, random_state=random_state)
    if noise_scale > 0 and numeric_columns:
        synthetic = apply_numeric_noise(
            synthetic,
            columns=numeric_columns,
            scale=noise_scale,
            random_state=random_state,
        )
    profile = build_statistical_profile(synthetic)
    return synthetic, profile
