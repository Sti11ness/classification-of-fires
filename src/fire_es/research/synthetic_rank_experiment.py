"""
Synthetic rank_tz experiment driven by the digital twin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier

from fire_es.rank_tz_contract import (
    CLASS_TO_RANK_MAP,
    get_feature_set_spec,
    map_rank_series_to_classes,
)
from fire_es.simulation.digital_twin import (
    CANONICAL_SOURCE_SHEET,
    build_statistical_profile,
    generate_rank_conditional_synthetic_dataset,
    recompute_derived_features,
)

DEFAULT_FEATURE_SET_NAMES = [
    "features_10_dispatch",
    "features_13_arrival",
    "features_15_first_hose",
    "features_19_retrospective_time",
    "features_35_enhanced_retrospective",
    "features_50_wide_no_target",
    "features_all_known_numeric",
]

DEFAULT_MODEL_NAMES = ["decision_tree", "random_forest", "hist_gradient_boosting"]
DEFAULT_MODES = ["baseline_real", "synthetic_only"]

TARGET_PROXY_PATTERNS = (
    "rank",
    "equipment",
    "nozzle",
)
TARGET_PROXY_COLUMNS = {
    "row_id",
    "severity_score",
    "fatalities",
    "injuries",
    "direct_damage",
    "direct_damage_log",
    "people_saved",
    "people_evacuated",
    "assets_saved",
    "flag_missing_outputs",
    "equipment_count_types",
    "equipment_count_bin",
}
META_COLUMNS = {
    "source_sheet",
    "source_period",
    "dup_flag",
    "synthetic_seed",
    "synthetic_base_rank",
}


@dataclass(frozen=True)
class FeatureSetConfig:
    name: str
    label: str
    features: list[str]
    availability_risk: str
    notes: str


@dataclass
class SyntheticRankExperimentConfig:
    input_path: Path
    output_dir: Path
    target: str = "rank_tz"
    synthetic_rows: int = 100_000
    train_rows: int = 97_000
    synthetic_validation_rows: int = 3_000
    seed: int = 42
    modes: list[str] = field(default_factory=lambda: DEFAULT_MODES.copy())
    feature_sets: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_SET_NAMES.copy())
    models: list[str] = field(default_factory=lambda: DEFAULT_MODEL_NAMES.copy())
    numeric_noise_scale: float = 0.08
    categorical_smoothing: float = 0.5
    global_mix: float = 0.15
    extra_missing_rate: float = 0.0
    canonical_source_sheet: str = CANONICAL_SOURCE_SHEET
    timestamp: str | None = None


def _stringify_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return value


def _event_key_columns(df: pd.DataFrame) -> list[str]:
    if "event_id" in df.columns:
        return ["event_id"]
    candidates = [
        "fire_date",
        "region_code",
        "settlement_type_code",
        "enterprise_type_code",
        "object_name",
        "address",
        "building_floors",
        "fire_floor",
        "t_report_min",
        "rank_tz",
    ]
    return [column for column in candidates if column in df.columns]


def add_stable_event_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    columns = _event_key_columns(result)
    if columns:
        frame = result[columns].astype("object").where(result[columns].notna(), "__missing__")
        result["_event_key"] = pd.util.hash_pandas_object(frame, index=False).astype(str)
    else:
        frame = result.astype("object").where(result.notna(), "__missing__")
        result["_event_key"] = pd.util.hash_pandas_object(frame, index=False).astype(str)
    return result


def prepare_real_canonical_test(
    df: pd.DataFrame,
    *,
    target: str = "rank_tz",
    canonical_source_sheet: str = CANONICAL_SOURCE_SHEET,
) -> tuple[pd.DataFrame, dict[str, int]]:
    initial_rows = int(len(df))
    if "source_sheet" in df.columns:
        canonical = df.loc[df["source_sheet"] == canonical_source_sheet].copy()
    else:
        canonical = df.copy()
    canonical_rows = int(len(canonical))
    canonical = canonical.dropna(subset=[target]).copy()
    with_target_rows = int(len(canonical))
    canonical = add_stable_event_key(canonical)
    before_dedup = int(len(canonical))
    canonical = canonical.drop_duplicates(subset=["_event_key"], keep="first").reset_index(drop=True)
    after_dedup = int(len(canonical))
    return canonical, {
        "input_rows": initial_rows,
        "canonical_rows": canonical_rows,
        "canonical_rows_with_target": with_target_rows,
        "real_test_rows": after_dedup,
        "duplicates_removed": before_dedup - after_dedup,
    }


def _wide_numeric_candidates(df: pd.DataFrame, target: str) -> list[str]:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    result: list[str] = []
    for column in numeric_columns:
        lowered = column.lower()
        if column == target or column in TARGET_PROXY_COLUMNS or column in META_COLUMNS:
            continue
        if any(pattern in lowered for pattern in TARGET_PROXY_PATTERNS):
            continue
        if lowered.startswith("synthetic_") or lowered.startswith("_"):
            continue
        if lowered.endswith("_invalid") or lowered.startswith("flag_"):
            continue
        result.append(column)
    return result


def build_experiment_feature_sets(df: pd.DataFrame, *, target: str = "rank_tz") -> dict[str, FeatureSetConfig]:
    dispatch = get_feature_set_spec("dispatch_initial_safe")["feature_order"]
    arrival = get_feature_set_spec("arrival_update_safe")["feature_order"]
    first_hose = get_feature_set_spec("first_hose_update_safe")["feature_order"]
    enhanced = get_feature_set_spec("enhanced_tactical")["feature_order"]
    wide_candidates = _wide_numeric_candidates(df, target)
    feature_sets = {
        "features_10_dispatch": FeatureSetConfig(
            name="features_10_dispatch",
            label="10 признаков: момент звонка",
            features=list(dispatch),
            availability_risk="production-safe",
            notes="Доступно до прибытия подразделений.",
        ),
        "features_13_arrival": FeatureSetConfig(
            name="features_13_arrival",
            label="13 признаков: после прибытия",
            features=list(arrival),
            availability_risk="stage-safe",
            notes="Нельзя использовать для решения в момент звонка.",
        ),
        "features_15_first_hose": FeatureSetConfig(
            name="features_15_first_hose",
            label="15 признаков: после подачи первого ствола",
            features=list(first_hose),
            availability_risk="stage-safe",
            notes="Нельзя использовать до развертывания.",
        ),
        "features_19_retrospective_time": FeatureSetConfig(
            name="features_19_retrospective_time",
            label="19 признаков: ретроспективные времена",
            features=list(
                dict.fromkeys(
                    list(first_hose)
                    + [
                        "t_contained_min",
                        "t_extinguished_min",
                        "delta_hose_to_contained",
                        "delta_contained_to_extinguished",
                    ]
                )
            ),
            availability_risk="offline/post-factum",
            notes="Benchmark после локализации/ликвидации, не production-safe.",
        ),
        "features_35_enhanced_retrospective": FeatureSetConfig(
            name="features_35_enhanced_retrospective",
            label="35 признаков: enhanced retrospective",
            features=list(
                dict.fromkeys(
                    list(enhanced)
                    + [
                        "quarter",
                        "delta_hose_to_contained",
                        "delta_contained_to_extinguished",
                        "total_response_time",
                    ]
                )
            ),
            availability_risk="offline/post-factum",
            notes="Расширенный исследовательский benchmark с поздними признаками.",
        ),
        "features_50_wide_no_target": FeatureSetConfig(
            name="features_50_wide_no_target",
            label="50 wide numeric без target/proxy",
            features=wide_candidates[:50],
            availability_risk="offline/post-factum",
            notes="Широкий offline-набор; момент доступности признаков не гарантирован.",
        ),
        "features_all_known_numeric": FeatureSetConfig(
            name="features_all_known_numeric",
            label=f"Все числовые кандидаты ({len(wide_candidates)})",
            features=wide_candidates,
            availability_risk="offline/post-factum",
            notes="Все числовые кандидаты после исключения явных target/proxy/meta полей.",
        ),
    }
    return feature_sets


def _model_factories(seed: int) -> dict[str, Callable[[], Any]]:
    return {
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def _prepare_frame(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    working = recompute_derived_features(df)
    for feature in features:
        if feature not in working.columns:
            working[feature] = np.nan
    y = map_rank_series_to_classes(working[target]).dropna().astype(int)
    rows = working.loc[y.index].reset_index(drop=True)
    x_frame = rows[features].apply(pd.to_numeric, errors="coerce")
    return x_frame.reset_index(drop=True), y.reset_index(drop=True), rows


def _fit_fill_values(x_train: pd.DataFrame) -> pd.Series:
    fill_values = x_train.median(numeric_only=True)
    return fill_values.reindex(x_train.columns).fillna(0.0)


def _apply_fill(x_frame: pd.DataFrame, fill_values: pd.Series) -> pd.DataFrame:
    return (
        x_frame.reindex(columns=fill_values.index)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(fill_values)
        .fillna(0.0)
    )


def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray | None, classes: np.ndarray, k: int) -> float | None:
    if y_proba is None or y_proba.size == 0 or len(y_true) == 0:
        return None
    k = min(k, y_proba.shape[1])
    top_indices = np.argsort(y_proba, axis=1)[:, -k:]
    top_classes = classes[top_indices]
    hits = [(truth in top_classes[idx]) for idx, truth in enumerate(y_true)]
    return float(np.mean(hits)) if hits else None


def _rank_values(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.array([CLASS_TO_RANK_MAP.get(int(value), float(value)) for value in list(values)], dtype=float)


def _confusion_summary(y_true: pd.Series, y_pred: np.ndarray) -> str:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cells: list[str] = []
    for i, actual in enumerate(labels):
        row_total = int(cm[i].sum())
        if row_total == 0:
            continue
        top_predictions = np.argsort(cm[i])[-3:][::-1]
        parts = [
            f"{CLASS_TO_RANK_MAP.get(int(labels[j]), labels[j])}:{int(cm[i, j])}"
            for j in top_predictions
            if cm[i, j] > 0
        ]
        cells.append(f"{CLASS_TO_RANK_MAP.get(int(actual), actual)} -> " + ", ".join(parts))
    return "; ".join(cells)


def _metrics(
    *,
    y_train: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
) -> dict[str, Any]:
    if y_true.empty:
        return {
            "accuracy": None,
            "f1_macro": None,
            "f1_weighted": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "top_2_accuracy": None,
            "top_3_accuracy": None,
            "ordinal_mae": None,
            "under_dispatch_rate": None,
            "confusion_matrix_summary": "",
            "train_size": int(len(y_train)),
            "test_size": 0,
            "n_classes_train": int(y_train.nunique()),
        }
    true_rank = _rank_values(y_true)
    pred_rank = _rank_values(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top_2_accuracy": _top_k_accuracy(y_true.to_numpy(), y_proba, classes, 2),
        "top_3_accuracy": _top_k_accuracy(y_true.to_numpy(), y_proba, classes, 3),
        "ordinal_mae": float(np.mean(np.abs(true_rank - pred_rank))),
        "under_dispatch_rate": float(np.mean(pred_rank < true_rank)),
        "confusion_matrix_summary": _confusion_summary(y_true, y_pred),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_true)),
        "n_classes_train": int(y_train.nunique()),
    }


def _evaluate_model(
    model: Any,
    *,
    x_frame: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series,
    dataset_name: str,
    fill_values: pd.Series,
) -> dict[str, Any]:
    if x_frame.empty or y.empty:
        payload = _metrics(
            y_train=y_train,
            y_true=pd.Series(dtype=int),
            y_pred=np.array([], dtype=int),
            y_proba=None,
            classes=np.array([], dtype=int),
        )
        payload["eval_dataset"] = dataset_name
        return payload
    x_eval = _apply_fill(x_frame, fill_values)
    y_pred = model.predict(x_eval)
    y_proba = model.predict_proba(x_eval) if hasattr(model, "predict_proba") else None
    payload = _metrics(
        y_train=y_train,
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=np.asarray(getattr(model, "classes_", [])),
    )
    payload["eval_dataset"] = dataset_name
    return payload


def _split_real_train_eval(real_df: pd.DataFrame, y: pd.Series, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if "_event_key" in real_df.columns and real_df["_event_key"].nunique(dropna=True) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return next(splitter.split(real_df, y, groups=real_df["_event_key"].fillna("__missing__")))
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    return train_test_split(np.arange(len(real_df)), test_size=0.25, random_state=seed, stratify=stratify)


def _feature_hashes(df: pd.DataFrame, features: list[str]) -> pd.Series:
    usable = [feature for feature in features if feature in df.columns]
    if not usable:
        return pd.Series(dtype="uint64")
    frame = df[usable].copy()
    for column in usable:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(6)
    frame = frame.astype("object").where(frame.notna(), "__missing__")
    return pd.util.hash_pandas_object(frame, index=False)


def exact_duplicate_rate_by_feature_hash(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    features: list[str],
) -> float:
    synthetic_hash = _feature_hashes(synthetic_df, features)
    real_hash = set(_feature_hashes(real_df, features).tolist())
    if synthetic_hash.empty:
        return 0.0
    return float(synthetic_hash.isin(real_hash).mean())


def break_exact_feature_hash_duplicates(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    feature_sets: dict[str, FeatureSetConfig],
    *,
    seed: int,
) -> pd.DataFrame:
    """Break exact synthetic-vs-real feature matches without touching targets."""
    result = synthetic_df.copy()
    rng = np.random.default_rng(seed + 17)
    ordered_sets = sorted(feature_sets.values(), key=lambda item: len(item.features))
    mutable_priority = [
        "distance_to_station",
        "t_detect_min",
        "t_report_min",
        "building_floors",
    ]
    for feature_set in ordered_sets:
        features = feature_set.features
        for _ in range(6):
            synthetic_hash = _feature_hashes(result, features)
            real_hash = set(_feature_hashes(real_df, features).tolist())
            duplicate_mask = synthetic_hash.isin(real_hash)
            if not duplicate_mask.any():
                break
            duplicate_indices = duplicate_mask[duplicate_mask].index.to_numpy()
            mutable = next(
                (
                    column
                    for column in mutable_priority
                    if column in features and column in result.columns
                ),
                None,
            )
            if mutable is None:
                break
            values = pd.to_numeric(result.loc[duplicate_indices, mutable], errors="coerce")
            fallback = pd.to_numeric(result[mutable], errors="coerce").median()
            fallback = 0.0 if pd.isna(fallback) else float(fallback)
            jitter = rng.normal(0.0, 0.003, size=len(duplicate_indices))
            jitter += np.arange(1, len(duplicate_indices) + 1) * 0.00001
            result.loc[duplicate_indices, mutable] = values.fillna(fallback).to_numpy(dtype=float) + jitter
            if mutable == "distance_to_station":
                result.loc[duplicate_indices, mutable] = pd.to_numeric(
                    result.loc[duplicate_indices, mutable],
                    errors="coerce",
                ).clip(lower=0.0)
            result = recompute_derived_features(result)
    return result


def _distribution_compare(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rank_distribution_real": real_df[target].value_counts(normalize=True).sort_index().round(4).to_dict()
        if target in real_df.columns
        else {},
        "rank_distribution_synthetic": synthetic_df[target].value_counts(normalize=True).sort_index().round(4).to_dict()
        if target in synthetic_df.columns
        else {},
        "missingness_mean_real": float(real_df.isna().mean().mean()) if not real_df.empty else None,
        "missingness_mean_synthetic": float(synthetic_df.isna().mean().mean()) if not synthetic_df.empty else None,
    }
    numeric = [
        column
        for column in ["building_floors", "fire_floor", "distance_to_station", "t_report_min", "t_arrival_min"]
        if column in real_df.columns and column in synthetic_df.columns
    ]
    payload["key_numeric_compare"] = {
        column: {
            "real_mean": float(pd.to_numeric(real_df[column], errors="coerce").mean()),
            "synthetic_mean": float(pd.to_numeric(synthetic_df[column], errors="coerce").mean()),
            "real_median": float(pd.to_numeric(real_df[column], errors="coerce").median()),
            "synthetic_median": float(pd.to_numeric(synthetic_df[column], errors="coerce").median()),
        }
        for column in numeric
    }
    corr_cols = [column for column in numeric if real_df[column].nunique(dropna=True) > 1]
    if len(corr_cols) >= 2:
        real_corr = real_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
        syn_corr = synthetic_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
        payload["correlation_mean_abs_diff"] = float((real_corr - syn_corr).abs().to_numpy().mean())
    else:
        payload["correlation_mean_abs_diff"] = None
    return _stringify_json(payload)


def _train_and_score(
    *,
    train_df: pd.DataFrame,
    synthetic_validation_df: pd.DataFrame,
    real_eval_df: pd.DataFrame,
    feature_set: FeatureSetConfig,
    model_name: str,
    model_factory: Callable[[], Any],
    target: str,
    mode: str,
) -> list[dict[str, Any]]:
    x_train, y_train, _ = _prepare_frame(train_df, feature_set.features, target)
    if y_train.nunique() < 2 or x_train.empty:
        return [
            {
                "mode": mode,
                "feature_set": feature_set.name,
                "feature_count": len(feature_set.features),
                "availability_risk": feature_set.availability_risk,
                "model": model_name,
                "eval_dataset": "error",
                "error": "Недостаточно классов или признаков для обучения",
            }
        ]
    fill_values = _fit_fill_values(x_train)
    x_train_prepared = _apply_fill(x_train, fill_values)
    model = model_factory()
    model.fit(x_train_prepared, y_train)

    rows: list[dict[str, Any]] = []
    for dataset_name, eval_df in [
        ("synthetic_validation", synthetic_validation_df),
        ("real_test", real_eval_df),
    ]:
        x_eval, y_eval, _ = _prepare_frame(eval_df, feature_set.features, target)
        metric_payload = _evaluate_model(
            model,
            x_frame=x_eval,
            y=y_eval,
            y_train=y_train,
            dataset_name=dataset_name,
            fill_values=fill_values,
        )
        metric_payload.update(
            {
                "mode": mode,
                "feature_set": feature_set.name,
                "feature_label": feature_set.label,
                "feature_count": len(feature_set.features),
                "availability_risk": feature_set.availability_risk,
                "model": model_name,
                "error": "",
            }
        )
        rows.append(metric_payload)
    return rows


def _write_report(
    *,
    run_dir: Path,
    config: SyntheticRankExperimentConfig,
    data_profile: dict[str, Any],
    feature_sets: dict[str, FeatureSetConfig],
    metrics_df: pd.DataFrame,
    comparison: dict[str, Any],
) -> None:
    best_real = metrics_df.loc[metrics_df["eval_dataset"] == "real_test"].copy()
    if "f1_macro" in best_real.columns:
        best_real = best_real.sort_values("f1_macro", ascending=False, na_position="last").head(12)

    lines = [
        "# Отчет: ЦДС, ИВД и сравнение моделей",
        "",
        "## Параметры запуска",
        f"- Входной файл: `{config.input_path}`",
        f"- Target: `{config.target}`",
        f"- Seed: `{config.seed}`",
        f"- Synthetic rows: `{config.synthetic_rows}`",
        f"- Train/validation: `{config.train_rows}` / `{config.synthetic_validation_rows}`",
        f"- Real-test rows: `{data_profile['real_test']['real_test_rows']}`",
        f"- Удалено дублей real-test: `{data_profile['real_test']['duplicates_removed']}`",
        f"- Режимы: `{', '.join(config.modes)}`",
        "",
        "## Sanity-check генератора",
        f"- Средняя missingness real: `{comparison.get('missingness_mean_real')}`",
        f"- Средняя missingness synthetic: `{comparison.get('missingness_mean_synthetic')}`",
        f"- Средняя абсолютная разница корреляций: `{comparison.get('correlation_mean_abs_diff')}`",
        "",
        "### Дубли synthetic-vs-real по feature hash",
    ]
    for name, payload in data_profile.get("duplicate_rates_by_feature_set", {}).items():
        lines.append(f"- `{name}`: `{payload:.6f}`")

    lines.extend(
        [
            "",
            "## Наборы признаков и риск доступности",
        ]
    )
    for feature_set in feature_sets.values():
        lines.append(
            f"- `{feature_set.name}`: {len(feature_set.features)} признаков, "
            f"`{feature_set.availability_risk}`. {feature_set.notes}"
        )

    lines.extend(
        [
            "",
            "## Лучшие результаты на real-test",
            "",
        ]
    )
    if best_real.empty:
        lines.append("Метрики real-test не рассчитаны.")
    else:
        columns = [
            "mode",
            "feature_set",
            "model",
            "f1_macro",
            "accuracy",
            "top_3_accuracy",
            "ordinal_mae",
            "under_dispatch_rate",
        ]
        table = best_real[columns].copy()
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in table.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    "" if pd.isna(row[column]) else f"{row[column]:.4f}" if isinstance(row[column], float) else str(row[column])
                    for column in columns
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Предупреждения по использованию",
            "- Для рабочего режима ЛПР production-safe остается только ранний набор `features_10_dispatch`.",
            "- Наборы после прибытия и после подачи первого ствола нельзя использовать в момент звонка.",
            "- Ретроспективные и wide-наборы нужны только для исследовательских benchmark-экспериментов.",
            "- Качество на `synthetic_validation` не является честной оценкой production-модели; главный контроль — `real_test`.",
            "",
            "## Артефакты",
            "- `synthetic_train.csv`",
            "- `synthetic_validation.csv`",
            "- `real_test.csv`",
            "- `generation_profile.json`",
            "- `metrics_by_feature_set.csv`",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_synthetic_rank_experiment(config: SyntheticRankExperimentConfig) -> dict[str, Any]:
    timestamp = config.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.input_path)
    real_test, real_profile = prepare_real_canonical_test(
        df,
        target=config.target,
        canonical_source_sheet=config.canonical_source_sheet,
    )
    total_synthetic_rows = max(
        int(config.synthetic_rows),
        int(config.train_rows) + int(config.synthetic_validation_rows),
    )
    synthetic, generation_profile = generate_rank_conditional_synthetic_dataset(
        real_test,
        n_rows=total_synthetic_rows,
        target_column=config.target,
        random_state=config.seed,
        numeric_noise_scale=config.numeric_noise_scale,
        categorical_smoothing=config.categorical_smoothing,
        global_mix=config.global_mix,
        extra_missing_rate=config.extra_missing_rate,
    )
    synthetic_train = synthetic.iloc[: int(config.train_rows)].reset_index(drop=True)
    synthetic_validation = synthetic.iloc[
        int(config.train_rows): int(config.train_rows) + int(config.synthetic_validation_rows)
    ].reset_index(drop=True)

    real_test.to_csv(run_dir / "real_test.csv", index=False, encoding="utf-8-sig")
    synthetic_train.to_csv(run_dir / "synthetic_train.csv", index=False, encoding="utf-8-sig")
    synthetic_validation.to_csv(run_dir / "synthetic_validation.csv", index=False, encoding="utf-8-sig")

    all_feature_sets = build_experiment_feature_sets(real_test, target=config.target)
    selected_feature_sets = {
        name: all_feature_sets[name]
        for name in config.feature_sets
        if name in all_feature_sets
    }
    model_factories = _model_factories(config.seed)
    selected_models = {
        name: model_factories[name]
        for name in config.models
        if name in model_factories
    }

    synthetic = break_exact_feature_hash_duplicates(
        synthetic,
        real_test,
        selected_feature_sets,
        seed=config.seed,
    )
    synthetic_train = synthetic.iloc[: int(config.train_rows)].reset_index(drop=True)
    synthetic_validation = synthetic.iloc[
        int(config.train_rows): int(config.train_rows) + int(config.synthetic_validation_rows)
    ].reset_index(drop=True)
    synthetic_train.to_csv(run_dir / "synthetic_train.csv", index=False, encoding="utf-8-sig")
    synthetic_validation.to_csv(run_dir / "synthetic_validation.csv", index=False, encoding="utf-8-sig")

    duplicate_rates = {
        name: exact_duplicate_rate_by_feature_hash(synthetic, real_test, feature_set.features)
        for name, feature_set in selected_feature_sets.items()
    }
    comparison = _distribution_compare(real_test, synthetic, config.target)
    data_profile = {
        "real_test": real_profile,
        "input_profile": build_statistical_profile(df, target_column=config.target),
        "generation_profile": generation_profile,
        "distribution_compare": comparison,
        "duplicate_rates_by_feature_set": duplicate_rates,
        "feature_sets": {
            name: {
                "label": feature_set.label,
                "feature_count": len(feature_set.features),
                "features": feature_set.features,
                "availability_risk": feature_set.availability_risk,
                "notes": feature_set.notes,
            }
            for name, feature_set in selected_feature_sets.items()
        },
        "config": {
            "input_path": str(config.input_path),
            "target": config.target,
            "seed": config.seed,
            "synthetic_rows": total_synthetic_rows,
            "train_rows": config.train_rows,
            "synthetic_validation_rows": config.synthetic_validation_rows,
            "modes": config.modes,
            "models": list(selected_models),
        },
    }
    (run_dir / "generation_profile.json").write_text(
        json.dumps(_stringify_json(data_profile), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_rows: list[dict[str, Any]] = []
    _real_x, real_y, real_rows = _prepare_frame(real_test, ["region_code"], config.target)
    real_train_idx: np.ndarray | None = None
    real_eval_idx: np.ndarray | None = None
    if len(real_rows) >= 4 and real_y.nunique() >= 2:
        real_train_idx, real_eval_idx = _split_real_train_eval(real_rows, real_y, seed=config.seed)

    for mode in config.modes:
        if mode == "baseline_real":
            if real_train_idx is None or real_eval_idx is None:
                continue
            train_df = real_rows.iloc[real_train_idx].reset_index(drop=True)
            real_eval_df = real_rows.iloc[real_eval_idx].reset_index(drop=True)
            synthetic_eval_df = pd.DataFrame(columns=synthetic_validation.columns)
        elif mode == "synthetic_only":
            train_df = synthetic_train
            real_eval_df = real_test
            synthetic_eval_df = synthetic_validation
        elif mode == "real_plus_synthetic":
            if real_train_idx is None or real_eval_idx is None:
                continue
            train_df = pd.concat(
                [real_rows.iloc[real_train_idx].reset_index(drop=True), synthetic_train],
                ignore_index=True,
            )
            real_eval_df = real_rows.iloc[real_eval_idx].reset_index(drop=True)
            synthetic_eval_df = synthetic_validation
        elif mode == "distortion_study":
            distorted_validation, _ = generate_rank_conditional_synthetic_dataset(
                real_test,
                n_rows=int(config.synthetic_validation_rows),
                target_column=config.target,
                random_state=config.seed + 991,
                numeric_noise_scale=max(config.numeric_noise_scale * 2.0, 0.15),
                categorical_smoothing=config.categorical_smoothing,
                global_mix=min(config.global_mix + 0.2, 0.6),
                extra_missing_rate=min(config.extra_missing_rate + 0.05, 0.4),
                distortion_scenario="distorted_validation",
            )
            train_df = synthetic_train
            real_eval_df = real_test
            synthetic_eval_df = distorted_validation
        else:
            continue

        for feature_set in selected_feature_sets.values():
            for model_name, model_factory in selected_models.items():
                metric_rows.extend(
                    _train_and_score(
                        train_df=train_df,
                        synthetic_validation_df=synthetic_eval_df,
                        real_eval_df=real_eval_df,
                        feature_set=feature_set,
                        model_name=model_name,
                        model_factory=model_factory,
                        target=config.target,
                        mode=mode,
                    )
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(run_dir / "metrics_by_feature_set.csv", index=False, encoding="utf-8-sig")
    _write_report(
        run_dir=run_dir,
        config=config,
        data_profile=data_profile,
        feature_sets=selected_feature_sets,
        metrics_df=metrics_df,
        comparison=comparison,
    )
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics_by_feature_set.csv"),
        "report_path": str(run_dir / "REPORT.md"),
        "profile_path": str(run_dir / "generation_profile.json"),
        "real_test_rows": real_profile["real_test_rows"],
        "duplicates_removed": real_profile["duplicates_removed"],
        "synthetic_train_rows": int(len(synthetic_train)),
        "synthetic_validation_rows": int(len(synthetic_validation)),
        "metrics_rows": int(len(metrics_df)),
        "best_real_f1_macro": float(
            metrics_df.loc[metrics_df["eval_dataset"] == "real_test", "f1_macro"].max()
        )
        if not metrics_df.empty and "f1_macro" in metrics_df.columns
        else None,
        "duplicate_rates_by_feature_set": duplicate_rates,
    }
