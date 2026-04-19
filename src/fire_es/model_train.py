"""
Обучение деревьев решений для прогнозирования ранга и ресурсов.

Модуль реализует:
- Обучение DecisionTreeClassifier для rank_tz
- Обучение DecisionTreeRegressor для ресурсов
- Визуализация дерева (PNG)
- Сохранение моделей (joblib)
- Метрики: accuracy, F1, confusion matrix
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Тип для моделей дерева
TreeModel = Union[DecisionTreeClassifier, DecisionTreeRegressor]


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Признаки из разделов 1 и 2 (по ТЗ)
FEATURE_SETS = {
    "minimal": [
        "region_code",
        "building_floors",
        "fire_floor",
    ],
    "managed": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "risk_category_code",
        "enterprise_type_code",
        "fpo_class_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
    ],
    "observed": [
        "region_code",
        "building_floors",
        "fire_floor",
        "fatalities",
        "injuries",
        "direct_damage",
        "people_saved",
        "people_evacuated",
    ],
    "time": [
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
        "t_first_hose_min",
        "t_contained_min",
        "t_extinguished_min",
    ],
    "full": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "risk_category_code",
        "enterprise_type_code",
        "fpo_class_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        "fatalities",
        "injuries",
        "direct_damage",
        "people_saved",
        "people_evacuated",
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
        "t_first_hose_min",
        "t_contained_min",
        "t_extinguished_min",
    ],
    # Online режимы (без утечки будущего)
    # Важно: risk_category_code/fpo_class_code исключены из value-фичей из-за 91%+ пропусков
    "online_dispatch": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
    ],
    "online_early": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
    ],
    "online_tactical": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
        "t_first_hose_min",
    ],
    # Enhanced наборы с feature engineering
    "enhanced_dispatch": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        # Temporal
        "day_of_week",
        "month",
        "season",
        "is_weekend",
        # Building risk
        "is_high_rise",
        "is_super_high_rise",
        "basement_fire",
        "top_floor_fire",
        "fire_floor_ratio",
        # Distance
        "is_far",
        "is_very_far",
        "distance_category",
        # Binned
        "building_floors_bin",
        # Missing indicators
        "risk_category_code_missing",
        "fpo_class_code_missing",
    ],
    "enhanced_early": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        # Temporal
        "day_of_week",
        "month",
        "season",
        "is_weekend",
        # Building risk
        "is_high_rise",
        "is_super_high_rise",
        "basement_fire",
        "top_floor_fire",
        "fire_floor_ratio",
        # Distance
        "is_far",
        "is_very_far",
        "distance_category",
        # Binned
        "building_floors_bin",
        # Missing indicators
        "risk_category_code_missing",
        "fpo_class_code_missing",
        # Response time
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
        "delta_detect_to_report",
        "delta_report_to_arrival",
    ],
    "enhanced_tactical": [
        "region_code",
        "settlement_type_code",
        "fire_protection_code",
        "enterprise_type_code",
        "building_floors",
        "fire_floor",
        "fire_resistance_code",
        "source_item_code",
        "distance_to_station",
        # Temporal
        "day_of_week",
        "month",
        "season",
        "is_weekend",
        # Building risk
        "is_high_rise",
        "is_super_high_rise",
        "basement_fire",
        "top_floor_fire",
        "fire_floor_ratio",
        # Distance
        "is_far",
        "is_very_far",
        "distance_category",
        # Binned
        "building_floors_bin",
        # Missing indicators
        "risk_category_code_missing",
        "fpo_class_code_missing",
        # Response time
        "t_detect_min",
        "t_report_min",
        "t_arrival_min",
        "t_first_hose_min",
        "delta_detect_to_report",
        "delta_report_to_arrival",
        "delta_arrival_to_hose",
    ],
}

# Целевые переменные
TARGET_RANK = "rank_tz"
TARGET_RESOURCES = "equipment_count"

# Политика по пропускам и источнику
ONLINE_FEATURE_SETS = {"online_dispatch", "online_early", "online_tactical"}
DEFAULT_INDICATOR_COLUMNS = ["risk_category_code", "fpo_class_code"]
CANONICAL_SOURCE_SHEET = "БД-1...2000--2020 (1+2)"


def _indicator_name(col: str) -> str:
    """Имя индикатора пропуска для колонки."""
    if col.endswith("_code"):
        return f"{col[:-5]}_missing"
    return f"{col}_missing"


# ============================================================================
# ФУНКЦИИ
# ============================================================================


def prepare_data(
    df: pd.DataFrame,
    feature_set: str = "managed",
    target: str = TARGET_RANK,
    fill_na: bool = True,
    fill_value: float = 0,
    fill_strategy: str = "constant",
    add_missing_indicators: bool = False,
    indicator_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Подготовка данных для обучения.
    
    Args:
        df: DataFrame с данными
        feature_set: набор признаков (minimal, managed, observed, time, full, online_*)
        target: целевая переменная
        fill_na: заполнять ли пропуски
        fill_value: значение для заполнения пропусков
        fill_strategy: стратегия заполнения ('constant', 'median', 'mode', 'drop')
        add_missing_indicators: добавить индикаторы пропусков для indicator_columns
        indicator_columns: колонки, для которых добавляются индикаторы
        
    Returns:
        (X, y, feature_names)
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Неизвестный набор признаков: {feature_set}")
    
    feature_names = FEATURE_SETS[feature_set]
    
    # Проверка наличия колонок
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки: {missing_cols}")
    
    if target not in df.columns:
        raise ValueError(f"Отсутствует целевая переменная: {target}")
    
    X = df[feature_names].copy()
    y = df[target].copy()
    
    # Удаление строк с пропусками в target
    mask = y.notna()
    X = X[mask].copy()
    y = y[mask].copy()
    
    # Замена бесконечных значений
    X = X.replace([np.inf, -np.inf], np.nan)

    # Для online-наборов дефолт: специальный маркер пропуска -1
    if feature_set in ONLINE_FEATURE_SETS and fill_strategy == "constant" and fill_value == 0:
        fill_value = -1

    # Добавление индикаторов пропусков
    if add_missing_indicators:
        cols_for_indicators = indicator_columns or DEFAULT_INDICATOR_COLUMNS
        for col in cols_for_indicators:
            if col in X.columns:
                X[_indicator_name(col)] = X[col].isna().astype(int)

    # Старый флаг совместимости
    if not fill_na:
        fill_strategy = "drop"

    if fill_strategy == "constant":
        X = X.fillna(fill_value)
    elif fill_strategy == "median":
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
    elif fill_strategy == "mode":
        for col in X.columns:
            if X[col].isna().any():
                mode_values = X[col].mode(dropna=True)
                mode_value = mode_values.iloc[0] if not mode_values.empty else fill_value
                X[col] = X[col].fillna(mode_value)
    elif fill_strategy == "drop":
        X = X.dropna()
        y = y.loc[X.index]
    else:
        raise ValueError(f"Неизвестная стратегия заполнения: {fill_strategy}")

    feature_names = X.columns.tolist()
    return X, y, feature_names


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """
    Обучение DecisionTreeClassifier.
    """
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Обучение модели
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    
    # Прогноз
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_classes": len(model.classes_),
        "classes": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train_random_forest_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    class_weight: str = "balanced",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Обучение RandomForestClassifier.
    
    Args:
        X: признаки
        y: целевая переменная
        n_estimators: количество деревьев
        max_depth: максимальная глубина деревьев
        min_samples_split: минимальное количество образцов для разделения
        min_samples_leaf: минимальное количество образцов в листе
        random_state: seed для воспроизводимости
        test_size: доля тестовой выборки
        class_weight: балансировка классов
        n_jobs: количество потоков
    
    Returns:
        dict с моделью, метриками и прогнозами
    """
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    # Прогноз
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_classes": len(model.classes_),
        "classes": model.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def train_random_forest_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Обучение RandomForestRegressor.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Обучение модели
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)

    # Прогноз
    y_pred = model.predict(X_test)

    # Метрики
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_type": "RandomForestRegressor",
        "n_estimators": n_estimators,
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """
    Обучение DecisionTreeRegressor.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Обучение модели
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    
    # Прогноз
    y_pred = model.predict(X_test)
    
    # Метрики
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    
    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "X_test": X_test,
        "y_test": y_test,
    }


def evaluate_feature_set_cv(
    df: pd.DataFrame,
    feature_set: str,
    target: str = TARGET_RANK,
    fill_strategy: str = "constant",
    fill_value: float = 0,
    add_missing_indicators: bool = False,
    indicator_columns: Optional[list[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Кросс-валидация для сравнения feature sets.
    """
    X, y, feature_names = prepare_data(
        df=df,
        feature_set=feature_set,
        target=target,
        fill_strategy=fill_strategy,
        fill_value=fill_value,
        add_missing_indicators=add_missing_indicators,
        indicator_columns=indicator_columns,
    )

    if target == TARGET_RANK:
        y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
        y_cv = y.map(y_classes).fillna(1).astype(int)
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
        }
    else:
        y_cv = y
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
        )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }

    scores = cross_validate(model, X, y_cv, cv=cv, scoring=scoring)
    metrics: dict[str, float] = {}
    for k, v in scores.items():
        if not k.startswith("test_"):
            continue
        metric_name = k.replace("test_", "")
        val = np.mean(v)
        if metric_name in {"mae", "rmse"}:
            val = -val
        metrics[f"{metric_name}_mean"] = float(val)
        metrics[f"{metric_name}_std"] = float(np.std(v))

    return {
        "feature_set": feature_set,
        "target": target,
        "n_samples": int(len(X)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "metrics": metrics,
    }


def visualize_tree(
    model: TreeModel,
    feature_names: list[str],
    class_names: Optional[list[str]] = None,
    output_path: str = "tree.png",
    max_depth: int = 3,
    figsize: tuple[int, int] = (20, 10),
) -> str:
    """
    Визуализация дерева решений.
    
    Args:
        model: обученная модель
        feature_names: имена признаков
        class_names: имена классов (для классификации)
        output_path: путь для сохранения
        max_depth: максимальная глубина для визуализации
        figsize: размер фигуры
        
    Returns:
        путь к сохранённому файлу
    """
    import matplotlib

    # Для фоновых потоков/CI используем неблокирующий backend.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    
    plt.title("Дерево решений", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path


def save_model(
    model: TreeModel,
    metrics: dict,
    feature_set: str,
    model_type: str,
    output_dir: str = "reports/models",
) -> dict[str, Any]:
    """
    Сохранение модели и метаданных.
    
    Args:
        model: обученная модель
        metrics: метрики
        feature_set: набор признаков
        model_type: тип модели ('rank' или 'resource')
        output_dir: директория для сохранения
        
    Returns:
        информация о сохранённой модели
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохранение модели
    model_filename = f"model_{model_type}_{feature_set}_{timestamp}.joblib"
    model_path = output_path / model_filename
    joblib.dump(model, model_path)
    
    # Сохранение метаданных
    metadata = {
        "model_type": model_type,
        "feature_set": feature_set,
        "feature_names": FEATURE_SETS.get(feature_set, []),
        "metrics": metrics,
        "timestamp": timestamp,
        "model_path": str(model_path),
    }
    
    metadata_filename = f"model_{model_type}_{feature_set}_{timestamp}.json"
    metadata_path = output_path / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def load_model(model_path: str) -> TreeModel:
    """Загрузка модели из файла."""
    return joblib.load(model_path)


def predict_with_proba(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Прогноз с вероятностями (top-K).
    
    Args:
        model: обученная модель
        X: признаки для прогноза
        top_k: количество лучших вариантов
        
    Returns:
        DataFrame с прогнозами и вероятностями
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    classes = model.classes_
    
    # Top-K вероятности
    top_k_indices = np.argsort(y_proba, axis=1)[:, -top_k:][:, ::-1]
    
    results = []
    for i in range(len(X)):
        result = {
            "predicted_rank": y_pred[i],
        }
        for j in range(top_k):
            class_idx = top_k_indices[i, j]
            result[f"top{j+1}_rank"] = classes[class_idx]
            result[f"top{j+1}_prob"] = y_proba[i, class_idx]
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================================
# CLI
# ============================================================================


def train_rank_model(
    data_path: str = "clean_df_with_rank.csv",
    feature_set: str = "online_tactical",
    output_dir: str = "reports/models",
    source_sheet: str = CANONICAL_SOURCE_SHEET,
    fill_strategy: str = "constant",
    fill_value: Optional[float] = None,
    add_missing_indicators: bool = True,
    indicator_columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Обучение модели для прогнозирования ранга.
    """
    # Загрузка данных
    df = pd.read_csv(data_path)

    # Канонический источник фактов (если есть колонка source_sheet)
    if source_sheet and "source_sheet" in df.columns:
        before = len(df)
        df = df[df["source_sheet"] == source_sheet].copy()
        print(f"📂 Фильтрация по source_sheet='{source_sheet}': {before} -> {len(df)}")
        if df.empty:
            raise ValueError(
                f"После фильтрации source_sheet='{source_sheet}' данных не осталось."
            )

    if fill_value is None:
        fill_value = -1 if feature_set in ONLINE_FEATURE_SETS else 0
    
    # Подготовка данных
    X, y, feature_names = prepare_data(
        df,
        feature_set=feature_set,
        target=TARGET_RANK,
        fill_strategy=fill_strategy,
        fill_value=fill_value,
        add_missing_indicators=add_missing_indicators,
        indicator_columns=indicator_columns,
    )
    
    # Преобразование rank_tz в целочисленные классы
    # 1.0->1, 1.5->2, 2.0->3, 3.0->4, 4.0->5, 5.0->6
    y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
    y_int = y.map(y_classes).fillna(1).astype(int)
    
    print(f"📊 Данные для обучения:")
    print(f"  Признаков: {len(feature_names)}")
    print(f"  Образцов: {len(X)}")
    print(f"  Классов: {y_int.nunique()}")
    print(f"  Распределение: {y_int.value_counts().sort_index().to_dict()}")
    
    # Обучение
    print(f"\n🔧 Обучение модели...")
    result = train_classifier(X, y_int, max_depth=10, min_samples_split=20)
    
    # Восстановление имён классов для отчёта
    class_names = {v: k for k, v in y_classes.items()}
    result["metrics"]["classes"] = [class_names.get(c, c) for c in result["metrics"]["classes"]]
    
    print(f"\n📈 Метрики:")
    for key, value in result["metrics"].items():
        if key not in ["confusion_matrix", "classes"]:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Сохранение
    print(f"\n💾 Сохранение модели...")
    saved = save_model(
        result["model"],
        result["metrics"],
        feature_set=feature_set,
        model_type="rank",
        output_dir=output_dir,
    )
    
    # Визуализация
    print(f"\n🎨 Визуализация дерева...")
    class_names_display = [f"Ранг {c}" for c in result["metrics"]["classes"]]
    viz_path = visualize_tree(
        result["model"],
        feature_names,
        class_names=class_names_display,
        output_path=Path(output_dir) / f"tree_rank_{feature_set}.png",
        max_depth=3,
    )
    print(f"  Сохранено: {viz_path}")
    
    return saved


def train_resource_model(
    data_path: str = "clean_df_with_rank.csv",
    feature_set: str = "online_tactical",
    output_dir: str = "reports/models",
    source_sheet: str = CANONICAL_SOURCE_SHEET,
    fill_strategy: str = "constant",
    fill_value: Optional[float] = None,
    add_missing_indicators: bool = True,
    indicator_columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Обучение модели для прогнозирования ресурсов.
    
    Args:
        data_path: путь к данным
        feature_set: набор признаков
        output_dir: директория для сохранения
        
    Returns:
        информация о сохранённой модели
    """
    # Загрузка данных
    df = pd.read_csv(data_path)

    # Канонический источник фактов (если есть колонка source_sheet)
    if source_sheet and "source_sheet" in df.columns:
        before = len(df)
        df = df[df["source_sheet"] == source_sheet].copy()
        print(f"📂 Фильтрация по source_sheet='{source_sheet}': {before} -> {len(df)}")
        if df.empty:
            raise ValueError(
                f"После фильтрации source_sheet='{source_sheet}' данных не осталось."
            )

    if fill_value is None:
        fill_value = -1 if feature_set in ONLINE_FEATURE_SETS else 0
    
    # Подготовка данных
    X, y, feature_names = prepare_data(
        df,
        feature_set=feature_set,
        target=TARGET_RESOURCES,
        fill_strategy=fill_strategy,
        fill_value=fill_value,
        add_missing_indicators=add_missing_indicators,
        indicator_columns=indicator_columns,
    )
    
    print(f"📊 Данные для обучения:")
    print(f"  Признаков: {len(feature_names)}")
    print(f"  Образцов: {len(X)}")
    
    # Обучение
    print(f"\n🔧 Обучение модели...")
    result = train_regressor(X, y, max_depth=10, min_samples_split=20)
    
    print(f"\n📈 Метрики:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Сохранение
    print(f"\n💾 Сохранение модели...")
    saved = save_model(
        result["model"],
        result["metrics"],
        feature_set=feature_set,
        model_type="resource",
        output_dir=output_dir,
    )
    
    # Визуализация
    print(f"\n🎨 Визуализация дерева...")
    viz_path = visualize_tree(
        result["model"],
        feature_names,
        output_path=Path(output_dir) / f"tree_resource_{feature_set}.png",
        max_depth=3,
    )
    print(f"  Сохранено: {viz_path}")
    
    return saved


if __name__ == "__main__":
    print("=" * 60)
    print("Обучение модели для прогнозирования ранга")
    print("=" * 60)
    train_rank_model()
    
    print()
    print("=" * 60)
    print("Обучение модели для прогнозирования ресурсов")
    print("=" * 60)
    train_resource_model()
