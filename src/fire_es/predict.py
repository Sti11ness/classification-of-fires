"""
Прогнозирование с доверительностью (Эпик 5).

Модуль реализует:
- Прогноз ранга с top-K вероятностями
- Бутстрап-ансамбль для доверительных интервалов
- Прогноз ресурсов с квантилями P20/P50/P80
- Связь с нормативным рангом
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fire_es.model_train import FEATURE_SETS, prepare_data


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Маппинг классов rank_tz
RANK_CLASSES = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}
RANK_NAMES = {
    1.0: "1",
    1.5: "1-бис",
    2.0: "2",
    3.0: "3",
    4.0: "4",
    5.0: "5",
}


# ============================================================================
# ФУНКЦИИ
# ============================================================================


def load_model(model_path: str) -> Union[DecisionTreeClassifier, RandomForestClassifier]:
    """
    Загрузка модели из файла.
    
    Args:
        model_path: путь к файлу модели (.joblib)
    
    Returns:
        загруженная модель
    """
    return joblib.load(model_path)


def predict_rank_topk(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    X: pd.DataFrame,
    top_k: int = 3,
    class_mapping: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Прогноз ранга с top-K вероятностями.
    
    Args:
        model: обученная модель классификации
        X: признаки для прогноза
        top_k: количество лучших вариантов
        class_mapping: маппинг индексов классов в значения рангов
    
    Returns:
        DataFrame с прогнозами и вероятностями
    """
    if class_mapping is None:
        class_mapping = RANK_CLASSES
    
    # Прогноз
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    classes = model.classes_
    
    # Top-K вероятности
    top_k_indices = np.argsort(y_proba, axis=1)[:, -top_k:][:, ::-1]
    
    results = []
    for i in range(len(X)):
        pred_label = classes[np.argmax(y_proba[i])]
        result = {
            "predicted_rank": class_mapping.get(pred_label, pred_label),
            "predicted_rank_idx": pred_label,
        }

        for j in range(top_k):
            class_idx = int(top_k_indices[i, j])
            class_label = classes[class_idx]
            rank_value = class_mapping.get(class_label, class_label)
            result[f"top{j+1}_rank_idx"] = class_label
            result[f"top{j+1}_rank"] = rank_value
            result[f"top{j+1}_rank_name"] = RANK_NAMES.get(rank_value, str(rank_value))
            result[f"top{j+1}_prob"] = float(y_proba[i, top_k_indices[i, j]])
        
        results.append(result)
    
    return pd.DataFrame(results)


def bootstrap_predict_rank(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    X: pd.DataFrame,
    n_bootstrap: int = 30,
    sample_ratio: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Бутстрап-ансамбль для прогноза ранга с доверительными интервалами.
    
    Args:
        model: обученная модель
        X: признаки для прогноза
        n_bootstrap: количество бутстрап-выборок
        sample_ratio: доля образцов в каждой бутстрап-выборке
        random_state: seed для воспроизводимости
    
    Returns:
        DataFrame с прогнозами и доверительными интервалами
    """
    np.random.seed(random_state)

    # Базовые вероятности всегда доступны
    base_probs = model.predict_proba(X)
    classes = model.classes_

    # Если есть ансамбль деревьев, считаем распределение по деревьям
    prob_samples = None
    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
        estimators = list(model.estimators_)
        if n_bootstrap > 0 and len(estimators) > n_bootstrap:
            sampled_indices = np.random.choice(
                len(estimators),
                size=n_bootstrap,
                replace=False,
            )
            estimators = [estimators[i] for i in sampled_indices]

        probs_list = []
        for est in estimators:
            if hasattr(est, "predict_proba"):
                probs_list.append(est.predict_proba(X))

        if probs_list:
            # shape: (n_estimators, n_samples, n_classes)
            prob_samples = np.stack(probs_list, axis=0)

    if prob_samples is None:
        # Fallback: без оценки неопределенности
        mean_probs = base_probs
        std_probs = np.zeros_like(base_probs)
        ci_lower = base_probs
        ci_upper = base_probs
    else:
        mean_probs = np.mean(prob_samples, axis=0)
        std_probs = np.std(prob_samples, axis=0)
        ci_lower = np.percentile(prob_samples, 5, axis=0)
        ci_upper = np.percentile(prob_samples, 95, axis=0)

    final_results = []
    for i in range(len(X)):
        best_idx = int(np.argmax(mean_probs[i]))
        pred_label = classes[best_idx]

        result = {
            "predicted_rank_idx": pred_label,
            "predicted_rank": RANK_CLASSES.get(pred_label, pred_label),
            "mean_prob_class": float(np.max(mean_probs[i])),
            "std_prob_class": float(np.max(std_probs[i])),
            "ci_lower_prob": float(np.max(ci_lower[i])),
            "ci_upper_prob": float(np.max(ci_upper[i])),
        }
        final_results.append(result)

    return pd.DataFrame(final_results)


def predict_resources_quantiles(
    model: Union[DecisionTreeRegressor, RandomForestRegressor],
    X: pd.DataFrame,
    quantiles: list[float] = None,
) -> pd.DataFrame:
    """
    Прогноз ресурсов с квантилями (P20/P50/P80).
    
    Для Random Forest используем предсказания отдельных деревьев
    для оценки квантилей.
    
    Args:
        model: обученная модель регрессии
        X: признаки для прогноза
        quantiles: список квантилей
    
    Returns:
        DataFrame с прогнозами и квантилями
    """
    if quantiles is None:
        quantiles = [0.2, 0.5, 0.8]
    
    # Базовый прогноз
    y_pred = model.predict(X)
    
    # Для Random Forest можно использовать предсказания отдельных деревьев
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # Предсказания каждого дерева
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Квантили по деревьям
        results = {
            "predicted_mean": y_pred,
            "predicted_std": np.std(tree_predictions, axis=0),
        }
        
        for q in quantiles:
            q_name = f"resource_p{int(q*100)}"
            results[q_name] = np.percentile(tree_predictions, q*100, axis=0)
        
        # Мин/макс по деревьям
        results["resource_min"] = np.min(tree_predictions, axis=0)
        results["resource_max"] = np.max(tree_predictions, axis=0)
        
    else:
        # Для одиночного дерева - только прогноз
        # Используем эвриристическую оценку неопределённости
        results = {
            "predicted_mean": y_pred,
            "predicted_std": np.zeros_like(y_pred),  # Неопределённость неизвестна
            "resource_min": y_pred,
            "resource_max": y_pred,
        }
        
        # Добавляем квантили как копию прогноза (без оценки неопределённости)
        for q in quantiles:
            q_name = f"resource_p{int(q*100)}"
            results[q_name] = y_pred.copy()
    
    return pd.DataFrame(results)


def rank_from_resources(
    resources: Union[int, float, pd.Series],
    normative_table: Optional[pd.DataFrame] = None,
) -> Union[float, pd.Series]:
    """
    Определение ранга по количеству ресурсов (упрощённая методика).
    
    По ТЗ 2.5.1: ранг определяется по минимальному евклидову расстоянию
    до нормативного вектора. Здесь используется упрощённая версия
    по количеству техники.
    
    Args:
        resources: количество техники (int, float или Series)
        normative_table: нормативная таблица (если None, используется стандартная)
    
    Returns:
        ранг (float или Series)
    """
    # Упрощённая нормативная таблица (количество техники по рангам)
    # На основе данных из БД МЧС
    normative = {
        1.0: 2,    # Ранг 1: 2 единицы техники
        1.5: 2,    # Ранг 1-бис: 2 единицы
        2.0: 3,    # Ранг 2: 3 единицы
        3.0: 5,    # Ранг 3: 5 единиц
        4.0: 8,    # Ранг 4: 8 единиц
        5.0: 12,   # Ранг 5: 12 единиц
    }
    
    if isinstance(resources, (int, float)):
        # Для скаляра
        min_dist = float('inf')
        best_rank = 1.0
        
        for rank, norm_count in normative.items():
            dist = abs(resources - norm_count)
            if dist < min_dist:
                min_dist = dist
                best_rank = rank
        
        return best_rank
    
    elif isinstance(resources, pd.Series):
        # Для Series
        def find_best_rank(count):
            min_dist = float('inf')
            best_rank = 1.0
            for rank, norm_count in normative.items():
                dist = abs(count - norm_count)
                if dist < min_dist:
                    min_dist = dist
                    best_rank = rank
            return best_rank
        
        return resources.apply(find_best_rank)
    
    else:
        raise ValueError("Неподдерживаемый тип resources")


def predict_with_confidence(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    X: pd.DataFrame,
    top_k: int = 3,
    use_bootstrap: bool = True,
    n_bootstrap: int = 30,
) -> dict[str, Any]:
    """
    Полный прогноз с доверительностью.
    
    Args:
        model: обученная модель
        X: признаки
        top_k: количество вариантов
        use_bootstrap: использовать ли бутстрап
        n_bootstrap: количество бутстрап-выборок
    
    Returns:
        dict с прогнозами и статистикой
    """
    # Top-K прогноз
    df_topk = predict_rank_topk(model, X, top_k=top_k)
    
    result = {
        "predictions": df_topk,
        "top_k": top_k,
        "model_type": type(model).__name__,
        "n_samples": len(X),
    }
    
    # Бутстрап (опционально)
    if use_bootstrap:
        df_bootstrap = bootstrap_predict_rank(
            model, X, n_bootstrap=n_bootstrap
        )
        
        # Объединение с top-K
        df_combined = pd.concat([
            df_topk.reset_index(drop=True),
            df_bootstrap.drop(columns=["predicted_rank_idx", "predicted_rank"], errors="ignore")
        ], axis=1)
        
        result["predictions"] = df_combined
        result["bootstrap"] = True
        result["n_bootstrap"] = n_bootstrap
    
    return result


def save_prediction_results(
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: str = "reports/predictions",
) -> dict[str, str]:
    """
    Сохранение результатов прогноза.
    
    Args:
        predictions: DataFrame с прогнозами
        metadata: метаданные прогноза
        output_dir: директория для сохранения
    
    Returns:
        пути к сохранённым файлам
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохранение прогнозов
    predictions_csv = output_path / f"predictions_{timestamp}.csv"
    predictions.to_csv(predictions_csv, index=False)
    
    # Сохранение метаданных
    metadata["timestamp"] = timestamp
    metadata_path = output_path / f"metadata_{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return {
        "predictions_csv": str(predictions_csv),
        "metadata_json": str(metadata_path),
    }


# ============================================================================
# CLI
# ============================================================================


def predict_rank_cli(
    data_path: str = "clean_df_enhanced.csv",
    model_path: str = None,
    feature_set: str = "online_tactical",
    top_k: int = 3,
    use_bootstrap: bool = True,
    n_bootstrap: int = 30,
    output_dir: str = "reports/predictions",
) -> dict[str, Any]:
    """
    Прогноз ранга из CLI.
    
    Args:
        data_path: путь к данным
        model_path: путь к модели (если None, ищется последняя в reports/models)
        feature_set: набор признаков
        top_k: количество вариантов
        use_bootstrap: использовать ли бутстрап
        n_bootstrap: количество бутстрап-выборок
        output_dir: директория для сохранения
    
    Returns:
        dict с результатами
    """
    import glob
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    print(f"📊 Загружено: {len(df)} записей")
    
    # Фильтрация по источнику
    if 'source_sheet' in df.columns:
        before = len(df)
        df = df[df["source_sheet"] == "БД-1...2000--2020 (1+2)"].copy()
        print(f"📂 Фильтрация по источнику: {before} -> {len(df)} записей")
    
    # Поиск модели
    if model_path is None:
        model_files = glob.glob("reports/models/model_rank_rf_*.joblib")
        if model_files:
            model_path = sorted(model_files)[-1]  # Последняя по времени
            print(f"🔍 Найдена модель: {model_path}")
        else:
            raise FileNotFoundError("Модель не найдена в reports/models/")
    
    # Загрузка модели
    model = load_model(model_path)
    print(f"✅ Загружена модель: {type(model).__name__}")
    
    # Подготовка данных
    X, y, features = prepare_data(
        df,
        feature_set=feature_set,
        target='rank_tz',
        fill_na=True,
        fill_value=-1,
        fill_strategy='constant'
    )
    
    print(f"📈 Признаков: {len(features)}, Образцов: {len(X)}")
    
    # Прогноз
    print(f"\n🔮 Прогнозирование (top_k={top_k}, bootstrap={use_bootstrap})...")
    result = predict_with_confidence(
        model, X,
        top_k=top_k,
        use_bootstrap=use_bootstrap,
        n_bootstrap=n_bootstrap
    )
    
    # Сохранение
    print(f"\n💾 Сохранение результатов...")
    metadata = {
        "model_path": model_path,
        "feature_set": feature_set,
        "top_k": top_k,
        "bootstrap": use_bootstrap,
        "n_bootstrap": n_bootstrap,
        "n_features": len(features),
        "n_samples": len(X),
    }
    
    saved = save_prediction_results(
        result["predictions"],
        metadata,
        output_dir=output_dir
    )
    
    print(f"  Прогнозы: {saved['predictions_csv']}")
    print(f"  Метаданные: {saved['metadata_json']}")
    
    # Статистика
    print(f"\n📊 СТАТИСТИКА ПРОГНОЗОВ:")
    df_pred = result["predictions"]
    
    if "top1_prob" in df_pred.columns:
        print(f"\nРаспределение top-1 прогнозов:")
        print(df_pred["top1_rank_name"].value_counts().sort_index())
        
        print(f"\nВероятности top-1:")
        print(df_pred["top1_prob"].describe())
        
        if "mean_prob_class" in df_pred.columns:
            print(f"\nСредняя вероятность (бутстрап):")
            print(df_pred["mean_prob_class"].describe())
    
    print("\n" + "=" * 70)
    print("✅ Прогнозирование завершено!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("ПРОГНОЗИРОВАНИЕ РАНГА С ДОВЕРИТЕЛЬНОСТЬЮ")
    print("=" * 70)
    
    predict_rank_cli()
