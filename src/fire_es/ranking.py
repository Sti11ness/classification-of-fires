"""
Расчёт ранга по ТЗ п.2.5.1.

Этот модуль реализует:
1. Нормирование векторов ресурсов [0;1]
2. Расчёт евклидова расстояния до нормативов
3. Определение ранга по минимальному расстоянию

ПРИМЕЧАНИЕ: В данных отсутствуют типы техники (Табл.19), поэтому используется
упрощённая методика: ранг определяется по количеству техники (equipment_count).
См. reports/equipment_data_issue.md
"""

from typing import Optional

import numpy as np
import pandas as pd

# Нормативная таблица ресурсов по рангам
# Основана на рис.3 ТЗ и типовой практике
# Формат: {ранг: минимальное_количество_техники}
RANK_BY_EQUIPMENT_COUNT = {
    1: 1,      # 1 единица техники
    1.5: 2,    # 1-бис
    2: 2,      # 2 единицы
    3: 3,      # 3 единицы
    4: 4,      # 4 единицы
    5: 5,      # 5+ единиц
}

# Нормативные векторы для расчёта расстояния (если бы были типы техники)
# Формат: {ранг: {тип: количество}}
NORMATIVE_VECTORS = {
    1: {"AC": 1},
    1.5: {"AC": 2},
    2: {"AC": 2, "AL": 1},
    3: {"AC": 3, "AL": 1, "ARM": 1},
    4: {"AC": 4, "AL": 2, "ARM": 1},
    5: {"AC": 5, "AL": 2, "ARM": 2, "APS": 1},
}


def normalize_vector(
    vector: pd.Series,
    max_values: Optional[dict] = None
) -> pd.Series:
    """
    Нормирование вектора к диапазону [0;1].

    Args:
        vector: исходный вектор (Series с именами категорий как индекс)
        max_values: максимумы для каждой категории

    Returns:
        нормированный вектор
    """
    if max_values is None:
        # Автоматическое определение максимумов из нормативов
        max_values = {}
        for resources in NORMATIVE_VECTORS.values():
            for cat, count in resources.items():
                max_values[cat] = max(max_values.get(cat, 0), count)

    normalized = {}
    for cat, value in vector.items():
        max_val = max_values.get(cat, 1)
        if max_val > 0:
            normalized[cat] = min(value / max_val, 1.0)
        else:
            normalized[cat] = 0.0

    return pd.Series(normalized)


def euclidean_distance(vec1: pd.Series, vec2: pd.Series) -> float:
    """
    Расчёт евклидова расстояния между двумя векторами.
    """
    # Объединение индексов
    all_idx = vec1.index.union(vec2.index)
    v1 = vec1.reindex(all_idx, fill_value=0.0)
    v2 = vec2.reindex(all_idx, fill_value=0.0)

    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def calculate_rank_by_vector(
    equipment_vector: pd.Series,
    normative_vectors: Optional[dict] = None
) -> tuple[float, float]:
    """
    Расчёт ранга по вектору ресурсов (евклидово расстояние).

    Args:
        equipment_vector: нормированный вектор ресурсов пожара
        normative_vectors: нормативные векторы {ранг: вектор}

    Returns:
        (best_rank, min_distance)
    """
    if normative_vectors is None:
        normative_vectors = NORMATIVE_VECTORS

    min_distance = float('inf')
    best_rank = 1

    # Предварительное нормирование всех нормативных векторов
    max_values = {}
    for resources in normative_vectors.values():
        for cat, count in resources.items():
            max_values[cat] = max(max_values.get(cat, 0), count)

    for rank, norm_vector in normative_vectors.items():
        norm_series = pd.Series(norm_vector)
        # Нормирование нормативного вектора теми же максимумами
        norm_series_norm = normalize_vector(norm_series, max_values)
        # Нормирование вектора пожара
        equip_series_norm = normalize_vector(equipment_vector, max_values)

        distance = euclidean_distance(equip_series_norm, norm_series_norm)

        if distance < min_distance:
            min_distance = distance
            best_rank = rank

    return best_rank, min_distance


def calculate_rank_by_count(equipment_count: float) -> tuple[float, float]:
    """
    Расчёт ранга по количеству техники (упрощённая методика).

    Используется когда типы техники неизвестны.

    Args:
        equipment_count: количество техники

    Returns:
        (rank, distance)
        distance = отклонение от границы ранга
    """
    if pd.isna(equipment_count) or equipment_count < 1:
        return 1.0, 0.0

    count = int(equipment_count)

    # Границы рангов
    if count == 1:
        return 1.0, 0.0
    elif count == 2:
        return 1.5, 0.0
    elif count == 3:
        return 2.0, 0.0
    elif count == 4:
        return 3.0, 0.0
    elif count == 5:
        return 4.0, 0.0
    else:
        # 5+ техники -> ранг 5
        distance = (count - 5) * 0.1  # Штраф за превышение
        return 5.0, distance


def assign_rank_tz(
    df: pd.DataFrame,
    equipment_count_col: str = 'equipment_count',
    method: str = 'count'
) -> pd.DataFrame:
    """
    Добавление поля rank_tz в DataFrame.

    Args:
        df: DataFrame с данными
        equipment_count_col: колонка с количеством техники
        method: 'count' (по количеству) или 'vector' (по вектору)

    Returns:
        DataFrame с добавленными полями:
        - rank_tz: ранг по ТЗ
        - rank_distance: расстояние до норматива
    """
    df = df.copy()

    if method == 'count':
        # Упрощённая методика по количеству
        result = df[equipment_count_col].apply(calculate_rank_by_count)
        df['rank_tz'] = result.apply(lambda x: x[0])
        df['rank_distance'] = result.apply(lambda x: x[1])

    elif method == 'vector':
        # Методика по вектору (требует equipment_vector)
        if 'equipment_vector_norm' not in df.columns:
            raise ValueError(
                "Для method='vector' требуется колонка 'equipment_vector_norm'. "
                "Сначала выполните process_equipment_column()."
            )

        def calc_rank(vec):
            rank, dist = calculate_rank_by_vector(vec)
            return pd.Series({'rank_tz': rank, 'rank_distance': dist})

        result = df['equipment_vector_norm'].apply(calc_rank)
        df['rank_tz'] = result['rank_tz']
        df['rank_distance'] = result['rank_distance']

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    return df


def get_rank_description(rank: float) -> str:
    """
    Текстовое описание ранга.
    """
    descriptions = {
        1.0: "Минимальный ранг (1 единица техники)",
        1.5: "Ранг 1-бис (2 единицы техники)",
        2.0: "Ранг 2 (2-3 единицы техники)",
        3.0: "Ранг 3 (3-4 единицы техники)",
        4.0: "Ранг 4 (4-5 единиц техники)",
        5.0: "Ранг 5 (5+ единиц техники)",
    }
    return descriptions.get(rank, f"Ранг {rank}")


def validate_rank_distribution(df: pd.DataFrame) -> dict:
    """
    Валидация распределения рангов.

    Returns:
        dict со статистикой
    """
    if 'rank_tz' not in df.columns:
        return {"error": "Колонка rank_tz отсутствует"}

    rank_counts = df['rank_tz'].value_counts().sort_index()

    return {
        "total": len(df),
        "distribution": rank_counts.to_dict(),
        "shares": (rank_counts / len(df)).to_dict(),
        "min_rank": float(rank_counts.index.min()),
        "max_rank": float(rank_counts.index.max()),
    }


if __name__ == "__main__":
    # Тестирование
    print("Тестирование модуля ranking...")

    # Тест 1: расчёт по количеству
    print("\n=== Тест 1: calculate_rank_by_count ===")
    for count in [1, 2, 3, 4, 5, 6, 10]:
        rank, dist = calculate_rank_by_count(count)
        print(f"  Техники: {count} -> Ранг: {rank}, Расстояние: {dist:.2f}")

    # Тест 2: нормирование вектора
    print("\n=== Тест 2: normalize_vector ===")
    vec = pd.Series({"AC": 2, "AL": 1, "ARM": 0})
    norm_vec = normalize_vector(vec)
    print(f"  Исходный: {vec.to_dict()}")
    print(f"  Нормированный: {norm_vec.to_dict()}")

    # Тест 3: расчёт по вектору
    print("\n=== Тест 3: calculate_rank_by_vector ===")
    for rank_test, resources in NORMATIVE_VECTORS.items():
        vec = pd.Series(resources)
        calc_rank, dist = calculate_rank_by_vector(vec)
        print(f"  Ранг {rank_test} (вектор {resources}): рассчитан как {calc_rank}, dist={dist:.2f}")

    print("\n✅ Тестирование завершено")
