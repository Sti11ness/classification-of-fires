"""
Парсер ресурсов (техники) из поля equipment.

Этот модуль реализует:
1. Загрузку справочника техники с кодами
2. Парсинг поля equipment (коды или текст)
3. Построение вектора ресурсов по типам
4. Нормирование вектора [0;1]
5. Расчёт ранга по евклидову расстоянию до нормативов
"""

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Словарь кодов техники (код -> категория для группировки)
# На основе Табл.19 из Excel-файла
EQUIPMENT_CODES = {
    # Автоцистерны
    11: "AC",  # АЦ
    49: "AC",  # АЦЛ
    52: "AC",  # АЦПК
    53: "AC",  # АЦБ
    54: "AC",  # АЦ-К
    55: "AC",  # АЦ-ВС
    56: "AC",  # АЦ-СПК
    57: "AC_T",  # АЦ-Т (особо тяжелая)
    # Пожарно-спасательные автомобили
    58: "APS",  # АПС
    59: "APS",  # АПСЛ
    60: "APS",  # АПС-К
    61: "APS",  # АПМ
    45: "APP",  # АПП
    14: "ANR",  # АНР
    50: "ANVD",  # С насосом высокого давления
    # Целевого применения
    15: "APT",  # АПТ
    16: "AP",  # АП (порошковый)
    17: "AKT",  # АКТ
    35: "AGT",  # АГТ
    19: "AGVT",  # АГВТ
    12: "PNS",  # ПНС
    44: "PPP",  # ППП
    62: "PPP",  # ПППЦ
    18: "AA",  # АА
    63: "APS_T",  # АПС-Т
    64: "APS_TR",  # АПС-Тр
    65: "APS_E",  # АПС-Э
    66: "PSAAM",  # ПСААМ
    67: "PNRK",  # ПНРК
    68: "PANRK",  # ПАНРК
    69: "AC_L",  # АЦ(л)
    # Автолестницы и подъемники
    23: "AL",  # АЛ
    24: "APK",  # АПК
    70: "APKL",  # АПКЛ
    71: "ALC",  # АЛЦ
    72: "APKC",  # АПКЦ
    73: "APTM",  # АПТМ
    # Специальные
    43: "ASA",  # АСА
    74: "AT",  # АТ
    28: "ASO",  # АСО
    27: "AG",  # АГ (газодымозащитный)
    33: "AR",  # АР
    26: "AD",  # АД
    75: "AOPT",  # АОПТ
    51: "ABG",  # АБГ
    76: "PKS",  # ПКС
    29: "ASH",  # АШ
    34: "ALP",  # АЛП
    77: "SPEKL",  # СПЭКЛ
    78: "AOS",  # АОС
    79: "APRSS",  # АПРСС
    80: "ADPT",  # АДПТ
    25: "APTS",  # АПТС
    81: "PAKM",  # ПАКМ
    82: "AVZ",  # АВЗ
    # Другие мобильные средства
    20: "TRAIN",  # Поезд
    21: "SHIP",  # Судно
    22: "HELI",  # Вертолет
    40: "PLANE",  # Самолет
    30: "MOTORPUMP",  # Мотопомпа
    48: "ROBOT_G",  # Робот наземный
    13: "PNS_TRAILER",  # Прицепная станция
    31: "CUSTOM",  # Переоборудованная
    83: "ROBOT_A",  # Робот воздушный
    84: "ROBOT_W",  # Робот подводный
    # Устройства и оборудование
    39: "SMOKE_EXTRACTOR",  # Дымосос
    42: "TRACTOR",  # Трактор
    32: "OTHER",  # Другое
    46: "RESCUE",  # Спасательные устройства
    47: "RESCUE_EQ",  # Спасательное оборудование
}

# Обратный словарь: категория -> список кодов
CATEGORY_CODES = {}
for code, cat in EQUIPMENT_CODES.items():
    if cat not in CATEGORY_CODES:
        CATEGORY_CODES[cat] = []
    CATEGORY_CODES[cat].append(code)

# Нормативная таблица ресурсов по рангам
# Формат: {ранг: {категория: количество}}
# Основана на типовой практике и рис.3 ТЗ
# Обновлена для лучшего соответствия реальным данным
NORMATIVE_RESOURCES = {
    1: {"AC": 1},  # Минимальный ранг: 1 автоцистерна
    1.5: {"AC": 2},  # 1-бис: 2 автоцистерны
    2: {"AC": 2, "AL": 1},  # Ранг 2: 2 АЦ + 1 АЛ
    3: {"AC": 3, "AL": 1, "APS": 1},  # Ранг 3: 3 АЦ + 1 АЛ + 1 АПС
    4: {"AC": 4, "AL": 2, "APS": 1, "AG": 1},  # Ранг 4: 4 АЦ + 2 АЛ + 1 АПС + 1 АГ
    5: {"AC": 5, "AL": 2, "APS": 2, "AG": 1, "AR": 1},  # Ранг 5: 5 АЦ + 2 АЛ + 2 АПС + 1 АГ + 1 АР
}


def load_equipment_dict(path: str = "data/normative/equipment_types_full.csv") -> pd.DataFrame:
    """
    Загрузка справочника техники.
    
    Returns:
        DataFrame с колонками: code, name, category
    """
    df = pd.read_csv(path)
    return df


def parse_equipment_field(value: Any) -> dict:
    """
    Парсинг поля equipment.
    
    Поддерживаемые форматы:
    - "11, 29, 31" (коды через запятую)
    - "АЦ, АЛ-30" (названия/сокращения)
    - "11 (АЦ), 29 (АЛ)" (коды с названиями)
    
    Returns:
        dict: {category: count}
        Например: {"AC": 2, "AL": 1}
    """
    if pd.isna(value) or value is None:
        return {}
    
    result = {}
    value_str = str(value).strip()
    
    if not value_str or value_str == 'nan':
        return {}
    
    # Разделение по запятой или точке с запятой
    parts = re.split(r'[,;]', value_str)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Извлечение кода (число в начале)
        code_match = re.match(r'^(\d+)', part)
        if code_match:
            code = int(code_match.group(1))
            if code in EQUIPMENT_CODES:
                category = EQUIPMENT_CODES[code]
                result[category] = result.get(category, 0) + 1
                continue
        
        # Поиск сокращения в скобках
        abbrev_match = re.search(r'\(([^)]+)\)', part)
        if abbrev_match:
            abbrev = abbrev_match.group(1).strip().upper()
            # Поиск по сокращению в названиях
            for code, cat in EQUIPMENT_CODES.items():
                # Получаем название из справочника
                pass  # Упрощённо: ищем по ключевым словам
            # Простой поиск по сокращениям
            abbrev_map = {
                "АЦ": "AC", "АЦЛ": "AC", "АЦПК": "AC", "АЦБ": "AC",
                "АПС": "APS", "АПСЛ": "APS", "АЛ": "AL", "АРМ": "ARM",
                "АГ": "AG", "АШ": "ASH", "АСО": "ASO", "АР": "AR",
            }
            if abbrev in abbrev_map:
                category = abbrev_map[abbrev]
                result[category] = result.get(category, 0) + 1
                continue
        
        # Поиск по ключевым словам в тексте
        part_upper = part.upper()
        for abbrev, category in [("АЦ", "AC"), ("АЛ", "AL"), ("АПС", "APS"), ("АГ", "AG")]:
            if abbrev in part_upper:
                result[category] = result.get(category, 0) + 1
                break
    
    return result


def build_resource_vector(equipment_dict: dict, all_categories: Optional[list] = None) -> pd.Series:
    """
    Построение вектора ресурсов из словаря техники.
    
    Args:
        equipment_dict: {category: count}
        all_categories: список всех категорий для полноты вектора
    
    Returns:
        pd.Series с индексом = категории
    """
    if all_categories is None:
        all_categories = list(set(EQUIPMENT_CODES.values()))
    
    vector = {cat: equipment_dict.get(cat, 0) for cat in all_categories}
    return pd.Series(vector)


def normalize_vector(
    vector: pd.Series,
    max_values: Optional[dict] = None
) -> pd.Series:
    """
    Нормирование вектора к диапазону [0;1].
    
    Args:
        vector: исходный вектор
        max_values: максимумы для каждой категории (из нормативов)
    
    Returns:
        нормированный вектор
    """
    if max_values is None:
        max_values = {}
        for resources in NORMATIVE_RESOURCES.values():
            for cat, count in resources.items():
                max_values[cat] = max(max_values.get(cat, 0), count)
    
    normalized = {}
    for cat, value in vector.items():
        max_val = max_values.get(cat, 1)
        normalized[cat] = min(value / max_val, 1.0) if max_val > 0 else 0.0
    
    return pd.Series(normalized)


def euclidean_distance(vec1: pd.Series, vec2: pd.Series) -> float:
    """Расчёт евклидова расстояния между двумя векторами."""
    all_idx = vec1.index.union(vec2.index)
    v1 = vec1.reindex(all_idx, fill_value=0.0)
    v2 = vec2.reindex(all_idx, fill_value=0.0)
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def calculate_rank(equipment_vector: pd.Series) -> tuple[float, float]:
    """
    Расчёт ранга по минимальному евклидову расстоянию до нормативов.
    
    Args:
        equipment_vector: нормированный вектор ресурсов пожара
    
    Returns:
        (best_rank, min_distance)
    """
    min_distance = float('inf')
    best_rank = 1
    
    # Нормирование нормативных векторов
    max_values = {}
    for resources in NORMATIVE_RESOURCES.values():
        for cat, count in resources.items():
            max_values[cat] = max(max_values.get(cat, 0), count)
    
    for rank, resources in NORMATIVE_RESOURCES.items():
        norm_series = pd.Series(resources)
        norm_series_norm = normalize_vector(norm_series, max_values)
        equip_series_norm = normalize_vector(equipment_vector, max_values)
        
        distance = euclidean_distance(equip_series_norm, norm_series_norm)
        
        if distance < min_distance:
            min_distance = distance
            best_rank = rank
    
    return best_rank, min_distance


def process_equipment_column(
    df: pd.DataFrame,
    equipment_col: str = 'equipment'
) -> pd.DataFrame:
    """
    Обработка колонки equipment и добавление полей:
    - equipment_vector: dict {category: count}
    - equipment_vector_norm: нормированный вектор
    - rank_tz: ранг по ТЗ
    - rank_distance: расстояние до норматива
    """
    df = df.copy()
    
    # Парсинг поля equipment
    df['equipment_vector'] = df[equipment_col].apply(parse_equipment_field)
    
    # Построение векторов
    all_categories = list(set(EQUIPMENT_CODES.values()))
    
    def build_vector(equipment_dict):
        if not equipment_dict:
            return pd.Series({cat: 0 for cat in all_categories})
        return build_resource_vector(equipment_dict, all_categories)
    
    df['equipment_vector_norm'] = df['equipment_vector'].apply(
        lambda d: normalize_vector(build_vector(d))
    )
    
    # Расчёт ранга
    def calc_rank(vec):
        rank, dist = calculate_rank(vec)
        return pd.Series({'rank_tz': rank, 'rank_distance': dist})
    
    rank_df = df['equipment_vector_norm'].apply(calc_rank)
    df['rank_tz'] = rank_df['rank_tz']
    df['rank_distance'] = rank_df['rank_distance']
    
    return df


if __name__ == "__main__":
    print("Тестирование парсера техники...")
    
    # Примеры
    test_cases = [
        "11, 23, 29",  # Коды: АЦ, АЛ, АШ
        "АЦ, АЛ-30",  # Сокращения
        "11 (АЦ), 23 (АЛ)",  # Коды с названиями
    ]
    
    for test in test_cases:
        result = parse_equipment_field(test)
        print(f"  {test!r} -> {result}")
    
    print("\n✅ Тестирование завершено")
