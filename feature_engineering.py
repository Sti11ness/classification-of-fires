"""
Feature Engineering для улучшения метрик моделей.

Создаёт новые признаки на основе имеющихся данных:
1. Временные признаки (день недели, месяц, сезон)
2. Признаки времени реагирования (дельты между этапами)
3. Бинирование числовых признаков
4. Полиномиальные признаки
5. Взаимодействия между признаками
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, 'src')

from fire_es.schema import EN_COLS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить временные признаки из fire_date.
    
    Новые признаки:
    - day_of_week: день недели (0-6)
    - month: месяц (1-12)
    - quarter: квартал (1-4)
    - season: сезон (1=зима, 2=весна, 3=лето, 4=осень)
    - is_weekend: выходной ли день
    - is_night: ночное ли время (если есть время суток)
    """
    df = df.copy()
    
    if 'fire_date' in df.columns:
        # Парсинг даты
        dates = pd.to_datetime(df['fire_date'], errors='coerce')
        
        # День недели (0=понедельник, 6=воскресенье)
        df['day_of_week'] = dates.dt.dayofweek
        
        # Месяц
        df['month'] = dates.dt.month
        
        # Квартал
        df['quarter'] = dates.dt.quarter
        
        # Сезон (1=зима, 2=весна, 3=лето, 4=осень)
        df['season'] = df['month'].map({
            12: 1, 1: 1, 2: 1,  # зима
            3: 2, 4: 2, 5: 2,   # весна
            6: 3, 7: 3, 8: 3,   # лето
            9: 4, 10: 4, 11: 4  # осень
        })
        
        # Выходной ли день
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def add_response_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить признаки времени реагирования.
    
    Новые признаки:
    - delta_detect_to_report: время от обнаружения до сообщения
    - delta_report_to_arrival: время от сообщения до прибытия
    - delta_arrival_to_hose: время от прибытия до подачи ствола
    - delta_hose_to_contained: время локализации
    - delta_contained_to_extinguished: время ликвидации
    - total_response_time: общее время реагирования
    """
    df = df.copy()
    
    time_cols = ['t_detect_min', 't_report_min', 't_arrival_min', 
                 't_first_hose_min', 't_contained_min', 't_extinguished_min']
    
    # delta_detect_to_report
    if 't_detect_min' in df.columns and 't_report_min' in df.columns:
        df['delta_detect_to_report'] = df['t_report_min'] - df['t_detect_min']
    
    # delta_report_to_arrival
    if 't_report_min' in df.columns and 't_arrival_min' in df.columns:
        df['delta_report_to_arrival'] = df['t_arrival_min'] - df['t_report_min']
    
    # delta_arrival_to_hose
    if 't_arrival_min' in df.columns and 't_first_hose_min' in df.columns:
        df['delta_arrival_to_hose'] = df['t_first_hose_min'] - df['t_arrival_min']
    
    # delta_hose_to_contained
    if 't_first_hose_min' in df.columns and 't_contained_min' in df.columns:
        df['delta_hose_to_contained'] = df['t_contained_min'] - df['t_first_hose_min']
    
    # delta_contained_to_extinguished
    if 't_contained_min' in df.columns and 't_extinguished_min' in df.columns:
        df['delta_contained_to_extinguished'] = df['t_extinguished_min'] - df['t_contained_min']
    
    # total_response_time
    if 't_detect_min' in df.columns and 't_extinguished_min' in df.columns:
        df['total_response_time'] = df['t_extinguished_min'] - df['t_detect_min']
    
    return df


def add_building_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить признаки риска здания.
    
    Новые признаки:
    - fire_floor_ratio: отношение этажа пожара к этажности
    - is_high_rise: высотное ли здание (>9 этажей)
    - is_super_high_rise: сверхвысотное ли здание (>25 этажей)
    - basement_fire: пожар в подвале (fire_floor < 0)
    - top_floor_fire: пожар на верхнем этаже
    """
    df = df.copy()
    
    if 'fire_floor' in df.columns and 'building_floors' in df.columns:
        # Отношение этажа пожара к этажности
        df['fire_floor_ratio'] = df['fire_floor'] / df['building_floors'].replace(0, np.nan)
        df['fire_floor_ratio'] = df['fire_floor_ratio'].clip(-1, 2)  # Обработка выбросов
        
        # Высотное здание (>9 этажей по СНиП)
        df['is_high_rise'] = (df['building_floors'] > 9).astype(int)
        
        # Сверхвысотное здание (>25 этажей)
        df['is_super_high_rise'] = (df['building_floors'] > 25).astype(int)
        
        # Пожар в подвале
        df['basement_fire'] = (df['fire_floor'] < 0).astype(int)
        
        # Пожар на верхнем этаже
        df['top_floor_fire'] = (df['fire_floor'] == df['building_floors']).astype(int)
    
    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить признаки расстояния до пожарной части.
    
    Новые признаки:
    - is_far: далеко ли (>5 км)
    - is_very_far: очень далеко ли (>10 км)
    - distance_category: категория расстояния (близко/средне/далеко)
    """
    df = df.copy()
    
    if 'distance_to_station' in df.columns:
        # Далеко ли (>5 км)
        df['is_far'] = (df['distance_to_station'] > 5).astype(int)
        
        # Очень далеко (>10 км)
        df['is_very_far'] = (df['distance_to_station'] > 10).astype(int)
        
        # Категория расстояния (0=близко, 1=средне, 2=далеко, 3=очень далеко)
        df['distance_category'] = pd.cut(
            df['distance_to_station'],
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    return df


def add_severity_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить индикаторы тяжести пожара (для online-режимов).
    
    Важно: эти признаки могут быть недоступны в реальном времени!
    Использовать только для исследовательских моделей.
    
    Новые признаки:
    - has_fatalities: есть ли погибшие
    - has_injuries: есть ли пострадавшие
    - has_damage: есть ли ущерб
    - severity_level: уровень тяжести (0-3)
    """
    df = df.copy()
    
    # Индикаторы
    if 'fatalities' in df.columns:
        df['has_fatalities'] = (df['fatalities'] > 0).astype(int)
    
    if 'injuries' in df.columns:
        df['has_injuries'] = (df['injuries'] > 0).astype(int)
    
    if 'direct_damage' in df.columns:
        df['has_damage'] = (df['direct_damage'] > 0).astype(int)
    
    # Уровень тяжести (сумма индикаторов)
    severity_cols = []
    if 'has_fatalities' in df.columns:
        severity_cols.append('has_fatalities')
    if 'has_injuries' in df.columns:
        severity_cols.append('has_injuries')
    if 'has_damage' in df.columns:
        severity_cols.append('has_damage')
    
    if severity_cols:
        df['severity_level'] = df[severity_cols].sum(axis=1)
    
    return df


def add_missing_indicators(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Добавить индикаторы пропусков для указанных колонок.
    
    Args:
        df: DataFrame
        columns: список колонок для создания индикаторов
    
    Returns:
        DataFrame с добавленными индикаторами
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            indicator_name = f"{col}_missing"
            df[indicator_name] = df[col].isna().astype(int)
    
    return df


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить бинированные версии числовых признаков.
    
    Новые признаки:
    - building_floors_bin: этажность (низкая/средняя/высокая)
    - equipment_count_bin: количество техники (мало/средне/много)
    - distance_bin: расстояние (близко/средне/далеко)
    """
    df = df.copy()
    
    # Бинирование этажности
    if 'building_floors' in df.columns:
        df['building_floors_bin'] = pd.cut(
            df['building_floors'],
            bins=[-np.inf, 2, 5, 9, 16, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
    
    # Бинирование количества техники
    if 'equipment_count' in df.columns:
        df['equipment_count_bin'] = pd.cut(
            df['equipment_count'],
            bins=[-np.inf, 0, 2, 5, 10, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить признаки взаимодействия между переменными.
    
    Новые признаки:
    - floors_x_distance: этажность * расстояние
    - weekend_x_night: выходной * ночь
    - high_rise_x_distance: высотное * расстояние
    """
    df = df.copy()
    
    # floors_x_distance
    if 'building_floors' in df.columns and 'distance_to_station' in df.columns:
        df['floors_x_distance'] = df['building_floors'] * df['distance_to_station']
    
    # weekend_x_detect_time
    if 'is_weekend' in df.columns and 't_detect_min' in df.columns:
        df['weekend_x_detect'] = df['is_weekend'] * df['t_detect_min']
    
    # high_rise_x_floors
    if 'is_high_rise' in df.columns and 'building_floors' in df.columns:
        df['high_rise_x_floors'] = df['is_high_rise'] * df['building_floors']
    
    return df


def add_all_features(df: pd.DataFrame, add_severity: bool = False) -> pd.DataFrame:
    """
    Добавить все признаки сразу.
    
    Args:
        df: исходный DataFrame
        add_severity: добавлять ли признаки тяжести (недоступны в реальном времени)
    
    Returns:
        DataFrame с добавленными признаками
    """
    # Последовательное добавление признаков
    df = add_temporal_features(df)
    df = add_response_time_features(df)
    df = add_building_risk_features(df)
    df = add_distance_features(df)
    df = add_missing_indicators(df, ['risk_category_code', 'fpo_class_code'])
    df = add_binned_features(df)
    
    # Признаки взаимодействия (после всех остальных)
    df = add_interaction_features(df)
    
    # Признаки тяжести (только для исследовательских моделей!)
    if add_severity:
        df = add_severity_indicators(df)
    
    return df


def get_feature_groups() -> dict[str, list[str]]:
    """
    Получить группы признаков для анализа важности.
    
    Returns:
        dict: группы признаков
    """
    return {
        'temporal': ['day_of_week', 'month', 'quarter', 'season', 'is_weekend'],
        'response_time': [
            'delta_detect_to_report', 'delta_report_to_arrival',
            'delta_arrival_to_hose', 'delta_hose_to_contained',
            'delta_contained_to_extinguished', 'total_response_time'
        ],
        'building_risk': [
            'fire_floor_ratio', 'is_high_rise', 'is_super_high_rise',
            'basement_fire', 'top_floor_fire'
        ],
        'distance': ['is_far', 'is_very_far', 'distance_category'],
        'missing_indicators': ['risk_category_code_missing', 'fpo_class_code_missing'],
        'binned': ['building_floors_bin', 'equipment_count_bin'],
        'interaction': ['floors_x_distance', 'weekend_x_detect', 'high_rise_x_floors'],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE ENGINEERING: Создание новых признаков")
    print("=" * 70)
    
    # Загрузка данных
    data_path = Path("clean_df_with_rank.csv")
    if not data_path.exists():
        print(f"❌ Файл {data_path} не найден!")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"\n📊 Исходные данные: {df.shape[0]} записей, {df.shape[1]} признаков")
    
    # Добавление признаков
    print("\n🔧 Добавление новых признаков...")
    df_enhanced = add_all_features(df, add_severity=False)
    
    print(f"\n✅ После feature engineering: {df_enhanced.shape[0]} записей, {df_enhanced.shape[1]} признаков")
    
    # Список новых признаков
    original_cols = set(df.columns)
    new_cols = [c for c in df_enhanced.columns if c not in original_cols]
    
    print(f"\n📈 Добавлено {len(new_cols)} новых признаков:")
    for col in new_cols:
        missing_pct = df_enhanced[col].isna().mean() * 100
        print(f"  - {col} (пропуски: {missing_pct:.1f}%)")
    
    # Сохранение
    output_path = Path("clean_df_enhanced.csv")
    df_enhanced.to_csv(output_path, index=False)
    print(f"\n💾 Сохранено в {output_path}")
    
    # Статистика по группам признаков
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ПО ГРУППАМ ПРИЗНАКОВ")
    print("=" * 70)
    
    feature_groups = get_feature_groups()
    for group_name, cols in feature_groups.items():
        available_cols = [c for c in cols if c in df_enhanced.columns]
        if available_cols:
            print(f"\n{group_name}: {len(available_cols)} признаков")
            for col in available_cols:
                non_missing = df_enhanced[col].notna().sum()
                pct = non_missing / len(df_enhanced) * 100
                print(f"  - {col}: {non_missing} записей ({pct:.1f}%)")
