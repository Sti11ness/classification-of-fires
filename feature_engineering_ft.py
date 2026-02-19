"""
Feature Engineering с помощью Featuretools.

Автоматическая генерация признаков через Deep Feature Synthesis (DFS).
"""

import pandas as pd
import numpy as np
import featuretools as ft
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

print("=" * 70)
print("FEATURE ENGINEERING С FEATURETOOLS")
print("=" * 70)

# Загрузка данных
data_path = Path("clean_df_with_rank.csv")
df = pd.read_csv(data_path)
print(f"\n📊 Загружено: {df.shape[0]} записей, {df.shape[1]} признаков")

# Фильтрация по каноническому источнику
if 'source_sheet' in df.columns:
    before = len(df)
    df = df[df["source_sheet"] == "БД-1...2000--2020 (1+2)"].copy()
    print(f"📂 Фильтрация по источнику: {before} -> {len(df)} записей")

# ============================================================================
# 1. Подготовка данных для Featuretools
# ============================================================================
print("\n" + "=" * 70)
print("1. ПОДГОТОВКА ENTITYSET")
print("=" * 70)

# Выбираем числовые и категориальные признаки для генерации
numeric_cols = [
    'building_floors', 'fire_floor', 'distance_to_station',
    't_detect_min', 't_report_min', 't_arrival_min', 't_first_hose_min',
    'equipment_count', 'nozzle_count',
]

categorical_cols = [
    'region_code', 'settlement_type_code', 'fire_protection_code',
    'enterprise_type_code', 'fire_resistance_code', 'source_item_code',
    'month', 'day_of_week', 'season',
]

# Создаём копию с нужными колонками + target
cols_to_use = numeric_cols + categorical_cols + ['rank_tz', 'row_id']
cols_to_use = [c for c in cols_to_use if c in df.columns]

df_ft = df[cols_to_use].copy()

# Создаём уникальный индекс
df_ft['ft_index'] = range(len(df_ft))

# Заполнение пропусков в числовых колонках медианой
for col in numeric_cols:
    if col in df_ft.columns:
        df_ft[col] = df_ft[col].fillna(df_ft[col].median())

# Заполнение пропусков в категориальных колонках модой
for col in categorical_cols:
    if col in df_ft.columns:
        mode_val = df_ft[col].mode(dropna=True)
        if len(mode_val) > 0:
            df_ft[col] = df_ft[col].fillna(mode_val.iloc[0])
        else:
            df_ft[col] = df_ft[col].fillna(-1)

print(f"\n✅ Пропуски заполнены")
print(f"   Числовых колонок: {len(numeric_cols)}")
print(f"   Категориальных колонок: {len(categorical_cols)}")

# Создание EntitySet
es = ft.EntitySet(id="fires")

# Добавляем основную таблицу
es = es.add_dataframe(
    dataframe_name="fires",
    dataframe=df_ft,
    index="ft_index",
)

print(f"\n✅ EntitySet создан")
print(f"   Таблиц: {len(es.dataframes)}")
print(f"   Колонок: {len(es['fires'].columns)}")

# ============================================================================
# 2. Deep Feature Synthesis
# ============================================================================
print("\n" + "=" * 70)
print("2. DEEP FEATURE SYNTHESIS")
print("=" * 70)

# Параметры DFS
max_depth = 2
max_features = 50

print(f"\n🔧 Параметры DFS:")
print(f"   max_depth: {max_depth}")
print(f"   max_features: {max_features}")

# Запуск DFS
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="fires",
    trans_primitives=['absolute', 'negate', 'multiply_numeric', 'divide_numeric', 'modulo_numeric'],
    agg_primitives=[],  # Нет агрегаций, т.к. одна таблица
    max_depth=max_depth,
    max_features=max_features,
    drop_contains=['ft_index'],
    verbose=False,
)

print(f"\n✅ Сгенерировано признаков: {len(feature_defs)}")
print(f"   Размер feature_matrix: {feature_matrix.shape}")

# ============================================================================
# 3. Анализ новых признаков
# ============================================================================
print("\n" + "=" * 70)
print("3. АНАЛИЗ НОВЫХ ПРИЗНАКОВ")
print("=" * 70)

# Исходные колонки
original_cols = set(df_ft.columns) - {'rank_tz', 'row_id'}
new_cols = set(feature_matrix.columns) - original_cols

print(f"\n📈 Исходных признаков: {len(original_cols)}")
print(f"📈 Новых признаков: {len(new_cols)}")

# Топ признаков по дисперсии
feature_matrix_clean = feature_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
variances = feature_matrix_clean.var()
top_features = variances.sort_values(ascending=False).head(20)

print(f"\n📊 Топ-20 признаков по дисперсии:")
for feat, var in top_features.items():
    if feat not in original_cols:
        print(f"   {feat}: {var:.4f}")

# ============================================================================
# 4. Сохранение результатов
# ============================================================================
print("\n" + "=" * 70)
print("4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Сохраняем feature_matrix с target
feature_matrix['rank_tz'] = df_ft['rank_tz'].values

# Добавляем исходные колонки, которые могли быть удалены
for col in ['region_code', 'settlement_type_code', 'fire_protection_code', 
            'enterprise_type_code', 'fire_resistance_code', 'source_item_code']:
    if col in df.columns and col not in feature_matrix.columns:
        feature_matrix[col] = df[col].values

# Удаляем технический индекс
if 'ft_index' in feature_matrix.columns:
    feature_matrix = feature_matrix.drop(columns=['ft_index'])

output_path = Path("clean_df_featuretools.csv")
feature_matrix.to_csv(output_path, index=False)

print(f"\n💾 Сохранено в {output_path}")
print(f"   Размер: {feature_matrix.shape[0]} записей × {feature_matrix.shape[1]} признаков")

# Сохраняем список признаков
feature_list_path = Path("reports/tables/featuretools_features.csv")
feature_list_path.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    'feature': feature_matrix.columns.tolist(),
    'type': ['categorical' if str(feature_matrix[c].dtype) == 'object' else 'numeric' 
             for c in feature_matrix.columns]
}).to_csv(feature_list_path, index=False)

print(f"💾 Список признаков: {feature_list_path}")

# ============================================================================
# 5. Сравнение с исходными данными
# ============================================================================
print("\n" + "=" * 70)
print("5. СРАВНЕНИЕ С ИСХОДНЫМИ ДАННЫМИ")
print("=" * 70)

print(f"\n📊 Исходные данные:")
print(f"   Записей: {len(df)}")
print(f"   Признаков: {len(df.columns)}")

print(f"\n📊 После Featuretools:")
print(f"   Записей: {len(feature_matrix)}")
print(f"   Признаков: {len(feature_matrix.columns)}")

print(f"\n📈 Прирост признаков: +{len(feature_matrix.columns) - len(df.columns)}")

# Проверка на пропуски
missing_before = df.isna().sum().sum()
missing_after = feature_matrix.isna().sum().sum()

print(f"\n📊 Пропуски:")
print(f"   До: {missing_before}")
print(f"   После: {missing_after}")

print("\n" + "=" * 70)
print("✅ FEATURE ENGINEERING ЗАВЕРШЁН")
print("=" * 70)
