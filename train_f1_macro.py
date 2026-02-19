"""
Обучение с оптимизацией под F1 Macro (основная метрика).

F1 Macro одинаково учитывает все классы, что важно для несбалансированных данных.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ОБУЧЕНИЕ С ОПТИМИЗАЦИЕЙ ПОД F1 MACRO")
print("=" * 70)

# Загрузка данных
df_orig = pd.read_csv("clean_df_with_rank.csv")
df_enhanced = pd.read_csv("clean_df_enhanced.csv")

# Фильтрация по источнику
for df in [df_orig, df_enhanced]:
    if 'source_sheet' in df.columns:
        mask = df["source_sheet"] == "БД-1...2000--2020 (1+2)"
        df[:] = df[mask].copy()

print(f"\n📊 Original: {df_orig.shape}")
print(f"📊 Enhanced: {df_enhanced.shape}")

# Целевая переменная
def prepare_target(df):
    y = df['rank_tz'].copy()
    y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
    y_int = y.map(y_classes).fillna(1).astype(int)
    return y_int

# Проверка распределения
y_test = prepare_target(df_orig)
print(f"\n📊 Распределение классов:")
print(y_test.value_counts().sort_index())

# ============================================================================
# Наборы признаков
# ============================================================================
print("\n" + "=" * 70)
print("НАБОРЫ ПРИЗНАКОВ")
print("=" * 70)

# 1. Original (базовый)
original_features = [
    'region_code', 'settlement_type_code', 'fire_protection_code',
    'enterprise_type_code', 'building_floors', 'fire_floor',
    'fire_resistance_code', 'source_item_code', 'distance_to_station',
    't_detect_min', 't_report_min', 't_arrival_min', 't_first_hose_min'
]
original_features = [c for c in original_features if c in df_orig.columns]
print(f"1. Original: {len(original_features)} признаков")

# 2. Enhanced (ручной feature engineering)
enhanced_features = [
    'region_code', 'settlement_type_code', 'fire_protection_code',
    'enterprise_type_code', 'building_floors', 'fire_floor',
    'fire_resistance_code', 'source_item_code', 'distance_to_station',
    't_detect_min', 't_report_min', 't_arrival_min', 't_first_hose_min',
    'day_of_week', 'month', 'season', 'is_weekend',
    'is_high_rise', 'is_super_high_rise', 'basement_fire', 'top_floor_fire',
    'fire_floor_ratio', 'is_far', 'is_very_far', 'distance_category',
    'building_floors_bin', 'risk_category_code_missing', 'fpo_class_code_missing'
]
enhanced_features = [c for c in enhanced_features if c in df_enhanced.columns]
print(f"2. Enhanced: {len(enhanced_features)} признаков")

# ============================================================================
# Обучение с разными параметрами
# ============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ (CV=5, scoring=f1_macro)")
print("=" * 70)

results = []

def train_eval(X, y, model, name, model_type):
    cv_f1m = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    cv_f1w = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    result = {
        'name': name,
        'model_type': model_type,
        'n_features': X.shape[1],
        'f1_macro_mean': cv_f1m.mean(),
        'f1_macro_std': cv_f1m.std(),
        'f1_weighted_mean': cv_f1w.mean(),
        'accuracy_mean': cv_acc.mean(),
    }
    results.append(result)
    
    print(f"\n{name}")
    print(f"  Признаков: {X.shape[1]}")
    print(f"  F1 Macro:  {cv_f1m.mean():.4f} (+/- {cv_f1m.std()*2:.4f}) ⭐")
    print(f"  F1 Weighted: {cv_f1w.mean():.4f}")
    print(f"  Accuracy: {cv_acc.mean():.4f}")
    
    return result

y = prepare_target(df_orig)

# ============================================================================
# 1. Original + разные модели
# ============================================================================
print("\n" + "=" * 70)
print("1. ORIGINAL ПРИЗНАКИ")
print("=" * 70)

X = df_orig[original_features].fillna(-1)

# Decision Tree (baseline)
train_eval(X, y, DecisionTreeClassifier(
    max_depth=10, min_samples_split=20, min_samples_leaf=10,
    random_state=42, class_weight='balanced'
), "Original + DT", "DecisionTree")

# Random Forest (стандартный)
train_eval(X, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF", "RandomForest")

# Random Forest (больше деревьев)
train_eval(X, y, RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF (200 trees)", "RandomForest")

# Random Forest (глубокий)
train_eval(X, y, RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF (deep)", "RandomForest")

# Logistic Regression (для сравнения)
X_norm = (X - X.mean()) / X.std()  # Нормализация
X_norm = X_norm.fillna(0)
train_eval(X_norm, y, LogisticRegression(
    max_iter=1000, random_state=42, class_weight='balanced', multi_class='multinomial'
), "Original + Logistic", "LogisticRegression")

# ============================================================================
# 2. Enhanced + разные модели
# ============================================================================
print("\n" + "=" * 70)
print("2. ENHANCED ПРИЗНАКИ")
print("=" * 70)

y = prepare_target(df_enhanced)
X = df_enhanced[enhanced_features].fillna(-1)

# Random Forest (стандартный)
train_eval(X, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Enhanced + RF", "RandomForest")

# Random Forest (больше деревьев)
train_eval(X, y, RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Enhanced + RF (200 trees)", "RandomForest")

# Random Forest (с подвыборкой)
train_eval(X, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', max_samples=0.8, n_jobs=-1
), "Enhanced + RF (subsample)", "RandomForest")

# ============================================================================
# 3. Сравнение с разными fill_value
# ============================================================================
print("\n" + "=" * 70)
print("3. РАЗНЫЕ СТРАТЕГИИ ЗАПОЛНЕНИЯ ПРОПУСКОВ")
print("=" * 70)

y = prepare_target(df_orig)
X_base = df_orig[original_features]

# fill_value = -1
X_m1 = X_base.fillna(-1)
train_eval(X_m1, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF (fill=-1)", "RandomForest")

# fill_value = 0
X_0 = X_base.fillna(0)
train_eval(X_0, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF (fill=0)", "RandomForest")

# fill_value = median
X_med = X_base.copy()
for col in X_med.columns:
    X_med[col] = X_med[col].fillna(X_med[col].median())
X_med = X_med.fillna(0)  # На случай если остались NaN

train_eval(X_med, y, RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1
), "Original + RF (fill=median)", "RandomForest")

# ============================================================================
# Итоговая таблица
# ============================================================================
print("\n" + "=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА (сортировка по F1 Macro ⭐)")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_macro_mean', ascending=False)

print("\n")
for _, row in results_df.iterrows():
    marker = "⭐" if row['name'] == results_df.iloc[0]['name'] else ""
    print(f"{row['name']:<30} | Feat: {row['n_features']:3} | "
          f"F1-M: {row['f1_macro_mean']:.4f} (+/- {row['f1_macro_std']*2:.4f}) {marker}| "
          f"F1-W: {row['f1_weighted_mean']:.4f} | Acc: {row['accuracy_mean']:.4f}")

# Сохранение
results_df.to_csv("reports/tables/model_comparison_f1_macro.csv", index=False)
print(f"\n💾 Сохранено: reports/tables/model_comparison_f1_macro.csv")

# ============================================================================
# Выводы
# ============================================================================
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

best = results_df.iloc[0]
print(f"\n✅ Лучшая модель: {best['name']}")
print(f"   F1 Macro: {best['f1_macro_mean']:.4f}")
print(f"   F1 Weighted: {best['f1_weighted_mean']:.4f}")
print(f"   Признаков: {best['n_features']}")

# Сравнение с baseline
baseline = results_df[results_df['name'] == "Original + DT"].iloc[0]
improvement_f1m = (best['f1_macro_mean'] - baseline['f1_macro_mean']) / baseline['f1_macro_mean'] * 100
improvement_f1w = (best['f1_weighted_mean'] - baseline['f1_weighted_mean']) / baseline['f1_weighted_mean'] * 100

print(f"\n📈 Улучшение относительно baseline (Original + DT):")
print(f"   F1 Macro:  {baseline['f1_macro_mean']:.4f} → {best['f1_macro_mean']:.4f} ({improvement_f1m:+.1f}%)")
print(f"   F1 Weighted: {baseline['f1_weighted_mean']:.4f} → {best['f1_weighted_mean']:.4f} ({improvement_f1w:+.1f}%)")

# Топ-3
print(f"\n📋 Топ-3 модели по F1 Macro:")
for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['name']} (F1-M: {row['f1_macro_mean']:.4f}, F1-W: {row['f1_weighted_mean']:.4f})")

print("\n" + "=" * 70)
