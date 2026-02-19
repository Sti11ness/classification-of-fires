"""
Финальное обучение с отобранными признаками.

Используем только безопасные признаки без утечки.
Оптимизация под F1 Weighted.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С БЕЗОПАСНЫМИ ПРИЗНАКАМИ")
print("=" * 70)

# Загрузка данных
df_orig = pd.read_csv("clean_df_with_rank.csv")
df_enhanced = pd.read_csv("clean_df_enhanced.csv")
df_ft_selected = pd.read_csv("clean_df_featuretools_selected.csv")

# Фильтрация по источнику
for df in [df_orig, df_enhanced, df_ft_selected]:
    if 'source_sheet' in df.columns:
        mask = df["source_sheet"] == "БД-1...2000--2020 (1+2)"
        df[:] = df[mask].copy()

print(f"\n📊 Original: {df_orig.shape}")
print(f"📊 Enhanced: {df_enhanced.shape}")
print(f"📊 FT Selected: {df_ft_selected.shape}")

# Целевая переменная
def prepare_target(df):
    y = df['rank_tz'].copy()
    y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
    y_int = y.map(y_classes).fillna(1).astype(int)
    return y_int

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

# 3. FT Selected (безопасные признаки featuretools)
ft_features = [c for c in df_ft_selected.columns if c != 'rank_tz']
print(f"3. FT Selected: {len(ft_features)} признаков")

# 4. Combined (Original + Enhanced + FT)
combined_features = list(set(original_features + enhanced_features))
combined_features = [c for c in combined_features if c in df_enhanced.columns]
# Добавляем FT признаки, исключая дубли
for c in ft_features:
    if c not in combined_features and c in df_ft_selected.columns:
        combined_features.append(c)
print(f"4. Combined: {len(combined_features)} признаков")

# ============================================================================
# Обучение
# ============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ (CV=5, scoring=f1_weighted)")
print("=" * 70)

results = []

rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}

dt_params = {
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}

def train_eval(X, y, model, name):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_f1m = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    
    result = {
        'name': name,
        'n_features': X.shape[1],
        'f1_weighted_mean': cv_scores.mean(),
        'f1_weighted_std': cv_scores.std(),
        'accuracy_mean': cv_acc.mean(),
        'f1_macro_mean': cv_f1m.mean(),
    }
    results.append(result)
    
    print(f"\n{name}")
    print(f"  Признаков: {X.shape[1]}")
    print(f"  F1 Weighted: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Accuracy: {cv_acc.mean():.4f}")
    print(f"  F1 Macro: {cv_f1m.mean():.4f}")
    
    return result

# 1. Original + RF
y = prepare_target(df_orig)
X = df_orig[original_features].fillna(-1)
train_eval(X, y, RandomForestClassifier(**rf_params), "Original + RF")

# 2. Original + DT
train_eval(X, y, DecisionTreeClassifier(**dt_params), "Original + DT")

# 3. Enhanced + RF
y = prepare_target(df_enhanced)
X = df_enhanced[enhanced_features].fillna(-1)
train_eval(X, y, RandomForestClassifier(**rf_params), "Enhanced + RF")

# 4. FT Selected + RF
y = prepare_target(df_ft_selected)
X = df_ft_selected[ft_features].replace([np.inf, -np.inf], np.nan).fillna(-1)
train_eval(X, y, RandomForestClassifier(**rf_params), "FT Selected + RF")

# 5. Combined + RF
y = prepare_target(df_enhanced)
X_combined = pd.DataFrame()
for c in combined_features:
    if c in df_enhanced.columns:
        X_combined[c] = df_enhanced[c].fillna(-1)
    elif c in df_ft_selected.columns:
        X_combined[c] = df_ft_selected[c].replace([np.inf, -np.inf], np.nan).fillna(-1)
    else:
        X_combined[c] = -1

X_combined = X_combined.replace([np.inf, -np.inf], np.nan).fillna(-1)
train_eval(X_combined, y, RandomForestClassifier(**rf_params), "Combined + RF")

# ============================================================================
# Итоговая таблица
# ============================================================================
print("\n" + "=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА (сортировка по F1 Weighted)")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_weighted_mean', ascending=False)

print("\n")
for _, row in results_df.iterrows():
    print(f"{row['name']:<25} | Feat: {row['n_features']:3} | "
          f"F1-W: {row['f1_weighted_mean']:.4f} (+/- {row['f1_weighted_std']*2:.4f}) | "
          f"Acc: {row['accuracy_mean']:.4f} | F1-M: {row['f1_macro_mean']:.4f}")

# Сохранение
results_df.to_csv("reports/tables/final_model_comparison.csv", index=False)
print(f"\n💾 Сохранено: reports/tables/final_model_comparison.csv")

# ============================================================================
# Выводы
# ============================================================================
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

best = results_df.iloc[0]
print(f"\n✅ Лучшая модель: {best['name']}")
print(f"   F1 Weighted: {best['f1_weighted_mean']:.4f}")
print(f"   Признаков: {best['n_features']}")

# Сравнение с baseline
baseline = results_df[results_df['name'] == "Original + DT"].iloc[0]
improvement = (best['f1_weighted_mean'] - baseline['f1_weighted_mean']) / baseline['f1_weighted_mean'] * 100
print(f"\n📈 Улучшение относительно baseline (Original + DT):")
print(f"   F1 Weighted: {baseline['f1_weighted_mean']:.4f} → {best['f1_weighted_mean']:.4f} ({improvement:+.1f}%)")

# Топ-3
print(f"\n📋 Топ-3 модели:")
for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['name']} (F1-W: {row['f1_weighted_mean']:.4f})")

print("\n" + "=" * 70)
