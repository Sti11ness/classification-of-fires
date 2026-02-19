"""
Обучение моделей с оптимизацией под F1 Weighted.

Сравнивает:
1. Исходные признаки (clean_df_with_rank.csv)
2. Признаки с featuretools (clean_df_featuretools.csv)
3. Признаки с ручным feature engineering (clean_df_enhanced.csv)

Основная метрика: F1 Weighted
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ОБУЧЕНИЕ С ОПТИМИЗАЦИЕЙ ПОД F1 WEIGHTED")
print("=" * 70)

# ============================================================================
# Загрузка данных
# ============================================================================
print("\n📂 ЗАГРУЗКА ДАННЫХ")

# Исходные данные
df_original = pd.read_csv("clean_df_with_rank.csv")
if 'source_sheet' in df_original.columns:
    df_original = df_original[df_original["source_sheet"] == "БД-1...2000--2020 (1+2)"].copy()
print(f"  Original: {df_original.shape}")

# Featuretools
df_ft = pd.read_csv("clean_df_featuretools.csv")
print(f"  Featuretools: {df_ft.shape}")

# Enhanced (ручной feature engineering)
df_enhanced = pd.read_csv("clean_df_enhanced.csv")
if 'source_sheet' in df_enhanced.columns:
    df_enhanced = df_enhanced[df_enhanced["source_sheet"] == "БД-1...2000--2020 (1+2)"].copy()
print(f"  Enhanced: {df_enhanced.shape}")

# ============================================================================
# Подготовка данных
# ============================================================================
print("\n🔧 ПОДГОТОВКА ДАННЫХ")

# Целевая переменная
def prepare_target(df):
    """Преобразование rank_tz в классы."""
    y = df['rank_tz'].copy()
    y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
    y_int = y.map(y_classes).fillna(1).astype(int)
    return y_int

# Признаки для каждого набора
# 1. Original: базовые признаки
base_features = [
    'region_code', 'settlement_type_code', 'fire_protection_code',
    'enterprise_type_code', 'building_floors', 'fire_floor',
    'fire_resistance_code', 'source_item_code', 'distance_to_station',
    't_detect_min', 't_report_min', 't_arrival_min', 't_first_hose_min'
]

# 2. Featuretools: все кроме target
ft_features = [c for c in df_ft.columns if c != 'rank_tz']

# 3. Enhanced: online_tactical + новые признаки
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

# Фильтрация признаков, которые есть в данных
enhanced_features = [c for c in enhanced_features if c in df_enhanced.columns]

print(f"  Базовых признаков: {len(base_features)}")
print(f"  Признаков Featuretools: {len(ft_features)}")
print(f"  Enhanced признаков: {len(enhanced_features)}")

# ============================================================================
# Функция обучения
# ============================================================================
def train_and_evaluate(X, y, model, cv=5):
    """Обучение и оценка модели."""
    # Кросс-валидация
    cv_scores_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_scores_f1_macro = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    cv_scores_f1_weighted = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    
    return {
        'accuracy_mean': cv_scores_acc.mean(),
        'accuracy_std': cv_scores_acc.std(),
        'f1_macro_mean': cv_scores_f1_macro.mean(),
        'f1_macro_std': cv_scores_f1_macro.std(),
        'f1_weighted_mean': cv_scores_f1_weighted.mean(),
        'f1_weighted_std': cv_scores_f1_weighted.std(),
    }

# ============================================================================
# Обучение моделей
# ============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

results = []

# Параметры моделей
rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'class_weight': 'balanced',  # Балансировка для F1!
    'n_jobs': -1
}

dt_params = {
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'class_weight': 'balanced'
}

# 1. Original + Random Forest
print("\n1️⃣ Original + Random Forest")
X_orig = df_original[base_features].fillna(-1)
y_orig = prepare_target(df_original)
model_rf_orig = RandomForestClassifier(**rf_params)
metrics_orig = train_and_evaluate(X_orig, y_orig, model_rf_orig)
metrics_orig['dataset'] = 'Original'
metrics_orig['model'] = 'RandomForest'
metrics_orig['n_features'] = len(base_features)
results.append(metrics_orig)

print(f"   Accuracy:  {metrics_orig['accuracy_mean']:.4f} (+/- {metrics_orig['accuracy_std']*2:.4f})")
print(f"   F1 Macro:  {metrics_orig['f1_macro_mean']:.4f} (+/- {metrics_orig['f1_macro_std']*2:.4f})")
print(f"   F1 Weighted: {metrics_orig['f1_weighted_mean']:.4f} (+/- {metrics_orig['f1_weighted_std']*2:.4f})")

# 2. Original + Decision Tree
print("\n2️⃣ Original + Decision Tree")
model_dt_orig = DecisionTreeClassifier(**dt_params)
metrics_dt_orig = train_and_evaluate(X_orig, y_orig, model_dt_orig)
metrics_dt_orig['dataset'] = 'Original'
metrics_dt_orig['model'] = 'DecisionTree'
metrics_dt_orig['n_features'] = len(base_features)
results.append(metrics_dt_orig)

print(f"   Accuracy:  {metrics_dt_orig['accuracy_mean']:.4f}")
print(f"   F1 Macro:  {metrics_dt_orig['f1_macro_mean']:.4f}")
print(f"   F1 Weighted: {metrics_dt_orig['f1_weighted_mean']:.4f}")

# 3. Featuretools + Random Forest
print("\n3️⃣ Featuretools + Random Forest")
X_ft = df_ft[ft_features].replace([np.inf, -np.inf], np.nan).fillna(-1)
y_ft = prepare_target(df_ft)
model_rf_ft = RandomForestClassifier(**rf_params)
metrics_ft = train_and_evaluate(X_ft, y_ft, model_rf_ft)
metrics_ft['dataset'] = 'Featuretools'
metrics_ft['model'] = 'RandomForest'
metrics_ft['n_features'] = len(ft_features)
results.append(metrics_ft)

print(f"   Accuracy:  {metrics_ft['accuracy_mean']:.4f} (+/- {metrics_ft['accuracy_std']*2:.4f})")
print(f"   F1 Macro:  {metrics_ft['f1_macro_mean']:.4f} (+/- {metrics_ft['f1_macro_std']*2:.4f})")
print(f"   F1 Weighted: {metrics_ft['f1_weighted_mean']:.4f} (+/- {metrics_ft['f1_weighted_std']*2:.4f})")

# 4. Enhanced + Random Forest
print("\n4️⃣ Enhanced + Random Forest")
X_enh = df_enhanced[enhanced_features].fillna(-1)
y_enh = prepare_target(df_enhanced)
model_rf_enh = RandomForestClassifier(**rf_params)
metrics_enh = train_and_evaluate(X_enh, y_enh, model_rf_enh)
metrics_enh['dataset'] = 'Enhanced'
metrics_enh['model'] = 'RandomForest'
metrics_enh['n_features'] = len(enhanced_features)
results.append(metrics_enh)

print(f"   Accuracy:  {metrics_enh['accuracy_mean']:.4f} (+/- {metrics_enh['accuracy_std']*2:.4f})")
print(f"   F1 Macro:  {metrics_enh['f1_macro_mean']:.4f} (+/- {metrics_enh['f1_macro_std']*2:.4f})")
print(f"   F1 Weighted: {metrics_enh['f1_weighted_mean']:.4f} (+/- {metrics_enh['f1_weighted_std']*2:.4f})")

# 5. Combined (Original + Enhanced) + Random Forest
print("\n5️⃣ Combined (Original + Enhanced) + Random Forest")
# Объединяем базовые и enhanced признаки
combined_features = list(set(base_features + enhanced_features))
combined_features = [c for c in combined_features if c in df_enhanced.columns]

X_comb = df_enhanced[combined_features].fillna(-1)
model_rf_comb = RandomForestClassifier(**rf_params)
metrics_comb = train_and_evaluate(X_comb, y_enh, model_rf_comb)
metrics_comb['dataset'] = 'Combined'
metrics_comb['model'] = 'RandomForest'
metrics_comb['n_features'] = len(combined_features)
results.append(metrics_comb)

print(f"   Accuracy:  {metrics_comb['accuracy_mean']:.4f} (+/- {metrics_comb['accuracy_std']*2:.4f})")
print(f"   F1 Macro:  {metrics_comb['f1_macro_mean']:.4f} (+/- {metrics_comb['f1_macro_std']*2:.4f})")
print(f"   F1 Weighted: {metrics_comb['f1_weighted_mean']:.4f} (+/- {metrics_comb['f1_weighted_std']*2:.4f})")

# 6. Combined + Featuretools + Random Forest
print("\n6️⃣ Combined + Featuretools + Random Forest")
# Объединяем enhanced и featuretools признаки
# Исключаем дубли и target
ft_cols = [c for c in ft_features if c not in combined_features]
all_features = combined_features + ft_cols
all_features = [c for c in all_features if c in df_ft.columns or c in df_enhanced.columns]

# Создаём объединённый DataFrame
X_combined = pd.DataFrame()
for col in combined_features:
    if col in df_enhanced.columns:
        X_combined[col] = df_enhanced[col].fillna(-1)
    else:
        X_combined[col] = -1

for col in ft_cols:
    if col in df_ft.columns:
        X_combined[col] = df_ft[col].fillna(-1)
    else:
        X_combined[col] = -1

# Заполняем бесконечные значения
X_combined = X_combined.replace([np.inf, -np.inf], -1)

model_rf_all = RandomForestClassifier(**rf_params)
metrics_all = train_and_evaluate(X_combined, y_enh, model_rf_all)
metrics_all['dataset'] = 'All Features'
metrics_all['model'] = 'RandomForest'
metrics_all['n_features'] = len(all_features)
results.append(metrics_all)

print(f"   Accuracy:  {metrics_all['accuracy_mean']:.4f} (+/- {metrics_all['accuracy_std']*2:.4f})")
print(f"   F1 Macro:  {metrics_all['f1_macro_mean']:.4f} (+/- {metrics_all['f1_macro_std']*2:.4f})")
print(f"   F1 Weighted: {metrics_all['f1_weighted_mean']:.4f} (+/- {metrics_all['f1_weighted_std']*2:.4f})")

# ============================================================================
# Сводная таблица
# ============================================================================
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА (сортировка по F1 Weighted)")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_weighted_mean', ascending=False)

# Форматирование
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n")
for idx, row in results_df.iterrows():
    print(f"{row['dataset']:15} + {row['model']:15} | "
          f"Feat: {row['n_features']:3} | "
          f"Acc: {row['accuracy_mean']:.4f} | "
          f"F1-M: {row['f1_macro_mean']:.4f} | "
          f"F1-W: {row['f1_weighted_mean']:.4f}")

# Сохранение
results_df.to_csv("reports/tables/model_comparison_f1_weighted.csv", index=False)
print(f"\n💾 Сохранено: reports/tables/model_comparison_f1_weighted.csv")

# ============================================================================
# Выводы
# ============================================================================
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

best = results_df.iloc[0]
print(f"\n✅ Лучшая комбинация: {best['dataset']} + {best['model']}")
print(f"   F1 Weighted: {best['f1_weighted_mean']:.4f}")
print(f"   Признаков: {best['n_features']}")

# Сравнение с baseline
baseline = results_df[
    (results_df['dataset'] == 'Original') & 
    (results_df['model'] == 'DecisionTree')
].iloc[0]

improvement = (best['f1_weighted_mean'] - baseline['f1_weighted_mean']) / baseline['f1_weighted_mean'] * 100
print(f"\n📈 Улучшение относительно baseline:")
print(f"   F1 Weighted: {baseline['f1_weighted_mean']:.4f} → {best['f1_weighted_mean']:.4f} ({improvement:+.1f}%)")

# Топ-3
print(f"\n📋 Топ-3 комбинации:")
for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['dataset']} + {row['model']} (F1-W: {row['f1_weighted_mean']:.4f})")

print("\n" + "=" * 70)
