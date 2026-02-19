"""
Финальное обучение моделей с лучшими признаками.

Сохраняет модели и метрики в reports/models/
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from fire_es.model_train import (
    prepare_data, 
    train_random_forest_classifier,
    save_model,
    visualize_tree,
    FEATURE_SETS,
    CANONICAL_SOURCE_SHEET,
)

print("=" * 70)
print("ФИНАЛЬНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

# Загрузка данных
data_path = 'clean_df_enhanced.csv'
df = pd.read_csv(data_path)
print(f"\n📊 Загружено: {len(df)} записей")

# Фильтрация по каноническому источнику
if 'source_sheet' in df.columns:
    before = len(df)
    df = df[df["source_sheet"] == CANONICAL_SOURCE_SHEET].copy()
    print(f"📂 Фильтрация по источнику: {before} -> {len(df)} записей")

# ============================================================================
# 1. Обучение Random Forest для rank_tz
# ============================================================================
print("\n" + "=" * 70)
print("1. ОБУЧЕНИЕ RANDOM FOREST ДЛЯ rank_tz")
print("=" * 70)

best_feature_set = 'online_tactical'  # Лучший по CV

X, y, features = prepare_data(
    df,
    feature_set=best_feature_set,
    target='rank_tz',
    fill_na=True,
    fill_value=-1,
    fill_strategy='constant'
)

# Преобразование в классы
y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
y_int = y.map(y_classes).fillna(1).astype(int)

print(f"\n📈 Данные для обучения:")
print(f"  Признаков: {len(features)}")
print(f"  Образцов: {len(X)}")
print(f"  Распределение классов: {y_int.value_counts().sort_index().to_dict()}")

# Обучение
print(f"\n🔧 Обучение Random Forest...")
result = train_random_forest_classifier(
    X, y_int,
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

# Восстановление имён классов для отчёта
class_names = {v: k for k, v in y_classes.items()}
result["metrics"]["classes"] = [class_names.get(c, c) for c in result["metrics"]["classes"]]

print(f"\n📈 Метрики на тестовой выборке:")
for key, value in result["metrics"].items():
    if key not in ["confusion_matrix", "classes"]:
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

# Сохранение модели
print(f"\n💾 Сохранение модели...")
saved = save_model(
    result["model"],
    result["metrics"],
    feature_set=best_feature_set,
    model_type="rank_rf",  # Random Forest
    output_dir="reports/models"
)

print(f"  Модель: {saved['model_path']}")
print(f"  Метрики: {saved['metadata_path']}")

# Визуализация одного дерева
print(f"\n🎨 Визуализация одного дерева...")
class_names_display = [f"Ранг {c}" for c in result["metrics"]["classes"]]
viz_path = visualize_tree(
    result["model"].estimators_[0],  # Первое дерево
    features,
    class_names=class_names_display,
    output_path=Path("reports/models") / f"tree_rank_{best_feature_set}_rf.png",
    max_depth=3,
    figsize=(20, 10)
)
print(f"  Сохранено: {viz_path}")

# ============================================================================
# 2. Обучение Decision Tree для сравнения
# ============================================================================
print("\n" + "=" * 70)
print("2. ОБУЧЕНИЕ DECISION TREE ДЛЯ rank_tz (baseline)")
print("=" * 70)

from fire_es.model_train import train_classifier

print(f"\n🔧 Обучение Decision Tree...")
result_dt = train_classifier(
    X, y_int,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

result_dt["metrics"]["classes"] = [class_names.get(c, c) for c in result_dt["metrics"]["classes"]]

print(f"\n📈 Метрики на тестовой выборке:")
for key, value in result_dt["metrics"].items():
    if key not in ["confusion_matrix", "classes"]:
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

# Сохранение модели
print(f"\n💾 Сохранение модели...")
saved_dt = save_model(
    result_dt["model"],
    result_dt["metrics"],
    feature_set=best_feature_set,
    model_type="rank_dt",  # Decision Tree
    output_dir="reports/models"
)

print(f"  Модель: {saved_dt['model_path']}")
print(f"  Метрики: {saved_dt['metadata_path']}")

# Визуализация
print(f"\n🎨 Визуализация дерева...")
viz_path_dt = visualize_tree(
    result_dt["model"],
    features,
    class_names=class_names_display,
    output_path=Path("reports/models") / f"tree_rank_{best_feature_set}_dt.png",
    max_depth=3,
    figsize=(20, 10)
)
print(f"  Сохранено: {viz_path_dt}")

# ============================================================================
# 3. Сравнение моделей
# ============================================================================
print("\n" + "=" * 70)
print("3. СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

comparison = {
    "feature_set": best_feature_set,
    "n_samples": len(X),
    "n_features": len(features),
    "random_forest": {
        "accuracy": result["metrics"]["accuracy"],
        "f1_macro": result["metrics"]["f1_macro"],
        "f1_weighted": result["metrics"]["f1_weighted"],
    },
    "decision_tree": {
        "accuracy": result_dt["metrics"]["accuracy"],
        "f1_macro": result_dt["metrics"]["f1_macro"],
        "f1_weighted": result_dt["metrics"]["f1_weighted"],
    },
    "improvement": {
        "accuracy": (result["metrics"]["accuracy"] - result_dt["metrics"]["accuracy"]) / result_dt["metrics"]["accuracy"] * 100,
        "f1_macro": (result["metrics"]["f1_macro"] - result_dt["metrics"]["f1_macro"]) / result_dt["metrics"]["f1_macro"] * 100,
        "f1_weighted": (result["metrics"]["f1_weighted"] - result_dt["metrics"]["f1_weighted"]) / result_dt["metrics"]["f1_weighted"] * 100,
    }
}

print(f"\n📊 Сравнение метрик:")
print(f"  {'Метрика':<15} {'Decision Tree':<15} {'Random Forest':<15} {'Улучшение':<15}")
print(f"  {'-'*60}")
for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
    dt_val = result_dt["metrics"][metric]
    rf_val = result["metrics"][metric]
    imp = comparison["improvement"][metric]
    sign = "+" if imp > 0 else ""
    print(f"  {metric:<15} {dt_val:<15.4f} {rf_val:<15.4f} {sign}{imp:>6.1f}%")

# Сохранение сравнения
with open(Path("reports/models") / "model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

print(f"\n💾 Сохранено: reports/models/model_comparison.json")

# ============================================================================
# 4. Важность признаков (для Random Forest)
# ============================================================================
print("\n" + "=" * 70)
print("4. ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest)")
print("=" * 70)

importances = result["model"].feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nТоп-10 важных признаков:")
print(feature_importance.head(10).to_string(index=False))

# Сохранение
feature_importance.to_csv(Path("reports/models") / "feature_importance.csv", index=False)
print(f"\n💾 Сохранено: reports/models/feature_importance.csv")

# ============================================================================
# 5. Итоговый отчёт
# ============================================================================
print("\n" + "=" * 70)
print("5. ИТОГОВЫЙ ОТЧЁТ")
print("=" * 70)

print(f"\n✅ Обучение завершено!")
print(f"\n📁 Сохранённые файлы:")
print(f"  - {saved['model_path']} (Random Forest модель)")
print(f"  - {saved['metadata_path']} (метрики RF)")
print(f"  - {saved_dt['model_path']} (Decision Tree модель)")
print(f"  - {saved_dt['metadata_path']} (метрики DT)")
print(f"  - {viz_path} (визуализация дерева RF)")
print(f"  - {viz_path_dt} (визуализация дерева DT)")
print(f"  - reports/models/model_comparison.json (сравнение)")
print(f"  - reports/models/feature_importance.csv (важность признаков)")

print(f"\n📈 Итоговые метрики (Random Forest):")
print(f"  Accuracy:  {result['metrics']['accuracy']:.4f}")
print(f"  F1 Macro:  {result['metrics']['f1_macro']:.4f}")
print(f"  F1 Weighted: {result['metrics']['f1_weighted']:.4f}")

print(f"\n📈 Улучшение относительно Decision Tree:")
print(f"  Accuracy:  {comparison['improvement']['accuracy']:+.1f}%")
print(f"  F1 Macro:  {comparison['improvement']['f1_macro']:+.1f}%")
print(f"  F1 Weighted: {comparison['improvement']['f1_weighted']:+.1f}%")

print("\n" + "=" * 70)
