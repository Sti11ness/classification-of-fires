"""
Сохранение лучшей модели (Random Forest, F1 Macro=0.2908).
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import sys
sys.path.insert(0, 'src')

from fire_es.model_train import save_model, visualize_tree

print("=" * 70)
print("СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ")
print("=" * 70)

# Загрузка данных
df = pd.read_csv("clean_df_with_rank.csv")
if 'source_sheet' in df.columns:
    df = df[df["source_sheet"] == "БД-1...2000--2020 (1+2)"].copy()

print(f"\n📊 Загружено: {len(df)} записей")

# Признаки
features = [
    'region_code', 'settlement_type_code', 'fire_protection_code',
    'enterprise_type_code', 'building_floors', 'fire_floor',
    'fire_resistance_code', 'source_item_code', 'distance_to_station',
    't_detect_min', 't_report_min', 't_arrival_min', 't_first_hose_min'
]

# Подготовка данных
X = df[features].fillna(0)  # Лучшая стратегия

y = df['rank_tz'].copy()
y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
y_int = y.map(y_classes).fillna(1).astype(int)

print(f"📈 Признаков: {len(features)}")
print(f"📈 Образцов: {len(X)}")

# ============================================================================
# Обучение лучшей модели
# ============================================================================
print("\n🔧 Обучение модели...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Кросс-валидация
cv_f1m = cross_val_score(model, X, y_int, cv=5, scoring='f1_macro')
cv_f1w = cross_val_score(model, X, y_int, cv=5, scoring='f1_weighted')
cv_acc = cross_val_score(model, X, y_int, cv=5, scoring='accuracy')

print(f"\n📈 CV метрики:")
print(f"  F1 Macro:  {cv_f1m.mean():.4f} (+/- {cv_f1m.std()*2:.4f})")
print(f"  F1 Weighted: {cv_f1w.mean():.4f} (+/- {cv_f1w.std()*2:.4f})")
print(f"  Accuracy:  {cv_acc.mean():.4f} (+/- {cv_acc.std()*2:.4f})")

# Обучение на всех данных
model.fit(X, y_int)

# ============================================================================
# Сохранение
# ============================================================================
print("\n💾 Сохранение модели...")

# Метрики на тесте (hold-out 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

model_test = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model_test.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

y_pred = model_test.predict(X_test)

test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
}

print(f"\n📈 Метрики на тесте (20% hold-out):")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

# Сохранение модели
saved = save_model(
    model,
    {
        'cv_f1_macro_mean': float(cv_f1m.mean()),
        'cv_f1_macro_std': float(cv_f1m.std()),
        'cv_f1_weighted_mean': float(cv_f1w.mean()),
        'cv_accuracy_mean': float(cv_acc.mean()),
        'test_metrics': test_metrics,
        'classes': [1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    },
    feature_set='online_tactical',
    model_type='rank_rf_best',
    output_dir='reports/models'
)

print(f"\n✅ Модель сохранена: {saved['model_path']}")
print(f"✅ Метрики сохранены: {saved['metadata_path']}")

# Визуализация
print(f"\n🎨 Визуализация дерева...")
class_names_display = ["Ранг 1", "Ранг 1-бис", "Ранг 2", "Ранг 3", "Ранг 4", "Ранг 5"]
viz_path = visualize_tree(
    model.estimators_[0],  # Первое дерево
    features,
    class_names=class_names_display,
    output_path=Path("reports/models") / "tree_rank_best.png",
    max_depth=3,
    figsize=(20, 10)
)
print(f"✅ Визуализация: {viz_path}")

# Сохранение важности признаков
import pandas as pd
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv("reports/models/feature_importance_best.csv", index=False)
print(f"✅ Важность признаков: reports/models/feature_importance_best.csv")

# ============================================================================
# Итоговый отчёт
# ============================================================================
print("\n" + "=" * 70)
print("ИТОГОВЫЙ ОТЧЁТ")
print("=" * 70)

report = {
    'model_type': 'RandomForestClassifier',
    'feature_set': 'online_tactical',
    'n_features': len(features),
    'n_samples': len(X),
    'cv_metrics': {
        'f1_macro': float(cv_f1m.mean()),
        'f1_weighted': float(cv_f1w.mean()),
        'accuracy': float(cv_acc.mean()),
    },
    'test_metrics': test_metrics,
    'model_path': saved['model_path'],
    'metadata_path': saved['metadata_path'],
    'timestamp': datetime.now().isoformat(),
}

report_path = Path("reports/models/best_model_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n📁 Сохранённые файлы:")
print(f"  - {saved['model_path']}")
print(f"  - {saved['metadata_path']}")
print(f"  - {viz_path}")
print(f"  - reports/models/feature_importance_best.csv")
print(f"  - {report_path}")

print(f"\n📈 Итоговые метрики:")
print(f"  CV F1 Macro:  {cv_f1m.mean():.4f}")
print(f"  CV F1 Weighted: {cv_f1w.mean():.4f}")
print(f"  CV Accuracy:  {cv_acc.mean():.4f}")

print("\n" + "=" * 70)
print("✅ МОДЕЛЬ СОХРАНЕНА")
print("=" * 70)
