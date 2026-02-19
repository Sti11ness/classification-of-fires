"""
Тесты для модуля model_train.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fire_es.model_train import (
    FEATURE_SETS,
    TARGET_RANK,
    TARGET_RESOURCES,
    evaluate_feature_set_cv,
    prepare_data,
    predict_with_proba,
    train_classifier,
    train_regressor,
)


@pytest.fixture
def sample_df():
    """Пример данных для тестов."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "region_code": np.random.randint(1, 100, n),
        "building_floors": np.random.randint(1, 10, n),
        "fire_floor": np.random.randint(1, 10, n),
        "rank_tz": np.random.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0], n),
        "equipment_count": np.random.randint(1, 10, n),
    })


class TestPrepareData:
    """Тесты для prepare_data."""
    
    def test_prepare_minimal(self, sample_df):
        """Тест с minimal набором."""
        X, y, features = prepare_data(
            sample_df, feature_set="minimal", target=TARGET_RANK
        )
        assert len(features) == 3
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)
    
    def test_prepare_unknown_feature_set(self, sample_df):
        """Тест с неизвестным набором."""
        with pytest.raises(ValueError, match="Неизвестный набор"):
            prepare_data(sample_df, feature_set="unknown")
    
    def test_prepare_missing_target(self, sample_df):
        """Тест с отсутствующей целевой переменной."""
        with pytest.raises(ValueError, match="Отсутствует целевая"):
            prepare_data(sample_df, feature_set="minimal", target="missing")
    
    def test_prepare_with_na(self, sample_df):
        """Тест с пропусками."""
        sample_df.loc[0, "region_code"] = np.nan
        X, y, features = prepare_data(
            sample_df, feature_set="minimal", target=TARGET_RANK, fill_na=True
        )
        assert X["region_code"].iloc[0] == 0  # Заполнено 0

    def test_prepare_online_defaults_to_minus_one(self):
        """Для online-set дефолтное заполнение должно быть -1."""
        df = pd.DataFrame(
            {
                "region_code": [1, 2],
                "settlement_type_code": [1, np.nan],
                "fire_protection_code": [1, 2],
                "enterprise_type_code": [11, 20],
                "building_floors": [5, np.nan],
                "fire_floor": [2, 1],
                "fire_resistance_code": [2, np.nan],
                "source_item_code": [10, 11],
                "distance_to_station": [1.0, np.nan],
                "t_detect_min": [100, np.nan],
                "t_report_min": [120, 130],
                "t_arrival_min": [150, np.nan],
                "t_first_hose_min": [160, np.nan],
                "rank_tz": [1.0, 1.5],
            }
        )
        X, _, _ = prepare_data(df, feature_set="online_tactical", target=TARGET_RANK)
        assert (X == -1).any().any()

    def test_prepare_add_missing_indicators(self):
        """Проверка генерации индикаторов пропусков."""
        df = pd.DataFrame(
            {
                "region_code": [1, 2],
                "settlement_type_code": [1, 1],
                "fire_protection_code": [1, 2],
                "risk_category_code": [np.nan, 3],
                "enterprise_type_code": [11, 20],
                "fpo_class_code": [np.nan, 2],
                "building_floors": [5, 4],
                "fire_floor": [2, 1],
                "fire_resistance_code": [2, 3],
                "source_item_code": [10, 11],
                "distance_to_station": [1.0, 2.0],
                "rank_tz": [1.0, 1.5],
            }
        )
        X, _, _ = prepare_data(
            df,
            feature_set="managed",
            target=TARGET_RANK,
            add_missing_indicators=True,
            indicator_columns=["risk_category_code", "fpo_class_code"],
            fill_value=-1,
        )
        assert "risk_category_missing" in X.columns
        assert "fpo_class_missing" in X.columns
        assert X["risk_category_missing"].tolist() == [1, 0]
        assert X["fpo_class_missing"].tolist() == [1, 0]

    def test_prepare_unknown_fill_strategy(self, sample_df):
        with pytest.raises(ValueError, match="Неизвестная стратегия заполнения"):
            prepare_data(sample_df, feature_set="minimal", fill_strategy="bad")


class TestTrainClassifier:
    """Тесты для train_classifier."""
    
    def test_train_classifier(self, sample_df):
        """Тест обучения классификатора."""
        X, y, _ = prepare_data(sample_df, feature_set="minimal", target=TARGET_RANK)
        
        # Преобразование в целые классы
        y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
        y_int = y.map(y_classes).fillna(1).astype(int)
        
        result = train_classifier(X, y_int, max_depth=5)
        
        assert "model" in result
        assert "metrics" in result
        assert result["metrics"]["accuracy"] > 0
        assert result["metrics"]["f1"] > 0


class TestTrainRegressor:
    """Тесты для train_regressor."""
    
    def test_train_regressor(self, sample_df):
        """Тест обучения регрессора."""
        X, y, _ = prepare_data(
            sample_df, feature_set="minimal", target=TARGET_RESOURCES
        )
        
        result = train_regressor(X, y, max_depth=5)
        
        assert "model" in result
        assert "metrics" in result
        assert "mae" in result["metrics"]
        assert "r2" in result["metrics"]


class TestPredictWithProba:
    """Тесты для predict_with_proba."""
    
    def test_predict_with_proba(self, sample_df):
        """Тест прогноза с вероятностями."""
        X, y, _ = prepare_data(sample_df, feature_set="minimal", target=TARGET_RANK)
        y_classes = {1.0: 1, 1.5: 2, 2.0: 3, 3.0: 4, 4.0: 5, 5.0: 6}
        y_int = y.map(y_classes).fillna(1).astype(int)
        
        # Обучение
        result = train_classifier(X, y_int, max_depth=5)
        model = result["model"]
        
        # Прогноз
        X_test = X.head(10)
        predictions = predict_with_proba(model, X_test, top_k=3)
        
        assert "predicted_rank" in predictions.columns
        assert "top1_rank" in predictions.columns
        assert "top1_prob" in predictions.columns
        assert len(predictions) == 10


class TestFeatureSets:
    """Тесты для наборов признаков."""
    
    def test_feature_sets_defined(self):
        """Тест наличия наборов."""
        assert "minimal" in FEATURE_SETS
        assert "managed" in FEATURE_SETS
        assert "observed" in FEATURE_SETS
        assert "time" in FEATURE_SETS
        assert "full" in FEATURE_SETS
        assert "online_dispatch" in FEATURE_SETS
        assert "online_early" in FEATURE_SETS
        assert "online_tactical" in FEATURE_SETS
    
    def test_feature_sets_non_empty(self):
        """Тест непустоты наборов."""
        for name, features in FEATURE_SETS.items():
            assert len(features) > 0, f"Набор {name} пуст"

    def test_online_sets_have_no_leakage_features(self):
        """Online-наборы не должны содержать признаки последствий (утечку будущего)."""
        leakage = {
            "fatalities",
            "injuries",
            "direct_damage",
            "people_saved",
            "people_evacuated",
            "t_contained_min",
            "t_extinguished_min",
        }
        for name in ["online_dispatch", "online_early", "online_tactical"]:
            assert not leakage.intersection(FEATURE_SETS[name])

    def test_online_sets_exclude_high_missing_value_columns(self):
        """risk/fpo должны быть исключены из value-фич online-наборов."""
        for name in ["online_dispatch", "online_early", "online_tactical"]:
            assert "risk_category_code" not in FEATURE_SETS[name]
            assert "fpo_class_code" not in FEATURE_SETS[name]


class TestCrossValidation:
    def test_evaluate_feature_set_cv_rank(self):
        n = 120
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "region_code": rng.integers(1, 100, n),
                "settlement_type_code": rng.integers(1, 6, n),
                "fire_protection_code": rng.integers(1, 4, n),
                "enterprise_type_code": rng.integers(10, 30, n),
                "building_floors": rng.integers(1, 20, n),
                "fire_floor": rng.integers(1, 20, n),
                "fire_resistance_code": rng.integers(1, 6, n),
                "source_item_code": rng.integers(1, 50, n),
                "distance_to_station": rng.uniform(0, 30, n),
                "t_detect_min": rng.integers(0, 1440, n),
                "t_report_min": rng.integers(0, 1440, n),
                "t_arrival_min": rng.integers(0, 1440, n),
                "t_first_hose_min": rng.integers(0, 1440, n),
                "rank_tz": rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, 5.0], n),
            }
        )
        result = evaluate_feature_set_cv(
            df,
            feature_set="online_tactical",
            target=TARGET_RANK,
            fill_value=-1,
            n_splits=3,
        )
        assert result["feature_set"] == "online_tactical"
        assert "f1_macro_mean" in result["metrics"]
