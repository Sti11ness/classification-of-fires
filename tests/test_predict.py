"""
Тесты для модуля predict.py

Проверяет:
- Прогноз с top-K вероятностями
- Бутстрап-ансамбль
- Прогноз ресурсов с квантилями
- Определение ранга по ресурсам
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import sys
sys.path.insert(0, 'src')

from fire_es.predict import (
    predict_rank_topk,
    bootstrap_predict_rank,
    predict_resources_quantiles,
    rank_from_resources,
    predict_with_confidence,
    RANK_CLASSES,
    RANK_NAMES,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_X():
    """Пример признаков для прогноза."""
    np.random.seed(42)
    return pd.DataFrame({
        'region_code': np.random.randint(1, 10, 100),
        'building_floors': np.random.randint(1, 20, 100),
        'fire_floor': np.random.randint(1, 10, 100),
        'distance_to_station': np.random.uniform(1, 20, 100),
        't_detect_min': np.random.uniform(1, 30, 100),
        't_report_min': np.random.uniform(2, 40, 100),
        't_arrival_min': np.random.uniform(5, 50, 100),
        't_first_hose_min': np.random.uniform(10, 60, 100),
        'source_item_code': np.random.randint(1, 5, 100),
        'enterprise_type_code': np.random.randint(1, 5, 100),
        'fire_protection_code': np.random.randint(1, 5, 100),
        'settlement_type_code': np.random.randint(1, 5, 100),
        'fire_resistance_code': np.random.randint(1, 5, 100),
    })


@pytest.fixture
def sample_y_rank():
    """Пример целевой переменной для ранга."""
    np.random.seed(42)
    # Классы: 1, 2, 3, 4, 5, 6
    return pd.Series(np.random.choice([1, 2, 3, 4, 5, 6], 100))


@pytest.fixture
def sample_y_resources():
    """Пример целевой переменной для ресурсов."""
    np.random.seed(42)
    return pd.Series(np.random.randint(1, 15, 100))


@pytest.fixture
def trained_classifier(sample_X, sample_y_rank):
    """Обученная модель классификации."""
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(sample_X, sample_y_rank)
    return model


@pytest.fixture
def trained_regressor(sample_X, sample_y_resources):
    """Обученная модель регрессии."""
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(sample_X, sample_y_resources)
    return model


# ============================================================================
# ТЕСТЫ: predict_rank_topk
# ============================================================================

class TestPredictRankTopk:
    """Тесты для predict_rank_topk."""
    
    def test_topk_output_shape(self, trained_classifier, sample_X):
        """Проверка формы выходных данных."""
        result = predict_rank_topk(trained_classifier, sample_X, top_k=3)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_X)
        assert "predicted_rank" in result.columns
        assert "predicted_rank_idx" in result.columns
    
    def test_topk_columns(self, trained_classifier, sample_X):
        """Проверка наличия колонок top-K."""
        result = predict_rank_topk(trained_classifier, sample_X, top_k=3)
        
        # Проверка колонок для top-3
        for i in range(1, 4):
            assert f"top{i}_rank_idx" in result.columns
            assert f"top{i}_rank" in result.columns
            assert f"top{i}_rank_name" in result.columns
            assert f"top{i}_prob" in result.columns
    
    def test_topk_probabilities_sum(self, trained_classifier, sample_X):
        """Проверка, что вероятности корректны (не отрицательные)."""
        result = predict_rank_topk(trained_classifier, sample_X, top_k=3)
        
        # Все вероятности должны быть неотрицательными
        for i in range(1, 4):
            assert (result[f"top{i}_prob"] >= 0).all()
            assert (result[f"top{i}_prob"] <= 1).all()
    
    def test_topk_rank_values(self, trained_classifier, sample_X):
        """Проверка значений рангов."""
        result = predict_rank_topk(trained_classifier, sample_X, top_k=3)
        
        # Ранги должны быть из допустимого набора
        valid_ranks = set(RANK_CLASSES.values())
        predicted_ranks = set(result["predicted_rank"].unique())
        assert predicted_ranks.issubset(valid_ranks)
    
    def test_topk_order(self, trained_classifier, sample_X):
        """Проверка, что вероятности убывают."""
        result = predict_rank_topk(trained_classifier, sample_X, top_k=3)
        
        # top1_prob >= top2_prob >= top3_prob
        assert (result["top1_prob"] >= result["top2_prob"]).all()
        assert (result["top2_prob"] >= result["top3_prob"]).all()
    
    def test_topk_custom_mapping(self, trained_classifier, sample_X):
        """Проверка с custom маппингом классов."""
        custom_mapping = {1: 1.0, 2: 2.0, 3: 3.0}
        result = predict_rank_topk(
            trained_classifier, sample_X, top_k=3, class_mapping=custom_mapping
        )
        
        assert "predicted_rank" in result.columns

    def test_topk_handles_models_with_fewer_classes_than_requested(self):
        X = pd.DataFrame({"x": [0, 1, 0, 1], "y": [1, 1, 2, 2]})
        y = pd.Series([1, 1, 2, 2])
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        result = predict_rank_topk(model, X.head(1), top_k=3)
        assert "top1_rank" in result.columns
        assert "top2_rank" in result.columns
        assert "top3_rank" not in result.columns


# ============================================================================
# ТЕСТЫ: bootstrap_predict_rank
# ============================================================================

class TestBootstrapPredictRank:
    """Тесты для bootstrap_predict_rank."""
    
    def test_bootstrap_output_shape(self, trained_classifier, sample_X):
        """Проверка формы выходных данных."""
        result = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_X)
    
    def test_bootstrap_columns(self, trained_classifier, sample_X):
        """Проверка наличия колонок доверительных интервалов."""
        result = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10
        )
        
        required_cols = [
            "predicted_rank_idx",
            "predicted_rank",
            "mean_prob_class",
            "std_prob_class",
            "ci_lower_prob",
            "ci_upper_prob",
        ]
        
        for col in required_cols:
            assert col in result.columns
    
    def test_bootstrap_confidence_interval(self, trained_classifier, sample_X):
        """Проверка корректности доверительных интервалов."""
        result = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10
        )
        
        # CI lower <= mean <= CI upper
        assert (result["ci_lower_prob"] <= result["mean_prob_class"]).all()
        assert (result["mean_prob_class"] <= result["ci_upper_prob"]).all()
    
    def test_bootstrap_std_non_negative(self, trained_classifier, sample_X):
        """Проверка, что std неотрицательный."""
        result = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10
        )
        
        assert (result["std_prob_class"] >= 0).all()
    
    def test_bootstrap_reproducibility(self, trained_classifier, sample_X):
        """Проверка воспроизводимости результатов."""
        result1 = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10, random_state=42
        )
        result2 = bootstrap_predict_rank(
            trained_classifier, sample_X, n_bootstrap=10, random_state=42
        )
        
        # Результаты должны совпадать
        pd.testing.assert_frame_equal(result1, result2)


# ============================================================================
# ТЕСТЫ: predict_resources_quantiles
# ============================================================================

class TestPredictResourcesQuantiles:
    """Тесты для predict_resources_quantiles."""
    
    def test_quantiles_output_shape(self, trained_regressor, sample_X):
        """Проверка формы выходных данных."""
        result = predict_resources_quantiles(trained_regressor, sample_X)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_X)
    
    def test_quantiles_columns(self, trained_regressor, sample_X):
        """Проверка наличия колонок квантилей."""
        result = predict_resources_quantiles(trained_regressor, sample_X)
        
        assert "predicted_mean" in result.columns
        assert "p20" in result.columns or "resource_p20" in result.columns
        assert "p50" in result.columns or "resource_p50" in result.columns
        assert "p80" in result.columns or "resource_p80" in result.columns
    
    def test_quantiles_order(self, trained_regressor, sample_X):
        """Проверка, что квантили упорядочены."""
        result = predict_resources_quantiles(trained_regressor, sample_X)
        
        # p20 <= p50 <= p80
        p20_col = "resource_p20" if "resource_p20" in result.columns else "p20"
        p50_col = "resource_p50" if "resource_p50" in result.columns else "p50"
        p80_col = "resource_p80" if "resource_p80" in result.columns else "p80"
        
        assert (result[p20_col] <= result[p50_col]).all()
        assert (result[p50_col] <= result[p80_col]).all()
    
    def test_quantiles_custom(self, trained_regressor, sample_X):
        """Проверка с custom квантилями."""
        custom_quantiles = [0.1, 0.5, 0.9]
        result = predict_resources_quantiles(
            trained_regressor, sample_X, quantiles=custom_quantiles
        )
        
        assert "resource_p10" in result.columns
        assert "resource_p50" in result.columns
        assert "resource_p90" in result.columns


# ============================================================================
# ТЕСТЫ: rank_from_resources
# ============================================================================

class TestRankFromResources:
    """Тесты для rank_from_resources."""
    
    def test_rank_scalar(self):
        """Проверка для скалярного значения."""
        assert rank_from_resources(2) == 1.5  # 2 техники -> ранг 1-бис
        assert rank_from_resources(3) == 2.0  # 3 техники -> ранг 2
        assert rank_from_resources(5) == 3.0  # 5 техники -> ранг 3
        assert rank_from_resources(8) == 4.0  # 8 техники -> ранг 4
    
    def test_rank_series(self):
        """Проверка для Series."""
        resources = pd.Series([2, 3, 5, 8, 12])
        ranks = rank_from_resources(resources)
        
        expected = pd.Series([1.5, 2.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(ranks, expected)
    
    def test_rank_edge_cases(self):
        """Проверка граничных значений."""
        # Очень маленькое значение
        assert rank_from_resources(0) == 1.0
        # Очень большое значение
        assert rank_from_resources(20) == 5.0
    
    def test_rank_invalid_type(self):
        """Проверка обработки неверного типа."""
        with pytest.raises(ValueError):
            rank_from_resources("invalid")


# ============================================================================
# ТЕСТЫ: predict_with_confidence
# ============================================================================

class TestPredictWithConfidence:
    """Тесты для predict_with_confidence."""
    
    def test_predict_with_confidence_output(self, trained_classifier, sample_X):
        """Проверка структуры выходных данных."""
        result = predict_with_confidence(
            trained_classifier, sample_X, top_k=3, use_bootstrap=True
        )
        
        assert "predictions" in result
        assert "top_k" in result
        assert "model_type" in result
        assert "n_samples" in result
        
        assert isinstance(result["predictions"], pd.DataFrame)
        assert result["top_k"] == 3
        assert result["n_samples"] == len(sample_X)
    
    def test_predict_with_confidence_no_bootstrap(self, trained_classifier, sample_X):
        """Проверка без бутстрапа."""
        result = predict_with_confidence(
            trained_classifier, sample_X, top_k=3, use_bootstrap=False
        )
        
        assert result.get("bootstrap") is None or not result["bootstrap"]
        assert "predictions" in result
    
    def test_predict_with_confidence_metadata(self, trained_classifier, sample_X):
        """Проверка метаданных."""
        result = predict_with_confidence(
            trained_classifier, sample_X, top_k=5, use_bootstrap=True, n_bootstrap=20
        )
        
        assert result["top_k"] == 5
        assert result["n_bootstrap"] == 20
        assert result["bootstrap"] is True


# ============================================================================
# ТЕСТЫ: интеграционные
# ============================================================================

class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_pipeline(self, sample_X, sample_y_rank):
        """Проверка полного цикла: обучение -> прогноз."""
        # Обучение
        model = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(sample_X, sample_y_rank)
        
        # Прогноз
        result = predict_with_confidence(
            model, sample_X, top_k=3, use_bootstrap=True, n_bootstrap=10
        )
        
        # Проверка
        assert len(result["predictions"]) == len(sample_X)
        assert "predicted_rank" in result["predictions"].columns
        assert "top1_prob" in result["predictions"].columns
    
    def test_random_forest_model(self, sample_X, sample_y_rank):
        """Проверка с Random Forest."""
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        )
        model.fit(sample_X, sample_y_rank)
        
        result = predict_rank_topk(model, sample_X, top_k=3)
        
        assert len(result) == len(sample_X)
        assert result["top1_prob"].mean() > 0.1  # Разумная вероятность


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
