"""
Тесты для модуля ranking.
"""

import numpy as np
import pandas as pd

from fire_es.ranking import (
    assign_rank_tz,
    calculate_rank_by_count,
    calculate_rank_by_vector,
    get_rank_description,
    normalize_vector,
    validate_rank_distribution,
)


class TestNormalizeVector:
    """Тесты для функции normalize_vector."""

    def test_simple_normalization(self):
        vec = pd.Series({"AC": 2, "AL": 1})
        norm = normalize_vector(vec)
        assert norm["AC"] == 0.4  # 2/5
        assert norm["AL"] == 0.5  # 1/2

    def test_with_max_values(self):
        vec = pd.Series({"AC": 5, "AL": 2})
        max_vals = {"AC": 5, "AL": 2}
        norm = normalize_vector(vec, max_vals)
        assert norm["AC"] == 1.0
        assert norm["AL"] == 1.0

    def test_clip_to_one(self):
        vec = pd.Series({"AC": 10})
        max_vals = {"AC": 5}
        norm = normalize_vector(vec, max_vals)
        assert norm["AC"] == 1.0  # Клиппинг


class TestCalculateRankByCount:
    """Тесты для функции calculate_rank_by_count."""

    def test_rank_1(self):
        rank, dist = calculate_rank_by_count(1)
        assert rank == 1.0

    def test_rank_1bis(self):
        rank, dist = calculate_rank_by_count(2)
        assert rank == 1.5

    def test_rank_2(self):
        rank, dist = calculate_rank_by_count(3)
        assert rank == 2.0

    def test_rank_3(self):
        rank, dist = calculate_rank_by_count(4)
        assert rank == 3.0

    def test_rank_4(self):
        rank, dist = calculate_rank_by_count(5)
        assert rank == 4.0

    def test_rank_5(self):
        rank, dist = calculate_rank_by_count(6)
        assert rank == 5.0

    def test_nan_input(self):
        rank, dist = calculate_rank_by_count(np.nan)
        assert rank is None
        assert dist is None

    def test_zero_input(self):
        rank, dist = calculate_rank_by_count(0)
        assert rank is None
        assert dist is None


class TestCalculateRankByVector:
    """Тесты для функции calculate_rank_by_vector."""

    def test_exact_match(self):
        vec = pd.Series({"AC": 1})
        rank, dist = calculate_rank_by_vector(vec)
        assert rank == 1
        assert dist == 0.0

    def test_exact_match_rank_3(self):
        vec = pd.Series({"AC": 3, "AL": 1, "APS": 1})
        rank, dist = calculate_rank_by_vector(vec)
        assert rank == 3
        assert dist == 0.0

    def test_closest_match(self):
        vec = pd.Series({"AC": 2, "AL": 0})
        rank, dist = calculate_rank_by_vector(vec)
        # Должен быть ближе к рангу 1.5 или 2
        assert rank in [1.5, 2]


class TestAssignRankTz:
    """Тесты для функции assign_rank_tz."""

    def test_assign_rank_count_method(self):
        df = pd.DataFrame({"equipment_count": [1, 2, 3, 4, 5, 6]})
        result = assign_rank_tz(df, method="count")

        assert "rank_tz" in result.columns
        assert "rank_tz_count_proxy" in result.columns
        assert "rank_distance" in result.columns
        assert result["rank_tz"].tolist() == [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    def test_assign_rank_with_nan(self):
        df = pd.DataFrame({"equipment_count": [1, np.nan, 3]})
        result = assign_rank_tz(df, method="count")

        assert result["rank_tz"].iloc[0] == 1.0
        assert pd.isna(result["rank_tz"].iloc[1])
        assert result["rank_tz"].iloc[2] == 2.0

    def test_assign_rank_vector_missing_resources_sets_quality_flag(self):
        df = pd.DataFrame({"equipment": [None, "11, 23"]})
        result = assign_rank_tz(df, target_definition="vector")
        assert pd.isna(result["rank_tz"].iloc[0])
        assert "missing_or_unparsed_resources" in result["rank_quality_flags"].iloc[0]
        assert result["rank_tz"].iloc[1] == 2.0


class TestGetRankDescription:
    """Тесты для функции get_rank_description."""

    def test_rank_1(self):
        desc = get_rank_description(1.0)
        assert "1 единица" in desc

    def test_rank_1bis(self):
        desc = get_rank_description(1.5)
        assert "1-бис" in desc

    def test_rank_5(self):
        desc = get_rank_description(5.0)
        assert "5+" in desc


class TestValidateRankDistribution:
    """Тесты для функции validate_rank_distribution."""

    def test_valid_distribution(self):
        df = pd.DataFrame({"rank_tz": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.0]})
        result = validate_rank_distribution(df)

        assert result["total"] == 7
        assert 1.0 in result["distribution"]
        assert 5.0 in result["distribution"]

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = validate_rank_distribution(df)

        assert "error" in result
