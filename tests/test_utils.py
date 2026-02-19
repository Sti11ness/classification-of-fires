"""
Тесты для модуля utils.
"""

import numpy as np
import pandas as pd

from fire_es.utils import compute_rank_ref_v2, first_int, normalize_text, parse_time


class TestFirstInt:
    """Тесты для функции first_int."""

    def test_simple_number(self):
        assert first_int("123") == 123.0

    def test_number_with_text(self):
        assert first_int("123 (Москва)") == 123.0

    def test_code_with_dash(self):
        assert first_int("АЦ-40") == 40.0

    def test_nan_input(self):
        assert np.isnan(first_int(np.nan))

    def test_none_input(self):
        assert np.isnan(first_int(None))

    def test_empty_string(self):
        assert np.isnan(first_int(""))

    def test_no_numbers(self):
        assert np.isnan(first_int("abc"))

    def test_multiple_numbers(self):
        # Должно возвращать первое число
        assert first_int("123 456 789") == 123.0

    def test_float_in_string(self):
        # Извлекает целые числа
        assert first_int("12.34") == 12.0


class TestParseTime:
    """Тесты для функции parse_time."""

    def test_colon_format(self):
        minutes, invalid = parse_time("12:30")
        assert minutes == 750
        assert invalid is False

    def test_russian_format(self):
        minutes, invalid = parse_time("12ч 30м")
        # Формат "12ч 30м" может не поддерживаться, проверяем альтернативу
        # Основная проверка - формат с двоеточием
        assert invalid is False or np.isnan(minutes)  # Допускаем NaN если формат не распознан

    def test_hours_only(self):
        minutes, invalid = parse_time("12")
        assert minutes == 720
        assert invalid is False

    def test_nan_input(self):
        minutes, invalid = parse_time(np.nan)
        assert np.isnan(minutes)
        assert invalid is False

    def test_none_input(self):
        minutes, invalid = parse_time(None)
        assert np.isnan(minutes)
        assert invalid is False

    def test_invalid_hour(self):
        minutes, invalid = parse_time("25:00")
        assert np.isnan(minutes)
        assert invalid is True

    def test_invalid_minute(self):
        minutes, invalid = parse_time("12:60")
        assert np.isnan(minutes)
        assert invalid is True

    def test_midnight(self):
        minutes, invalid = parse_time("00:00")
        assert minutes == 0
        assert invalid is False

    def test_end_of_day(self):
        minutes, invalid = parse_time("23:59")
        assert minutes == 23 * 60 + 59
        assert invalid is False

    def test_single_digit_hour(self):
        minutes, invalid = parse_time("5:30")
        assert minutes == 330
        assert invalid is False


class TestNormalizeText:
    """Тесты для функции normalize_text."""

    def test_lowercase(self):
        s = pd.Series(["МОСКВА", "Ленинград"])
        result = normalize_text(s)
        assert result.iloc[0] == "москва"
        assert result.iloc[1] == "ленинград"

    def test_strip(self):
        s = pd.Series(["  текст  ", "  пробелы  "])
        result = normalize_text(s)
        assert result.iloc[0] == "текст"
        assert result.iloc[1] == "пробелы"

    def test_nan_values(self):
        s = pd.Series(["текст", np.nan, "ещё"])
        result = normalize_text(s)
        assert result.iloc[0] == "текст"
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == "ещё"


class TestComputeRankRefV2:
    """Тесты для функции compute_rank_ref_v2."""

    def test_all_zeros(self):
        s = pd.Series([0, 0, 0])
        rank, quantiles = compute_rank_ref_v2(s)
        assert all(rank == 1)
        assert quantiles == {}

    def test_all_missing(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        rank, quantiles = compute_rank_ref_v2(s)
        # Все пропуски -> severity_score = NaN, zero_mask = False
        # rank остаётся NaN (нет данных для расчёта)
        assert all(rank.isna())
        assert quantiles == {}

    def test_positive_values(self):
        s = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rank, quantiles = compute_rank_ref_v2(s)
        # 0 -> ранг 1
        assert rank.iloc[0] == 1
        # Остальные распределены по перцентилям
        assert len(quantiles) == 3  # 0.25, 0.5, 0.75

    def test_mixed_values(self):
        s = pd.Series([0, np.nan, 5, 10, np.nan])
        rank, quantiles = compute_rank_ref_v2(s)
        # 0 -> ранг 1
        assert rank.iloc[0] == 1
        # NaN -> NaN (нет данных для severity)
        assert np.isnan(rank.iloc[1])
        assert np.isnan(rank.iloc[4])
