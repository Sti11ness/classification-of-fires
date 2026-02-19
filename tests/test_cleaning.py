"""
Тесты для модуля cleaning.
"""

import pandas as pd
import pytest

from fire_es.cleaning import clean_fire_data, sheet_period


class TestSheetPeriod:
    """Тесты для функции sheet_period."""

    def test_period_2000_2008(self):
        assert sheet_period("1...2000-2008") == "2000-2008"

    def test_period_2009_2020(self):
        assert sheet_period("2...2009-2020") == "2009-2020"

    def test_period_combined(self):
        assert sheet_period("БД-1...2000--2020 (1+2)") == "2000-2020 (склеено)"

    def test_period_other(self):
        assert sheet_period("Табл.1") == "other"


class TestCleanFireData:
    """Тесты для функции clean_fire_data."""

    @pytest.fixture
    def sample_df(self):
        """Создание тестового DataFrame."""
        data = {
            "N п/п": [1, 2, 3, 4],
            "Субъекты РФ": ["171 (Москва)", "1105 (Приморский край)", "172 (СПб)", "1132 (Краснодар)"],
            "Дата возникновения пожара": ["2010-05-15", "2015-08-20", "1999-01-01", "2018-12-31"],
            "Вид нас. Пункта": ["1 (город)", "2 (село)", "1 (город)", "1 (город)"],
            "Вид пож. Охраны": ["1 (ФПС)", "2 (ДПС)", "1 (ФПС)", "1 (ФПС)"],
            "Категория риска объекта пожара": ["1", "2", "3", "1"],
            "Тип предприятия": ["11 (жилое)", "20 (склад)", "27 (офис)", "11 (жилое)"],
            "Класс ФПО": ["Ф1.1", "Ф5.2", "Ф4.3", "Ф1.3"],
            "Этажность здания": [5, 1, 10, 2],
            "Этаж на котором возник пожар": [3, 1, 5, 1],
            "Степень огнестойкости": ["II", "III", "II", "II"],
            "Изделие, устройство": ["электропроводка", "печь", "неизвестно", "кухня"],
            "Расстояние до пожарной части": [2, 5, 3, 1],
            "Погибло людей: Всего": [0, 1, 0, 0],
            "Получили травмы: Всего": [0, 0, 1, 0],
            "Прямой ущерб": [10000, 50000, 20000, 5000],
            "Спасено на пожаре людей": [0, 0, 0, 0],
            "Эвакуировано на пожаре людей": [10, 0, 5, 2],
            "Материальных ценностей": [100000, 50000, 200000, 50000],
            "Время обнаружения": ["10:30", "14:00", "08:15", "20:45"],
            "Время сообщения": ["10:35", "14:05", "08:20", "20:50"],
            "Время прибытия 1-го пож.подразд-ния": ["10:45", "14:20", "08:35", "21:00"],
            "Время подачи 1-го ствола": ["10:50", "14:25", "08:40", "21:05"],
            "Время локализации": ["11:30", "15:00", "09:20", "21:40"],
            "Время ликвидации": ["12:00", "15:30", "09:50", "22:10"],
            "Время ликвидации посл. пожара, час. мин.": ["12:00", "15:30", "09:50", "22:10"],
            "Техника": ["АЦ-40", "АЦ-40, АЛ-30", "АЦ-40", "АЦ-40"],
            "Количество техники": [1, 2, 1, 1],
            "Виды стволов": ["РС-50", "РС-70", "РС-50", "РС-50"],
            "Количество стволов": [1, 2, 1, 1],
            "Огнетушащие средства": ["вода", "вода, пена", "вода", "вода"],
            "Первичные средства": ["нет", "нет", "нет", "нет"],
            "Использование СИЗОД": ["да", "да", "нет", "нет"],
            "Водоисточники": ["водопровод", "водопровод", "водопровод", "водопровод"],
            "Вид АУП": ["нет", "нет", "нет", "нет"],
            "Наименование объекта": ["жилой дом", "склад", "офис", "жилой дом"],
            "Почтовый адрес": ["ул. Ленина 1", "ул. Мира 10", "пр. Мира 5", "ул. Кирова 3"],
        }
        return pd.DataFrame(data)

    def test_clean_data_shape(self, sample_df):
        """Проверка формы результата."""
        clean_df, issues = clean_fire_data(sample_df)
        # Количество строк не меняется
        assert len(clean_df) == len(sample_df)
        # Количество колонок увеличивается (добавляются флаги, _code, _min, и т.д.)
        assert len(clean_df.columns) > len(sample_df.columns)

    def test_date_validation(self, sample_df):
        """Проверка валидации дат (1999 год должен быть помечен)."""
        clean_df, issues = clean_fire_data(sample_df)
        # Третья строка (1999 год) должна быть помечена как outlier
        assert clean_df.iloc[2]["flag_date_outlier"]
        # Остальные строки - валидные
        assert not clean_df.iloc[0]["flag_date_outlier"]
        assert not clean_df.iloc[1]["flag_date_outlier"]

    def test_code_extraction(self, sample_df):
        """Проверка извлечения кодов."""
        clean_df, issues = clean_fire_data(sample_df)
        # region_code должно извлекаться
        assert clean_df.iloc[0]["region_code"] == 171.0
        assert clean_df.iloc[1]["region_code"] == 1105.0

    def test_time_parsing(self, sample_df):
        """Проверка парсинга времени."""
        clean_df, issues = clean_fire_data(sample_df)
        # t_detect_min должно быть в минутах
        assert clean_df.iloc[0]["t_detect_min"] == 10 * 60 + 30
        assert clean_df.iloc[1]["t_detect_min"] == 14 * 60

    def test_severity_score(self, sample_df):
        """Проверка расчёта severity_score."""
        clean_df, issues = clean_fire_data(sample_df)
        # severity_score должен быть рассчитан
        assert "severity_score" in clean_df.columns
        # Для строки с fatalities=1 severity должен быть выше
        assert clean_df.iloc[1]["severity_score"] > clean_df.iloc[0]["severity_score"]

    def test_floor_outlier_flag(self, sample_df):
        """Проверка флага выброса этажности."""
        # Добавим строку с этажом > 150
        sample_df.loc[0, "Этажность здания"] = 200
        clean_df, issues = clean_fire_data(sample_df)
        assert clean_df.iloc[0]["flag_floor_outlier"]

    def test_distance_outlier_flag(self, sample_df):
        """Проверка флага выброса расстояния."""
        # Добавим строку с расстоянием > 1000
        sample_df.loc[0, "Расстояние до пожарной части"] = 1500
        clean_df, issues = clean_fire_data(sample_df)
        assert clean_df.iloc[0]["flag_distance_outlier"]

    def test_floor_inconsistent_flag(self, sample_df):
        """Проверка флага несогласованности этажей."""
        # fire_floor > building_floors
        sample_df.loc[0, "Этажность здания"] = 2
        sample_df.loc[0, "Этаж на котором возник пожар"] = 5
        clean_df, issues = clean_fire_data(sample_df)
        assert clean_df.iloc[0]["flag_floor_inconsistent"]

    def test_quality_report(self, sample_df):
        """Проверка отчёта о качестве."""
        clean_df, issues = clean_fire_data(sample_df)
        # Отчёт должен содержать ключи
        assert "invalid_time_counts" in issues
        assert "floor_outliers" in issues
        assert "missing_outputs" in issues
