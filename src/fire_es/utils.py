"""
Утилиты: парсинг времени, извлечение чисел, нормализация текста.

Функции:
- norm_col: нормализация имени колонки
- map_col: маппинг колонки на каноническое имя
- first_int: извлечение первого целого числа из строки
- parse_time: парсинг времени (часы:минуты) в минуты
- normalize_text: нормализация текста (lowercase, strip)
- compute_rank_ref_v2: вычисление ранга по severity (исследовательский)
"""

import re
from typing import Any, Optional

import numpy as np
import pandas as pd


def norm_col(s: str) -> str:
    """
    Нормализация имени колонки: удаление лишних пробелов,
    замена битых символов.
    """
    s = str(s).replace("\ufffd", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def map_col(col: str) -> Optional[str]:
    """
    Маппинг колонки на каноническое русское имя.

    Возвращает None, если колонка не распознана.
    """
    c = norm_col(col).lower().replace("ё", "е")

    if c in ("n п/п", "n п/п."):
        return "N п/п"
    if "код региона" in c or "субъект" in c:
        return "Субъекты РФ"
    if c.startswith("дата возник"):
        return "Дата возникновения пожара"
    if "вид нас" in c:
        return "Вид нас. Пункта"
    if "вид пож" in c or "пож. охран" in c:
        return "Вид пож. Охраны"
    if "категория риска" in c:
        return "Категория риска объекта пожара"
    if "тип предпр" in c:
        return "Тип предприятия"
    if "класс фпо" in c:
        return "Класс ФПО"
    if "этажност" in c:
        return "Этажность здания"
    if "этаж" in c and "возник" in c:
        return "Этаж на котором возник пожар"
    if "огнестойк" in c:
        return "Степень огнестойкости"
    if "изделие" in c:
        return "Изделие, устройство"
    if ("расст" in c and "част" in c) or "пож.-спас" in c:
        return "Расстояние до пожарной части"
    if "погибло" in c:
        return "Погибло людей: Всего"
    if "травмы" in c:
        return "Получили травмы: Всего"
    if "прямой ущерб" in c:
        return "Прямой ущерб"
    if "спасено" in c and "люд" in c:
        return "Спасено на пожаре людей"
    if "эвакуир" in c:
        return "Эвакуировано на пожаре людей"
    if "материальных ценност" in c:
        return "Материальных ценностей"
    if "обнаруж" in c:
        return "Время обнаружения"
    if "сообщ" in c:
        return "Время сообщения"
    if "прибытия" in c:
        return "Время прибытия 1-го пож.подразд-ния"
    if "подачи" in c and "ствол" in c:
        return "Время подачи 1-го ствола"
    if "локализ" in c:
        return "Время локализации"
    if "ликвидации" in c and "посл" in c:
        return "Время ликвидации посл. пожара, час. мин."
    if "ликвидации" in c:
        return "Время ликвидации"
    if c.startswith("техник"):
        return "Техника"
    if "количество техник" in c or (c.startswith("количество") and "ствол" not in c):
        return "Количество техники"
    if "ствол" in c and "количество" not in c:
        return "Виды стволов"
    if "количество ствол" in c or "количество .1" in c:
        return "Количество стволов"
    if "огнетуш" in c:
        return "Огнетушащие средства"
    if "первич" in c:
        return "Первичные средства"
    if "сизод" in c:
        return "Использование СИЗОД"
    if "водоснабж" in c or "водоисточ" in c:
        return "Водоисточники"
    if "ауп" in c or "пожарной автоматик" in c:
        return "Вид АУП"
    if "наименован" in c:
        return "Наименование объекта"
    if "почтовый адрес" in c:
        return "Почтовый адрес"

    return None


def first_int(val: Any) -> float:
    """
    Извлечение первого целого числа из строки.

    Примеры:
    - "123 (Москва)" -> 123.0
    - "АЦ-40" -> 40.0
    - NaN -> NaN
    - "" -> NaN
    """
    if pd.isna(val):
        return np.nan
    nums = re.findall(r"\d+", str(val))
    return float(nums[0]) if nums else np.nan


def parse_time(val: Any) -> tuple[float, bool]:
    """
    Парсинг времени в минуты от начала суток.

    Поддерживаемые форматы:
    - "12:30" -> 750, False
    - "12ч 30м" -> 750, False
    - "12" (часы) -> 720, False
    - datetime -> минуты, False
    - Некорректное время -> NaN, True

    Возвращает кортеж: (минуты, флаг_некорректности)
    """
    if pd.isna(val):
        return (np.nan, False)

    s = str(val).strip()

    # Формат: часы:минуты или часыч минутым
    m = re.search(r"(?P<h>\d{1,2})\D+(?P<m>\d{1,2})$", s)
    if m:
        h = int(m.group("h"))
        mi = int(m.group("m"))
        invalid = h >= 24 or mi >= 60
        return ((np.nan if invalid else h * 60 + mi), invalid)

    # Только часы (одно-два числа)
    if re.fullmatch(r"\d{1,2}", s):
        h = int(s)
        invalid = h >= 24
        return ((np.nan if invalid else h * 60), invalid)

    # Попытка парсинга как datetime
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        h, mi = dt.hour, dt.minute
        # Если год не 1899/1900, это не чистое время
        invalid = dt.year not in (1899, 1900)
        invalid = invalid or h >= 24 or mi >= 60
        return ((np.nan if invalid else h * 60 + mi), invalid)

    # Не распознано
    return (np.nan, True)


def normalize_text(series: pd.Series) -> pd.Series:
    """
    Нормализация текста: lowercase, strip, NaN для пропусков.
    """
    s = series.astype(str).str.strip().str.lower()
    s = s.where(series.notna(), np.nan)
    return s


def compute_rank_ref_v2(
    severity_series: pd.Series,
) -> tuple[pd.Series, dict]:
    """
    Вычисление исследовательского ранга по severity score.

    Severity = fatalities*5 + injuries*2 + log1p(direct_damage)

    Ранг:
    - 1: все нули/пропуски
    - 2: <= 25-го перцентиля
    - 3: <= 50-го перцентиля
    - 4: <= 75-го перцентиля
    - 5: > 75-го перцентиля

    Возвращает:
    - rank: Series с рангами
    - quantiles: dict с перцентилями (для отчёта)
    """
    s = severity_series.copy()
    rank = pd.Series(np.nan, index=s.index)

    # Нулевые значения -> ранг 1
    zero_mask = s.fillna(np.nan).eq(0)
    rank.loc[zero_mask] = 1

    # Положительные значения
    pos = s[s > 0].dropna()

    if len(pos) > 0:
        q = pos.quantile([0.25, 0.50, 0.75]).to_dict()

        def assign_pos(val: float) -> int:
            if val <= q[0.25]:
                return 2
            if val <= q[0.50]:
                return 3
            if val <= q[0.75]:
                return 4
            return 5

        rank.loc[pos.index] = pos.apply(assign_pos)
        return rank, q

    return rank, {}
