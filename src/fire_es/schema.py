"""
Схема данных: константы колонок, маппинг RU→EN, конфигурация.

Этот модуль содержит:
- RU_COLS: список колонок на русском языке (исходный Excel)
- EN_COLS: список колонок на английском (нормализованные)
- RU_TO_EN: словарь маппинга
- TIME_COLS: временные колонки
- Константы валидации (MAX_FLOORS, MAX_DISTANCE и др.)
"""

# Исходные колонки на русском (порядок важен для дедупликации)
RU_COLS = [
    "N п/п",
    "Субъекты РФ",
    "Дата возникновения пожара",
    "Вид нас. Пункта",
    "Вид пож. Охраны",
    "Категория риска объекта пожара",
    "Тип предприятия",
    "Класс ФПО",
    "Этажность здания",
    "Этаж на котором возник пожар",
    "Степень огнестойкости",
    "Изделие, устройство",
    "Расстояние до пожарной части",
    "Погибло людей: Всего",
    "Получили травмы: Всего",
    "Прямой ущерб",
    "Спасено на пожаре людей",
    "Эвакуировано на пожаре людей",
    "Материальных ценностей",
    "Время обнаружения",
    "Время сообщения",
    "Время прибытия 1-го пож.подразд-ния",
    "Время подачи 1-го ствола",
    "Время локализации",
    "Время ликвидации",
    "Время ликвидации посл. пожара, час. мин.",
    "Техника",
    "Количество техники",
    "Виды стволов",
    "Количество стволов",
    "Огнетушащие средства",
    "Первичные средства",
    "Использование СИЗОД",
    "Водоисточники",
    "Вид АУП",
    "Наименование объекта",
    "Почтовый адрес",
]

# Нормализованные колонки на английском
EN_COLS = [
    "row_id",
    "region",
    "fire_date",
    "settlement_type",
    "fire_protection",
    "risk_category",
    "enterprise_type",
    "fpo_class",
    "building_floors",
    "fire_floor",
    "fire_resistance",
    "source_item",
    "distance_to_station",
    "fatalities",
    "injuries",
    "direct_damage",
    "people_saved",
    "people_evacuated",
    "assets_saved",
    "t_detect",
    "t_report",
    "t_arrival",
    "t_first_hose",
    "t_contained",
    "t_extinguished",
    "t_final_extinguish",
    "equipment",
    "equipment_count",
    "nozzle_types",
    "nozzle_count",
    "extinguishing_agents",
    "initial_means",
    "respirators_use",
    "water_sources",
    "alarm_type",
    "object_name",
    "address",
]

# Словарь маппинга RU → EN
RU_TO_EN = dict(zip(RU_COLS, EN_COLS))

# Временные колонки (для парсинга)
TIME_COLS = [
    "t_detect",
    "t_report",
    "t_arrival",
    "t_first_hose",
    "t_contained",
    "t_extinguished",
    "t_final_extinguish",
]

# Колонки-коды (для извлечения кода из текста)
CODE_COLS = [
    "region",
    "settlement_type",
    "fire_protection",
    "risk_category",
    "enterprise_type",
    "fpo_class",
    "fire_resistance",
    "source_item",
    "equipment",
    "nozzle_types",
    "extinguishing_agents",
    "initial_means",
    "respirators_use",
    "water_sources",
    "alarm_type",
]

# Числовые колонки для валидации
NUM_COLS = [
    "building_floors",
    "fire_floor",
    "distance_to_station",
    "fatalities",
    "injuries",
    "direct_damage",
    "people_saved",
    "people_evacuated",
    "assets_saved",
    "equipment_count",
    "nozzle_count",
]

# Константы валидации
MAX_FLOORS = 150  # Максимальное допустимое количество этажей
MAX_DISTANCE = 1000  # Максимальное расстояние до пожарной части (км)
MIN_YEAR = 2000  # Минимальный год
MAX_YEAR = 2020  # Максимальный год

# Колонки для расчёта тяжести (severity)
SEVERITY_COLS = ["fatalities", "injuries", "direct_damage"]

# Колонки для определения пропусков выходов
OUTPUT_COLS = ["fatalities", "injuries", "direct_damage"]

# Факт-листы (префиксы для определения)
FACT_SHEET_PREFIXES = ["БД-1", "1...", "2..."]

# Периоды по имени листа
PERIOD_MAPPING = {
    "1...": "2000-2008",
    "2...": "2009-2020",
    "БД-1": "2000-2020 (склеено)",
}

# Номер строки заголовка для разных листов
HEADER_ROWS = {
    "БД-1": 3,  # Заголовок на строке 4 (0-indexed: 3)
    "default": 1,  # Заголовок на строке 2 (0-indexed: 1)
}
