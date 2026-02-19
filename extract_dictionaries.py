"""Извлечение справочников из Excel и сохранение в CSV."""
import pandas as pd
import os

xl = pd.ExcelFile('БД-1_(МЧС_ТЭП 2000-2020_v.20.1_.xlsx')

# Создадим директорию для справочников
os.makedirs('data/normative', exist_ok=True)

def extract_table(table_name, output_file, key_col='Unnamed: 2'):
    """Извлечение таблицы и сохранение."""
    df = xl.parse(table_name)
    df_clean = df.iloc[2:].dropna(subset=[key_col])
    df_clean.columns = ['code', 'name', 'id']
    df_clean = df_clean[['code', 'name']]
    df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{table_name} сохранена в {output_file}")
    return df_clean

# Табл.19 - Вид техники
print("=== Табл.19 (техника) ===")
df19 = extract_table('Табл.19', 'data/normative/equipment_types.csv')
print(df19.head(10))

# Табл.20 - Виды стволов
print("\n=== Табл.20 (стволы) ===")
df20 = extract_table('Табл.20', 'data/normative/nozzle_types.csv')

# Табл.21 - Огнетушащие средства
print("\n=== Табл.21 (Огнетушащие средства) ===")
df21 = extract_table('Табл.21', 'data/normative/extinguishing_agents.csv')

# Табл.23 - СИЗОД
print("\n=== Табл.23 (СИЗОД) ===")
df23 = extract_table('Табл.23', 'data/normative/respirators.csv')

# Табл.24 - Водоисточники
print("\n=== Табл.24 (Водоисточники) ===")
df24 = extract_table('Табл.24', 'data/normative/water_sources.csv')

# Табл.25 - АУП
print("\n=== Табл.25 (АУП) ===")
df25 = extract_table('Табл.25', 'data/normative/alarm_types.csv')

print("\n✅ Все справочники сохранены в data/normative/")
