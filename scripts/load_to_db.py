"""
Загрузка данных в БД SQLite.

Использование:
    python scripts/load_to_db.py [--csv PATH] [--db PATH]
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from fire_es.db import init_db, DatabaseManager, Fire


def main():
    parser = argparse.ArgumentParser(description="Загрузка данных в БД SQLite")
    parser.add_argument(
        "--csv",
        default="clean_df_with_rank.csv",
        help="Путь к CSV файлу (по умолчанию: clean_df_with_rank.csv)"
    )
    parser.add_argument(
        "--db",
        default="fire_es.sqlite",
        help="Путь к БД SQLite (по умолчанию: fire_es.sqlite)"
    )
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    db_path = Path(args.db)
    
    if not csv_path.exists():
        print(f"❌ CSV файл не найден: {csv_path}")
        return
    
    print(f"📂 CSV файл: {csv_path}")
    print(f"💾 БД: {db_path}")
    print()
    
    # Инициализация БД
    print("🔧 Инициализация БД...")
    db = init_db(str(db_path))
    
    # Загрузка данных
    print("📥 Загрузка данных из CSV...")
    
    # Получаем список колонок модели Fire
    fire_columns = [c.name for c in Fire.__table__.columns]
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Фильтруем только нужные колонки
    cols_to_load = [c for c in df.columns if c in fire_columns]
    print(f"  Колонки для загрузки: {len(cols_to_load)} из {len(df.columns)}")
    
    df_filtered = df[cols_to_load].copy()
    
    # Преобразуем fire_date в datetime
    if 'fire_date' in df_filtered.columns:
        df_filtered['fire_date'] = pd.to_datetime(df_filtered['fire_date'], errors='coerce')
    
    added = 0
    skipped = 0
    
    session = db.get_session()
    try:
        for i, row in df_filtered.iterrows():
            try:
                data = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                fire = Fire(**data)
                session.add(fire)
                added += 1
                
                if added % 500 == 0:
                    session.commit()
                    print(f"  Загружено {added} записей...")
                    
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  Пропущена строка {i}: {e}")
        
        session.commit()
        
    finally:
        session.close()
    
    print()
    print("📊 Статистика загрузки:")
    print(f"  ✅ Добавлено: {added}")
    print(f"  ⚠️  Пропущено: {skipped}")
    print(f"  📄 Всего: {len(df_filtered)}")
    
    # Статистика БД
    print()
    print("📊 Статистика БД:")
    db_stats = db.get_stats()
    print(f"  🔥 Пожаров: {db_stats['fires_count']}")
    print(f"  📋 Нормативов: {db_stats['normatives_count']}")
    print(f"  💡 Решений ЛПР: {db_stats['lpr_decisions_count']}")
    print(f"  🤖 Моделей: {db_stats['models_count']}")
    
    print()
    print(f"✅ Загрузка завершена! БД: {db_path}")


if __name__ == "__main__":
    main()
