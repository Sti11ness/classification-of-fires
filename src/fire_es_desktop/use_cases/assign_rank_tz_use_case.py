# src/fire_es_desktop/use_cases/assign_rank_tz_use_case.py
"""
AssignRankTzUseCase — разметка ранга пожара по нормативам.

Согласно spec_first.md раздел 3.1 и spec_second.md раздел 11.4:
- Массовое вычисление rank_tz по нормативной таблице
- Сохранение в БД
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .base_use_case import BaseUseCase, UseCaseResult, UseCaseStatus

# Импорт из domain слоя
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fire_es"))

from fire_es.ranking import assign_rank_tz, calculate_rank_by_vector
from fire_es.db import DatabaseManager, Fire
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("AssignRankTzUseCase")


class AssignRankTzUseCase(BaseUseCase):
    """
    Сценарий разметки rank_tz.

    Шаги:
    1. Загрузка данных из БД
    2. Расчёт rank_tz по нормативам
    3. Статистика разметки
    4. Запись результатов в БД
    """

    def __init__(self, db_path: Path):
        super().__init__(
            name="AssignRankTz",
            description="Разметка ранга пожара по нормативам"
        )
        self.db_path = db_path

    def execute(
        self,
        use_vector: bool = True,
        batch_size: int = 500
    ) -> UseCaseResult:
        """
        Выполнить разметку rank_tz.

        Args:
            use_vector: Использовать векторную методику (или по количеству).
            batch_size: Размер пакета для обновления БД.

        Returns:
            Результат разметки.
        """
        self.status = UseCaseStatus.RUNNING
        self._cancel_requested = False
        warnings = []

        try:
            db = DatabaseManager(str(self.db_path))

            # Шаг 1: Загрузка данных из БД
            self.report_progress(1, 4, "Загрузка данных из БД")
            self.check_cancelled()

            # Получить все пожары без rank_tz или пересчитать все
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{self.db_path}")

            # Загрузить данные
            df = pd.read_sql(
                "SELECT id, equipment, equipment_count, nozzle_count, "
                "direct_damage, fatalities, injuries "
                "FROM fires",
                engine
            )

            if df.empty:
                db.close()
                return UseCaseResult(
                    success=False,
                    message="Нет данных в БД",
                    warnings=warnings
                )

            logger.info(f"Loaded {len(df)} fires for ranking")

            # Шаг 2: Расчёт rank_tz
            self.report_progress(2, 4, "Расчёт rank_tz по нормативам")
            self.check_cancelled()

            if use_vector:
                # Векторная методика (по типам техники)
                # Требует распарсенного поля equipment
                ranks = []
                for idx, row in df.iterrows():
                    self.check_cancelled()
                    rank = calculate_rank_by_vector(
                        equipment=row.get('equipment', ''),
                        equipment_count=row.get('equipment_count', 0)
                    )
                    ranks.append(rank)

                df['rank_tz'] = ranks
            else:
                # Упрощённая методика (по количеству)
                df['rank_tz'] = df['equipment_count'].apply(
                    lambda x: self._rank_by_count(x)
                )

            # Статистика разметки
            rank_distribution = df['rank_tz'].value_counts().to_dict()
            null_count = df['rank_tz'].isna().sum()

            if null_count > 0:
                warnings.append(
                    f"Не размечено {null_count} записей ({null_count/len(df)*100:.1f}%)"
                )

            logger.info(f"Rank distribution: {rank_distribution}")

            # Шаг 3: Запись в БД
            self.report_progress(3, 4, "Запись rank_tz в БД")
            self.check_cancelled()

            Session = sessionmaker(bind=engine)
            updated_count = 0

            with Session() as session:
                # Обновление пакетами
                for i in range(0, len(df), batch_size):
                    self.check_cancelled()

                    batch = df.iloc[i:i + batch_size]
                    for _, row in batch.iterrows():
                        session.query(Fire).filter(
                            Fire.id == row['id']
                        ).update({
                            'rank_tz': row['rank_tz']
                        })

                    session.commit()
                    updated_count += len(batch)
                    self.report_progress(
                        3 + (updated_count / len(df)) * 0.25,
                        4,
                        f"Обновлено {updated_count} из {len(df)}"
                    )

            db.close()

            # Шаг 4: Завершение
            self.report_progress(4, 4, "Разметка завершена")

            return UseCaseResult(
                success=True,
                message=f"Размечено {updated_count} записей",
                data={
                    "total_records": len(df),
                    "updated_records": updated_count,
                    "rank_distribution": rank_distribution,
                    "null_count": null_count
                },
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Rank assignment failed: {e}", exc_info=True)
            self.status = UseCaseStatus.FAILED

            return UseCaseResult(
                success=False,
                message=f"Ошибка разметки: {str(e)}",
                error=str(e),
                warnings=warnings
            )

    def _rank_by_count(self, count: int) -> str:
        """
        Упрощённое определение ранга по количеству техники.

        Args:
            count: Количество техники.

        Returns:
            Ранг (строка).
        """
        if pd.isna(count) or count is None:
            return None

        if count <= 0:
            return None
        elif count <= 2:
            return "1"
        elif count == 3:
            return "2"
        elif count <= 5:
            return "3"
        elif count <= 8:
            return "4"
        else:
            return "5"
