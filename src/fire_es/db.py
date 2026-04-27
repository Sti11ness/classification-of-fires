"""
База данных (SQLite) для экспертной системы пожаров.

Модуль реализует:
- SQLAlchemy модели (Fire, Normative, LPRDecision, Model)
- CRUD операции для Fire
- Загрузку данных из CSV
- Сохранение решений ЛПР
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text, JSON,
    Table, create_engine, ForeignKey, UniqueConstraint, inspect, text
)
from sqlalchemy.orm import (
    Session, declarative_base, relationship, sessionmaker
)

from .normatives import (
    get_normative_rank_table,
    load_rank_resource_normatives,
)

# Базовый класс для моделей
Base = declarative_base()


# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

class Fire(Base):
    """
    Таблица пожаров (fires).
    
    Содержит очищенные и размеченные данные о пожарах.
    """
    __tablename__ = "fires"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Исходные данные
    row_id = Column(Integer, index=True)  # Исходный ID строки
    source_sheet = Column(String(100))  # Источник (лист Excel)
    source_period = Column(String(50))  # Период
    source_file = Column(String(500))  # Исходный файл

    # Раздел I: Общие сведения
    region_code = Column(Integer)  # Код региона
    region_text = Column(String(200))  # Текст региона
    fire_date = Column(DateTime)  # Дата пожара
    year = Column(Integer)  # Год
    month = Column(Integer)  # Месяц
    settlement_type_code = Column(Integer)  # Тип населённого пункта
    fire_protection_code = Column(Integer)  # Вид пожарной охраны
    
    # Раздел II: Объект пожара
    risk_category_code = Column(Integer)  # Категория риска
    enterprise_type_code = Column(Integer)  # Тип предприятия
    fpo_class_code = Column(Integer)  # Класс ФПО
    building_floors = Column(Integer)  # Этажность здания
    fire_floor = Column(Integer)  # Этаж пожара
    fire_resistance_code = Column(Integer)  # Степень огнестойкости
    source_item_code = Column(Integer)  # Источник зажигания
    distance_to_station = Column(Float)  # Расстояние до пожарной части
    object_name = Column(String(500))  # Наименование объекта
    address = Column(String(500))  # Адрес

    # Event identity / duplicate handling
    event_id = Column(String(128), index=True)
    event_fingerprint = Column(String(512))
    duplicate_group_id = Column(String(128), index=True)
    is_canonical_event_record = Column(Boolean, default=True)
    source_priority = Column(Integer, default=0)
    duplicate_policy = Column(String(64))
    event_id_low_confidence = Column(Boolean, default=False)
    
    # Раздел III: Последствия
    fatalities = Column(Integer)  # Погибло
    injuries = Column(Integer)  # Травмировано
    direct_damage = Column(Float)  # Прямой ущерб
    direct_damage_log = Column(Float)  # Логарифм ущерба
    
    # Раздел IV: Спасено
    people_saved = Column(Integer)  # Спасено людей
    people_evacuated = Column(Integer)  # Эвакуировано
    assets_saved = Column(Float)  # Спасено ценностей
    
    # Раздел V: Времена (в минутах)
    t_detect_min = Column(Float)  # Обнаружение
    t_report_min = Column(Float)  # Сообщение
    t_arrival_min = Column(Float)  # Прибытие
    t_first_hose_min = Column(Float)  # Первый ствол
    t_contained_min = Column(Float)  # Локализация
    t_extinguished_min = Column(Float)  # Ликвидация
    
    # Раздел VI: Ресурсы
    equipment = Column(String(500))  # Техника (текст)
    equipment_count = Column(Float)  # Количество техники
    nozzle_types = Column(String(500))  # Стволы
    nozzle_count = Column(Float)  # Количество стволов
    extinguishing_agents_code = Column(Integer)  # Средства тушения
    initial_means_code = Column(Integer)  # Первичные средства
    respirators_use_code = Column(Integer)  # СИЗОД
    water_sources_code = Column(Integer)  # Водоисточники
    alarm_type_code = Column(Integer)  # АУП
    
    # Ранг (ТЗ п.2.5.1)
    rank_tz = Column(Float)  # Ранг по ТЗ
    rank_distance = Column(Float)  # Расстояние до норматива
    rank_tz_vector = Column(Float)  # Canonical semantic target for mode 2.5.1
    rank_tz_count_proxy = Column(Float)  # Auxiliary legacy proxy label
    rank_ref = Column(Float)  # Исследовательский ранг (severity)
    severity_score = Column(Float)  # Severity score
    rank_label_source = Column(String(64))
    rank_normative_version = Column(String(64))
    rank_quality_flags = Column(Text)
    human_verified = Column(Boolean, default=False)
    usable_for_training = Column(Boolean, default=False)
    predicted_rank_at_decision = Column(Float)
    
    # Флаги качества
    flag_date_outlier = Column(Boolean, default=False)
    flag_floor_outlier = Column(Boolean, default=False)
    flag_distance_outlier = Column(Boolean, default=False)
    flag_floor_inconsistent = Column(Boolean, default=False)
    flag_negative_values = Column(Boolean, default=False)
    flag_damage_outlier = Column(Boolean, default=False)
    flag_time_invalid = Column(Boolean, default=False)
    flag_missing_outputs = Column(Boolean, default=False)
    
    # Метаданные
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Связи
    lpr_decisions = relationship("LPRDecision", back_populates="fire", cascade="all, delete-orphan")
    
    def to_dict(self) -> dict:
        """Преобразование в словарь."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


FIRES_HISTORICAL_TABLE = Fire.__table__.to_metadata(Base.metadata, name="fires_historical")
FIRES_LPR_TRAIN_TABLE = Fire.__table__.to_metadata(Base.metadata, name="fires_lpr_train")
FIRES_LPR_TRAIN_TABLE.append_column(Column("source_decision_id", Integer, unique=True, index=True))
FIRES_LPR_TRAIN_TABLE.append_column(Column("promoted_at", DateTime, default=datetime.utcnow))
FIRES_LPR_TRAIN_TABLE.append_column(Column("promoted_by", String(128)))
FIRES_LPR_TRAIN_TABLE.append_column(Column("train_enabled", Boolean, default=True))

TRAIN_SYNTHETIC_TABLE = Table(
    "train_synthetic",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("train_run_id", String(64), nullable=False, index=True),
    Column("row_number", Integer, nullable=False),
    Column("generator_method", String(64), nullable=False),
    Column("generator_params", Text),
    Column("base_source_scope", Text),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("source_feature_set", String(128)),
    Column("semantic_target", String(64)),
    Column("is_synthetic", Boolean, default=True),
    Column("target_class", Integer),
    Column("target_rank", Float),
    Column("features_json", Text, nullable=False),
)


class Normative(Base):
    """
    Таблица нормативов (normatives).
    
    Хранит нормативную таблицу ресурсов по рангам (ТЗ рис.3).
    """
    __tablename__ = "normatives"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    rank = Column(Float, nullable=False, index=True)  # Ранг (1, 1.5, 2, 3, 4, 5)
    resource_type = Column(String(50), nullable=False)  # Тип ресурса (AC, AL, APS...)
    quantity = Column(Integer, nullable=False)  # Количество

    description = Column(String(500))  # Описание
    label = Column(String(50))
    sort_order = Column(Integer, default=0)
    min_equipment_count = Column(Integer)
    normative_version = Column(String(64))
    is_active = Column(Boolean, default=True)  # Актуальность
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('rank', 'resource_type', name='uq_rank_resource'),
    )
    
    def to_dict(self) -> dict:
        """Преобразование в словарь."""
        return {
            "id": self.id,
            "rank": self.rank,
            "resource_type": self.resource_type,
            "quantity": self.quantity,
            "description": self.description,
            "label": self.label,
            "sort_order": self.sort_order,
            "min_equipment_count": self.min_equipment_count,
            "normative_version": self.normative_version,
            "is_active": self.is_active,
        }


class LPRDecision(Base):
    """
    Таблица решений ЛПР (lpr_decisions).
    
    Хранит решения, принятые ЛПР (не обязательно совпадающие с прогнозом).
    """
    __tablename__ = "lpr_decisions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    fire_id = Column(Integer, ForeignKey("fires.id"), nullable=False, index=True)
    
    # Решение ЛПР
    decision_rank = Column(Float)  # Выбранный ранг
    decision_resources = Column(JSON)  # Выбранные ресурсы (JSON)
    
    # Прогноз модели (на момент решения)
    predicted_rank = Column(Float)  # Прогнозируемый ранг
    predicted_probabilities = Column(JSON)  # Вероятности (top-K)
    
    # Метаданные
    comment = Column(Text)  # Комментарий ЛПР
    save_to_db = Column(Boolean, default=True)  # Сохранять ли для обучения
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Связи
    fire = relationship("Fire", back_populates="lpr_decisions")
    
    def to_dict(self) -> dict:
        """Преобразование в словарь."""
        return {
            "id": self.id,
            "fire_id": self.fire_id,
            "decision_rank": self.decision_rank,
            "decision_resources": self.decision_resources,
            "predicted_rank": self.predicted_rank,
            "predicted_probabilities": self.predicted_probabilities,
            "comment": self.comment,
            "save_to_db": self.save_to_db,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Model(Base):
    """
    Таблица моделей (models).
    
    Хранит метаданные обученных моделей деревьев решений.
    """
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    model_type = Column(String(50), nullable=False)  # 'rank' или 'resource'
    model_path = Column(String(500), nullable=False)  # Путь к файлу модели
    
    # Параметры обучения
    feature_set = Column(String(200))  # Набор признаков
    hyperparams = Column(JSON)  # Гиперпараметры
    
    # Метрики
    metrics = Column(JSON)  # Метрики качества (accuracy, F1, MAE...)
    
    # Версия
    version = Column(String(50))  # Версия модели
    is_active = Column(Boolean, default=False)  # Активная модель
    
    # Данные для обучения
    train_size = Column(Integer)  # Размер обучающей выборки
    test_size = Column(Integer)  # Размер тестовой выборки
    includes_lpr_decisions = Column(Boolean, default=False)  # Включает решения ЛПР
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Преобразование в словарь."""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "feature_set": self.feature_set,
            "hyperparams": self.hyperparams,
            "metrics": self.metrics,
            "version": self.version,
            "is_active": self.is_active,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "includes_lpr_decisions": self.includes_lpr_decisions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# УПРАВЛЕНИЕ БД
# ============================================================================

class DatabaseManager:
    """
    Менеджер базы данных.
    
    Реализует:
    - Создание/подключение к БД
    - CRUD операции для Fire
    - Загрузку данных из CSV
    - Сохранение решений ЛПР
    """
    
    def __init__(self, db_path: str = "fire_es.sqlite"):
        """
        Инициализация менеджера БД.
        
        Args:
            db_path: путь к файлу SQLite БД
        """
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Создание всех таблиц."""
        Base.metadata.create_all(bind=self.engine)
        self._ensure_schema_migrations()
        self._seed_normatives_if_empty()
        
    def drop_tables(self):
        """Удаление всех таблиц."""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """Получение сессии."""
        return self.SessionLocal()

    def close(self) -> None:
        """Закрыть подключения SQLAlchemy engine."""
        self.engine.dispose()

    def _ensure_schema_migrations(self) -> None:
        """Apply additive schema migrations required by the desktop runtime."""
        fire_columns = {
            "source_file": "TEXT",
            "event_id": "TEXT",
            "event_fingerprint": "TEXT",
            "duplicate_group_id": "TEXT",
            "is_canonical_event_record": "BOOLEAN DEFAULT 1",
            "source_priority": "INTEGER DEFAULT 0",
            "duplicate_policy": "TEXT",
            "event_id_low_confidence": "BOOLEAN DEFAULT 0",
            "rank_tz_vector": "REAL",
            "rank_tz_count_proxy": "REAL",
            "rank_label_source": "TEXT",
            "rank_normative_version": "TEXT",
            "rank_quality_flags": "TEXT",
            "human_verified": "BOOLEAN DEFAULT 0",
            "usable_for_training": "BOOLEAN DEFAULT 0",
            "predicted_rank_at_decision": "REAL",
        }
        norm_columns = {
            "label": "TEXT",
            "sort_order": "INTEGER DEFAULT 0",
            "min_equipment_count": "INTEGER",
            "normative_version": "TEXT",
        }
        self._ensure_columns("fires", fire_columns)
        self._ensure_columns("normatives", norm_columns)
        with self.engine.begin() as conn:
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_fires_event_id ON fires(event_id)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_fires_duplicate_group_id ON fires(duplicate_group_id)")
            )

    def _ensure_columns(self, table_name: str, columns: dict[str, str]) -> None:
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            return
        existing = {column["name"] for column in inspector.get_columns(table_name)}
        with self.engine.begin() as conn:
            for name, ddl in columns.items():
                if name in existing:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {name} {ddl}"))

    def _seed_normatives_if_empty(self) -> None:
        """Load canonical normatives into SQLite when the table is empty."""
        with self.engine.begin() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM normatives")).scalar() or 0
            if total:
                return
            payload = load_rank_resource_normatives()
            rank_table = get_normative_rank_table(payload)
            rows = []
            for _, row in rank_table.iterrows():
                resource_vector = row["resource_vector"] or {}
                for resource_type, quantity in resource_vector.items():
                    rows.append(
                        {
                            "rank": float(row["rank"]),
                            "resource_type": str(resource_type),
                            "quantity": int(quantity),
                            "description": row["description"],
                            "label": row["label"],
                            "sort_order": int(row["sort_order"]),
                            "min_equipment_count": (
                                int(row["min_equipment_count"])
                                if row["min_equipment_count"] is not None
                                else None
                            ),
                            "normative_version": payload["normative_version"],
                            "is_active": True,
                            "created_at": datetime.utcnow(),
                        }
                    )
            if rows:
                conn.execute(
                    text(
                        "INSERT INTO normatives (rank, resource_type, quantity, description, label, "
                        "sort_order, min_equipment_count, normative_version, is_active, created_at) "
                        "VALUES (:rank, :resource_type, :quantity, :description, :label, :sort_order, "
                        ":min_equipment_count, :normative_version, :is_active, :created_at)"
                    ),
                    rows,
                )
    
    # ========================================================================
    # CRUD для Fire
    # ========================================================================
    
    def add_fire(self, fire_data: dict) -> Fire:
        """
        Добавление записи о пожаре.
        
        Args:
            fire_data: словарь с данными пожара
            
        Returns:
            Добавленная запись Fire
        """
        session = self.get_session()
        try:
            fire = Fire(**fire_data)
            session.add(fire)
            session.commit()
            session.refresh(fire)
            return fire
        finally:
            session.close()
    
    def get_fire(self, fire_id: int) -> Optional[Fire]:
        """Получение записи о пожаре по ID."""
        session = self.get_session()
        try:
            return session.query(Fire).filter(Fire.id == fire_id).first()
        finally:
            session.close()
    
    def get_all_fires(self) -> list[Fire]:
        """Получение всех записей о пожарах."""
        session = self.get_session()
        try:
            return session.query(Fire).all()
        finally:
            session.close()
    
    def update_fire(self, fire_id: int, update_data: dict) -> Optional[Fire]:
        """Обновление записи о пожаре."""
        session = self.get_session()
        try:
            fire = session.query(Fire).filter(Fire.id == fire_id).first()
            if fire:
                for key, value in update_data.items():
                    setattr(fire, key, value)
                session.commit()
                session.refresh(fire)
            return fire
        finally:
            session.close()
    
    def delete_fire(self, fire_id: int) -> bool:
        """Удаление записи о пожаре."""
        session = self.get_session()
        try:
            fire = session.query(Fire).filter(Fire.id == fire_id).first()
            if fire:
                session.delete(fire)
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    # ========================================================================
    # Загрузка из CSV
    # ========================================================================
    
    def load_from_csv(self, csv_path: str, batch_size: int = 100) -> dict:
        """
        Загрузка данных из CSV файла.
        
        Args:
            csv_path: путь к CSV файлу
            batch_size: размер пакета для коммита
            
        Returns:
            Статистика загрузки
        """
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        session = self.get_session()
        try:
            added = 0
            skipped = 0
            
            for i, row in df.iterrows():
                try:
                    # Преобразование NaN в None
                    data = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                    
                    # Пропуск служебных колонок
                    skip_cols = ['source_sheet', 'source_period', 'dup_flag', 'period_group']
                    for col in skip_cols:
                        data.pop(col, None)
                    
                    fire = Fire(**data)
                    session.add(fire)
                    added += 1
                    
                    if added % batch_size == 0:
                        session.commit()
                        
                except Exception as e:
                    skipped += 1
                    print(f"  Пропущена строка {i}: {e}")
            
            session.commit()
            
            return {
                "added": added,
                "skipped": skipped,
                "total": len(df),
            }
        finally:
            session.close()
    
    # ========================================================================
    # Решения ЛПР
    # ========================================================================
    
    def add_lpr_decision(
        self,
        fire_id: int,
        decision_rank: Optional[float] = None,
        decision_resources: Optional[dict] = None,
        predicted_rank: Optional[float] = None,
        predicted_probabilities: Optional[list] = None,
        comment: Optional[str] = None,
        save_to_db: bool = True,
    ) -> LPRDecision:
        """
        Добавление решения ЛПР.
        
        Args:
            fire_id: ID пожара
            decision_rank: выбранный ранг
            decision_resources: выбранные ресурсы
            predicted_rank: прогнозируемый ранг
            predicted_probabilities: вероятности прогноза
            comment: комментарий
            save_to_db: сохранять ли для обучения
            
        Returns:
            Добавленная запись LPRDecision
        """
        session = self.get_session()
        try:
            decision = LPRDecision(
                fire_id=fire_id,
                decision_rank=decision_rank,
                decision_resources=decision_resources,
                predicted_rank=predicted_rank,
                predicted_probabilities=predicted_probabilities,
                comment=comment,
                save_to_db=save_to_db,
            )
            session.add(decision)
            session.commit()
            session.refresh(decision)
            return decision
        finally:
            session.close()
    
    def get_lpr_decisions(self, fire_id: Optional[int] = None) -> list[LPRDecision]:
        """Получение решений ЛПР."""
        session = self.get_session()
        try:
            query = session.query(LPRDecision)
            if fire_id:
                query = query.filter(LPRDecision.fire_id == fire_id)
            return query.all()
        finally:
            session.close()
    
    def get_lpr_decisions_for_training(self) -> list[LPRDecision]:
        """Получение решений ЛПР для обучения (save_to_db=True)."""
        session = self.get_session()
        try:
            return session.query(LPRDecision).filter(
                LPRDecision.save_to_db == True
            ).all()
        finally:
            session.close()
    
    # ========================================================================
    # Модели
    # ========================================================================
    
    def add_model(
        self,
        model_type: str,
        model_path: str,
        feature_set: Optional[str] = None,
        hyperparams: Optional[dict] = None,
        metrics: Optional[dict] = None,
        version: Optional[str] = None,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        includes_lpr_decisions: bool = False,
    ) -> Model:
        """Добавление записи о модели."""
        session = self.get_session()
        try:
            model = Model(
                model_type=model_type,
                model_path=model_path,
                feature_set=feature_set,
                hyperparams=hyperparams,
                metrics=metrics,
                version=version,
                train_size=train_size,
                test_size=test_size,
                includes_lpr_decisions=includes_lpr_decisions,
            )
            session.add(model)
            session.commit()
            session.refresh(model)
            return model
        finally:
            session.close()
    
    def get_active_model(self, model_type: str) -> Optional[Model]:
        """Получение активной модели по типу."""
        session = self.get_session()
        try:
            return session.query(Model).filter(
                Model.model_type == model_type,
                Model.is_active == True
            ).first()
        finally:
            session.close()
    
    def set_model_active(self, model_id: int):
        """Установка модели как активной."""
        session = self.get_session()
        try:
            # Сначала деактивируем все модели этого типа
            model_type = session.query(Model).filter(Model.id == model_id).first()
            if model_type:
                session.query(Model).filter(
                    Model.model_type == model_type.model_type
                ).update({"is_active": False})
            
            # Активируем нужную
            session.query(Model).filter(Model.id == model_id).update({"is_active": True})
            session.commit()
        finally:
            session.close()
    
    # ========================================================================
    # Статистика
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Получение статистики БД."""
        session = self.get_session()
        try:
            return {
                "fires_count": session.query(Fire).count(),
                "normatives_count": session.query(Normative).count(),
                "lpr_decisions_count": session.query(LPRDecision).count(),
                "models_count": session.query(Model).count(),
                "db_path": str(self.db_path),
            }
        finally:
            session.close()


# ============================================================================
# УДОБНЫЕ ФУНКЦИИ
# ============================================================================

def init_db(db_path: str = "fire_es.sqlite") -> DatabaseManager:
    """
    Инициализация БД (создание таблиц).
    
    Args:
        db_path: путь к файлу БД
        
    Returns:
        DatabaseManager
    """
    db = DatabaseManager(db_path)
    db.create_tables()
    return db


def load_data_to_db(
    csv_path: str,
    db_path: str = "fire_es.sqlite"
) -> dict:
    """
    Загрузка данных из CSV в БД.
    
    Args:
        csv_path: путь к CSV файлу
        db_path: путь к файлу БД
        
    Returns:
        Статистика загрузки
    """
    db = init_db(db_path)
    return db.load_from_csv(csv_path)


if __name__ == "__main__":
    # Тестирование
    print("Тестирование модуля db...")
    
    # Создание тестовой БД
    db = init_db("test_fire_es.sqlite")
    
    # Статистика
    stats = db.get_stats()
    print(f"Статистика: {stats}")
    
    # Добавление норматива
    norm = Normative(rank=1, resource_type="AC", quantity=1, description="Автоцистерна")
    session = db.get_session()
    session.add(norm)
    session.commit()
    session.close()
    
    print("✅ Тестирование завершено")
    print("  Тестовая БД: test_fire_es.sqlite")
