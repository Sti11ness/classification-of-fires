"""
Тесты для модуля db (БД SQLite).
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from fire_es.db import (
    Base, DatabaseManager, Fire, LPRDecision, Model, Normative,
    init_db, load_data_to_db
)


@pytest.fixture
def test_db_path(tmp_path):
    """Путь к тестовой БД."""
    return str(tmp_path / "test_fire_es.sqlite")


@pytest.fixture
def db(test_db_path):
    """Инициализированный менеджер БД."""
    db_manager = init_db(test_db_path)
    return db_manager


@pytest.fixture
def sample_fire_data():
    """Пример данных пожара для тестов."""
    from datetime import datetime
    
    return {
        "row_id": 1,
        "region_code": 171,
        "region_text": "Москва",
        "fire_date": datetime(2010, 5, 15),  # datetime объект, не строка
        "year": 2010,
        "month": 5,
        "building_floors": 5,
        "fire_floor": 3,
        "fatalities": 0,
        "injuries": 1,
        "direct_damage": 10000.0,
        "equipment_count": 2,
        "rank_tz": 1.5,
        "rank_distance": 0.0,
    }


class TestDatabaseManager:
    """Тесты для DatabaseManager."""
    
    def test_init(self, test_db_path):
        """Тест инициализации."""
        db = DatabaseManager(test_db_path)
        assert str(db.db_path) == test_db_path
        
    def test_create_tables(self, db):
        """Тест создания таблиц."""
        stats = db.get_stats()
        assert "fires_count" in stats
        assert "normatives_count" in stats
        
    def test_add_fire(self, db, sample_fire_data):
        """Тест добавления пожара."""
        fire = db.add_fire(sample_fire_data)
        assert fire.id is not None
        assert fire.row_id == 1
        assert fire.region_code == 171
        
    def test_get_fire(self, db, sample_fire_data):
        """Тест получения пожара по ID."""
        fire = db.add_fire(sample_fire_data)
        retrieved = db.get_fire(fire.id)
        assert retrieved is not None
        assert retrieved.row_id == 1
        
    def test_get_all_fires(self, db, sample_fire_data):
        """Тест получения всех пожаров."""
        db.add_fire(sample_fire_data)
        db.add_fire(sample_fire_data)
        fires = db.get_all_fires()
        assert len(fires) == 2
        
    def test_update_fire(self, db, sample_fire_data):
        """Тест обновления пожара."""
        fire = db.add_fire(sample_fire_data)
        db.update_fire(fire.id, {"fatalities": 2})
        updated = db.get_fire(fire.id)
        assert updated.fatalities == 2
        
    def test_delete_fire(self, db, sample_fire_data):
        """Тест удаления пожара."""
        fire = db.add_fire(sample_fire_data)
        result = db.delete_fire(fire.id)
        assert result is True
        assert db.get_fire(fire.id) is None
        
    def test_load_from_csv(self, db, tmp_path):
        """Тест загрузки из CSV."""
        # Создание тестового CSV
        csv_path = tmp_path / "test_fires.csv"
        df = pd.DataFrame([
            {"row_id": 1, "region_code": 171, "fatalities": 0, "rank_tz": 1.0},
            {"row_id": 2, "region_code": 172, "fatalities": 1, "rank_tz": 2.0},
        ])
        df.to_csv(csv_path, index=False)
        
        # Загрузка
        stats = db.load_from_csv(str(csv_path))
        assert stats["added"] == 2
        assert stats["skipped"] == 0
        
        # Проверка
        fires = db.get_all_fires()
        assert len(fires) == 2


class TestNormative:
    """Тесты для нормативной таблицы."""
    
    def test_add_normative(self, db):
        """Тест добавления норматива."""
        session = db.get_session()
        baseline_count = session.query(Normative).count()
        norm = Normative(rank=1, resource_type="TEST_AC", quantity=1)
        session.add(norm)
        session.commit()
        norms = session.query(Normative).all()
        assert len(norms) == baseline_count + 1
        assert any(item.resource_type == "TEST_AC" and item.rank == 1 for item in norms)
        session.close()


class TestLPRDecision:
    """Тесты для решений ЛПР."""
    
    def test_add_lpr_decision(self, db, sample_fire_data):
        """Тест добавления решения ЛПР."""
        # Добавляем пожар
        fire = db.add_fire(sample_fire_data)
        
        # Добавляем решение
        decision = db.add_lpr_decision(
            fire_id=fire.id,
            decision_rank=2.0,
            predicted_rank=1.5,
            predicted_probabilities=[0.3, 0.5, 0.2],
            comment="Тестовое решение",
            save_to_db=True,
        )
        
        assert decision.id is not None
        assert decision.decision_rank == 2.0
        assert decision.predicted_rank == 1.5
        
    def test_get_lpr_decisions(self, db, sample_fire_data):
        """Тест получения решений ЛПР."""
        fire = db.add_fire(sample_fire_data)
        
        db.add_lpr_decision(fire_id=fire.id, decision_rank=1.0)
        db.add_lpr_decision(fire_id=fire.id, decision_rank=2.0)
        
        decisions = db.get_lpr_decisions(fire_id=fire.id)
        assert len(decisions) == 2
        
    def test_get_lpr_decisions_for_training(self, db, sample_fire_data):
        """Тест получения решений для обучения."""
        fire = db.add_fire(sample_fire_data)
        
        db.add_lpr_decision(fire_id=fire.id, decision_rank=1.0, save_to_db=True)
        db.add_lpr_decision(fire_id=fire.id, decision_rank=2.0, save_to_db=False)
        
        training_decisions = db.get_lpr_decisions_for_training()
        assert len(training_decisions) == 1
        assert training_decisions[0].save_to_db is True


class TestModel:
    """Тесты для моделей."""
    
    def test_add_model(self, db):
        """Тест добавления модели."""
        model = db.add_model(
            model_type="rank",
            model_path="models/tree_rank.joblib",
            feature_set="managed",
            metrics={"accuracy": 0.85, "f1": 0.82},
            version="1.0",
        )
        
        assert model.id is not None
        assert model.model_type == "rank"
        assert model.is_active is False
        
    def test_set_model_active(self, db):
        """Тест активации модели."""
        model1 = db.add_model(model_type="rank", model_path="m1.joblib")
        model2 = db.add_model(model_type="rank", model_path="m2.joblib")
        
        db.set_model_active(model2.id)
        
        active = db.get_active_model("rank")
        assert active.id == model2.id
        
    def test_get_active_model(self, db):
        """Тест получения активной модели."""
        model1 = db.add_model(model_type="rank", model_path="m1.joblib")
        db.set_model_active(model1.id)
        
        active = db.get_active_model("rank")
        assert active is not None
        assert active.id == model1.id


class TestDatabaseStats:
    """Тесты статистики БД."""
    
    def test_get_stats(self, db, sample_fire_data):
        """Тест получения статистики."""
        db.add_fire(sample_fire_data)
        
        stats = db.get_stats()
        assert stats["fires_count"] == 1
        assert "db_path" in stats


class TestInitDb:
    """Тесты инициализации БД."""
    
    def test_init_db(self, test_db_path):
        """Тест функции init_db."""
        db = init_db(test_db_path)
        assert isinstance(db, DatabaseManager)
        assert db.db_path.exists()
        
    def test_load_data_to_db(self, test_db_path, tmp_path):
        """Тест функции load_data_to_db."""
        # Создание CSV
        csv_path = tmp_path / "test.csv"
        pd.DataFrame([{"row_id": 1, "rank_tz": 1.0}]).to_csv(csv_path, index=False)
        
        # Загрузка
        stats = load_data_to_db(str(csv_path), test_db_path)
        assert stats["added"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
