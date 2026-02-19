# src/fire_es_desktop/workspace/workspace_manager.py
"""
Workspace Manager — управление рабочими папками проекта.

Реализует контракт Workspace согласно spec_second.md:
- fire_es.sqlite
- config.json
- reports/
- logs/
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

from fire_es.db import DatabaseManager


class WorkspaceManager:
    def __init__(self):
        self.current_workspace: Optional[Path] = None
        self.logger = logging.getLogger("WorkspaceManager")
    
    def create_workspace(self, path: Path) -> bool:
        """Создать новую рабочую папку."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Создать структуру
            (path / "reports" / "models").mkdir(parents=True, exist_ok=True)
            (path / "reports" / "figs").mkdir(parents=True, exist_ok=True)
            (path / "reports" / "tables").mkdir(parents=True, exist_ok=True)
            (path / "logs").mkdir(parents=True, exist_ok=True)
            
            # Создать конфиг
            config = {
                "active_model_id": None,
                "last_updated": None,
                "created_at": str(path.stat().st_ctime),
                "version": "0.1.0"
            }
            
            with open(path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Инициализировать БД и создать таблицы.
            db_path = path / "fire_es.sqlite"
            if not self._ensure_db_schema(db_path):
                return False
            
            self.current_workspace = path
            self.logger.info(f"Workspace created at: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create workspace: {e}")
            return False
    
    def open_workspace(self, path: Path) -> bool:
        """Открыть существующую рабочую папку."""
        if not path.exists():
            self.logger.error(f"Workspace path does not exist: {path}")
            return False
        
        required_files = ["config.json", "fire_es.sqlite"]
        for file in required_files:
            if not (path / file).exists():
                self.logger.error(f"Required file missing: {file} in {path}")
                return False

        # Для старых/поврежденных workspace: гарантируем наличие схемы.
        db_path = path / "fire_es.sqlite"
        if not self._ensure_db_schema(db_path):
            return False

        self.current_workspace = path
        self.logger.info(f"Workspace opened: {path}")
        return True
    
    def get_current_workspace(self) -> Optional[Path]:
        """Получить текущую рабочую папку."""
        return self.current_workspace
    
    def get_db_path(self) -> Optional[Path]:
        """Получить путь к БД."""
        if self.current_workspace:
            return self.current_workspace / "fire_es.sqlite"
        return None
    
    def get_config_path(self) -> Optional[Path]:
        """Получить путь к конфигу."""
        if self.current_workspace:
            return self.current_workspace / "config.json"
        return None
    
    def get_reports_path(self) -> Optional[Path]:
        """Получить путь к reports."""
        if self.current_workspace:
            return self.current_workspace / "reports"
        return None
    
    def get_logs_path(self) -> Optional[Path]:
        """Получить путь к logs."""
        if self.current_workspace:
            return self.current_workspace / "logs"
        return None
    
    def validate_workspace(self) -> tuple[bool, str]:
        """Проверить целостность Workspace."""
        if not self.current_workspace:
            return False, "No workspace opened"
        
        if not self.current_workspace.exists():
            return False, "Workspace directory does not exist"
        
        # Проверка обязательных файлов
        if not (self.current_workspace / "fire_es.sqlite").exists():
            return False, "fire_es.sqlite not found"
        
        if not (self.current_workspace / "config.json").exists():
            return False, "config.json not found"
        
        # Проверка структуры
        reports = self.current_workspace / "reports"
        if not reports.exists():
            return False, "reports directory not found"
        
        logs = self.current_workspace / "logs"
        if not logs.exists():
            return False, "logs directory not found"

        db_path = self.current_workspace / "fire_es.sqlite"
        if not self._has_table(db_path, "fires"):
            return False, "fires table not found in fire_es.sqlite"

        return True, "Workspace is valid"

    def _ensure_db_schema(self, db_path: Path) -> bool:
        """Создать обязательные таблицы в SQLite (идемпотентно)."""
        try:
            db = DatabaseManager(str(db_path))
            db.create_tables()
            db.engine.dispose()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DB schema at {db_path}: {e}")
            return False

    def _has_table(self, db_path: Path, table_name: str) -> bool:
        """Проверить наличие таблицы в SQLite."""
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                return cur.fetchone() is not None
        except Exception:
            return False
