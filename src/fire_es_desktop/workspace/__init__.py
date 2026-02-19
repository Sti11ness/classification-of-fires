# src/fire_es_desktop/workspace/__init__.py
"""
Workspace — рабочая папка проекта (контракт).
Содержит:
- fire_es.sqlite (БД)
- config.json (конфигурация)
- reports/ (артефакты)
- logs/ (журналы)

Интерфейс для создания, открытия и валидации Workspace.
"""

from pathlib import Path
from typing import Optional

class Workspace:
    def __init__(self, path: Path):
        self.path = path
        self.db_path = path / "fire_es.sqlite"
        self.config_path = path / "config.json"
        self.reports_path = path / "reports"
        self.logs_path = path / "logs"
    
    def create(self) -> bool:
        """Создать структуру Workspace."""
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            self.reports_path.mkdir(exist_ok=True)
            self.logs_path.mkdir(exist_ok=True)
            
            # Создать пустой config.json
            if not self.config_path.exists():
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write('{"active_model_id": null, "last_updated": null}')
            
            return True
        except Exception:
            return False
    
    def validate(self) -> tuple[bool, str]:
        """Проверить целостность Workspace."""
        if not self.path.exists():
            return False, "Workspace папка не существует"
        
        if not self.db_path.exists():
            return False, "fire_es.sqlite не найден"
        
        if not self.config_path.exists():
            return False, "config.json не найден"
        
        return True, "OK"