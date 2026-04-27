"""Тесты для WorkspaceManager."""

import json
import sqlite3
from pathlib import Path

from fire_es_desktop.workspace.workspace_manager import WorkspaceManager


def _table_exists(db_path: Path, table: str) -> bool:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cur.fetchone() is not None


def test_create_workspace_initializes_db_schema(tmp_path: Path):
    ws_path = tmp_path / "workspace"
    manager = WorkspaceManager()

    assert manager.create_workspace(ws_path) is True

    db_path = ws_path / "fire_es.sqlite"
    assert db_path.exists()
    assert _table_exists(db_path, "fires")
    assert _table_exists(db_path, "models")
    assert _table_exists(db_path, "lpr_decisions")


def test_open_workspace_bootstraps_empty_db_file(tmp_path: Path):
    ws_path = tmp_path / "workspace_empty_db"
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "reports" / "models").mkdir(parents=True, exist_ok=True)
    (ws_path / "reports" / "figs").mkdir(parents=True, exist_ok=True)
    (ws_path / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    (ws_path / "logs").mkdir(parents=True, exist_ok=True)
    (ws_path / "fire_es.sqlite").write_bytes(b"")
    (ws_path / "config.json").write_text(
        json.dumps({"active_model_id": None, "version": "0.1.0"}),
        encoding="utf-8",
    )

    manager = WorkspaceManager()
    assert manager.open_workspace(ws_path) is True
    assert _table_exists(ws_path / "fire_es.sqlite", "fires")


def test_resolve_workspace_path_accepts_parent_directory(tmp_path: Path):
    parent = tmp_path / "parent"
    ws_path = parent / "fire_es_workspace"
    manager = WorkspaceManager()

    assert manager.create_workspace(ws_path) is True
    assert manager.resolve_workspace_path(parent) == ws_path
    assert manager.open_workspace(parent) is True
    assert manager.get_current_workspace() == ws_path
