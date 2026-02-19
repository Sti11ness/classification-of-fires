"""Smoke/E2E import into an isolated temporary workspace.

Usage:
  python scripts/smoke_import_workspace.py \
      --excel-glob "БД-1_*xlsx" \
      --temp-workspace-root build/auto_test_runtime
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_import_paths(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _find_excel_file(repo_root: Path, excel_glob: str) -> Path:
    matches = sorted(repo_root.glob(excel_glob))
    if not matches:
        raise FileNotFoundError(f"No Excel files matched: {excel_glob}")
    return matches[0]


def _get_fires_count(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM fires")
        row = cur.fetchone()
    return int(row[0]) if row else 0


def run_smoke_import(excel_glob: str, temp_workspace_root: Path) -> dict:
    repo_root = _resolve_repo_root()
    _bootstrap_import_paths(repo_root)

    # Deferred imports after sys.path bootstrap.
    from fire_es_desktop.use_cases.import_data_use_case import ImportDataUseCase
    from fire_es_desktop.workspace.workspace_manager import WorkspaceManager

    excel_path = _find_excel_file(repo_root, excel_glob)
    temp_workspace_root.mkdir(parents=True, exist_ok=True)

    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}_",
            dir=str(temp_workspace_root),
        )
    )
    workspace_path = run_dir / "workspace"

    manager = WorkspaceManager()
    if not manager.create_workspace(workspace_path):
        raise RuntimeError(f"Failed to create workspace: {workspace_path}")

    db_path = workspace_path / "fire_es.sqlite"
    use_case = ImportDataUseCase(db_path)
    result = use_case.execute(
        excel_path=excel_path,
        sheet_name=None,
        clean=True,
        save_to_db=True,
    )
    if not result.success:
        raise RuntimeError(result.message)

    fires_count = _get_fires_count(db_path)
    if fires_count <= 0:
        raise RuntimeError("Smoke import finished, but fires table is empty")

    summary = {
        "workspace": str(workspace_path),
        "db_path": str(db_path),
        "excel_path": str(excel_path),
        "added_to_db": result.data.get("added_to_db") if result.data else None,
        "fires_count": fires_count,
        "success": True,
    }

    # Keep the directory for postmortem if needed by the caller.
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke import in isolated workspace")
    parser.add_argument(
        "--excel-glob",
        default="БД-1_*xlsx",
        help="Glob pattern (repo-root relative) for source Excel",
    )
    parser.add_argument(
        "--temp-workspace-root",
        default="build/auto_test_runtime",
        help="Directory used for temporary smoke workspaces",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove created temp run directory after success",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    temp_workspace_root = (repo_root / args.temp_workspace_root).resolve()

    try:
        summary = run_smoke_import(
            excel_glob=args.excel_glob,
            temp_workspace_root=temp_workspace_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.cleanup:
            workspace_path = Path(summary["workspace"])
            run_dir = workspace_path.parent
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
        return 0
    except Exception as exc:
        error_summary = {
            "success": False,
            "error": str(exc),
            "excel_glob": args.excel_glob,
            "temp_workspace_root": str(temp_workspace_root),
        }
        print(json.dumps(error_summary, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

