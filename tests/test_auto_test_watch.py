"""Tests for scripts/auto_test_watch.py."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "auto_test_watch.py"
    spec = importlib.util.spec_from_file_location("auto_test_watch", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_run_id_format():
    module = _load_module()
    run_id = module.make_run_id(datetime(2026, 2, 19, 3, 4, 5))
    assert re.fullmatch(r"\d{8}_\d{6}", run_id)
    assert run_id == "20260219_030405"


def test_strict_green_pass():
    module = _load_module()
    ok_step = module.StepResult(
        name="ok",
        command="cmd",
        success=True,
        return_code=0,
        duration_sec=0.1,
        log_path="x.log",
    )
    bad_step = module.StepResult(
        name="bad",
        command="cmd",
        success=False,
        return_code=1,
        duration_sec=0.1,
        log_path="y.log",
    )
    assert module.strict_green_pass([ok_step, ok_step], [])
    assert not module.strict_green_pass([ok_step, bad_step], [])
    assert not module.strict_green_pass([ok_step, ok_step], ["error"])


def test_iteration_markdown_required_sections():
    module = _load_module()
    record = module.IterationRecord(
        iteration=1,
        status="FAILED",
        strict_green_passed=False,
        decision="continue",
        started_at="2026-02-19T03:00:00",
        ended_at="2026-02-19T03:01:00",
        duration_sec=60.0,
        failed_steps=["Pytest"],
        trigger_changes=["src/x.py"],
        changed_files=["src/x.py"],
        warnings=[],
        report_path="iter-0001.md",
        artifacts_dir="artifacts/iter-0001",
        failure_context_path="artifacts/iter-0001/failure-context.json",
    )
    step = module.StepResult(
        name="Pytest",
        command="pytest -q",
        success=False,
        return_code=1,
        duration_sec=1.0,
        log_path="artifacts/iter-0001/step1.log",
        details="failed",
    )
    content = module.build_iteration_markdown(
        record=record,
        step_results=[step],
        detected_errors=["traceback"],
        blocked_reasons=[],
        failure_context_rel="artifacts/iter-0001/failure-context.json",
        git_warnings=[],
    )
    assert "## Что сделано" in content
    assert "## Что обнаружено" in content
    assert "## Что нужно исправить" in content
    assert "## Статус шагов" in content
    assert "## Измененные файлы" in content
    assert "## Ссылки на raw-логи" in content
    assert "## Решение по итерации" in content


def test_watch_trigger_ignores_report_dir(tmp_path):
    module = _load_module()
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "reports" / "auto_test_runs" / "run1").mkdir(parents=True)
    src_file = repo / "src" / "a.py"
    report_file = repo / "reports" / "auto_test_runs" / "run1" / "iter-0001.md"
    src_file.write_text("print(1)\n", encoding="utf-8")
    report_file.write_text("old\n", encoding="utf-8")

    snap1 = module.snapshot_repo(repo, module.DEFAULT_IGNORE_PATTERNS)
    report_file.write_text("new\n", encoding="utf-8")
    snap2 = module.snapshot_repo(repo, module.DEFAULT_IGNORE_PATTERNS)
    changes_after_report = module.detect_changes(snap1, snap2)
    assert not any(path.startswith("reports/auto_test_runs/") for path in changes_after_report)

    src_file.write_text("print(2)\n", encoding="utf-8")
    snap3 = module.snapshot_repo(repo, module.DEFAULT_IGNORE_PATTERNS)
    changes_after_src = module.detect_changes(snap2, snap3)
    assert "src/a.py" in changes_after_src


def test_execute_iteration_success_and_failure(tmp_path, monkeypatch):
    module = _load_module()
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "reports" / "auto_test_runs" / "run_x"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir(parents=True)

    args = argparse.Namespace(
        strict_green=True,
        watch=True,
        commit_each_green=False,
        excel_glob="*.xlsx",
        temp_workspace_root="build/auto_test_runtime",
    )

    ok_step = module.StepResult(
        name="Pytest",
        command="pytest -q",
        success=True,
        return_code=0,
        duration_sec=1.2,
        log_path="artifacts/iter-0001/step1.log",
        details="",
    )
    fail_step = module.StepResult(
        name="Pytest",
        command="pytest -q",
        success=False,
        return_code=1,
        duration_sec=1.2,
        log_path="artifacts/iter-0002/step1.log",
        details="boom",
    )

    def fake_run_pipeline_green(*, args, repo_root, iter_artifacts_dir):
        return module.PipelineResult(
            steps=[ok_step],
            detected_errors=[],
            warnings=[],
            blocked_reasons=[],
        )

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline_green)
    monkeypatch.setattr(module, "get_changed_files", lambda _: ["src/changed.py"])

    rec1 = module.execute_iteration(
        args=args,
        repo_root=repo_root,
        run_dir=run_dir,
        iteration=1,
        trigger_changes=["initial"],
    )
    assert rec1.status == "GREEN"
    assert Path(rec1.report_path).exists()
    assert rec1.failure_context_path == ""

    def fake_run_pipeline_fail(*, args, repo_root, iter_artifacts_dir):
        return module.PipelineResult(
            steps=[fail_step],
            detected_errors=["Traceback something"],
            warnings=[],
            blocked_reasons=[],
        )

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline_fail)
    rec2 = module.execute_iteration(
        args=args,
        repo_root=repo_root,
        run_dir=run_dir,
        iteration=2,
        trigger_changes=["src/changed.py"],
    )
    assert rec2.status == "FAILED"
    assert rec2.decision == "continue"
    assert Path(rec2.report_path).exists()
    assert rec2.failure_context_path
    assert Path(rec2.failure_context_path).exists()

    module.write_index_report(run_dir, [rec1, rec2])
    assert (run_dir / "index.md").exists()

    summary_path = module.write_final_summary(
        run_dir=run_dir,
        records=[rec1, rec2],
        reason="max iterations reached",
        started_at=datetime(2026, 2, 19, 3, 0, 0),
        ended_at=datetime(2026, 2, 19, 3, 10, 0),
    )
    assert summary_path.exists()
