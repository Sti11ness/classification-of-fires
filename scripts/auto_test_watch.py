"""Automatic test orchestration with per-iteration Markdown reports.

Default run:
  python scripts/auto_test_watch.py --watch --max-iterations 20 --max-hours 6 \
      --strict-green --commit-each-green --reports-dir reports/auto_test_runs
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

ERROR_PATTERNS = (
    re.compile(r"\bERROR\b"),
    re.compile(r"Traceback \(most recent call last\):"),
)

DEFAULT_IGNORE_PATTERNS = [
    "reports/auto_test_runs/**",
    "build/auto_test_runtime/**",
    "dist/**",
    "dist_new/**",
    "build/pyinstaller*/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    ".venv/**",
    ".ruff_cache/**",
    ".git/**",
]

FORBIDDEN_DELETE_PATTERNS = [
    "src/**",
    "tests/**",
    "*.xlsx",
    "*.xls",
    "*.csv",
    "*.parquet",
    "data/**",
    "WS/**",
    "README.md",
    "README_DESKTOP.md",
    "pyproject.toml",
]


@dataclass
class StepResult:
    name: str
    command: str
    success: bool
    return_code: int
    duration_sec: float
    log_path: str
    details: str = ""


@dataclass
class PipelineResult:
    steps: list[StepResult] = field(default_factory=list)
    detected_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)


@dataclass
class IterationRecord:
    iteration: int
    status: str
    strict_green_passed: bool
    decision: str
    started_at: str
    ended_at: str
    duration_sec: float
    failed_steps: list[str]
    trigger_changes: list[str]
    changed_files: list[str]
    warnings: list[str]
    report_path: str
    artifacts_dir: str
    failure_context_path: str = ""


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_src_path(repo_root: Path) -> None:
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def make_run_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def _to_posix(path: Path) -> str:
    return path.as_posix()


def should_ignore_rel_path(rel_posix: str, ignore_patterns: Sequence[str]) -> bool:
    rel = PurePosixPath(rel_posix)
    for pattern in ignore_patterns:
        if rel.match(pattern):
            return True
    return False


def snapshot_repo(root: Path, ignore_patterns: Sequence[str]) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for current_root, dirs, files in os.walk(root, topdown=True):
        current_root_path = Path(current_root)
        rel_root = current_root_path.relative_to(root)

        pruned_dirs = []
        for directory in dirs:
            rel_dir = rel_root / directory
            rel_dir_posix = _to_posix(rel_dir)
            if should_ignore_rel_path(rel_dir_posix, ignore_patterns):
                continue
            pruned_dirs.append(directory)
        dirs[:] = pruned_dirs

        for file_name in files:
            rel_file = rel_root / file_name
            rel_file_posix = _to_posix(rel_file)
            if should_ignore_rel_path(rel_file_posix, ignore_patterns):
                continue
            abs_file = current_root_path / file_name
            try:
                stat = abs_file.stat()
            except OSError:
                continue
            snapshot[rel_file_posix] = (stat.st_mtime_ns, stat.st_size)

    return snapshot


def detect_changes(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[str]:
    changed: list[str] = []
    keys = set(before) | set(after)
    for key in sorted(keys):
        if before.get(key) != after.get(key):
            changed.append(key)
    return changed


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def run_command(
    *,
    name: str,
    command: Sequence[str],
    workdir: Path,
    log_path: Path,
    timeout_sec: int | None = None,
) -> StepResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cmd_str = _format_command(command)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"$ {cmd_str}\n\n")
        try:
            proc = subprocess.run(
                list(command),
                cwd=str(workdir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            success = proc.returncode == 0
            details = ""
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            success = False
            return_code = 124
            details = f"Timeout after {timeout_sec}s"
            log_file.write(f"\n{details}\n")
        except Exception as exc:  # pragma: no cover - defensive path
            success = False
            return_code = 1
            details = f"Command execution failed: {exc}"
            log_file.write(f"\n{details}\n")

    elapsed = time.perf_counter() - started
    return StepResult(
        name=name,
        command=cmd_str,
        success=success,
        return_code=return_code,
        duration_sec=elapsed,
        log_path=_to_posix(log_path),
        details=details,
    )


def run_exe_smoke_role(
    *,
    exe_path: Path,
    role: str,
    wait_sec: int,
    log_path: Path,
) -> StepResult:
    name = f"Smoke exe ({role})"
    command = [str(exe_path), "--role", role]
    cmd_str = _format_command(command)
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not exe_path.exists():
        return StepResult(
            name=name,
            command=cmd_str,
            success=False,
            return_code=1,
            duration_sec=0.0,
            log_path=_to_posix(log_path),
            details=f"Executable not found: {exe_path}",
        )

    proc: subprocess.Popen | None = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"$ {cmd_str}\n\n")
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(wait_sec)
            poll = proc.poll()
            if poll is not None:
                output = proc.stdout.read() if proc.stdout else ""
                if output:
                    log_file.write(output)
                details = f"Process exited early with code {poll}"
                success = False
                return_code = poll
            else:
                details = f"Process stayed alive for {wait_sec}s"
                success = True
                return_code = 0
        except Exception as exc:
            success = False
            return_code = 1
            details = f"Smoke launch failed: {exc}"
            log_file.write(details + "\n")
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if proc is not None and proc.stdout:
                output = proc.stdout.read()
                if output:
                    log_file.write(output)
            log_file.write(details + "\n")

    elapsed = time.perf_counter() - started
    return StepResult(
        name=name,
        command=cmd_str,
        success=success,
        return_code=return_code,
        duration_sec=elapsed,
        log_path=_to_posix(log_path),
        details=details,
    )


def analyze_logs_for_errors(
    *,
    log_paths: Sequence[Path],
    log_path: Path,
) -> tuple[StepResult, list[str]]:
    name = "Analyze logs (ERROR/Traceback)"
    cmd_str = "scan logs for ERROR and Traceback"
    started = time.perf_counter()
    findings: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as out:
        out.write("$ " + cmd_str + "\n\n")
        for path in log_paths:
            if not path.exists():
                continue
            out.write(f"[scan] {path}\n")
            with path.open("r", encoding="utf-8", errors="replace") as log_file:
                for idx, line in enumerate(log_file, start=1):
                    text = line.rstrip("\n")
                    if any(pattern.search(text) for pattern in ERROR_PATTERNS):
                        finding = f"{path}:{idx}: {text}"
                        findings.append(finding)
            out.write("\n")

        if findings:
            out.write("[findings]\n")
            for finding in findings[:200]:
                out.write(f"- {finding}\n")
        else:
            out.write("[findings] none\n")

    elapsed = time.perf_counter() - started
    step = StepResult(
        name=name,
        command=cmd_str,
        success=not findings,
        return_code=0 if not findings else 1,
        duration_sec=elapsed,
        log_path=_to_posix(log_path),
        details=f"{len(findings)} findings",
    )
    return step, findings


def _parse_deleted_paths(git_status_output: str) -> list[str]:
    deleted: list[str] = []
    for line in git_status_output.splitlines():
        if len(line) < 4:
            continue
        state = line[:2]
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        path_part = path_part.strip().replace("\\", "/")
        if "D" in state:
            deleted.append(path_part)
    return deleted


def _is_forbidden_delete(path_str: str, patterns: Sequence[str]) -> bool:
    path = PurePosixPath(path_str)
    return any(path.match(pattern) for pattern in patterns)


def safety_preflight(repo_root: Path) -> tuple[bool, list[str]]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Non-git directory or unavailable git: do not block pipeline.
        return True, [f"preflight warning: git status failed ({proc.returncode})"]

    deleted_paths = _parse_deleted_paths(proc.stdout)
    violations = [
        path
        for path in deleted_paths
        if _is_forbidden_delete(path, FORBIDDEN_DELETE_PATTERNS)
    ]
    if violations:
        reasons = [f"Forbidden deletion detected: {path}" for path in violations]
        return False, reasons
    return True, []


def get_changed_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []

    changed: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        changed.append(path_part.strip().replace("\\", "/"))
    return sorted(set(changed))


def strict_green_pass(step_results: Sequence[StepResult], detected_errors: Sequence[str]) -> bool:
    if detected_errors:
        return False
    return all(step.success for step in step_results)


def build_iteration_markdown(
    *,
    record: IterationRecord,
    step_results: Sequence[StepResult],
    detected_errors: Sequence[str],
    blocked_reasons: Sequence[str],
    failure_context_rel: str,
    git_warnings: Sequence[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Iteration {record.iteration:04d} - {record.status}")
    lines.append("")
    lines.append("## Что сделано")
    lines.append("- Выполнен полный pipeline автотестирования (pytest, smoke import, build, exe smoke, log scan).")
    lines.append(f"- Сформированы raw-логи в `{record.artifacts_dir}`.")
    lines.append(f"- Зафиксированы изменения рабочего дерева: {len(record.changed_files)} файлов.")
    lines.append("")
    lines.append("## Что обнаружено")
    if blocked_reasons:
        for reason in blocked_reasons:
            lines.append(f"- BLOCKED: {reason}")
    failed_steps = [step for step in step_results if not step.success]
    if failed_steps:
        for step in failed_steps:
            details = f" ({step.details})" if step.details else ""
            lines.append(f"- Шаг неуспешен: `{step.name}`{details}")
    if detected_errors:
        lines.append(f"- Найдено лог-ошибок: {len(detected_errors)}")
        for finding in detected_errors[:10]:
            lines.append(f"  - `{finding}`")
    if not blocked_reasons and not failed_steps and not detected_errors:
        lines.append("- Критичных ошибок не обнаружено.")
    lines.append("")
    lines.append("## Что нужно исправить")
    if record.status == "GREEN":
        lines.append("- Исправления не требуются, критерий strict-green достигнут.")
    else:
        if blocked_reasons:
            lines.append("- Устранить policy violations (запрещенные удаления) перед следующей итерацией.")
        for step in failed_steps:
            lines.append(f"- Исправить причину падения шага `{step.name}` и повторить цикл.")
        if detected_errors:
            lines.append("- Устранить `ERROR`/`Traceback` из логов текущей итерации.")
        if failure_context_rel:
            lines.append(f"- См. детальный failure-context: `{failure_context_rel}`.")
    lines.append("")
    lines.append("## Статус шагов")
    lines.append("| Шаг | Статус | Код | Длительность, c | Лог |")
    lines.append("|---|---|---:|---:|---|")
    for step in step_results:
        status = "OK" if step.success else "FAIL"
        lines.append(
            f"| {step.name} | {status} | {step.return_code} | {step.duration_sec:.2f} | `{step.log_path}` |"
        )
    lines.append("")
    lines.append("## Измененные файлы")
    if record.changed_files:
        for changed in record.changed_files[:200]:
            lines.append(f"- `{changed}`")
    else:
        lines.append("- Нет изменений.")
    lines.append("")
    lines.append("## Ссылки на raw-логи")
    for step in step_results:
        lines.append(f"- `{step.log_path}`")
    if failure_context_rel:
        lines.append(f"- `{failure_context_rel}`")
    lines.append("")
    lines.append("## Решение по итерации")
    lines.append(f"- `{record.decision}`")
    if git_warnings:
        lines.append("")
        lines.append("## Git warnings")
        for warning in git_warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def write_index_report(run_dir: Path, records: Sequence[IterationRecord]) -> None:
    index_path = run_dir / "index.md"
    lines = [
        f"# Auto Test Run {run_dir.name}",
        "",
        "| Iteration | Status | Strict green | Failed steps | Decision | Report |",
        "|---:|---|---|---|---|---|",
    ]
    for rec in records:
        failed = ", ".join(rec.failed_steps) if rec.failed_steps else "-"
        report_rel = _to_posix(Path(rec.report_path).relative_to(run_dir))
        lines.append(
            f"| {rec.iteration:04d} | {rec.status} | "
            f"{'yes' if rec.strict_green_passed else 'no'} | {failed} | "
            f"{rec.decision} | `{report_rel}` |"
        )
    lines.append("")
    index_path.write_text("\n".join(lines), encoding="utf-8")


def write_final_summary(
    *,
    run_dir: Path,
    records: Sequence[IterationRecord],
    reason: str,
    started_at: datetime,
    ended_at: datetime,
) -> Path:
    summary_path = run_dir / "FINAL_SUMMARY.md"
    lines = [
        f"# Auto Test Final Summary ({run_dir.name})",
        "",
        f"- Start: `{started_at.isoformat(timespec='seconds')}`",
        f"- End: `{ended_at.isoformat(timespec='seconds')}`",
        f"- Duration: `{(ended_at - started_at)}`",
        f"- Stop reason: `{reason}`",
        f"- Iterations: `{len(records)}`",
        "",
        "## Итог",
    ]
    if records and records[-1].status == "GREEN":
        lines.append("- Strict-green достигнут.")
    else:
        lines.append("- Strict-green не достигнут; см. список блокеров ниже.")
    lines.append("")
    lines.append("## Блокеры / последние ошибки")
    blockers_added = False
    for rec in reversed(records):
        if rec.failure_context_path:
            context_path = Path(rec.failure_context_path)
            if context_path.exists():
                data = json.loads(context_path.read_text(encoding="utf-8"))
                failed_steps = data.get("failed_steps", [])
                blocked_reasons = data.get("blocked_reasons", [])
                detected_errors = data.get("detected_errors", [])
                lines.append(f"- Iteration {rec.iteration:04d}:")
                for item in blocked_reasons:
                    lines.append(f"  - blocked: {item}")
                for item in failed_steps:
                    lines.append(f"  - failed step: {item}")
                if detected_errors:
                    lines.append(f"  - detected errors: {len(detected_errors)}")
                blockers_added = True
                break
    if not blockers_added:
        lines.append("- Нет.")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def commit_green_iteration(
    *,
    repo_root: Path,
    iteration: int,
) -> list[str]:
    warnings: list[str] = []
    add_proc = subprocess.run(
        ["git", "add", "-A"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if add_proc.returncode != 0:
        warnings.append(
            f"git add failed ({add_proc.returncode}): {(add_proc.stderr or add_proc.stdout).strip()}"
        )
        return warnings

    message = f"autotest(iter-{iteration:04d}): strict green"
    commit_proc = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_proc.returncode != 0:
        output = (commit_proc.stderr or commit_proc.stdout).strip()
        warnings.append(f"git commit warning ({commit_proc.returncode}): {output}")
    return warnings


def run_pipeline(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    iter_artifacts_dir: Path,
) -> PipelineResult:
    result = PipelineResult()

    safe, preflight_messages = safety_preflight(repo_root)
    if preflight_messages:
        result.warnings.extend(preflight_messages)
    if not safe:
        result.blocked_reasons.extend(preflight_messages)
        return result

    step_logs: list[Path] = []

    step1 = run_command(
        name="Pytest",
        command=[sys.executable, "-m", "pytest", "-q"],
        workdir=repo_root,
        log_path=iter_artifacts_dir / "step1_pytest.log",
    )
    result.steps.append(step1)
    step_logs.append(Path(step1.log_path))

    step2 = run_command(
        name="Smoke import workspace",
        command=[
            sys.executable,
            str(repo_root / "scripts" / "smoke_import_workspace.py"),
            "--excel-glob",
            args.excel_glob,
            "--temp-workspace-root",
            args.temp_workspace_root,
            "--cleanup",
        ],
        workdir=repo_root,
        log_path=iter_artifacts_dir / "step2_smoke_import.log",
    )
    result.steps.append(step2)
    step_logs.append(Path(step2.log_path))

    step3 = run_command(
        name="Build exe",
        command=["cmd", "/c", "scripts\\build_exe_fast.bat"],
        workdir=repo_root,
        log_path=iter_artifacts_dir / "step3_build_exe.log",
    )
    result.steps.append(step3)
    step_logs.append(Path(step3.log_path))

    exe_path = repo_root / "dist_new" / "FireES" / "FireES.exe"
    step4 = run_exe_smoke_role(
        exe_path=exe_path,
        role="analyst",
        wait_sec=6,
        log_path=iter_artifacts_dir / "step4_smoke_analyst.log",
    )
    result.steps.append(step4)
    step_logs.append(Path(step4.log_path))

    step5 = run_exe_smoke_role(
        exe_path=exe_path,
        role="lpr",
        wait_sec=6,
        log_path=iter_artifacts_dir / "step5_smoke_lpr.log",
    )
    result.steps.append(step5)
    step_logs.append(Path(step5.log_path))

    step6, findings = analyze_logs_for_errors(
        log_paths=step_logs,
        log_path=iter_artifacts_dir / "step6_log_scan.log",
    )
    result.steps.append(step6)
    result.detected_errors.extend(findings)
    return result


def execute_iteration(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    iteration: int,
    trigger_changes: Sequence[str],
) -> IterationRecord:
    iter_started = datetime.now()
    iter_artifacts_dir = run_dir / "artifacts" / f"iter-{iteration:04d}"
    iter_artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline = run_pipeline(
        args=args,
        repo_root=repo_root,
        iter_artifacts_dir=iter_artifacts_dir,
    )
    failed_steps = [step.name for step in pipeline.steps if not step.success]
    strict_ok = strict_green_pass(pipeline.steps, pipeline.detected_errors) and not pipeline.blocked_reasons

    if pipeline.blocked_reasons:
        status = "BLOCKED"
    elif args.strict_green and strict_ok:
        status = "GREEN"
    elif not args.strict_green and all(step.success for step in pipeline.steps):
        status = "GREEN"
    else:
        status = "FAILED"

    if status == "GREEN":
        decision = "green"
    elif status == "BLOCKED":
        decision = "blocked"
    elif args.watch:
        decision = "continue"
    else:
        decision = "stop_failed"

    changed_files = get_changed_files(repo_root)

    failure_context_path = ""
    if status != "GREEN":
        failure_context = {
            "iteration": iteration,
            "status": status,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "failed_steps": failed_steps,
            "blocked_reasons": pipeline.blocked_reasons,
            "detected_errors": pipeline.detected_errors[:200],
            "warnings": pipeline.warnings,
            "trigger_changes": list(trigger_changes),
            "changed_files": changed_files[:200],
            "steps": [asdict(step) for step in pipeline.steps],
        }
        failure_context_path = str(iter_artifacts_dir / "failure-context.json")
        Path(failure_context_path).write_text(
            json.dumps(failure_context, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    git_warnings: list[str] = []
    if status == "GREEN" and args.commit_each_green:
        git_warnings = commit_green_iteration(repo_root=repo_root, iteration=iteration)

    iter_ended = datetime.now()
    record = IterationRecord(
        iteration=iteration,
        status=status,
        strict_green_passed=strict_ok,
        decision=decision,
        started_at=iter_started.isoformat(timespec="seconds"),
        ended_at=iter_ended.isoformat(timespec="seconds"),
        duration_sec=(iter_ended - iter_started).total_seconds(),
        failed_steps=failed_steps,
        trigger_changes=list(trigger_changes),
        changed_files=changed_files,
        warnings=list(pipeline.warnings) + git_warnings,
        report_path=str(run_dir / f"iter-{iteration:04d}.md"),
        artifacts_dir=_to_posix(iter_artifacts_dir.relative_to(run_dir)),
        failure_context_path=failure_context_path,
    )

    failure_context_rel = ""
    if failure_context_path:
        failure_context_rel = _to_posix(Path(failure_context_path).relative_to(run_dir))

    markdown = build_iteration_markdown(
        record=record,
        step_results=pipeline.steps,
        detected_errors=pipeline.detected_errors,
        blocked_reasons=pipeline.blocked_reasons,
        failure_context_rel=failure_context_rel,
        git_warnings=git_warnings,
    )
    Path(record.report_path).write_text(markdown, encoding="utf-8")
    return record


def _safe_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(lock_path: Path, run_id: str) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", -1))
        except Exception:
            pid = -1
        if _safe_pid_running(pid):
            raise RuntimeError(f"Another auto-test process is running (pid={pid})")
        lock_path.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cleanup() -> None:
        try:
            if lock_path.exists():
                data = json.loads(lock_path.read_text(encoding="utf-8"))
                if int(data.get("pid", -1)) == os.getpid():
                    lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic test watch orchestrator")
    parser.add_argument("--watch", action="store_true", help="Watch repository and rerun on changes")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations per run")
    parser.add_argument("--max-hours", type=float, default=6.0, help="Max hours per run")
    parser.add_argument("--strict-green", dest="strict_green", action="store_true", default=True)
    parser.add_argument("--no-strict-green", dest="strict_green", action="store_false")
    parser.add_argument("--commit-each-green", dest="commit_each_green", action="store_true", default=True)
    parser.add_argument("--no-commit-each-green", dest="commit_each_green", action="store_false")
    parser.add_argument("--reports-dir", default="reports/auto_test_runs", help="Run reports directory")
    parser.add_argument("--excel-glob", default="БД-1_*xlsx", help="Glob for smoke import source Excel")
    parser.add_argument(
        "--temp-workspace-root",
        default="build/auto_test_runtime",
        help="Temp root for smoke import workspaces",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in watch mode")
    parser.add_argument("--debounce-seconds", type=float, default=5.0, help="Debounce for change events")
    parser.add_argument("--lock-file", default="", help="Optional lock file path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _resolve_repo_root()
    _bootstrap_src_path(repo_root)

    reports_root = (repo_root / args.reports_dir).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    run_dir = reports_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    lock_path = Path(args.lock_file).resolve() if args.lock_file else (reports_root / ".auto_test_watch.lock")
    acquire_lock(lock_path, run_id)

    started_at = datetime.now()
    budget_deadline = started_at + timedelta(hours=args.max_hours)
    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    records: list[IterationRecord] = []

    print(f"[auto-test] run_id={run_id}")
    print(f"[auto-test] reports={run_dir}")

    iteration = 1
    record = execute_iteration(
        args=args,
        repo_root=repo_root,
        run_dir=run_dir,
        iteration=iteration,
        trigger_changes=["initial-run"],
    )
    records.append(record)
    write_index_report(run_dir, records)
    print(f"[auto-test] iteration={iteration:04d} status={record.status}")

    if record.status == "GREEN" and args.strict_green:
        write_final_summary(
            run_dir=run_dir,
            records=records,
            reason="strict-green reached",
            started_at=started_at,
            ended_at=datetime.now(),
        )
        return 0

    if not args.watch:
        reason = "watch disabled"
        if record.status != "GREEN":
            reason = "watch disabled and non-green"
        write_final_summary(
            run_dir=run_dir,
            records=records,
            reason=reason,
            started_at=started_at,
            ended_at=datetime.now(),
        )
        return 0 if record.status == "GREEN" else 1

    previous_snapshot = snapshot_repo(repo_root, ignore_patterns)
    pending_changes: list[str] = []
    last_change_time = 0.0

    try:
        while True:
            now = datetime.now()
            if now >= budget_deadline:
                write_final_summary(
                    run_dir=run_dir,
                    records=records,
                    reason="budget exceeded",
                    started_at=started_at,
                    ended_at=now,
                )
                return 1
            if len(records) >= args.max_iterations:
                write_final_summary(
                    run_dir=run_dir,
                    records=records,
                    reason="max iterations reached",
                    started_at=started_at,
                    ended_at=now,
                )
                return 1

            current_snapshot = snapshot_repo(repo_root, ignore_patterns)
            changes = detect_changes(previous_snapshot, current_snapshot)
            previous_snapshot = current_snapshot
            if changes:
                pending_changes.extend(changes)
                last_change_time = time.monotonic()
                print(f"[auto-test] detected changes: {len(changes)}")

            should_run = (
                bool(pending_changes)
                and last_change_time > 0
                and (time.monotonic() - last_change_time) >= args.debounce_seconds
            )
            if should_run:
                iteration += 1
                dedup_changes = sorted(set(pending_changes))
                pending_changes.clear()
                record = execute_iteration(
                    args=args,
                    repo_root=repo_root,
                    run_dir=run_dir,
                    iteration=iteration,
                    trigger_changes=dedup_changes,
                )
                records.append(record)
                write_index_report(run_dir, records)
                print(f"[auto-test] iteration={iteration:04d} status={record.status}")

                if record.status == "GREEN" and args.strict_green:
                    write_final_summary(
                        run_dir=run_dir,
                        records=records,
                        reason="strict-green reached",
                        started_at=started_at,
                        ended_at=datetime.now(),
                    )
                    return 0
                if record.status == "BLOCKED":
                    write_final_summary(
                        run_dir=run_dir,
                        records=records,
                        reason="blocked by safety guard",
                        started_at=started_at,
                        ended_at=datetime.now(),
                    )
                    return 1

            time.sleep(max(args.poll_interval, 0.2))
    except KeyboardInterrupt:
        write_final_summary(
            run_dir=run_dir,
            records=records,
            reason="interrupted by user",
            started_at=started_at,
            ended_at=datetime.now(),
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
