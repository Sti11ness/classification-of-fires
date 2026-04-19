"""
Полный функциональный UI smoke/e2e тест Fire ES Desktop.

Сценарий проходит все разделы интерфейса:
- Проект (создать/открыть/закрыть Workspace)
- Импорт данных (выбор файла, предпросмотр, импорт)
- Обучение модели (обучить + активация)
- Модели (обновить/выбрать/активировать/открыть папку)
- Прогноз ЛПР (ввод, прогноз, выбор ранга, сохранение, отмена)
- История решений ЛПР (просмотр, детали, редактирование)
- Пакетный прогноз (выбор файла, запуск, экспорт)
- Журнал (фильтр, refresh, export, open log)
- Переключение ролей analyst/lpr в контекстной панели

Запуск:
  python scripts/full_ui_functional_test.py
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fire_es.model_train import FEATURE_SETS  # noqa: E402
from fire_es_desktop.ui import MainWindow  # noqa: E402


@dataclass
class UiTestResult:
    name: str
    success: bool
    details: str = ""


@dataclass
class UiTestContext:
    excel_path: Path
    run_root: Path
    workspace_parent: Path
    workspace_path: Path
    batch_input_path: Path
    export_log_path: Path
    created_files: List[Path] = field(default_factory=list)
    opened_paths: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)
    dialog_dir_queue: List[Path] = field(default_factory=list)
    open_file_queue: List[Path] = field(default_factory=list)
    save_file_queue: List[Path] = field(default_factory=list)


class UiDialogPatcher:
    """Патчит диалоги/сообщения, чтобы тест был полностью автоматический."""

    def __init__(self, ctx: UiTestContext):
        self.ctx = ctx
        self._originals: Dict[str, Any] = {}

    def install(self) -> None:
        self._originals["qmessage_information"] = QMessageBox.information
        self._originals["qmessage_warning"] = QMessageBox.warning
        self._originals["qmessage_critical"] = QMessageBox.critical
        self._originals["qfd_get_open"] = QFileDialog.getOpenFileName
        self._originals["qfd_get_save"] = QFileDialog.getSaveFileName
        self._originals["qfd_exec"] = QFileDialog.exec
        self._originals["qfd_selected"] = QFileDialog.selectedFiles
        self._originals["os_startfile"] = getattr(os, "startfile", None)

        def _msg(kind: str) -> Callable[..., int]:
            def _inner(parent, title, text, *args, **kwargs):  # type: ignore[no-untyped-def]
                self.ctx.messages.append(
                    {"kind": kind, "title": str(title), "text": str(text)}
                )
                return QMessageBox.Ok

            return _inner

        def _get_open_file_name(*args, **kwargs):  # type: ignore[no-untyped-def]
            if self.ctx.open_file_queue:
                path = self.ctx.open_file_queue.pop(0)
                return str(path), ""
            return "", ""

        def _get_save_file_name(*args, **kwargs):  # type: ignore[no-untyped-def]
            if self.ctx.save_file_queue:
                path = self.ctx.save_file_queue.pop(0)
                return str(path), ""
            return "", ""

        def _dialog_exec(dialog) -> bool:  # type: ignore[no-untyped-def]
            return bool(self.ctx.dialog_dir_queue)

        def _dialog_selected_files(dialog) -> List[str]:  # type: ignore[no-untyped-def]
            if self.ctx.dialog_dir_queue:
                return [str(self.ctx.dialog_dir_queue.pop(0))]
            return []

        def _startfile(path: str) -> None:
            self.ctx.opened_paths.append(str(path))

        QMessageBox.information = _msg("information")  # type: ignore[assignment]
        QMessageBox.warning = _msg("warning")  # type: ignore[assignment]
        QMessageBox.critical = _msg("critical")  # type: ignore[assignment]
        QFileDialog.getOpenFileName = _get_open_file_name  # type: ignore[assignment]
        QFileDialog.getSaveFileName = _get_save_file_name  # type: ignore[assignment]
        QFileDialog.exec = _dialog_exec  # type: ignore[assignment]
        QFileDialog.selectedFiles = _dialog_selected_files  # type: ignore[assignment]
        if hasattr(os, "startfile"):
            os.startfile = _startfile  # type: ignore[assignment]

    def restore(self) -> None:
        QMessageBox.information = self._originals["qmessage_information"]  # type: ignore[assignment]
        QMessageBox.warning = self._originals["qmessage_warning"]  # type: ignore[assignment]
        QMessageBox.critical = self._originals["qmessage_critical"]  # type: ignore[assignment]
        QFileDialog.getOpenFileName = self._originals["qfd_get_open"]  # type: ignore[assignment]
        QFileDialog.getSaveFileName = self._originals["qfd_get_save"]  # type: ignore[assignment]
        QFileDialog.exec = self._originals["qfd_exec"]  # type: ignore[assignment]
        QFileDialog.selectedFiles = self._originals["qfd_selected"]  # type: ignore[assignment]
        if hasattr(os, "startfile"):
            os.startfile = self._originals["os_startfile"]  # type: ignore[assignment]


def wait_until(
    app: QApplication,
    condition: Callable[[], bool],
    timeout_ms: int,
    step_ms: int = 50,
) -> None:
    deadline = datetime.now().timestamp() + timeout_ms / 1000
    while datetime.now().timestamp() < deadline:
        app.processEvents()
        if condition():
            return
        QTest.qWait(step_ms)
    raise TimeoutError(f"Condition timeout after {timeout_ms} ms")


def click(app: QApplication, button: QPushButton) -> None:
    button.click()
    app.processEvents()
    QTest.qWait(80)


def find_button_by_text(root, text: str) -> QPushButton:
    for btn in root.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    raise AssertionError(f"Кнопка не найдена: '{text}'")


def navigate_to(win: MainWindow, app: QApplication, text: str) -> None:
    for row in range(win.navigation.count()):
        item = win.navigation.item(row)
        if item and item.text() == text:
            win.navigation.setCurrentRow(row)
            app.processEvents()
            QTest.qWait(80)
            return
    raise AssertionError(f"Пункт навигации не найден: '{text}'")


def create_batch_input(ctx: UiTestContext, db_path: Path) -> None:
    cols = FEATURE_SETS["online_tactical"]
    query = f"SELECT {', '.join(cols)} FROM fires LIMIT 30"
    df = pd.read_sql(query, f"sqlite:///{db_path}")
    if df.empty:
        raise RuntimeError("Для batch-прогона не найдено строк в fires")
    ctx.batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(ctx.batch_input_path, index=False)
    ctx.created_files.append(ctx.batch_input_path)


def run_full_ui_test(ctx: UiTestContext) -> Tuple[bool, List[UiTestResult], str]:
    results: List[UiTestResult] = []
    app = QApplication.instance() or QApplication(sys.argv)
    win: Optional[MainWindow] = None
    patcher = UiDialogPatcher(ctx)

    try:
        patcher.install()

        def stage(name: str) -> None:
            print(f"[UI-STEP] {name}", flush=True)

        stage("Init MainWindow")
        win = MainWindow(role="analyst")
        win.show()
        app.processEvents()
        QTest.qWait(200)

        # 1. Проект: create/close/open
        stage("Project create")
        ctx.dialog_dir_queue.append(ctx.workspace_parent)
        click(app, win.project_page.create_btn)
        if not win.project_vm.is_workspace_open:
            raise AssertionError("Workspace не открылся после 'Создать Workspace'")
        if not ctx.workspace_path.exists():
            raise AssertionError(f"Workspace не создан: {ctx.workspace_path}")
        results.append(UiTestResult("Project: create workspace", True))

        stage("Project close")
        click(app, win.project_page.close_btn)
        if win.project_vm.is_workspace_open:
            raise AssertionError("Workspace не закрылся после 'Закрыть Workspace'")
        results.append(UiTestResult("Project: close workspace", True))

        stage("Project open")
        ctx.dialog_dir_queue.append(ctx.workspace_path)
        click(app, win.project_page.open_btn)
        if not win.project_vm.is_workspace_open:
            raise AssertionError("Workspace не открылся после 'Открыть Workspace'")
        results.append(UiTestResult("Project: open workspace", True))

        # 2. Импорт данных: select/preview/import
        stage("Import select/preview")
        import_page = win.import_page
        if import_page.viewmodel is None:
            raise AssertionError("ImportPage ViewModel не инициализирован после открытия workspace")

        ctx.open_file_queue.append(ctx.excel_path)
        click(app, import_page.select_file_btn)
        if import_page.sheets_list.count() == 0:
            raise AssertionError("После выбора файла список листов пуст")

        import_page.sheets_list.setCurrentRow(0)
        first_item = import_page.sheets_list.item(0)
        first_item.setSelected(True)
        click(app, import_page.preview_btn)
        if import_page.preview_table.rowCount() == 0:
            raise AssertionError("Предпросмотр не загрузил строки")
        if import_page.preview_table.horizontalScrollBarPolicy() != Qt.ScrollBarAsNeeded:
            raise AssertionError("Горизонтальный скролл предпросмотра не включен")

        stage("Import run")
        click(app, import_page.import_btn)
        wait_until(
            app,
            lambda: import_page.worker is not None and not import_page.worker.isRunning(),
            timeout_ms=300000,
        )
        if "Импортировано" not in import_page.status_label.text():
            raise AssertionError(
                f"Импорт не завершен успешно: '{import_page.status_label.text()}'"
            )
        results.append(UiTestResult("Import: select + preview + import", True))

        # 3. Обучение модели: train + activate
        stage("Training run")
        training_page = win.training_page
        if training_page.viewmodel is None:
            raise AssertionError("TrainingPage ViewModel не инициализирован")
        training_page.target_combo.setCurrentText("rank_tz")
        training_page.model_type_combo.setCurrentText("random_forest")
        training_page.feature_set_combo.setCurrentIndex(0)
        training_page.test_size_combo.setCurrentText("0.2")
        training_page.class_weight_combo.setCurrentText("balanced")

        click(app, training_page.train_btn)
        wait_until(
            app,
            lambda: training_page.worker is not None and not training_page.worker.isRunning(),
            timeout_ms=480000,
        )
        wait_until(
            app,
            lambda: "Обучение завершено" in training_page.results_text.toPlainText(),
            timeout_ms=10000,
        )
        if "Обучение завершено" not in training_page.results_text.toPlainText():
            raise AssertionError("Обучение не завершилось успешно")

        stage("Training activate")
        wait_until(
            app,
            lambda: training_page.activate_btn.isEnabled() and hasattr(training_page, "_current_model_id"),
            timeout_ms=10000,
        )
        click(app, training_page.activate_btn)
        active_model_id = training_page.viewmodel.get_active_model_id()
        if not active_model_id:
            raise AssertionError("Модель не стала активной после нажатия 'Сделать модель активной'")
        results.append(UiTestResult("Training: train + activate", True))

        # 4. Модели: refresh/select/activate/open_folder
        stage("Models page")
        models_page = win.models_page
        click(app, models_page.refresh_btn)
        if models_page.models_table.rowCount() == 0:
            raise AssertionError("В разделе 'Модели' пустая таблица после обучения")
        models_page.models_table.selectRow(0)
        models_page._on_cell_clicked(0, 0)
        if not models_page.info_text.toPlainText().strip():
            raise AssertionError("Информация о модели не отображается")
        click(app, models_page.activate_btn)
        click(app, models_page.open_folder_btn)
        results.append(UiTestResult("Models: refresh + select + activate + open", True))

        # 5. ЛПР: predict/save/cancel
        stage("LPR predict/save/cancel")
        lpr_page = win.lpr_page
        initial_comment = "Auto UI full test"
        updated_comment = "Auto UI full test updated"
        lpr_inputs = {
            "region_code": 77,
            "settlement_type_code": 1,
            "fire_protection_code": 2,
            "enterprise_type_code": 11,
            "building_floors": 9,
            "fire_floor": 3,
            "fire_resistance_code": 2,
            "source_item_code": 12,
            "distance_to_station": 2.5,
            "t_detect_min": 15,
            "t_report_min": 25,
            "t_arrival_min": 35,
            "t_first_hose_min": 45,
        }
        for field_name, value in lpr_inputs.items():
            lpr_page.input_widgets[field_name].setValue(value)

        click(app, lpr_page.predict_btn)
        wait_until(
            app,
            lambda: lpr_page.predict_worker is not None and not lpr_page.predict_worker.isRunning(),
            timeout_ms=180000,
        )
        if "Прогноз выполнен" not in lpr_page.status_label.text():
            raise AssertionError(f"Прогноз ЛПР не выполнен: '{lpr_page.status_label.text()}'")
        if not lpr_page.confidence_label.text().strip():
            raise AssertionError("После прогноза не заполнена метка уверенности")

        lpr_page.rank_combo.setCurrentText("2")
        lpr_page.comment_edit.setPlainText(initial_comment)
        click(app, lpr_page.save_btn)
        wait_until(
            app,
            lambda: lpr_page.save_worker is not None and not lpr_page.save_worker.isRunning(),
            timeout_ms=180000,
        )
        if "сохран" not in lpr_page.status_label.text().lower():
            raise AssertionError(f"Решение ЛПР не сохранено: '{lpr_page.status_label.text()}'")

        # 6. История решений ЛПР: refresh/select/edit
        stage("LPR history view/edit")
        navigate_to(win, app, "История решений ЛПР")
        history_page = win.lpr_history_page
        click(app, history_page.refresh_btn)
        if history_page.decisions_table.rowCount() == 0:
            raise AssertionError("История решений ЛПР пуста после сохранения решения")

        history_page.decisions_table.selectRow(0)
        history_page._on_table_selection_changed()
        app.processEvents()
        QTest.qWait(100)

        if history_page.decision_id_value.text() == "—":
            raise AssertionError("Карточка решения не открылась после выбора строки")
        if initial_comment not in history_page.comment_edit.toPlainText():
            raise AssertionError("В истории решений не найден только что сохраненный комментарий")
        if not history_page.probabilities_text.toPlainText().strip():
            raise AssertionError("В истории решений не отображаются вероятности top-k")
        if not history_page.input_snapshot_text.toPlainText().strip():
            raise AssertionError("В истории решений не отображаются входные данные кейса")

        history_page.decision_rank_combo.setCurrentText("3")
        history_page.comment_edit.setPlainText(updated_comment)
        click(app, history_page.save_btn)
        click(app, history_page.refresh_btn)
        history_page.decisions_table.selectRow(0)
        history_page._on_table_selection_changed()
        app.processEvents()
        QTest.qWait(100)

        if history_page.decision_rank_combo.currentText() != "3":
            raise AssertionError("Измененный ранг не сохранился в истории решений")
        if history_page.comment_edit.toPlainText() != updated_comment:
            raise AssertionError("Измененный комментарий не сохранился в истории решений")
        results.append(UiTestResult("LPR history: view + edit + persist", True))

        navigate_to(win, app, "Прогноз (ЛПР)")

        click(app, lpr_page.cancel_btn)
        if lpr_page.rank_combo.currentText() != "Не выбрано":
            raise AssertionError("Кнопка 'Отмена' не сбрасывает выбранный ранг")
        results.append(UiTestResult("LPR: predict + save + cancel", True))

        # 7. Batch predict: select/start/result
        stage("Batch predict")
        db_path = win.project_vm.get_db_path()
        if not db_path:
            raise AssertionError("Не удалось получить путь к БД workspace")
        create_batch_input(ctx, db_path)

        batch_page = win.batch_predict_page
        select_btn = find_button_by_text(batch_page, "Выбрать файл")
        ctx.open_file_queue.append(ctx.batch_input_path)
        click(app, select_btn)
        batch_page.source_combo.setCurrentIndex(0)
        batch_page.format_combo.setCurrentIndex(1)  # csv
        batch_page.feature_combo.setCurrentIndex(0)
        batch_page.top_k_spin.setValue(3)
        batch_page.bootstrap_check.setChecked(False)
        batch_page.n_bootstrap_spin.setValue(10)

        click(app, batch_page.start_btn)
        wait_until(
            app,
            lambda: batch_page.worker is not None and not batch_page.worker.isRunning(),
            timeout_ms=240000,
        )
        if "Завершено" not in batch_page.status_label.text():
            raise AssertionError(f"Batch прогноз не завершился: '{batch_page.status_label.text()}'")
        if not batch_page.viewmodel or not batch_page.viewmodel.state.output_path:
            raise AssertionError("Batch прогноз не вернул путь к результату")
        if not batch_page.viewmodel.state.output_path.exists():
            raise AssertionError(
                f"Batch результат не найден: {batch_page.viewmodel.state.output_path}"
            )
        results.append(UiTestResult("Batch predict: select + run + export", True))

        # 8. Журнал: refresh/filter/export/open-log
        stage("Log page")
        log_page = win.log_page
        if log_page.log_store:
            log_page.log_store.log_operation("full_ui_test marker", source="FullUiTest")
        click(app, log_page.refresh_btn)
        if log_page.logs_table.rowCount() > 0:
            log_page._on_cell_clicked(0, 0)
            if not log_page.details_text.toPlainText().strip():
                raise AssertionError("В журнале не открываются детали записи")

        log_page.level_combo.setCurrentText("INFO")
        click(app, log_page.filter_btn)

        ctx.export_log_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.save_file_queue.append(ctx.export_log_path)
        click(app, log_page.export_btn)
        if not ctx.export_log_path.exists():
            raise AssertionError("Экспорт журнала не создал файл")
        click(app, log_page.open_log_btn)
        results.append(UiTestResult("Log: filter + refresh + export + open", True))

        # 9. Переключение ролей и кнопка журнала в статусбаре
        stage("Role switch + status log")
        role_combo = win.context_panel.role_combo
        role_combo.setCurrentIndex(role_combo.findData("lpr"))
        app.processEvents()
        QTest.qWait(120)
        lpr_nav = [win.navigation.item(i).text() for i in range(win.navigation.count())]
        if win.role != "lpr" or win.navigation.count() != 4:
            raise AssertionError("Переключение в режим lpr не обновило навигацию")
        if lpr_nav != ["Проект", "Прогноз (ЛПР)", "История решений ЛПР", "Журнал"]:
            raise AssertionError(f"Некорректная навигация lpr: {lpr_nav}")

        role_combo.setCurrentIndex(role_combo.findData("analyst"))
        app.processEvents()
        QTest.qWait(120)
        analyst_nav = [win.navigation.item(i).text() for i in range(win.navigation.count())]
        if win.role != "analyst" or win.navigation.count() < 8:
            raise AssertionError("Переключение в режим analyst не обновило навигацию")
        if "История решений ЛПР" not in analyst_nav:
            raise AssertionError("В режиме analyst отсутствует страница истории решений ЛПР")

        click(app, win.status_bar.log_button)
        if win.pages_stack.currentIndex() != win.page_indices["log"]:
            raise AssertionError("Кнопка 'Журнал' в статусбаре не открыла страницу журнала")
        results.append(UiTestResult("Main window: role switch + status log button", True))

        return True, results, ""

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return False, results, err
    finally:
        patcher.restore()
        if win is not None:
            win.close()
        app.processEvents()


def write_report(
    ctx: UiTestContext,
    success: bool,
    results: List[UiTestResult],
    error_text: str,
) -> Path:
    report_path = ctx.run_root / "FULL_UI_TEST_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        f"# Full UI Functional Test Report ({now})",
        "",
        f"- Status: {'GREEN' if success else 'FAILED'}",
        f"- Passed steps: {passed}/{total}",
        f"- Excel: `{ctx.excel_path}`",
        f"- Workspace: `{ctx.workspace_path}`",
        "",
        "## Что сделано",
        "- Автоматически пройдены разделы: Проект, Импорт, Обучение, Модели, ЛПР, История решений ЛПР, Пакетный прогноз, Журнал, переключение ролей.",
        "- Протестированы клики по ключевым кнопкам и заполнение всех основных полей ввода на страницах.",
        "- Проверены фоновые операции (импорт/обучение/прогноз/сохранение) до завершения.",
        "",
        "## Статус шагов",
    ]

    for idx, item in enumerate(results, start=1):
        lines.append(f"{idx}. {'OK' if item.success else 'FAIL'} — {item.name}")
        if item.details:
            lines.append(f"   - {item.details}")

    lines.extend(
        [
            "",
            "## Что обнаружено",
        ]
    )
    if success:
        lines.append("- Критических ошибок по пройденным пользовательским сценариям не обнаружено.")
    else:
        lines.append("- Найден блокер при полном UI-прогоне:")
        lines.append("```text")
        lines.append(error_text.strip())
        lines.append("```")

    lines.extend(
        [
            "",
            "## Что нужно исправить",
        ]
    )
    if success:
        lines.append("- На этом прогоне обязательных исправлений не требуется.")
    else:
        lines.append("- Исправить блокер из секции 'Что обнаружено' и повторить полный UI-прогон.")

    if ctx.messages:
        lines.extend(["", "## Диалоги (QMessageBox)", ""])
        for msg in ctx.messages[:50]:
            lines.append(f"- `{msg['kind']}` | {msg['title']}: {msg['text']}")

    if ctx.opened_paths:
        lines.extend(["", "## Opened Paths (os.startfile)", ""])
        for p in ctx.opened_paths:
            lines.append(f"- `{p}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Полный автоматический UI тест Fire ES Desktop")
    parser.add_argument(
        "--excel-glob",
        default="БД-1_*xlsx",
        help="Глоб для входного Excel файла в корне проекта",
    )
    parser.add_argument(
        "--run-root",
        default="reports/full_ui_runs",
        help="Папка для артефактов прогона",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Принудительно установить QT_QPA_PLATFORM=offscreen",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    excel_matches = sorted(ROOT.glob(args.excel_glob))
    if not excel_matches:
        print(f"ERROR: Excel not found by glob: {args.excel_glob}")
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / args.run_root / run_id
    workspace_parent = run_root / "workspace_parent"
    workspace_path = workspace_parent / "fire_es_workspace"
    batch_input_path = run_root / "inputs" / "batch_small.xlsx"
    export_log_path = run_root / "exports" / "operations_export.json"

    ctx = UiTestContext(
        excel_path=excel_matches[0],
        run_root=run_root,
        workspace_parent=workspace_parent,
        workspace_path=workspace_path,
        batch_input_path=batch_input_path,
        export_log_path=export_log_path,
    )

    success, results, error_text = run_full_ui_test(ctx)
    report_path = write_report(ctx, success, results, error_text)

    summary = {
        "success": success,
        "report": str(report_path),
        "workspace": str(ctx.workspace_path),
        "messages_count": len(ctx.messages),
        "steps_passed": sum(1 for r in results if r.success),
        "steps_total": len(results),
    }
    print(summary)
    if not success:
        print(error_text)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
