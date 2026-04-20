# Fire ES — Technical README

Этот документ предназначен для инженера, который заходит в репозиторий впервые и должен быстро понять:

- из каких слоев состоит система;
- где проходит граница между domain-логикой и desktop-приложением;
- как устроены data flow, model flow и storage flow;
- где именно искать source of truth при изменениях.

Документ опирается на текущий код репозитория, тесты и документы в `docs/`. Если текст расходится со старым README или с устным описанием, приоритет имеет текущее состояние кода.

## 1. Source of truth

При изменениях в проекте ориентироваться в таком порядке:

1. код в `src/` и текущая структура репозитория;
2. тесты в `tests/`;
3. документы:
   - `docs/spec_first.md`
   - `docs/spec_second.md`
   - `docs/ARCHITECTURE_REPORT.md`
   - `docs/FULL_PROJECT_DESCRIPTION.md`
4. существующие README как вспомогательный слой.

Практически это означает:

- пользовательские обещания проверять по `src/fire_es_desktop`;
- ML-контракты и feature sets проверять по `src/fire_es` и тестам;
- layout Workspace проверять по `src/fire_es_desktop/workspace/workspace_manager.py`;
- фактический production-контракт модели проверять по `src/fire_es/rank_tz_contract.py`;
- canonical нормативы рангов проверять по `data/normatives/rank_resource_normatives.json` и `src/fire_es/normatives.py`.

## 2. Система в одном абзаце

Fire ES состоит из двух рабочих контуров:

- `src/fire_es` — domain/ML-слой: загрузка Excel, очистка, ранжирование, подготовка признаков, обучение, прогнозирование, схема SQLite;
- `src/fire_es_desktop` — desktop-слой: Workspace, инфраструктура, сценарии, MVVM, PySide6 UI, файловый реестр моделей и журналирование.

Исторические Excel-данные загружаются в SQLite, затем используются для разметки `rank_tz`, обучения модели, регистрации артефактов и последующего online inference для ЛПР.

## 3. Репозиторий верхнего уровня

| Путь | Назначение |
|---|---|
| `src/fire_es/` | domain- и ML-логика |
| `src/fire_es_desktop/` | desktop-приложение на PySide6 |
| `tests/` | pytest-набор для domain и desktop контрактов |
| `scripts/` | запуск, сборка, smoke- и audit-сценарии |
| `docs/` | требования, архитектурные и описательные документы |
| `build/` | `pyinstaller.spec` и сборочная конфигурация |
| `app/` | отдельный Streamlit UI-артефакт, не основной desktop entrypoint |
| `reports/` | отчеты и артефакты, созданные в рабочем дереве репозитория |
| `data/` | директория данных и нормативных файлов |
| `clean_df*.csv`, `*.parquet` | подготовленные производные датасеты и экспериментальные артефакты |
| `train_*.py`, `feature_engineering*.py`, `save_best_model.py` | root-level исследовательские и офлайн-экспериментальные скрипты |

## 4. Точки входа

### Основной desktop entrypoint

- `python -m fire_es_desktop.main --role analyst`
- `python -m fire_es_desktop.main --role lpr`

Файл: `src/fire_es_desktop/main.py`

Что делает:

- парсит аргументы `--role` и `--workspace`;
- создает `QApplication`;
- применяет глобальный стиль;
- поднимает `MainWindow`;
- запускает desktop-приложение.

### CLI entrypoint domain-пакета

- `fire-es`

Файл: `src/fire_es/cli.py`

Назначение:

- служебный CLI-слой для запуска notebook-oriented сценариев и вспомогательных команд.

### Сборка и запуск из scripts

| Файл | Назначение |
|---|---|
| `scripts/build_exe.bat` | полная сборка desktop `.exe` через PyInstaller |
| `scripts/build_exe_fast.bat` | быстрая пересборка в `dist_new` |
| `scripts/run_desktop_dev.bat` | запуск desktop из исходников |
| `scripts/run_ui.bat` | запуск отдельного Streamlit UI-сценария |
| `scripts/auto_test_watch.py` | цикл автоматических проверок, сборки и smoke-прогонов |
| `scripts/smoke_import_workspace.py` | smoke-сценарий для Workspace/import |
| `scripts/ml_rank_tz_audit.py` | аудит ML-контракта `rank_tz` |
| `scripts/load_to_db.py` | загрузочный скрипт для переноса данных в SQLite |

## 5. Архитектура по слоям

### 5.1. Domain слой: `src/fire_es`

Это основной source of truth для обработки данных, feature engineering contract, обучения и инференса.

| Файл | Ответственность |
|---|---|
| `schema.py` | константы колонок, маппинг RU→EN, ограничения схемы |
| `utils.py` | низкоуровневые утилиты: нормализация текста, извлечение кода, парсинг времени, исследовательский `rank_ref` |
| `cleaning.py` | загрузка Excel-листов, переименование колонок, валидация, числовые/временные преобразования, quality flags |
| `equipment_parse.py` | разбор поля `equipment`, canonical resource-vector helpers |
| `normatives.py` | единый source of truth для нормативов рангов и hash/version норматива |
| `ranking.py` | canonical vector-rank и auxiliary count-proxy logic |
| `rank_tz_contract.py` | production/offline контракт модели `rank_tz`: semantic target, availability stages, feature sets, preprocessor artifact |
| `model_selection.py` | leakage-safe split protocols (`group_shuffle`, `group_kfold`, `temporal_holdout`, legacy row split) |
| `metrics.py` | metrics passport для rank classifier |
| `cluster_analysis.py` | analyst-side cluster analysis block |
| `model_train.py` | подготовка выборки, классические feature sets, обучение деревьев и лесов, CV-оценка, сохранение модели |
| `predict.py` | Top-K прогноз, bootstrap confidence, resource quantiles, batch/result serialization |
| `db.py` | SQLAlchemy-модели `Fire`, `Normative`, `LPRDecision`, `Model` и менеджер `DatabaseManager` |
| `cli.py` | CLI-обертка над domain-контуром |

### 5.2. Desktop слой: `src/fire_es_desktop`

Desktop-слой организован как MVVM + Use Case + Infrastructure.

#### Workspace

| Файл | Ответственность |
|---|---|
| `workspace/workspace_manager.py` | создает, открывает и валидирует Workspace; гарантирует наличие структуры папок и схемы SQLite |

#### Infrastructure

| Файл | Ответственность |
|---|---|
| `infra/db_repository.py` | CRUD и summary-операции поверх `fire_es.db`, без SQLAlchemy-логики в UI |
| `infra/artifact_store.py` | файловое хранилище моделей, таблиц, графиков и метаданных в `reports/` |
| `infra/model_registry.py` | файловый реестр моделей в `reports/models/registry.json`, выбор активной production-модели |
| `infra/log_store.py` | `app.log`, `tasks.log`, `operations.jsonl`, фильтрация и экспорт журнала |

#### Use Cases

| Файл | Ответственность |
|---|---|
| `use_cases/base_use_case.py` | общая модель статуса, результата и отмены сценариев |
| `use_cases/import_data_use_case.py` | Excel import → clean → optional DB write |
| `use_cases/assign_rank_tz_use_case.py` | массовая разметка `rank_tz` по данным в БД |
| `use_cases/train_model_use_case.py` | training pipeline, packaging артефактов и подготовка данных для `ModelRegistry` |
| `use_cases/predict_use_case.py` | online production-safe inference для ЛПР |
| `use_cases/save_decision_use_case.py` | сохранение кейса и решения ЛПР в SQLite |
| `use_cases/batch_predict_export_use_case.py` | пакетный прогноз по Excel и экспорт результатов |

Ключевой follow-up после методологической ветки:

- форма ЛПР строится по `input_schema` активной production-safe модели;
- fallback schema = `dispatch_initial_safe`;
- `dispatch_initial_safe` не содержит `source_item_code`;
- `AssignRankTzUseCase` по умолчанию не перезаписывает `human_verified` / `lpr_decision` строки;
- train и predict используют один и тот же stage-aware feature engineering helper;
- parser техники пишет parse confidence/conflict metadata и может выводить строку из canonical training.

#### ViewModels

| Файл | Ответственность |
|---|---|
| `viewmodels/project_viewmodel.py` | lifecycle Workspace и подключение `DbRepository`, `ModelRegistry`, `LogStore` |
| `viewmodels/import_data_viewmodel.py` | выбор файла, листов, preview и orchestration import use case |
| `viewmodels/train_model_viewmodel.py` | параметры обучения, вызов train use case, регистрация и активация модели |
| `viewmodels/lpr_predict_viewmodel.py` | состояние формы прогноза, вызов inference и сохранения решения |
| `viewmodels/lpr_decision_history_viewmodel.py` | список решений, карточка решения, редактирование rank/comment |
| `viewmodels/batch_predict_viewmodel.py` | параметры пакетного прогноза и orchestration batch export |

#### UI

| Файл | Ответственность |
|---|---|
| `ui/main_window.py` | shell приложения: навигация, stacked pages, context panel, status bar, role-based navigation |
| `ui/pages/project_page.py` | создание и открытие Workspace |
| `ui/pages/import_page.py` | импорт Excel, выбор листов, preview и запуск загрузки |
| `ui/pages/training_page.py` | обучение, метрики, выбор и активация модели |
| `ui/pages/lpr_predict_page.py` | экран прогноза ЛПР, форма ввода, график вероятностей, нормативная таблица, блок решения |
| `ui/pages/lpr_decision_history_page.py` | история решений ЛПР |
| `ui/pages/batch_predict_page.py` | batch-инференс по Excel |
| `ui/pages/models_page.py` | просмотр и активация моделей |
| `ui/pages/log_page.py` | журнал операций и экспорт логов |

#### Tasks

| Файл | Ответственность |
|---|---|
| `tasks/task_runner.py` | фоновые задачи, прогресс, кооперативная отмена, результаты и task-лог |

## 6. Workspace contract

Workspace — переносимая рабочая папка проекта. Структура создается и валидируется `WorkspaceManager`.

```text
workspace/
  fire_es.sqlite
  config.json
  reports/
    models/
    figs/
    tables/
  logs/
    app.log
    tasks.log
```

Что важно:

- `fire_es.sqlite` — основная локальная БД проекта;
- `config.json` — метаданные Workspace;
- `reports/models/` — файловые ML-артефакты и `registry.json`;
- `logs/operations.jsonl` создается `LogStore` дополнительно к `app.log` и `tasks.log`.

`ProjectViewModel` после открытия Workspace поднимает три инфраструктурных сервиса:

- `DbRepository` на `fire_es.sqlite`;
- `ModelRegistry` на `reports/models`;
- `LogStore` на `logs/`.

## 7. Основные таблицы БД

Схема описана в `src/fire_es/db.py`.

### `fires`

Главная таблица с очищенными и размеченными записями о пожарах.

Содержит:

- исходные поля и метаданные источника;
- коды, даты, числа и времена;
- показатели последствий;
- поля ресурсов;
- `rank_tz`, `rank_distance`, `rank_ref`, `severity_score`;
- quality flags;
- `created_at` и `updated_at`.

### `normatives`

Таблица нормативов по рангам и типам ресурсов.

Практическая роль:

- хранение нормативных записей в SQLite;
- reference-слой для rank/resource логики.

### `lpr_decisions`

Таблица решений ЛПР.

Содержит:

- ссылку на запись в `fires`;
- выбранный ранг;
- прогноз модели на момент решения;
- вероятности;
- комментарий и метаданные сохранения.

### `models`

SQLAlchemy-модель для хранения метаданных моделей существует в схеме `db.py`, но текущий desktop workflow управляет активной моделью через файловый `ModelRegistry` в `reports/models/registry.json`.

Практическое следствие:

- для lifecycle активной модели смотреть `infra/model_registry.py`;
- на наличие SQLAlchemy-таблицы `models` не опираться как на основной production registry.

## 8. Data flow

### 8.1. Импорт исторических данных

Путь:

`Excel` → `ImportPage` → `ImportDataViewModel` → `ImportDataUseCase` → `cleaning.py` → `fires`

Подробно:

1. `ImportPage` выбирает Excel-файл и листы.
2. `ImportDataViewModel` готовит состояние импорта.
3. `ImportDataUseCase.execute()`:
   - открывает файл через `pd.ExcelFile`;
   - для каждого листа вызывает `load_fact_sheet()`;
   - склеивает листы в единый DataFrame;
   - при `clean=True` прогоняет `clean_fire_data()`;
   - конвертирует DataFrame в записи;
   - через SQLAlchemy bulk insert сохраняет записи в `fires`.
4. В `clean_fire_data()` происходят:
   - RU→EN rename;
   - валидация дат и годов;
   - извлечение кодов из текстовых полей;
   - преобразование числовых колонок;
   - парсинг времен в минуты;
   - расчет `direct_damage_log`, `severity_score`, `rank_ref`;
   - постановка quality flags.

### 8.2. Формирование признаков и производных датасетов

В проекте есть два уровня feature engineering.

#### Operational contract

Operational contract задается в `rank_tz_contract.py` и используется desktop pipeline для training/inference.

#### Исследовательский слой

Root-level скрипты:

- `feature_engineering.py` — ручные производные признаки: temporal, response-time, building-risk, distance, missing indicators;
- `feature_engineering_ft.py` — автоматическая генерация признаков через Featuretools;
- `train_f1_weighted.py`, `train_final_models.py`, `train_final_f1_weighted.py`, `train_f1_macro.py` — offline-эксперименты на подготовленных CSV.

Именно поэтому в репозитории одновременно присутствуют:

- `clean_df_with_rank.csv`
- `clean_df_enhanced.csv`
- `clean_df_featuretools.csv`
- другие производные CSV/Parquet-файлы.

### 8.3. Расчет `rank_tz`

Путь:

`fires` / DataFrame → `ranking.py` → `rank_tz`, `rank_distance`

Ключевой код:

- `assign_rank_tz()`
- `calculate_rank_by_count()`
- `calculate_rank_by_vector()`

Фактическое состояние:

- в `ranking.py` прямо зафиксировано, что из-за ограничений исходных данных operational path опирается на `equipment_count`;
- при наличии вектора ресурсов возможен векторный путь, но рабочая прикладная логика ранга в текущем коде упрощена.

Desktop-сценарий для пакетной разметки реализован в `AssignRankTzUseCase`. Он:

- читает существующие записи из `fires`;
- вычисляет `rank_tz`;
- пишет результат обратно в БД.

Отдельной UI-страницы для этой операции в текущем `MainWindow` не зарегистрировано, но сам use case в коде есть.

## 9. Model flow

### 9.1. Контракт признаков

В проекте есть два близких, но разных слоя feature-set definitions.

#### `model_train.py`

Содержит классические наборы признаков:

- `minimal`
- `managed`
- `observed`
- `time`
- `full`
- `online_dispatch`
- `online_early`
- `online_tactical`
- `enhanced_dispatch`
- `enhanced_early`
- `enhanced_tactical`

Это общий ML-слой для обучения и исследований.

#### `rank_tz_contract.py`

Это current source of truth для desktop training/inference.

Ключевые feature sets:

- `basic` — legacy baseline, offline only;
- `extended` — offline benchmark;
- `online_tactical` — production-safe deploy role для ЛПР;
- `enhanced_tactical` — offline experiment.

Production feature order для `online_tactical`:

1. `region_code`
2. `settlement_type_code`
3. `fire_protection_code`
4. `enterprise_type_code`
5. `building_floors`
6. `fire_floor`
7. `fire_resistance_code`
8. `source_item_code`
9. `distance_to_station`
10. `t_detect_min`
11. `t_report_min`
12. `t_arrival_min`
13. `t_first_hose_min`

### 9.2. Mapping рангов в классы

Контракт задается в `rank_tz_contract.py`:

- `1.0 -> 1`
- `1.5 -> 2`
- `2.0 -> 3`
- `3.0 -> 4`
- `4.0 -> 5`
- `5.0 -> 6`

Обратный mapping используется при возврате прогнозов в пользовательские значения ранга.

### 9.3. Preprocessor artifact

Функции:

- `build_preprocessor_artifact()`
- `apply_preprocessor_artifact()`
- `ensure_feature_frame()`

Артефакт включает:

- `schema_version`
- `target`
- `feature_set`
- `feature_order`
- `input_schema`
- `fill_strategy`
- `fill_values`
- `allowed_missing`
- `class_mapping`
- training metadata

Практическое значение:

- training и inference используют один и тот же сериализованный контракт;
- UI-форма ЛПР строится на `input_schema`;
- batch и online inference одинаково проходят через `apply_preprocessor_artifact()`.

### 9.4. Обучение модели

Путь:

`fires` → `TrainModelUseCase` → `rank_tz_contract` preprocessing → `RandomForestClassifier` или `DecisionTreeClassifier` → артефакты в `reports/models`

Подробно:

1. `TrainModelUseCase` читает `fires` с ненулевым `rank_tz`.
2. Если `rank_tz` отсутствует, пытается авторазметить через внутренний helper.
3. Применяет `add_rank_tz_engineered_features()` для выбранного feature set.
4. Формирует `raw_X` через `ensure_feature_frame()`.
5. Переводит `rank_tz` в классы.
6. Делит данные на train/test.
7. Строит `preprocessor_artifact` по train-части.
8. Обучает модель:
   - `DecisionTreeClassifier`, либо
   - `RandomForestClassifier`.
9. Считает метрики, permutation importance и benchmark payload.
10. Сохраняет артефакты в `reports/models`.

### 9.5. Артефакты модели и их форматы

Для каждой обученной модели `TrainModelUseCase` сохраняет:

| Файл | Формат | Содержимое |
|---|---|---|
| `model_<id>.joblib` | joblib | бинарная модель |
| `model_<id>_meta.json` | JSON | метаданные модели |
| `model_<id>_preprocessor.json` | JSON | сериализованный preprocessing contract |
| `model_<id>_metrics.csv` | CSV | плоская выгрузка метрик |
| `model_<id>_feature_importance.csv` | CSV | impurity/permutation importance |
| `model_<id>_benchmark.json` | JSON | benchmark payload и missingness summary |
| `registry.json` | JSON | файловый реестр моделей и active pointer |

### 9.6. Активация модели

`ModelRegistry`:

- регистрирует модель;
- хранит ее метаданные;
- переключает активную модель;
- блокирует активацию `rank_tz`-модели, если она не имеет `deployment_role = rank_tz_lpr_production`.

Это подтверждается и кодом, и тестом `test_model_registry_blocks_offline_rank_tz_activation`.

## 10. Inference flow

### 10.1. Online inference для ЛПР

Путь:

`LPRPredictPage` → `LPRPredictViewModel` → `PredictUseCase` → `ModelRegistry` → `apply_preprocessor_artifact` → `predict_with_confidence`

Подробно:

1. `LPRPredictPage` строит форму на основе `get_input_schema()` из `rank_tz_contract.py`.
2. `LPRPredictViewModel` собирает `input_data`.
3. `PredictUseCase` находит активную production-safe модель через `ModelRegistry`.
4. Загружаются:
   - `model_<id>.joblib`
   - `model_<id>_preprocessor.json`
5. `apply_preprocessor_artifact()` формирует DataFrame в нужном порядке и с теми же fill rules, что были на training.
6. `predict_with_confidence()` вызывает:
   - `predict_rank_topk()`
   - `bootstrap_predict_rank()` при bootstrap-сценарии
7. Use case возвращает:
   - `top_k_ranks`
   - `confidence`
   - `entropy`
   - `induced_rank_p50`
   - `all_probabilities`
   - метаданные активной модели.

### 10.2. Сохранение решения ЛПР

Путь:

`LPRPredictViewModel` → `SaveDecisionUseCase` → `fires` + `lpr_decisions`

Что происходит:

1. use case валидирует `input_data`, `prediction_data`, `decision_rank`;
2. создает новую запись `Fire` c источником `LPR_MANUAL_INPUT`;
3. сохраняет в ней ключевые входные поля и прогнозный `rank_tz`;
4. создает запись `LPRDecision` с:
   - выбранным рангом;
   - top-K вероятностями;
   - комментарием;
   - служебными метаданными.

### 10.3. Пакетный прогноз

Путь:

`BatchPredictPage` → `BatchPredictViewModel` → `BatchPredictExportUseCase` → Excel input → production model → `reports/tables`

Что делает use case:

- читает Excel;
- проверяет наличие production-safe модели;
- предупреждает о недостающих колонках входного файла;
- применяет тот же `preprocessor_artifact`, что и online inference;
- считает прогноз построчно;
- формирует DataFrame с исходными полями и результатами;
- экспортирует:
  - `.xlsx`, либо
  - `.csv`
  в `reports/tables`.

## 11. UI flow и orchestration

### Main window

`MainWindow` собирает shell приложения:

- left navigation;
- central `QStackedWidget`;
- right context panel;
- bottom status bar.

Навигация привязана к роли:

- `analyst`: проект, импорт, обучение, пакетный прогноз, прогноз, история решений, модели, журнал;
- `lpr`: прогноз, история решений, журнал.

### MVVM-связка

Рабочая связка в проекте такая:

`Page` → `ViewModel` → `UseCase` → `Domain / Infra`

Примеры:

- `ImportPage` → `ImportDataViewModel` → `ImportDataUseCase` → `cleaning.py` / `DatabaseManager`
- `TrainingPage` → `TrainModelViewModel` → `TrainModelUseCase` → `rank_tz_contract.py` / `ModelRegistry`
- `LPRPredictPage` → `LPRPredictViewModel` → `PredictUseCase` / `SaveDecisionUseCase`
- `BatchPredictPage` → `BatchPredictViewModel` → `BatchPredictExportUseCase`

## 12. Task execution и логирование

### TaskRunner

`tasks/task_runner.py` предоставляет:

- регистрацию задач;
- запуск в отдельном `threading.Thread`;
- callbacks прогресса;
- callbacks завершения;
- кооперативную отмену;
- нормализованный `TaskResult`.

### LogStore

`LogStore` создает и ведет:

- `logs/app.log`
- `logs/tasks.log`
- `logs/operations.jsonl`

Также он:

- хранит recent operations in-memory для UI;
- поддерживает фильтрацию;
- экспортирует журнал в JSON/CSV.

## 13. Тесты и что они страхуют

| Тест | Что страхует |
|---|---|
| `test_cleaning.py` | очистка и валидация данных |
| `test_ranking.py` | логика `rank_tz` и распределения рангов |
| `test_db.py` | схема и базовые операции БД |
| `test_model_train.py` | `prepare_data`, feature sets, CV и обучение |
| `test_predict.py` | Top-K, bootstrap confidence, resource quantiles |
| `test_workspace_manager.py` | контракт Workspace |
| `test_rank_tz_contract_desktop.py` | production contract, preprocessor artifact, registry rules, predict/batch consistency |
| `test_lpr_decision_history.py` | история решений ЛПР |
| `test_auto_test_watch.py` | автопрогон и orchestration-скрипты |

Если меняются:

- feature order;
- fill strategy;
- deploy role;
- registry semantics;
- output shape predict/batch;

то первым делом смотреть `test_rank_tz_contract_desktop.py`.

## 14. Что использовать как source of truth при изменениях

### Если меняется форма ввода ЛПР

Смотреть:

- `src/fire_es/rank_tz_contract.py`
- `src/fire_es_desktop/ui/pages/lpr_predict_page.py`
- `src/fire_es_desktop/use_cases/predict_use_case.py`

### Если меняется training contract

Смотреть:

- `src/fire_es_desktop/use_cases/train_model_use_case.py`
- `src/fire_es/rank_tz_contract.py`
- `tests/test_rank_tz_contract_desktop.py`

### Если меняется хранение модели

Смотреть:

- `src/fire_es_desktop/infra/model_registry.py`
- `src/fire_es_desktop/infra/artifact_store.py`
- `reports/models/registry.json` в конкретном Workspace

### Если меняется SQLite-схема

Смотреть:

- `src/fire_es/db.py`
- `src/fire_es_desktop/workspace/workspace_manager.py`
- `src/fire_es_desktop/infra/db_repository.py`

### Если меняется пользовательский workflow

Смотреть:

- `src/fire_es_desktop/ui/main_window.py`
- соответствующую `ui/pages/*`
- соответствующую `viewmodels/*`
- соответствующий `use_cases/*`

## 15. Практические замечания по репозиторию

- Основной поддерживаемый прикладной путь — desktop workflow через `fire_es_desktop.main`.
- `app/streamlit_app.py` существует в репозитории, но не является основным desktop entrypoint.
- В корне репозитория сохранено много исследовательских артефактов и подготовленных CSV/Parquet-файлов; это не то же самое, что переносимый Workspace.
- Для production inference опираться нужно не на все historical feature sets, а на контракт `rank_tz_contract.py`.
- Для lifecycle модели в desktop-приложении опорой служит файловый `ModelRegistry`, а не SQLAlchemy-таблица `models`.
