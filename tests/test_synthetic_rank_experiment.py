import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fire_es.research.synthetic_rank_experiment import (
    build_experiment_feature_sets,
    exact_duplicate_rate_by_feature_hash,
    prepare_real_canonical_test,
)
from fire_es.simulation.digital_twin import (
    CANONICAL_SOURCE_SHEET,
    TIME_ORDER_COLUMNS,
    generate_rank_conditional_synthetic_dataset,
)


def _sample_rank_df(rows: int = 72) -> pd.DataFrame:
    ranks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    payload = []
    for idx in range(rows):
        rank = ranks[idx % len(ranks)]
        floors = 2 + (idx % 12)
        fire_floor = min(floors, idx % 9)
        t_detect = 5 + idx % 20
        t_report = t_detect + 2
        t_arrival = t_report + 7
        t_hose = t_arrival + 4
        t_contained = t_hose + 20
        t_extinguished = t_contained + 30
        payload.append(
            {
                "row_id": idx + 1,
                "source_sheet": CANONICAL_SOURCE_SHEET,
                "source_period": "test",
                "fire_date": f"2020-01-{(idx % 28) + 1:02d}",
                "region_code": 70 + (idx % 8),
                "settlement_type_code": idx % 3 + 1,
                "fire_protection_code": idx % 4 + 1,
                "enterprise_type_code": 10 + (idx % 6),
                "building_floors": floors,
                "fire_floor": fire_floor,
                "fire_resistance_code": idx % 5 + 1,
                "source_item_code": idx % 7 + 1,
                "distance_to_station": 1.5 + (idx % 13) * 0.8,
                "t_detect_min": t_detect,
                "t_report_min": t_report,
                "t_arrival_min": t_arrival,
                "t_first_hose_min": t_hose,
                "t_contained_min": t_contained,
                "t_extinguished_min": t_extinguished,
                "risk_category_code": idx % 4,
                "fpo_class_code": idx % 5,
                "rank_tz": rank,
                "object_name": f"object-{idx % 11}",
                "address": f"address-{idx % 17}",
            }
        )
    return pd.DataFrame(payload)


def test_rank_conditional_generator_rows_target_duplicates_and_constraints():
    real = _sample_rank_df()
    synthetic, profile = generate_rank_conditional_synthetic_dataset(
        real,
        n_rows=120,
        random_state=7,
        numeric_noise_scale=0.12,
    )
    assert len(synthetic) == 120
    assert synthetic["rank_tz"].notna().all()
    assert set(synthetic["rank_tz"].unique()) == set(real["rank_tz"].unique())
    assert profile["synthetic_generation"]["rows"] == 120

    features = build_experiment_feature_sets(real)["features_10_dispatch"].features
    assert exact_duplicate_rate_by_feature_hash(synthetic, real, features) == 0.0

    assert (synthetic["fire_floor"] <= synthetic["building_floors"]).all()
    for left, right in zip(TIME_ORDER_COLUMNS[:-1], TIME_ORDER_COLUMNS[1:]):
        assert (synthetic[right] >= synthetic[left]).all()
    assert "delta_report_to_arrival" in synthetic.columns


def test_feature_sets_do_not_leak_for_safe_stages():
    feature_sets = build_experiment_feature_sets(_sample_rank_df())
    dispatch = set(feature_sets["features_10_dispatch"].features)
    arrival = set(feature_sets["features_13_arrival"].features)
    first_hose = set(feature_sets["features_15_first_hose"].features)

    forbidden_dispatch = {
        "source_item_code",
        "t_arrival_min",
        "t_first_hose_min",
        "t_contained_min",
        "t_extinguished_min",
        "fatalities",
        "injuries",
        "direct_damage",
        "equipment_count",
        "nozzle_count",
        "rank_tz",
    }
    assert not forbidden_dispatch.intersection(dispatch)
    assert "t_first_hose_min" not in arrival
    assert "t_contained_min" not in first_hose
    assert "t_extinguished_min" not in first_hose


def test_prepare_real_canonical_test_removes_duplicates():
    df = _sample_rank_df(12)
    duplicated = pd.concat([df, df.iloc[:3]], ignore_index=True)
    real_test, profile = prepare_real_canonical_test(duplicated)
    assert len(real_test) == 12
    assert profile["duplicates_removed"] == 3


def test_cli_smoke_creates_report_and_metrics(tmp_path: Path):
    input_path = tmp_path / "input.csv"
    _sample_rank_df(84).to_csv(input_path, index=False)
    output_dir = tmp_path / "reports"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_rank_experiment.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--synthetic-rows",
            "800",
            "--train-rows",
            "600",
            "--synthetic-validation-rows",
            "200",
            "--modes",
            "synthetic_only",
            "--feature-sets",
            "features_10_dispatch",
            "--models",
            "decision_tree",
            "--timestamp",
            "smoke",
        ],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    run_dir = output_dir / "smoke"
    assert (run_dir / "REPORT.md").exists()
    assert (run_dir / "metrics_by_feature_set.csv").exists()
