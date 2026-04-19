"""
Rank_tz ML audit for workspace models and production-safe baseline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fire_es.rank_tz_contract import (  # noqa: E402
    add_rank_tz_engineered_features,
    apply_preprocessor_artifact,
    build_preprocessor_artifact,
    ensure_feature_frame,
    get_feature_set_spec,
    get_manual_inference_feature_order,
    map_rank_series_to_classes,
)
from fire_es_desktop.infra import ModelRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rank_tz ML pipeline for a workspace")
    parser.add_argument(
        "--workspace",
        default="WS/fire_es_workspace",
        help="Path to workspace root",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/ml_audit",
        help="Directory for audit artifacts",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Specific model_id to audit instead of the active registry model",
    )
    return parser.parse_args()


def load_workspace_df(workspace: Path) -> pd.DataFrame:
    db_path = workspace / "fire_es.sqlite"
    return pd.read_sql("SELECT * FROM fires WHERE rank_tz IS NOT NULL", f"sqlite:///{db_path}")


def build_split(
    df: pd.DataFrame,
    feature_order: list[str],
    *,
    feature_set: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    working = add_rank_tz_engineered_features(df, feature_set)
    X = ensure_feature_frame(working, feature_order)
    y = map_rank_series_to_classes(working["rank_tz"])
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def evaluate_predictions(y_true: pd.Series, y_pred) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique().tolist()))
    per_class_recall = {}
    for idx, class_id in enumerate(sorted(y_true.unique().tolist())):
        total = cm[idx].sum()
        per_class_recall[str(class_id)] = round(float(cm[idx, idx] / total) if total else 0.0, 6)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
    }


def compute_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, dict[str, float]]:
    impurity = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    permutation = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    return {
        "impurity_top20": impurity.head(20).round(6).to_dict(),
        "permutation_top20": permutation.head(20).round(6).to_dict(),
    }


def evaluate_active_model(workspace: Path, df: pd.DataFrame, *, model_id: Optional[str]) -> dict[str, Any]:
    registry = ModelRegistry(workspace / "reports" / "models")
    active = registry.get_model_info(model_id) if model_id else registry.get_active_model_info()
    if not active:
        raise RuntimeError("Requested model was not found in workspace registry")

    test_size = float(active.get("params", {}).get("test_size", 0.25))
    random_state = int(active.get("params", {}).get("random_state", 42))
    feature_set = active.get("feature_set", "extended")
    feature_order = active["features"]
    X_train_raw, X_test_raw, y_train, y_test = build_split(
        df,
        feature_order,
        feature_set=feature_set,
        test_size=test_size,
        random_state=random_state,
    )

    model = joblib.load(workspace / "reports" / "models" / active["artifact_path"])
    preprocessor_path = active.get("preprocessor_path")
    if preprocessor_path:
        artifact = json.loads((workspace / "reports" / "models" / preprocessor_path).read_text(encoding="utf-8"))
        X_test = apply_preprocessor_artifact(X_test_raw, artifact)
    else:
        artifact = None
        X_test = X_test_raw.replace([float("inf"), float("-inf")], pd.NA).fillna(0)

    y_pred = model.predict(X_test)
    active_metrics = evaluate_predictions(y_test, y_pred)
    missing_pct = (X_test_raw.isna().mean() * 100).round(4).sort_values(ascending=False).head(20).to_dict()

    manual6 = set(get_manual_inference_feature_order())
    X_app = X_test.copy()
    for column in X_app.columns:
        if column not in manual6:
            X_app[column] = 0
    app_metrics = evaluate_predictions(y_test, model.predict(X_app))

    active_importance = compute_importance(model, X_test, y_test)

    preprocessing_comparison = []
    raw_all = add_rank_tz_engineered_features(df, feature_set)
    raw_X = ensure_feature_frame(raw_all, feature_order)
    y_all = map_rank_series_to_classes(raw_all["rank_tz"])
    valid_mask = y_all.notna()
    raw_X = raw_X.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask].astype(int)
    split = train_test_split(
        raw_X,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )
    X_train_raw_cmp, X_test_raw_cmp, y_train_cmp, y_test_cmp = split
    strategies = [
        ("current_fill0_balanced", "constant", 0.0, "balanced"),
        ("median_balanced", "median", None, "balanced"),
        ("median_no_class_weight", "median", None, None),
        ("fill_minus1_balanced", "constant", -1.0, "balanced"),
    ]
    for name, fill_strategy, fill_value, class_weight in strategies:
        artifact_cmp, X_train_cmp = build_preprocessor_artifact(
            X_train_raw_cmp,
            feature_order=feature_order,
            feature_set=feature_set,
            fill_strategy=fill_strategy,
            fill_value=fill_value,
            training_rows=len(raw_X),
            test_size=test_size,
            random_state=random_state,
        )
        X_test_cmp = apply_preprocessor_artifact(X_test_raw_cmp, artifact_cmp)
        model_cmp = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )
        model_cmp.fit(X_train_cmp, y_train_cmp)
        y_pred_cmp = model_cmp.predict(X_test_cmp)
        metrics_cmp = evaluate_predictions(y_test_cmp, y_pred_cmp)
        preprocessing_comparison.append(
            {
                "variant": name,
                "accuracy": metrics_cmp["accuracy"],
                "f1_macro": metrics_cmp["f1_macro"],
                "f1_weighted": metrics_cmp["f1_weighted"],
            }
        )

    return {
        "active_model": {
            "model_id": active["model_id"],
            "name": active.get("name", ""),
            "feature_set": feature_set,
            "deployment_role": active.get("deployment_role", "legacy"),
            "offline_only": active.get("offline_only", False),
            "test_size": test_size,
            "random_state": random_state,
            "full_metrics": active_metrics,
            "app_style_metrics": app_metrics,
            "missing_pct_top20": missing_pct,
            **active_importance,
            "preprocessing_comparison": preprocessing_comparison,
        }
    }


def evaluate_production_baseline(df: pd.DataFrame, *, feature_set: str) -> dict[str, Any]:
    spec = get_feature_set_spec(feature_set)
    X_train_raw, X_test_raw, y_train, y_test = build_split(
        df,
        spec["feature_order"],
        feature_set=feature_set,
        test_size=0.25,
        random_state=42,
    )
    artifact, X_train = build_preprocessor_artifact(
        X_train_raw,
        feature_order=spec["feature_order"],
        feature_set=feature_set,
        fill_strategy=spec["default_fill_strategy"],
        fill_value=spec["default_fill_value"],
        training_rows=len(df),
        test_size=0.25,
        random_state=42,
    )
    X_test = apply_preprocessor_artifact(X_test_raw, artifact)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    importance = compute_importance(model, X_test, y_test)
    return {
        "feature_set": feature_set,
        "deployment_role": spec["deployment_role"],
        "offline_only": spec["offline_only"],
        "metrics": metrics,
        **importance,
    }


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Rank_tz ML Audit",
        "",
        f"- Timestamp: {summary['generated_at']}",
        f"- Workspace: `{summary['workspace']}`",
        "",
        "## Key Metrics",
        "",
        f"- Active offline/full model `f1_macro`: {summary['active_model']['full_metrics']['f1_macro']:.4f}",
        f"- Active model in current app-style inference `f1_macro`: {summary['active_model']['app_style_metrics']['f1_macro']:.4f}",
        f"- Production baseline (`online_tactical`) `f1_macro`: {summary['production_online_tactical']['metrics']['f1_macro']:.4f}",
        "",
        "## Comparison Tracks",
        "",
        "| Track | Accuracy | F1 Macro | F1 Weighted |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| offline_extended/current_active_full | "
            f"{summary['active_model']['full_metrics']['accuracy']:.4f} | "
            f"{summary['active_model']['full_metrics']['f1_macro']:.4f} | "
            f"{summary['active_model']['full_metrics']['f1_weighted']:.4f} |"
        ),
        (
            f"| production_online_tactical | "
            f"{summary['production_online_tactical']['metrics']['accuracy']:.4f} | "
            f"{summary['production_online_tactical']['metrics']['f1_macro']:.4f} | "
            f"{summary['production_online_tactical']['metrics']['f1_weighted']:.4f} |"
        ),
        (
            f"| current_active_in_app_style | "
            f"{summary['active_model']['app_style_metrics']['accuracy']:.4f} | "
            f"{summary['active_model']['app_style_metrics']['f1_macro']:.4f} | "
            f"{summary['active_model']['app_style_metrics']['f1_weighted']:.4f} |"
        ),
        "",
        "## Active Model Missingness Top 10",
        "",
    ]

    for name, value in list(summary["active_model"]["missing_pct_top20"].items())[:10]:
        lines.append(f"- `{name}`: {value:.2f}%")

    lines.extend(
        [
            "",
            "## Production Baseline Permutation Importance Top 10",
            "",
        ]
    )
    for name, value in list(summary["production_online_tactical"]["permutation_top20"].items())[:10]:
        lines.append(f"- `{name}`: {value:.6f}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `reports/featuretools_report.md` should be treated as historical. Use this audit as the current source of truth.",
            "- `current_active_in_app_style` quantifies how badly a train/inference contract mismatch can degrade the deployed model.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = (ROOT / args.workspace).resolve()
    output_dir = (ROOT / args.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()

    df = load_workspace_df(workspace)
    active_summary = evaluate_active_model(workspace, df, model_id=args.model_id)
    production_summary = evaluate_production_baseline(df, feature_set="online_tactical")
    enhanced_summary = evaluate_production_baseline(
        add_rank_tz_engineered_features(df, "enhanced_tactical"),
        feature_set="enhanced_tactical",
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(workspace),
        **active_summary,
        "production_online_tactical": production_summary,
        "enhanced_tactical_experiment": enhanced_summary,
    }
    write_outputs(output_dir, summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
