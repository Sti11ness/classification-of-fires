"""
Classification metrics passport for leakage-safe rank_tz evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    if y_proba.size == 0:
        return 0.0
    top_indices = np.argsort(y_proba, axis=1)[:, -k:]
    top_classes = classes[top_indices]
    hits = [(truth in top_classes[idx]) for idx, truth in enumerate(y_true)]
    return float(np.mean(hits)) if hits else 0.0


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    if len(confidence) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not mask.any():
            continue
        bucket_conf = float(np.mean(confidence[mask]))
        bucket_acc = float(np.mean(y_true[mask] == y_pred[mask]))
        ece += (np.sum(mask) / len(confidence)) * abs(bucket_acc - bucket_conf)
    return float(ece)


def build_classification_metrics(
    *,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    class_to_rank_map: dict[int, float],
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the full metric passport for a classifier."""
    label_order = sorted(classes.tolist())
    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    confidence = y_proba.max(axis=1) if y_proba.size else np.zeros(len(y_pred))
    per_class_precision = {}
    per_class_recall = {}
    for idx, class_id in enumerate(label_order):
        rank_value = str(class_to_rank_map.get(int(class_id), class_id))
        predicted_total = cm[:, idx].sum()
        actual_total = cm[idx].sum()
        per_class_precision[rank_value] = float(cm[idx, idx] / predicted_total) if predicted_total else 0.0
        per_class_recall[rank_value] = float(cm[idx, idx] / actual_total) if actual_total else 0.0

    y_test_rank = np.array([class_to_rank_map.get(int(value), float(value)) for value in y_test.tolist()], dtype=float)
    y_pred_rank = np.array([class_to_rank_map.get(int(value), float(value)) for value in y_pred.tolist()], dtype=float)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "top_2_accuracy": _top_k_accuracy(y_test.to_numpy(), y_proba, classes, 2),
        "top_3_accuracy": _top_k_accuracy(y_test.to_numpy(), y_proba, classes, 3),
        "ordinal_mae": float(np.mean(np.abs(y_test_rank - y_pred_rank))) if len(y_test_rank) else 0.0,
        "under_dispatch_rate": float(np.mean(y_pred_rank < y_test_rank)) if len(y_test_rank) else 0.0,
        "calibration_error": _expected_calibration_error(y_test.to_numpy(), y_pred, confidence),
        "event_overlap_rate": float(split_metadata.get("event_overlap_rate", 0.0)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "n_classes": int(len(classes)),
        "classes": label_order,
        "classes_rank_values": [class_to_rank_map.get(int(value), float(value)) for value in label_order],
        "metric_primary": "f1_macro",
        "split_protocol": split_metadata.get("split_protocol"),
    }
