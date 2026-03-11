from __future__ import annotations

from typing import Any

import numpy as np


def _safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr  # type: ignore

        corr, _pvalue = spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return None
        return float(corr)
    except Exception:
        return None


def classification_metrics(
    y_true: np.ndarray,
    y_pred_label: np.ndarray,
    *,
    labels: list[str] | list[int] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred_label, labels=labels)
    report = classification_report(y_true, y_pred_label, labels=labels, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred_label)
    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def binary_classifier_metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y_pred = (y_score >= threshold).astype(int)
    out = classification_metrics(y_true, y_pred)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    out["threshold"] = float(threshold)
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    spearman = _safe_spearmanr(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }
