from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import spearmanr  # type: ignore
        r, _ = spearmanr(y_true, y_pred)
        if np.isnan(r):
            return None
        return float(r)
    except Exception:
        return None


def classification_metrics(y_true: np.ndarray, y_pred_label: np.ndarray, *, labels: Optional[list[str]] = None) -> Dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    cm = confusion_matrix(y_true, y_pred_label, labels=labels)
    report = classification_report(y_true, y_pred_label, labels=labels, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred_label)
    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def binary_classifier_metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    y_pred = (y_score >= threshold).astype(int)
    out = classification_metrics(y_true, y_pred)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    out["threshold"] = float(threshold)
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    sp = _safe_spearmanr(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman": sp,
    }
