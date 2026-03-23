from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr  # type: ignore

        corr, _pvalue = spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return None
        return float(corr)
    except Exception:
        return None


def safe_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan")


def safe_macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else float("nan")


def safe_mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def safe_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def safe_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((true_arr - pred_arr) ** 2))


def safe_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    mse = safe_mse(y_true, y_pred)
    return float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")


def safe_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_series = pd.Series(list(y_true), dtype="float64")
    pred_series = pd.Series(list(y_pred), dtype="float64")
    if true_series.nunique(dropna=True) < 2 or pred_series.nunique(dropna=True) < 2:
        return float("nan")
    value = true_series.corr(pred_series, method="pearson")
    return float(value) if pd.notna(value) else float("nan")


def safe_spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_series = pd.Series(list(y_true), dtype="float64")
    pred_series = pd.Series(list(y_pred), dtype="float64")
    if true_series.nunique(dropna=True) < 2 or pred_series.nunique(dropna=True) < 2:
        return float("nan")
    value = true_series.corr(pred_series, method="spearman")
    return float(value) if pd.notna(value) else float("nan")


def render_metric_value(value: Any) -> str:
    if isinstance(value, (int, float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        return f"{numeric:.4f}"
    return str(value)


def format_metric_report(label: str, metrics: dict[str, Any]) -> str:
    sections = [
        ("overall", ["model_score"]),
        ("entry_action", ["entry_support", "entry_action_accuracy", "entry_action_macro_f1"]),
        ("entry_return", ["entry_support", "entry_return_mae", "entry_return_rmse", "entry_return_spearman"]),
        (
            "entry_signed_return",
            ["entry_support", "entry_signed_return_mae", "entry_signed_return_rmse", "entry_signed_return_spearman"],
        ),
        ("entry_duration", ["entry_support", "entry_duration_mae", "entry_duration_rmse", "entry_duration_spearman"]),
        ("entry_reconstruction", ["entry_support", "entry_context_recon_cosine_mean", "entry_numeric_recon_mae"]),
        ("exit_action", ["exit_support", "exit_action_accuracy", "exit_action_macro_f1"]),
        ("exit_return", ["exit_support", "exit_return_mae", "exit_return_rmse", "exit_return_spearman"]),
        (
            "exit_signed_return",
            ["exit_support", "exit_signed_return_mae", "exit_signed_return_rmse", "exit_signed_return_spearman"],
        ),
        ("exit_duration", ["exit_support", "exit_duration_mae", "exit_duration_rmse", "exit_duration_spearman"]),
        ("exit_reconstruction", ["exit_support", "exit_context_recon_cosine_mean", "exit_numeric_recon_mae"]),
        ("pair_duration", ["pair_support", "pair_duration_mae", "pair_duration_rmse", "pair_duration_spearman"]),
        ("transition", ["transition_support", "transition_context_cosine_mean", "transition_numeric_recon_mae"]),
    ]
    lines = [label]
    covered = {item for _, keys in sections for item in keys}
    for section_name, keys in sections:
        rendered_items: list[str] = []
        for key in keys:
            if key not in metrics:
                continue
            short_key = "support" if key.endswith("_support") else (
                key.replace(section_name + "_", "") if key.startswith(section_name + "_") else key
            )
            rendered_items.append(f"{short_key}={render_metric_value(metrics[key])}")
        if rendered_items:
            lines.append(f"  {section_name}: " + " | ".join(rendered_items))
    remaining = [key for key in sorted(metrics) if key not in covered]
    if remaining:
        lines.append("  other: " + " | ".join(f"{key}={render_metric_value(metrics[key])}" for key in remaining))
    return "\n".join(lines)


def build_flair_action_report(task_name: str, y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report

    label_list = list(labels)
    report_text = classification_report(
        y_true,
        y_pred,
        digits=4,
        target_names=label_list,
        zero_division=0,
        labels=label_list,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_list,
        zero_division=0,
        output_dict=True,
        labels=label_list,
    )
    accuracy = round(accuracy_score(y_true, y_pred), 4) if len(y_true) else float("nan")
    if len(label_list) == 1 and "macro avg" in report_dict:
        report_dict["micro avg"] = report_dict["macro avg"]
    if "micro avg" not in report_dict and "macro avg" in report_dict:
        report_dict["micro avg"] = {}
        for metric_key in report_dict["macro avg"]:
            if metric_key != "support":
                report_dict["micro avg"][metric_key] = accuracy
            else:
                report_dict["micro avg"][metric_key] = report_dict["macro avg"]["support"]
    detailed_result = (
        "\nResults:"
        f"\n- F-score (micro) {round(report_dict['micro avg']['f1-score'], 4)}"
        f"\n- F-score (macro) {round(report_dict['macro avg']['f1-score'], 4)}"
        f"\n- Accuracy {accuracy}"
        "\n\nBy class:\n" + report_text
    )
    return {"task": task_name, "text": report_text, "dict": report_dict, "accuracy": accuracy, "detailed_result": detailed_result}


def build_action_classification_report_df(
    split: str,
    task_name: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    flair_report = build_flair_action_report(task_name, y_true, y_pred, labels)
    report = flair_report["dict"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        label_metrics = report.get(label, {})
        rows.append(
            {
                "split": split,
                "task": task_name,
                "label": label,
                "precision": float(label_metrics.get("precision", float("nan"))),
                "recall": float(label_metrics.get("recall", float("nan"))),
                "f1-score": float(label_metrics.get("f1-score", float("nan"))),
                "support": int(label_metrics.get("support", 0.0)),
            }
        )
    rows.append(
        {
            "split": split,
            "task": task_name,
            "label": "accuracy",
            "precision": float("nan"),
            "recall": float("nan"),
            "f1-score": float(flair_report["accuracy"]),
            "support": int(len(y_true)),
        }
    )
    for aggregate_label in ("micro avg", "macro avg", "weighted avg"):
        aggregate_metrics = report.get(aggregate_label, {})
        rows.append(
            {
                "split": split,
                "task": task_name,
                "label": aggregate_label,
                "precision": float(aggregate_metrics.get("precision", float("nan"))),
                "recall": float(aggregate_metrics.get("recall", float("nan"))),
                "f1-score": float(aggregate_metrics.get("f1-score", float("nan"))),
                "support": int(aggregate_metrics.get("support", len(y_true))),
            }
        )
    return pd.DataFrame(rows)


def build_flair_regression_result(task_name: str, y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    mse = safe_mse(y_true, y_pred)
    mae = safe_mae(y_true, y_pred)
    pearson = safe_pearson(y_true, y_pred)
    spearman = safe_spearman(y_true, y_pred)
    detailed_result = (
        f"AVG: mse: {mse:.4f} - "
        f"mae: {mae:.4f} - "
        f"pearson: {pearson:.4f} - "
        f"spearman: {spearman:.4f}"
    )
    return {
        "task": task_name,
        "support": int(len(y_true)),
        "mse": mse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "detailed_result": detailed_result,
    }


def build_context_numeric_result(task_name: str, support: int, context_cosine: float, numeric_recon_mae: float) -> dict[str, Any]:
    return {
        "task": task_name,
        "support": int(support),
        "context_cosine": float(context_cosine),
        "numeric_recon_mae": float(numeric_recon_mae),
        "detailed_result": (
            f"AVG: context_cosine: {float(context_cosine):.4f} - "
            f"numeric_recon_mae: {float(numeric_recon_mae):.4f}"
        ),
    }


def build_action_f1_comparison_df(
    split: str,
    metrics: dict[str, Any],
    task_buffers: dict[str, dict[str, Sequence[Any]]],
    task_labels: dict[str, Sequence[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name, labels in task_labels.items():
        flair_report = build_flair_action_report(
            task_name,
            task_buffers[task_name]["action_true"],
            task_buffers[task_name]["action_pred"],
            list(labels),
        )
        report_macro_f1 = float(flair_report["dict"]["macro avg"]["f1-score"])
        report_micro_f1 = float(flair_report["dict"]["micro avg"]["f1-score"])
        metric_macro_f1 = float(metrics[f"{task_name}_action_macro_f1"])
        rows.append(
            {
                "split": split,
                "task": task_name,
                "metric_macro_f1": metric_macro_f1,
                "report_macro_f1": report_macro_f1,
                "report_micro_f1": report_micro_f1,
                "difference": metric_macro_f1 - report_macro_f1,
            }
        )
    return pd.DataFrame(rows)


def build_regression_task_report_df(
    split: str,
    metrics: dict[str, Any],
    task_buffers: dict[str, dict[str, Sequence[Any]]],
    regression_specs: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    context_numeric_specs: Sequence[tuple[str, int, float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_name, y_true, y_pred in regression_specs:
        rows.append({"split": split, **build_flair_regression_result(task_name, y_true, y_pred)})
    for task_name, support, context_cosine, numeric_recon_mae in context_numeric_specs:
        rows.append(
            {
                "split": split,
                **build_context_numeric_result(task_name, support, context_cosine, numeric_recon_mae),
            }
        )
    return pd.DataFrame(rows)


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
