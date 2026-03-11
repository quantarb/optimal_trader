from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .diagnostics import _read_csv_artifact
from pipeline.models import Artifact
from .situation_similarity import load_market_situation_cluster_artifact


def _feature_family_signature(feature_families: list[str]) -> str:
    cleaned = [str(value).strip() for value in list(feature_families or []) if str(value).strip()]
    return " + ".join(cleaned) if cleaned else "unattributed"


def _bucket_hold_days(value: float) -> str:
    if value <= 10:
        return "hold_1_10"
    if value <= 30:
        return "hold_11_30"
    if value <= 90:
        return "hold_31_90"
    return "hold_91_plus"


def _score_column(df: pd.DataFrame) -> str:
    for column in ("signal_score", "prediction_score", "prediction", "raw_prediction"):
        if column in df.columns:
            return column
    raise ValueError("Prediction artifact is missing a score-like column.")


def _source_model_payload(prediction_artifact: Artifact) -> tuple[dict[str, Any], dict[str, Any]]:
    model_artifact_id = int((prediction_artifact.metadata or {}).get("source_model_artifact_id") or 0)
    model_artifact = Artifact.objects.filter(pk=model_artifact_id).first()
    if model_artifact is None:
        return {}, {}
    return dict(model_artifact.content or {}), dict(model_artifact.metadata or {})


def _prepare_label_frame(label_artifact: Artifact) -> pd.DataFrame:
    labels = _read_csv_artifact(label_artifact).copy()
    required = {"date", "symbol", "trade_return", "hold_days"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Label artifact #{label_artifact.id} is missing columns: {', '.join(missing)}")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    labels["symbol"] = labels["symbol"].astype(str).str.strip().str.upper()
    labels["trade_return"] = pd.to_numeric(labels["trade_return"], errors="coerce")
    labels["hold_days"] = pd.to_numeric(labels["hold_days"], errors="coerce")
    if "side" in labels.columns:
        labels["side"] = labels["side"].astype(str).str.strip().str.lower()
    else:
        labels["side"] = "unknown"
    if "freq" in labels.columns:
        labels["freq"] = labels["freq"].astype(str).str.strip().replace("", "unknown")
    else:
        labels["freq"] = "unknown"
    labels["k"] = pd.to_numeric(labels["k"], errors="coerce").fillna(0).astype(int) if "k" in labels.columns else 0
    labels = labels.dropna(subset=["date", "symbol", "trade_return", "hold_days"]).copy()
    labels["hold_bucket"] = labels["hold_days"].apply(lambda value: _bucket_hold_days(float(value)))
    bucket_count = min(4, int(labels["trade_return"].nunique()))
    if bucket_count >= 2:
        labels["return_bucket"] = pd.qcut(
            labels["trade_return"].rank(method="first"),
            q=bucket_count,
            duplicates="drop",
        ).astype(str)
    else:
        labels["return_bucket"] = "single_bucket"
    labels["cluster_key"] = (
        labels["side"].replace("", "unknown")
        + "|"
        + labels["freq"]
        + "|k="
        + labels["k"].astype(str)
        + "|"
        + labels["hold_bucket"]
        + "|"
        + labels["return_bucket"]
    )
    return labels


def _selected_subset(prediction_artifact: Artifact, labels: pd.DataFrame, selection_quantile: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_df = _read_csv_artifact(prediction_artifact).copy()
    score_col = _score_column(prediction_df)
    prediction_df["date"] = pd.to_datetime(prediction_df["date"], errors="coerce")
    prediction_df["symbol"] = prediction_df["symbol"].astype(str).str.strip().str.upper()
    prediction_df[score_col] = pd.to_numeric(prediction_df[score_col], errors="coerce")
    merged = prediction_df.merge(
        labels,
        on=["date", "symbol"],
        how="inner",
        suffixes=("", "_label"),
    )
    merged = merged.dropna(subset=[score_col, "trade_return", "hold_days"]).copy()
    if merged.empty:
        return merged, {"score_column": score_col, "threshold": None}
    quantile = min(max(float(selection_quantile), 0.01), 0.99)
    threshold = float(merged[score_col].quantile(quantile))
    selected = merged[merged[score_col] >= threshold].copy()
    return selected, {"score_column": score_col, "threshold": threshold}


def _oracle_summary(labels: pd.DataFrame) -> dict[str, Any]:
    return {
        "oracle_rows": int(len(labels)),
        "symbols": int(labels["symbol"].nunique()),
        "clusters": int(labels["cluster_key"].nunique()),
        "long_rows": int((labels["side"] == "long").sum()),
        "short_rows": int((labels["side"] == "short").sum()),
        "avg_trade_return": round(float(labels["trade_return"].mean()), 6) if not labels.empty else 0.0,
        "median_trade_return": round(float(labels["trade_return"].median()), 6) if not labels.empty else 0.0,
        "avg_hold_days": round(float(labels["hold_days"].mean()), 4) if not labels.empty else 0.0,
    }


def _model_row_for_prediction_artifact(
    *,
    labels: pd.DataFrame,
    prediction_artifact: Artifact,
    selection_quantile: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected, score_meta = _selected_subset(prediction_artifact, labels, selection_quantile)
    model_content, model_metadata = _source_model_payload(prediction_artifact)
    feature_families = [str(value).strip() for value in list(model_metadata.get("feature_families") or []) if str(value).strip()]
    feature_signature = _feature_family_signature(feature_families)
    model_name = str(model_content.get("name") or model_metadata.get("model_name") or prediction_artifact.pipeline_run.name or f"artifact_{prediction_artifact.id}")
    oracle_cluster_keys = [str(value).strip() for value in list(model_metadata.get("oracle_cluster_keys") or []) if str(value).strip()]
    oracle_cluster_scope = str(model_metadata.get("oracle_cluster_scope") or ("specialist" if oracle_cluster_keys else "generalist")).strip() or "generalist"

    oracle_rows = len(labels)
    selected_rows = len(selected)
    cluster_coverage = float(selected["cluster_key"].nunique() / max(labels["cluster_key"].nunique(), 1)) if selected_rows else 0.0
    symbol_coverage = float(selected["symbol"].nunique() / max(labels["symbol"].nunique(), 1)) if selected_rows else 0.0
    model_row = {
        "artifact_id": int(prediction_artifact.id),
        "artifact_type": str(prediction_artifact.artifact_type),
        "model_name": model_name,
        "feature_families": feature_families,
        "feature_family_signature": feature_signature,
        "oracle_cluster_scope": oracle_cluster_scope,
        "oracle_cluster_keys": oracle_cluster_keys,
        "selection_quantile": float(selection_quantile),
        "score_column": str(score_meta.get("score_column") or ""),
        "selection_threshold": score_meta.get("threshold"),
        "oracle_rows": int(oracle_rows),
        "selected_rows": int(selected_rows),
        "oracle_recall": round(float(selected_rows / float(max(oracle_rows, 1))), 6),
        "selected_avg_trade_return": round(float(selected["trade_return"].mean()), 6) if selected_rows else 0.0,
        "selected_median_trade_return": round(float(selected["trade_return"].median()), 6) if selected_rows else 0.0,
        "selected_avg_hold_days": round(float(selected["hold_days"].mean()), 4) if selected_rows else 0.0,
        "cluster_coverage_rate": round(float(cluster_coverage), 6),
        "symbol_coverage_rate": round(float(symbol_coverage), 6),
    }
    return model_row, selected


def summarize_prediction_artifact_set_oracle_coverage(
    *,
    label_artifact: Artifact,
    prediction_artifacts: list[Artifact],
    selection_quantile: float = 0.8,
) -> dict[str, Any]:
    labels = _prepare_label_frame(label_artifact)
    if labels.empty or not prediction_artifacts:
        return {
            "prediction_artifact_count": 0,
            "oracle_recall": 0.0,
            "oracle_cluster_coverage_rate": 0.0,
            "oracle_selected_avg_trade_return": 0.0,
            "oracle_selected_rows_mean": 0.0,
            "feature_family_signature": "",
        }

    model_rows: list[dict[str, Any]] = []
    feature_signatures: list[str] = []
    for artifact in prediction_artifacts:
        row, _selected = _model_row_for_prediction_artifact(
            labels=labels,
            prediction_artifact=artifact,
            selection_quantile=float(selection_quantile),
        )
        model_rows.append(row)
        signature = str(row.get("feature_family_signature") or "").strip()
        if signature:
            feature_signatures.append(signature)
    if not model_rows:
        return {
            "prediction_artifact_count": 0,
            "oracle_recall": 0.0,
            "oracle_cluster_coverage_rate": 0.0,
            "oracle_selected_avg_trade_return": 0.0,
            "oracle_selected_rows_mean": 0.0,
            "feature_family_signature": "",
        }

    family_signature = feature_signatures[0] if len(set(feature_signatures)) == 1 else " / ".join(sorted(dict.fromkeys(feature_signatures)))
    return {
        "prediction_artifact_count": int(len(model_rows)),
        "oracle_recall": round(float(sum(float(row.get("oracle_recall") or 0.0) for row in model_rows) / len(model_rows)), 6),
        "oracle_cluster_coverage_rate": round(float(sum(float(row.get("cluster_coverage_rate") or 0.0) for row in model_rows) / len(model_rows)), 6),
        "oracle_selected_avg_trade_return": round(float(sum(float(row.get("selected_avg_trade_return") or 0.0) for row in model_rows) / len(model_rows)), 6),
        "oracle_selected_rows_mean": round(float(sum(float(row.get("selected_rows") or 0.0) for row in model_rows) / len(model_rows)), 6),
        "feature_family_signature": family_signature,
    }


def build_oracle_trade_report(
    *,
    label_artifact: Artifact,
    prediction_artifacts: list[Artifact],
    market_situation_artifact: Artifact | None = None,
    selection_quantile: float = 0.8,
    top_cluster_count: int = 20,
) -> dict[str, Any]:
    labels = _prepare_label_frame(label_artifact)
    if labels.empty:
        raise ValueError("Label artifact did not contain any usable oracle trade rows.")

    model_rows: list[dict[str, Any]] = []
    cluster_recovery_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    market_situation_cluster_recovery_rows: list[dict[str, Any]] = []

    labels_by_cluster = labels.groupby("cluster_key", observed=True)
    cluster_baseline = (
        labels_by_cluster.agg(
            oracle_rows=("cluster_key", "size"),
            avg_trade_return=("trade_return", "mean"),
            median_trade_return=("trade_return", "median"),
            avg_hold_days=("hold_days", "mean"),
        )
        .reset_index()
    )

    family_aggregate: dict[str, dict[str, Any]] = {}
    family_presence_aggregate: dict[str, dict[str, Any]] = {}
    selected_index_sets: dict[int, set[tuple[str, str]]] = {}
    market_situation_assignments = pd.DataFrame()
    market_situation_baseline = pd.DataFrame()
    if market_situation_artifact is not None:
        try:
            market_bundle = load_market_situation_cluster_artifact(market_situation_artifact)
            market_situation_assignments = market_bundle.assignments.copy()
            market_situation_baseline = (
                market_situation_assignments.groupby(["cluster_id", "cluster_description"], observed=True)
                .agg(
                    oracle_rows=("cluster_id", "size"),
                    median_trade_return=("trade_return", "median"),
                    avg_hold_days=("hold_days", "mean"),
                    yearly_median_return_std=("trade_return", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) if len(s) > 1 else 0.0)),
                )
                .reset_index()
            )
        except Exception:
            market_situation_assignments = pd.DataFrame()
            market_situation_baseline = pd.DataFrame()

    for artifact in prediction_artifacts:
        model_row, selected = _model_row_for_prediction_artifact(
            labels=labels,
            prediction_artifact=artifact,
            selection_quantile=float(selection_quantile),
        )
        feature_families = list(model_row.get("feature_families") or [])
        feature_signature = str(model_row.get("feature_family_signature") or "")
        model_name = str(model_row.get("model_name") or "")
        model_rows.append(model_row)
        selected_index_sets[int(artifact.id)] = {
            (str(item[0])[:10], str(item[1]).strip().upper())
            for item in selected[["date", "symbol"]].itertuples(index=False, name=None)
        }

        for _, row in cluster_baseline.iterrows():
            cluster_key = str(row["cluster_key"])
            cluster_oracle_rows = int(row["oracle_rows"])
            selected_cluster = selected[selected["cluster_key"] == cluster_key]
            cluster_recovery_rows.append(
                {
                    "cluster_key": cluster_key,
                    "model_name": model_name,
                    "artifact_id": int(artifact.id),
                    "feature_family_signature": feature_signature,
                    "oracle_rows": cluster_oracle_rows,
                    "selected_rows": int(len(selected_cluster)),
                    "cluster_recall": round(float(len(selected_cluster) / float(max(cluster_oracle_rows, 1))), 6),
                    "selected_avg_trade_return": round(float(selected_cluster["trade_return"].mean()), 6) if len(selected_cluster) else 0.0,
                }
            )

        if not market_situation_baseline.empty and not selected.empty:
            join_cols = [column for column in ("date", "symbol", "side", "k") if column in selected.columns and column in market_situation_assignments.columns]
            selected_market = selected.merge(
                market_situation_assignments[join_cols + ["cluster_id", "cluster_description"]],
                on=join_cols,
                how="left",
            ) if join_cols else pd.DataFrame()
            if not selected_market.empty:
                model_row["market_situation_cluster_coverage_rate"] = round(
                    float(selected_market["cluster_id"].dropna().nunique() / max(market_situation_baseline["cluster_id"].nunique(), 1)),
                    6,
                )
                for _, base_row in market_situation_baseline.iterrows():
                    cluster_id = str(base_row["cluster_id"])
                    selected_cluster = selected_market[selected_market["cluster_id"].astype(str) == cluster_id]
                    market_situation_cluster_recovery_rows.append(
                        {
                            "cluster_id": cluster_id,
                            "cluster_description": str(base_row.get("cluster_description") or ""),
                            "model_name": model_name,
                            "oracle_rows": int(base_row.get("oracle_rows") or 0),
                            "selected_rows": int(len(selected_cluster)),
                            "cluster_recall": round(float(len(selected_cluster) / float(max(int(base_row.get("oracle_rows") or 0), 1))), 6),
                            "median_trade_return": round(float(base_row.get("median_trade_return") or 0.0), 6),
                            "avg_hold_days": round(float(base_row.get("avg_hold_days") or 0.0), 6),
                            "yearly_median_return_std": round(float(base_row.get("yearly_median_return_std") or 0.0), 6),
                        }
                    )
            else:
                model_row["market_situation_cluster_coverage_rate"] = 0.0
        else:
            model_row["market_situation_cluster_coverage_rate"] = 0.0

        signature_state = family_aggregate.setdefault(
            feature_signature,
            {"signature": feature_signature, "models": 0, "oracle_recall_sum": 0.0, "cluster_coverage_sum": 0.0, "selected_return_sum": 0.0},
        )
        signature_state["models"] += 1
        signature_state["oracle_recall_sum"] += float(model_row["oracle_recall"])
        signature_state["cluster_coverage_sum"] += float(model_row["cluster_coverage_rate"])
        signature_state["selected_return_sum"] += float(model_row["selected_avg_trade_return"])

        for family in feature_families:
            state = family_presence_aggregate.setdefault(
                family,
                {"family": family, "models": 0, "oracle_recall_sum": 0.0, "cluster_coverage_sum": 0.0, "selected_return_sum": 0.0},
            )
            state["models"] += 1
            state["oracle_recall_sum"] += float(model_row["oracle_recall"])
            state["cluster_coverage_sum"] += float(model_row["cluster_coverage_rate"])
            state["selected_return_sum"] += float(model_row["selected_avg_trade_return"])

    model_rows.sort(key=lambda row: (float(row["selected_avg_trade_return"]), float(row["cluster_coverage_rate"]), float(row["oracle_recall"])), reverse=True)

    best_cluster_by_model: dict[str, dict[str, Any]] = {}
    for row in cluster_recovery_rows:
        cluster_key = str(row["cluster_key"])
        current = best_cluster_by_model.get(cluster_key)
        if current is None or float(row["cluster_recall"]) > float(current["cluster_recall"]):
            best_cluster_by_model[cluster_key] = row

    cluster_rows: list[dict[str, Any]] = []
    missed_cluster_rows: list[dict[str, Any]] = []
    for _, row in cluster_baseline.sort_values(["avg_trade_return", "oracle_rows"], ascending=[False, False]).head(int(top_cluster_count)).iterrows():
        cluster_key = str(row["cluster_key"])
        best = dict(best_cluster_by_model.get(cluster_key) or {})
        cluster_row = {
            "cluster_key": cluster_key,
            "oracle_rows": int(row["oracle_rows"]),
            "avg_trade_return": round(float(row["avg_trade_return"]), 6),
            "median_trade_return": round(float(row["median_trade_return"]), 6),
            "avg_hold_days": round(float(row["avg_hold_days"]), 4),
            "best_model_name": str(best.get("model_name") or ""),
            "best_cluster_recall": round(float(best.get("cluster_recall") or 0.0), 6),
            "best_feature_family_signature": str(best.get("feature_family_signature") or ""),
        }
        cluster_rows.append(cluster_row)
        missed_cluster_rows.append(
            {
                **cluster_row,
                "miss_rate": round(float(1.0 - float(cluster_row["best_cluster_recall"])), 6),
            }
        )

    best_market_cluster_by_model: dict[str, dict[str, Any]] = {}
    for row in market_situation_cluster_recovery_rows:
        cluster_id = str(row["cluster_id"])
        current = best_market_cluster_by_model.get(cluster_id)
        if current is None or float(row["cluster_recall"]) > float(current["cluster_recall"]):
            best_market_cluster_by_model[cluster_id] = row
    market_situation_cluster_rows: list[dict[str, Any]] = []
    missed_market_situation_cluster_rows: list[dict[str, Any]] = []
    if not market_situation_baseline.empty:
        for _, row in market_situation_baseline.sort_values(["median_trade_return", "oracle_rows"], ascending=[False, False]).head(int(top_cluster_count)).iterrows():
            cluster_id = str(row["cluster_id"])
            best = dict(best_market_cluster_by_model.get(cluster_id) or {})
            cluster_row = {
                "cluster_id": cluster_id,
                "cluster_description": str(row.get("cluster_description") or ""),
                "oracle_rows": int(row.get("oracle_rows") or 0),
                "median_trade_return": round(float(row.get("median_trade_return") or 0.0), 6),
                "avg_hold_days": round(float(row.get("avg_hold_days") or 0.0), 6),
                "yearly_median_return_std": round(float(row.get("yearly_median_return_std") or 0.0), 6),
                "best_model_name": str(best.get("model_name") or ""),
                "best_cluster_recall": round(float(best.get("cluster_recall") or 0.0), 6),
            }
            market_situation_cluster_rows.append(cluster_row)
            missed_market_situation_cluster_rows.append(
                {
                    **cluster_row,
                    "miss_rate": round(float(1.0 - float(cluster_row["best_cluster_recall"])), 6),
                }
            )

    overlap_rows: list[dict[str, Any]] = []
    sorted_model_rows = list(model_rows)[: min(len(model_rows), 8)]
    for left_index, left_row in enumerate(sorted_model_rows):
        left_set = selected_index_sets.get(int(left_row.get("artifact_id") or 0), set())
        for right_row in sorted_model_rows[left_index + 1:]:
            right_set = selected_index_sets.get(int(right_row.get("artifact_id") or 0), set())
            union = len(left_set | right_set)
            if union <= 0:
                continue
            intersection = len(left_set & right_set)
            overlap_rows.append(
                {
                    "left_model_name": str(left_row.get("model_name") or ""),
                    "right_model_name": str(right_row.get("model_name") or ""),
                    "left_artifact_id": int(left_row.get("artifact_id") or 0),
                    "right_artifact_id": int(right_row.get("artifact_id") or 0),
                    "jaccard_overlap": round(float(intersection / union), 6),
                    "shared_selected_rows": int(intersection),
                    "union_selected_rows": int(union),
                }
            )
    overlap_rows.sort(key=lambda row: (float(row.get("jaccard_overlap") or 0.0), int(row.get("shared_selected_rows") or 0)), reverse=True)

    for state in family_aggregate.values():
        models = max(int(state["models"]), 1)
        family_rows.append(
            {
                "family_kind": "signature",
                "family_name": str(state["signature"]),
                "models": int(models),
                "avg_oracle_recall": round(float(state["oracle_recall_sum"] / models), 6),
                "avg_cluster_coverage_rate": round(float(state["cluster_coverage_sum"] / models), 6),
                "avg_selected_trade_return": round(float(state["selected_return_sum"] / models), 6),
            }
        )
    for state in family_presence_aggregate.values():
        models = max(int(state["models"]), 1)
        family_rows.append(
            {
                "family_kind": "family",
                "family_name": str(state["family"]),
                "models": int(models),
                "avg_oracle_recall": round(float(state["oracle_recall_sum"] / models), 6),
                "avg_cluster_coverage_rate": round(float(state["cluster_coverage_sum"] / models), 6),
                "avg_selected_trade_return": round(float(state["selected_return_sum"] / models), 6),
            }
        )
    family_rows.sort(key=lambda row: (float(row["avg_selected_trade_return"]), float(row["avg_cluster_coverage_rate"])), reverse=True)

    recommendations: list[str] = []
    if model_rows:
        top_model = model_rows[0]
        if float(top_model["cluster_coverage_rate"]) < 0.35:
            recommendations.append("Top model recovers only a narrow subset of oracle trade clusters. Add more diverse feature families or train cluster-specialist heads.")
        if float(top_model["oracle_recall"]) < 0.2:
            recommendations.append("Oracle recall is still low at the current threshold. Lower the score gate or calibrate the model before policy search.")
        if float(top_model.get("market_situation_cluster_coverage_rate") or 0.0) < 0.35 and not market_situation_baseline.empty:
            recommendations.append("Learned market situation clusters are still sparsely recovered. Add cluster-aware features or train cluster-conditional models.")
    specialist_rows = [row for row in model_rows if str(row.get("oracle_cluster_scope") or "") == "specialist"]
    generalist_rows = [row for row in model_rows if str(row.get("oracle_cluster_scope") or "generalist") != "specialist"]
    if specialist_rows and generalist_rows:
        best_specialist = max(specialist_rows, key=lambda row: float(row.get("selected_avg_trade_return") or 0.0))
        best_generalist = max(generalist_rows, key=lambda row: float(row.get("selected_avg_trade_return") or 0.0))
        if float(best_specialist.get("selected_avg_trade_return") or 0.0) > float(best_generalist.get("selected_avg_trade_return") or 0.0):
            recommendations.append("Best specialist model captures higher-return oracle subsets than the best generalist. Consider specialist heads or cluster-conditioned ranking.")
    if family_rows:
        top_family = family_rows[0]
        recommendations.append(f"Best coverage currently comes from '{top_family['family_name']}'. Use it as the baseline family set for the next MTL or RL pass.")
    if not recommendations:
        recommendations.append("No immediate gaps detected. Next step is to widen the prediction set and compare more feature-family signatures.")

    observations = [
        f"Oracle label set contains {len(labels)} recoverable trade rows across {labels['symbol'].nunique()} symbols and {labels['cluster_key'].nunique()} explainable clusters.",
        f"Compared {len(prediction_artifacts)} prediction artifacts against the same oracle trade set.",
    ]
    if specialist_rows:
        observations.append(f"{len(specialist_rows)} specialist model variants were compared against {len(generalist_rows)} generalist variants.")
    if not market_situation_baseline.empty:
        observations.append(f"Market situation taxonomy contributed {market_situation_baseline['cluster_id'].nunique()} learned clusters for coverage analysis.")

    return {
        "kind": "oracle_trade_report",
        "artifacts": {
            "labels": int(label_artifact.id),
            "prediction_artifacts": [int(artifact.id) for artifact in prediction_artifacts],
            "market_situation_artifact": int(market_situation_artifact.id) if market_situation_artifact is not None else 0,
        },
        "oracle_summary": _oracle_summary(labels),
        "model_rows": model_rows,
        "cluster_rows": cluster_rows,
        "missed_cluster_rows": sorted(missed_cluster_rows, key=lambda row: (float(row["best_cluster_recall"]), -int(row["oracle_rows"])))[: int(top_cluster_count)],
        "cluster_recovery_rows": cluster_recovery_rows,
        "market_situation_summary": {
            "clusters": int(market_situation_baseline["cluster_id"].nunique()) if not market_situation_baseline.empty else 0,
        },
        "market_situation_cluster_rows": market_situation_cluster_rows,
        "missed_market_situation_cluster_rows": sorted(missed_market_situation_cluster_rows, key=lambda row: (float(row["best_cluster_recall"]), -int(row["oracle_rows"])))[: int(top_cluster_count)],
        "model_overlap_rows": overlap_rows[:20],
        "feature_family_rows": family_rows,
        "selection_quantile": float(selection_quantile),
        "observations": observations,
        "recommendations": recommendations,
    }


def write_oracle_trade_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
