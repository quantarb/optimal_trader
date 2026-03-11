from __future__ import annotations

from django.shortcuts import render

from .report_catalog import (
    load_cohort_summary_files,
    load_diagnostic_report_files,
    load_feature_attribution_files,
    load_json_payload_from_uri,
    load_oracle_report_files,
)
from .view_support import _annotate_rows_with_bar_pct, _cohort_summary_detail, _to_int


def pipeline_cohorts_view(request, *, artifact_dir):
    summaries = load_cohort_summary_files(limit=30, artifact_dir=artifact_dir)
    selected_name = str(request.GET.get("summary") or "").strip()
    selected_summary = next((summary for summary in summaries if summary["name"] == selected_name), summaries[0] if summaries else None)
    detail = _cohort_summary_detail(dict(selected_summary.get("payload") or {})) if selected_summary is not None else {
        "rows": [],
        "rejected_rows": [],
        "label_artifact_id": 0,
        "feature_artifact_id": 0,
        "leaderboard": {},
        "report_summary": {},
        "research_profile": {},
    }
    all_detail_rows = list(detail["rows"])
    model_filter = str(request.GET.get("model") or "").strip().lower()
    family_filter = str(request.GET.get("family") or "").strip().lower()
    k_filter = str(request.GET.get("k") or "").strip()
    sort_by = str(request.GET.get("sort") or "final_equity").strip().lower()

    filtered_rows = all_detail_rows
    if model_filter:
        filtered_rows = [row for row in filtered_rows if model_filter in str(row.get("variant_name") or "").lower()]
    if family_filter:
        filtered_rows = [row for row in filtered_rows if any(family_filter in str(item).lower() for item in list(row.get("feature_families") or []))]
    if k_filter:
        filtered_rows = [row for row in filtered_rows if k_filter in {str(v) for v in list(row.get("label_ks") or [])}]

    sort_options = {
        "final_equity": lambda row: float(row.get("final_equity") or 0.0),
        "walk_forward_final_equity": lambda row: float(row.get("walk_forward_final_equity") or 0.0),
        "cumulative_return": lambda row: float(row.get("cumulative_return") or 0.0),
        "walk_forward_excess": lambda row: float(row.get("walk_forward_excess_cumulative_return") or row.get("excess_cumulative_return") or 0.0),
        "mean_fold_excess": lambda row: float(row.get("mean_fold_excess_cumulative_return") or 0.0),
        "fold_dispersion": lambda row: -float(row.get("fold_excess_cumulative_return_std") or 10**9),
        "valid_fold_rate": lambda row: float(row.get("valid_fold_rate") or 0.0),
        "max_drawdown": lambda row: float(row.get("max_drawdown") or -999.0),
        "fit_seconds": lambda row: -float(row.get("fit_seconds") or 10**9),
        "total_runtime": lambda row: -float(row.get("total_runtime_seconds") or 10**9),
        "return_to_drawdown": lambda row: float(row.get("return_to_drawdown") or -999.0),
    }
    filtered_rows.sort(key=sort_options.get(sort_by, sort_options["final_equity"]), reverse=True)
    show_all_variants = str(request.GET.get("show") or "").strip().lower() == "all"
    detail_rows = filtered_rows if show_all_variants else filtered_rows[:20]
    detail_rows = _annotate_rows_with_bar_pct(detail_rows, "mean_fold_excess_cumulative_return", "excess_bar_pct")
    detail_rows = _annotate_rows_with_bar_pct(detail_rows, "walk_forward_excess_cumulative_return", "walk_forward_excess_bar_pct")
    rejected_rows = _annotate_rows_with_bar_pct(detail["rejected_rows"][:20], "fold_excess_cumulative_return_std", "dispersion_bar_pct")
    available_families = sorted({str(family) for row in all_detail_rows for family in list(row.get("feature_families") or []) if str(family).strip()})
    available_ks = sorted({str(value) for row in all_detail_rows for value in list(row.get("label_ks") or []) if str(value).strip()}, key=lambda value: int(value))
    return render(
        request,
        "pipeline/cohorts.html",
        {
            "summaries": summaries,
            "selected_summary": selected_summary,
            "detail_rows": detail_rows,
            "detail_row_count": len(filtered_rows),
            "detail_hidden_count": max(len(filtered_rows) - len(detail_rows), 0),
            "show_all_variants": show_all_variants,
            "label_artifact_id": detail["label_artifact_id"],
            "feature_artifact_id": detail["feature_artifact_id"],
            "leaderboard": detail["leaderboard"],
            "rejected_rows": rejected_rows,
            "rejected_row_count": len(detail["rejected_rows"]),
            "report_summary": detail["report_summary"],
            "research_profile": detail["research_profile"],
            "model_filter": model_filter,
            "family_filter": family_filter,
            "k_filter": k_filter,
            "sort_by": sort_by,
            "available_families": available_families,
            "available_ks": available_ks,
        },
    )


def pipeline_research_reports_view(request, *, artifact_dir):
    summaries = [item for item in load_cohort_summary_files(limit=30, artifact_dir=artifact_dir) if item.get("kind") == "research_report"]
    selected_name = str(request.GET.get("summary") or "").strip()
    selected_summary = next((item for item in summaries if item["name"] == selected_name), summaries[0] if summaries else None)
    detail = _cohort_summary_detail(dict(selected_summary.get("payload") or {})) if selected_summary is not None else {
        "rows": [],
        "rejected_rows": [],
        "label_artifact_id": 0,
        "feature_artifact_id": 0,
        "leaderboard": {},
        "report_summary": {},
        "research_profile": {},
    }
    return render(
        request,
        "pipeline/research_reports.html",
        {
            "summaries": summaries,
            "selected_summary": selected_summary,
            "leaderboard_rows": _annotate_rows_with_bar_pct(detail["rows"][:20], "mean_fold_excess_cumulative_return", "excess_bar_pct"),
            "rejected_rows": _annotate_rows_with_bar_pct(detail["rejected_rows"][:20], "fold_excess_cumulative_return_std", "dispersion_bar_pct"),
            "oracle_variant_rows": _annotate_rows_with_bar_pct(list((selected_summary or {}).get("payload", {}).get("oracle_variant_rows") or [])[:12], "oracle_cluster_coverage_rate", "coverage_bar_pct"),
            "oracle_feature_family_rows": _annotate_rows_with_bar_pct(list((selected_summary or {}).get("payload", {}).get("oracle_feature_family_rows") or [])[:12], "avg_oracle_cluster_coverage_rate", "coverage_bar_pct"),
            "report_summary": detail["report_summary"],
            "research_profile": detail["research_profile"],
            "leaderboard": detail["leaderboard"],
            "label_artifact_id": detail["label_artifact_id"],
            "feature_artifact_id": detail["feature_artifact_id"],
            "specialist_leaderboard_rows": [
                row
                for row in _annotate_rows_with_bar_pct(detail["rows"][:50], "oracle_recall", "oracle_recall_bar_pct")
                if str(row.get("oracle_cluster_scope") or "generalist") == "specialist"
            ][:12],
        },
    )


def pipeline_diagnostic_reports_view(request, *, artifact_dir):
    summaries = load_diagnostic_report_files(limit=30, artifact_dir=artifact_dir)
    selected_name = str(request.GET.get("summary") or "").strip()
    selected_summary = next((item for item in summaries if item["name"] == selected_name), summaries[0] if summaries else None)
    payload = dict(selected_summary.get("payload") or {}) if selected_summary is not None else {}
    quantiles = dict(payload.get("prediction_quantiles") or {})
    combined_rows = _annotate_rows_with_bar_pct(list(quantiles.get("combined_rank_mean") or []), "avg_trade_return", "return_bar_pct")
    regressor_rows = _annotate_rows_with_bar_pct(list(quantiles.get("regressor_trade_return") or []), "avg_trade_return", "return_bar_pct")
    rl_rows = _annotate_rows_with_bar_pct(list(payload.get("rl_results") or []), "combined_total_return_pct", "return_bar_pct")
    return render(
        request,
        "pipeline/diagnostic_reports.html",
        {
            "summaries": summaries,
            "selected_summary": selected_summary,
            "payload": payload,
            "recommendations": list(payload.get("recommendations") or []),
            "observations": list(payload.get("observations") or []),
            "combined_rows": combined_rows,
            "regressor_rows": regressor_rows,
            "rl_rows": rl_rows,
            "candidate_rule": dict(payload.get("candidate_rule") or {}),
            "backtest_summary": dict(payload.get("backtest_summary") or {}),
            "backtest_config": dict(payload.get("backtest_config") or {}),
            "best_rl_result": dict(payload.get("best_rl_result") or {}),
            "ae_signal_bug_check": dict(payload.get("ae_signal_bug_check") or {}),
        },
    )


def pipeline_oracle_reports_view(request, *, artifact_dir):
    summaries = load_oracle_report_files(limit=30, artifact_dir=artifact_dir)
    selected_name = str(request.GET.get("summary") or "").strip()
    selected_summary = next((item for item in summaries if item["name"] == selected_name), summaries[0] if summaries else None)
    payload = dict(selected_summary.get("payload") or {}) if selected_summary is not None else {}
    model_rows = _annotate_rows_with_bar_pct(list(payload.get("model_rows") or []), "selected_avg_trade_return", "return_bar_pct")
    cluster_rows = _annotate_rows_with_bar_pct(list(payload.get("cluster_rows") or []), "best_cluster_recall", "recall_bar_pct")
    missed_cluster_rows = _annotate_rows_with_bar_pct(list(payload.get("missed_cluster_rows") or []), "miss_rate", "miss_bar_pct")
    family_rows = _annotate_rows_with_bar_pct(list(payload.get("feature_family_rows") or []), "avg_selected_trade_return", "return_bar_pct")
    overlap_rows = _annotate_rows_with_bar_pct(list(payload.get("model_overlap_rows") or []), "jaccard_overlap", "overlap_bar_pct")
    market_cluster_rows = _annotate_rows_with_bar_pct(list(payload.get("market_situation_cluster_rows") or []), "best_cluster_recall", "recall_bar_pct")
    missed_market_cluster_rows = _annotate_rows_with_bar_pct(list(payload.get("missed_market_situation_cluster_rows") or []), "miss_rate", "miss_bar_pct")
    return render(
        request,
        "pipeline/oracle_reports.html",
        {
            "summaries": summaries,
            "selected_summary": selected_summary,
            "payload": payload,
            "oracle_summary": dict(payload.get("oracle_summary") or {}),
            "model_rows": model_rows[:20],
            "cluster_rows": cluster_rows[:20],
            "missed_cluster_rows": missed_cluster_rows[:20],
            "market_cluster_rows": market_cluster_rows[:20],
            "missed_market_cluster_rows": missed_market_cluster_rows[:20],
            "model_overlap_rows": overlap_rows[:20],
            "family_rows": family_rows[:20],
            "recommendations": list(payload.get("recommendations") or []),
            "observations": list(payload.get("observations") or []),
        },
    )


def pipeline_feature_attribution_reports_view(request, *, artifact_dir):
    summaries = load_feature_attribution_files(limit=30, artifact_dir=artifact_dir)
    selected_name = str(request.GET.get("summary") or "").strip()
    selected_summary = next((item for item in summaries if item["name"] == selected_name), summaries[0] if summaries else None)
    payload = dict(selected_summary.get("payload") or {}) if selected_summary is not None else {}
    rows = _annotate_rows_with_bar_pct(list(payload.get("rows") or []), "oracle_cluster_coverage_rate", "coverage_bar_pct")
    rows = _annotate_rows_with_bar_pct(rows, "mean_fold_excess_cumulative_return", "excess_bar_pct")
    marginal_rows = _annotate_rows_with_bar_pct(list(payload.get("marginal_rows") or []), "delta_oracle_cluster_coverage_rate", "delta_coverage_bar_pct")
    marginal_rows = _annotate_rows_with_bar_pct(marginal_rows, "delta_mean_fold_excess_cumulative_return", "delta_excess_bar_pct")
    return render(
        request,
        "pipeline/feature_attribution_reports.html",
        {
            "summaries": summaries,
            "selected_summary": selected_summary,
            "payload": payload,
            "summary": dict(payload.get("summary") or {}),
            "rows": rows[:20],
            "marginal_rows": marginal_rows[:20],
            "recommendations": list(payload.get("recommendations") or []),
        },
    )


def pipeline_rl_policy_reports_view(request):
    from .models import Artifact

    artifacts = list(Artifact.objects.filter(artifact_type="RL_POLICY_RESULT").select_related("pipeline_run").order_by("-created_at", "-id")[:30])
    selected_id = _to_int(request.GET.get("artifact_id"))
    selected_artifact = next((artifact for artifact in artifacts if int(artifact.id) == selected_id), artifacts[0] if artifacts else None)
    payload = load_json_payload_from_uri(selected_artifact.uri) if selected_artifact is not None else {}
    summary_rows = _annotate_rows_with_bar_pct(list(payload.get("summary_rows") or []), "combined_total_return_pct", "return_bar_pct", absolute=True)
    yearly_rows = _annotate_rows_with_bar_pct(list(payload.get("yearly_rows") or []), "total_return_pct", "return_bar_pct", absolute=True)
    trade_log_preview = list(payload.get("trade_log_preview") or [])[:40]
    return render(
        request,
        "pipeline/rl_policy_reports.html",
        {
            "artifacts": artifacts,
            "selected_artifact": selected_artifact,
            "payload": payload,
            "summary_rows": summary_rows,
            "yearly_rows": yearly_rows,
            "trade_log_preview": trade_log_preview,
            "executed_action_counts": dict(payload.get("executed_action_counts") or {}),
            "action_counts": dict(payload.get("action_counts") or {}),
        },
    )


__all__ = [
    "pipeline_cohorts_view",
    "pipeline_diagnostic_reports_view",
    "pipeline_feature_attribution_reports_view",
    "pipeline_oracle_reports_view",
    "pipeline_research_reports_view",
    "pipeline_rl_policy_reports_view",
]
