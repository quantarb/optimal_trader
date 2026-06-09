from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.template import engines
from django.views.decorators.clickjacking import xframe_options_exempt

from fmp.models import EconomicIndicatorSeries, Symbol, SymbolSectionHistorical, TreasuryRateSeries
from pipeline.models import PipelineRun
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
    build_ta_classic_technical_features,
    build_ttm_financial_statement_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.naming import feature_display_name
from utils.workflow import (
    default_feature_symbol,
    latest_universe_artifact_id,
    selected_universe_artifact_id,
    universe_artifact_choices,
    universe_artifact_name,
    universe_symbols_from_artifact_id,
    workflow_symbols_from_request,
)

from .forms import FeaturePreviewForm


def _load_adjusted_prices(symbol_obj: Symbol, start_date, end_date) -> pd.DataFrame:
    qs = SymbolSectionHistorical.objects.filter(symbol=symbol_obj, section_key="prices_div_adj")
    if start_date is not None:
        qs = qs.filter(record_date__gte=start_date)
    if end_date is not None:
        qs = qs.filter(record_date__lte=end_date)
    qs = qs.order_by("record_date", "updated_at").only("record_date", "payload")
    rows: list[dict[str, Any]] = []
    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        date_val = payload.get("date") or (item.record_date.isoformat() if item.record_date else None)
        if not date_val:
            continue
        rows.append(
            {
                "date": str(date_val)[:10],
                "open": payload.get("adjOpen"),
                "high": payload.get("adjHigh"),
                "low": payload.get("adjLow"),
                "close": payload.get("adjClose"),
                "volume": payload.get("volume"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _render_feature_template(request, context: dict[str, Any]) -> HttpResponse:
    template_path = Path(__file__).resolve().parent / "templates" / "features" / "feature_form.html"
    if not template_path.exists():
        template_path = Path(__file__).resolve().parent.parent / "templates" / "features" / "feature_form.html"
    template = engines["django"].from_string(template_path.read_text(encoding="utf-8"))
    return HttpResponse(template.render(context=context, request=request))


def _default_feature_preview_data(symbol: str) -> dict[str, Any]:
    return {
        "symbol": str(symbol).strip().upper(),
        "include_price_technicals": True,
        "include_ta_classic_technicals": False,
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_ttm_financial_statements": False,
        "include_event_features": True,
        "include_ownership_features": True,
        "include_economic_indicators": True,
        "include_treasury_rates": True,
        "preview_rows": 100,
    }


def _default_feature_form_data() -> dict[str, Any]:
    return {
        "job_name": "",
        "include_price_technicals": True,
        "include_ta_classic_technicals": False,
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_ttm_financial_statements": False,
        "include_event_features": True,
        "include_ownership_features": True,
        "include_economic_indicators": True,
        "include_treasury_rates": True,
        "preview_rows": 100,
    }


def _feature_form_toggle_data(source: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(source or {})

    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    defaults = _default_feature_form_data()
    return {
        "include_price_technicals": _as_bool(raw.get("include_price_technicals"), bool(defaults["include_price_technicals"])),
        "include_ta_classic_technicals": _as_bool(
            raw.get("include_ta_classic_technicals"),
            bool(defaults["include_ta_classic_technicals"]),
        ),
        "include_fundamental_change": _as_bool(raw.get("include_fundamental_change"), bool(defaults["include_fundamental_change"])),
        "include_statement_quality": _as_bool(raw.get("include_statement_quality"), bool(defaults["include_statement_quality"])),
        "include_ttm_financial_statements": _as_bool(
            raw.get("include_ttm_financial_statements"),
            bool(defaults["include_ttm_financial_statements"]),
        ),
        "include_event_features": _as_bool(raw.get("include_event_features"), bool(defaults["include_event_features"])),
        "include_ownership_features": _as_bool(raw.get("include_ownership_features"), bool(defaults["include_ownership_features"])),
        "include_economic_indicators": _as_bool(raw.get("include_economic_indicators"), bool(defaults["include_economic_indicators"])),
        "include_treasury_rates": _as_bool(raw.get("include_treasury_rates"), bool(defaults["include_treasury_rates"])),
        "preview_rows": int(raw.get("preview_rows") or defaults["preview_rows"]),
    }


def _empty_feature_preview_result(section_order: list[str]) -> dict[str, Any]:
    return {
        "error": "",
        "summary": {},
        "feature_columns": [],
        "grouped_feature_columns": {key: [] for key in section_order},
        "grouped_feature_samples": {key: [] for key in section_order},
        "grouped_feature_tables": {key: {"columns": [], "labels": [], "rows": []} for key in section_order},
        "coverage_rows": [],
        "feature_sections": [],
    }


def _feature_section_metadata() -> tuple[list[str], dict[str, str]]:
    section_order = [
        "prices_div_adj",
        "technical_candles",
        "technical_cycles",
        "technical_math",
        "technical_momentum",
        "technical_overlap",
        "technical_performance",
        "key_metrics",
        "ratios",
        "key_metrics_ttm",
        "ratios_ttm",
        "income_statement_ttm",
        "cash_flow_ttm",
        "balance_sheet_ttm",
        "income_statement",
        "income_statement_growth",
        "cash_flow",
        "cash_flow_growth",
        "balance_sheet",
        "balance_sheet_growth",
        "financial_growth",
        "earnings",
        "analyst_estimates",
        "ratings_historical",
        "grades_historical",
        "insider_trading",
        "economic_indicators",
        "treasury_rates",
    ]
    section_labels = {
        "prices_div_adj": "Prices Div Adj",
        "technical_candles": "Technical Candles",
        "technical_cycles": "Technical Cycles",
        "technical_math": "Technical Math",
        "technical_momentum": "Technical Momentum",
        "technical_overlap": "Technical Overlap",
        "technical_performance": "Technical Performance",
        "key_metrics": "Key Metrics",
        "ratios": "Ratios",
        "key_metrics_ttm": "Key Metrics TTM",
        "ratios_ttm": "Ratios TTM",
        "income_statement_ttm": "Income Statement TTM",
        "cash_flow_ttm": "Cash Flow TTM",
        "balance_sheet_ttm": "Balance Sheet TTM",
        "income_statement": "Income Statement",
        "income_statement_growth": "Income Statement Growth",
        "cash_flow": "Cash Flow",
        "cash_flow_growth": "Cash Flow Growth",
        "balance_sheet": "Balance Sheet",
        "balance_sheet_growth": "Balance Sheet Growth",
        "financial_growth": "Financial Growth",
        "earnings": "Earnings",
        "analyst_estimates": "Analyst Estimates",
        "ratings_historical": "Ratings Historical",
        "grades_historical": "Grades Historical",
        "insider_trading": "Insider Trading",
        "economic_indicators": "Economic Indicators",
        "treasury_rates": "Treasury Rates",
    }
    return section_order, section_labels


def _build_symbol_table_rows(
    *,
    symbols: list[str],
    data: dict[str, Any],
    section_order: list[str],
    section_labels: dict[str, str],
    limit: int = 10,
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [section_labels[key] for key in section_order if bool(data.get(_section_toggle_name(key), True))]
    rows: list[dict[str, Any]] = []
    for symbol in list(symbols or [])[:limit]:
        symbol_obj = Symbol.objects.filter(symbol__iexact=symbol).first()
        if symbol_obj is None:
            continue
        result = _build_feature_preview_result(
            symbol_obj=symbol_obj,
            data=data,
            section_order=section_order,
            section_labels=section_labels,
        )
        coverage_map = {
            str(row.get("section_label") or ""): int(row.get("count") or 0)
            for row in list(result.get("coverage_rows") or [])
        }
        count_values = [int(coverage_map.get(label) or 0) for label in columns]
        rows.append(
            {
                "symbol": str(symbol_obj.symbol).strip().upper(),
                "count_values": count_values,
            }
        )
    return columns, rows


def _section_toggle_name(section_key: str) -> str:
    mapping = {
        "prices_div_adj": "include_price_technicals",
        "technical_candles": "include_ta_classic_technicals",
        "technical_cycles": "include_ta_classic_technicals",
        "technical_math": "include_ta_classic_technicals",
        "technical_momentum": "include_ta_classic_technicals",
        "technical_overlap": "include_ta_classic_technicals",
        "technical_performance": "include_ta_classic_technicals",
        "key_metrics": "include_fundamental_change",
        "ratios": "include_fundamental_change",
        "key_metrics_ttm": "include_ttm_financial_statements",
        "ratios_ttm": "include_ttm_financial_statements",
        "income_statement_ttm": "include_ttm_financial_statements",
        "cash_flow_ttm": "include_ttm_financial_statements",
        "balance_sheet_ttm": "include_ttm_financial_statements",
        "income_statement": "include_statement_quality",
        "income_statement_growth": "include_statement_quality",
        "cash_flow": "include_statement_quality",
        "cash_flow_growth": "include_statement_quality",
        "balance_sheet": "include_statement_quality",
        "balance_sheet_growth": "include_statement_quality",
        "financial_growth": "include_statement_quality",
        "earnings": "include_event_features",
        "analyst_estimates": "include_event_features",
        "ratings_historical": "include_event_features",
        "grades_historical": "include_event_features",
        "insider_trading": "include_ownership_features",
        "economic_indicators": "include_economic_indicators",
        "treasury_rates": "include_treasury_rates",
    }
    return mapping.get(section_key, "")


def _feature_artifact_statistics(uri: str) -> dict[str, Any]:
    path = Path(str(uri or "").strip())
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return {
            "min_date": None,
            "max_date": None,
            "column_count": 0,
            "feature_column_count": 0,
            "avg_rows_per_symbol": 0.0,
            "sample_symbols": [],
        }

    min_date = None
    max_date = None
    fieldnames: list[str] = []
    per_symbol: dict[str, dict[str, Any]] = {}
    total_rows = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            total_rows += 1
            symbol = str(row.get("symbol") or "").strip().upper()
            date_value = str(row.get("date") or "").strip()
            if date_value:
                min_date = date_value if min_date is None or date_value < min_date else min_date
                max_date = date_value if max_date is None or date_value > max_date else max_date
            if not symbol:
                continue
            bucket = per_symbol.setdefault(
                symbol,
                {"symbol": symbol, "rows": 0, "min_date": None, "max_date": None},
            )
            bucket["rows"] += 1
            if date_value:
                bucket["min_date"] = date_value if bucket["min_date"] is None or date_value < bucket["min_date"] else bucket["min_date"]
                bucket["max_date"] = date_value if bucket["max_date"] is None or date_value > bucket["max_date"] else bucket["max_date"]

    feature_columns = [name for name in fieldnames if name not in {"date", "symbol"}]
    sample_symbols = sorted(
        per_symbol.values(),
        key=lambda row: (-int(row["rows"]), str(row["symbol"])),
    )[:20]
    avg_rows_per_symbol = (float(total_rows) / float(len(per_symbol))) if per_symbol else 0.0
    return {
        "min_date": min_date,
        "max_date": max_date,
        "column_count": len(fieldnames),
        "feature_column_count": len(feature_columns),
        "avg_rows_per_symbol": avg_rows_per_symbol,
        "sample_symbols": sample_symbols,
    }


def _recent_feature_job_rows(limit: int = 20, *, include_statistics: bool = True) -> list[dict[str, Any]]:
    runs = (
        PipelineRun.objects.filter(requested_job="features")
        .prefetch_related("artifacts", "job_runs__input_artifacts")
        .order_by("-created_at", "-id")[:limit]
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        artifact = None
        for candidate in run.artifacts.all():
            if str(candidate.artifact_type) == "FEATURES":
                artifact = candidate
        content = dict((artifact.content if artifact is not None else {}) or {})
        metadata = dict((artifact.metadata if artifact is not None else {}) or {})
        universe_artifact_id = int(metadata.get("source_universe_artifact_id") or 0)
        if not universe_artifact_id:
            for job_run in run.job_runs.all():
                input_ids = [int(v) for v in job_run.input_artifacts.values_list("id", flat=True)]
                if input_ids:
                    universe_artifact_id = input_ids[0]
                    break
        symbols = universe_symbols_from_artifact_id(universe_artifact_id)
        rows.append(
            {
                "pipeline_run_id": int(run.id),
                "name": str(run.name or ""),
                "status": str(run.status or ""),
                "config": dict(run.config or {}),
                "source_universe_artifact_id": universe_artifact_id,
                "features_artifact_id": int(artifact.id) if artifact is not None else 0,
                "rows": int(content.get("rows") or 0),
                "symbols": int(content.get("symbols") or 0),
                "symbol_list": symbols,
                "statistics": (
                    _feature_artifact_statistics(str((artifact.uri if artifact is not None else "") or ""))
                    if include_statistics
                    else {}
                ),
                "uri": str((artifact.uri if artifact is not None else "") or ""),
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "error": str(run.error or ""),
            }
        )
    return rows


def _universe_job_name_defaults(artifact_choices: list[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for artifact_id, _label in artifact_choices:
        name = universe_artifact_name(int(artifact_id))
        if name:
            out[str(artifact_id)] = f"{name} + Features"
    return out


def feature_job_rows_json(request):
    try:
        limit = int(request.GET.get("limit") or 20)
    except Exception:
        limit = 20
    limit = max(1, min(100, limit))
    return JsonResponse({"rows": _recent_feature_job_rows(limit=limit, include_statistics=True)})


def feature_symbol_table_json(request):
    section_order, section_labels = _feature_section_metadata()
    artifact_id = selected_universe_artifact_id(request)
    if artifact_id <= 0:
        artifact_id = latest_universe_artifact_id()
    symbols = universe_symbols_from_artifact_id(artifact_id)
    data = _feature_form_toggle_data(request.GET)
    columns, rows = _build_symbol_table_rows(
        symbols=symbols,
        data=data,
        section_order=section_order,
        section_labels=section_labels,
    )
    return JsonResponse(
        {
            "columns": columns,
            "rows": rows,
            "universe_artifact_id": artifact_id,
        }
    )


def _build_feature_preview_result(
    *,
    symbol_obj: Symbol,
    data: dict[str, Any],
    section_order: list[str],
    section_labels: dict[str, str],
) -> dict[str, Any]:
    error = ""
    summary: dict[str, Any] = {}
    feature_columns: list[str] = []
    grouped_feature_columns: dict[str, list[str]] = {key: [] for key in section_order}
    grouped_feature_samples: dict[str, list[dict[str, str]]] = {key: [] for key in section_order}
    grouped_feature_tables: dict[str, dict[str, Any]] = {key: {"columns": [], "labels": [], "rows": []} for key in section_order}
    coverage_rows: list[dict[str, Any]] = []
    feature_sections: list[dict[str, Any]] = []

    symbol = str(symbol_obj.symbol).strip().upper()
    try:
        df_prices = _load_adjusted_prices(symbol_obj, None, None)
        if df_prices.empty:
            raise ValueError("No adjusted price data found for the selected symbol.")

        effective_start = df_prices.index.min().date()
        effective_end = df_prices.index.max().date()
        target_index = pd.MultiIndex.from_arrays(
            [df_prices.index, [symbol] * len(df_prices)],
            names=["date", "symbol"],
        )
        merged = pd.DataFrame(index=target_index)
        selected_sections: set[str] = set()
        source_counts = {key: 0 for key in section_order}

        if data.get("include_price_technicals"):
            selected_sections.add("prices_div_adj")
            built = build_price_technical_features(symbol, df_prices)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                grouped_feature_columns["prices_div_adj"] = list(built.feature_cols)
                source_counts["prices_div_adj"] = len(built.feature_cols)

        if data.get("include_ta_classic_technicals"):
            built_by_family = build_ta_classic_technical_features(symbol, df_prices)
            for family_name, built in built_by_family.items():
                selected_sections.add(family_name)
                if built.df.empty:
                    continue
                active_cols = [c for c in built.feature_cols if c in built.df.columns]
                if not active_cols:
                    continue
                merged = merged.join(built.df[active_cols], how="left")
                feature_columns.extend(active_cols)
                grouped_feature_columns[family_name] = list(active_cols)
                source_counts[family_name] = len(active_cols)

        if data.get("include_fundamental_change"):
            selected_sections.update({"key_metrics", "ratios"})
            built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                km_cols = [c for c in built.feature_cols if c.startswith("km__")]
                rt_cols = [c for c in built.feature_cols if c.startswith("rt__")]
                grouped_feature_columns["key_metrics"] = km_cols
                grouped_feature_columns["ratios"] = rt_cols
                source_counts["key_metrics"] = len(km_cols)
                source_counts["ratios"] = len(rt_cols)

        if data.get("include_statement_quality"):
            selected_sections.update(
                {
                    "income_statement",
                    "income_statement_growth",
                    "cash_flow",
                    "cash_flow_growth",
                    "balance_sheet",
                    "balance_sheet_growth",
                    "financial_growth",
                }
            )
            built = build_statement_quality_features(symbol_obj, target_index)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                grouped_feature_columns["income_statement"] = [c for c in built.feature_cols if c.startswith("is__")]
                grouped_feature_columns["income_statement_growth"] = [c for c in built.feature_cols if c.startswith("isg__")]
                grouped_feature_columns["cash_flow"] = [c for c in built.feature_cols if c.startswith("cf__")]
                grouped_feature_columns["cash_flow_growth"] = [c for c in built.feature_cols if c.startswith("cfg__")]
                grouped_feature_columns["balance_sheet"] = [c for c in built.feature_cols if c.startswith("bs__")]
                grouped_feature_columns["balance_sheet_growth"] = [c for c in built.feature_cols if c.startswith("bsg__")]
                grouped_feature_columns["financial_growth"] = [c for c in built.feature_cols if c.startswith("fg__")]
                for key in (
                    "income_statement",
                    "income_statement_growth",
                    "cash_flow",
                    "cash_flow_growth",
                    "balance_sheet",
                    "balance_sheet_growth",
                    "financial_growth",
                ):
                    source_counts[key] = len(grouped_feature_columns[key])

        if data.get("include_ttm_financial_statements"):
            ttm_sections = {
                "key_metrics_ttm": "km_ttm__",
                "ratios_ttm": "rt_ttm__",
                "income_statement_ttm": "is_ttm__",
                "cash_flow_ttm": "cf_ttm__",
                "balance_sheet_ttm": "bs_ttm__",
            }
            selected_sections.update(ttm_sections)
            built = build_ttm_financial_statement_features(symbol_obj, target_index, df_prices=df_prices)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                for key, prefix in ttm_sections.items():
                    grouped_feature_columns[key] = [col for col in built.feature_cols if col.startswith(prefix)]
                    source_counts[key] = len(grouped_feature_columns[key])

        if data.get("include_event_features"):
            selected_sections.update({"earnings", "analyst_estimates", "ratings_historical", "grades_historical"})
            built = build_event_features(symbol_obj, target_index)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                grouped_feature_columns["earnings"] = [c for c in built.feature_cols if c.startswith("evt__earn_")]
                grouped_feature_columns["analyst_estimates"] = [c for c in built.feature_cols if c.startswith("evt__ae_")]
                grouped_feature_columns["ratings_historical"] = [c for c in built.feature_cols if c.startswith("evt__rating_")]
                grouped_feature_columns["grades_historical"] = [c for c in built.feature_cols if c.startswith("evt__grade_")]
                for key in ("earnings", "analyst_estimates", "ratings_historical", "grades_historical"):
                    source_counts[key] = len(grouped_feature_columns[key])

        if data.get("include_ownership_features"):
            selected_sections.add("insider_trading")
            built = build_ownership_features(symbol_obj, target_index)
            if not built.df.empty:
                merged = merged.join(built.df[built.feature_cols], how="left")
                feature_columns.extend(built.feature_cols)
                grouped_feature_columns["insider_trading"] = [c for c in built.feature_cols if c.startswith("own__insider_")]
                source_counts["insider_trading"] = len(grouped_feature_columns["insider_trading"])

        if data.get("include_economic_indicators"):
            economic_series_codes = tuple(str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True))
            economic_df = fetch_economic_data_series(
                api_key="",
                start_date=effective_start.isoformat(),
                end_date=effective_end.isoformat(),
                config=EconomicDataConfig(
                    economic_indicator_series=economic_series_codes,
                    include_treasury_rates=False,
                ),
            )
            if not economic_df.empty:
                selected_sections.add("economic_indicators")
                economic_daily = broadcast_series_to_daily(economic_df, target_index)
                economic_cols = list(economic_daily.columns)
                merged = merged.join(economic_daily[economic_cols], how="left")
                feature_columns.extend(economic_cols)
                grouped_feature_columns["economic_indicators"] = economic_cols
                source_counts["economic_indicators"] = len(economic_cols)

        if data.get("include_treasury_rates"):
            treasury_series_codes = tuple(str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True))
            treasury_df = fetch_economic_data_series(
                api_key="",
                start_date=effective_start.isoformat(),
                end_date=effective_end.isoformat(),
                config=EconomicDataConfig(
                    economic_indicator_series=treasury_series_codes,
                    include_treasury_rates=False,
                ),
            )
            if not treasury_df.empty:
                selected_sections.add("treasury_rates")
                treasury_daily = broadcast_series_to_daily(treasury_df, target_index)
                treasury_cols = list(treasury_daily.columns)
                merged = merged.join(treasury_daily[treasury_cols], how="left")
                feature_columns.extend(treasury_cols)
                grouped_feature_columns["treasury_rates"] = treasury_cols
                source_counts["treasury_rates"] = len(treasury_cols)

        feature_columns = list(dict.fromkeys(feature_columns))
        preview_df = merged.reset_index()
        preview_df["date"] = preview_df["date"].dt.strftime("%Y-%m-%d")
        preview_df = preview_df.sort_values(["date"], ascending=False)
        preview_df = preview_df.head(int(data.get("preview_rows") or 100))

        for group_name in section_order:
            cols = [c for c in grouped_feature_columns[group_name] if c in preview_df.columns]
            grouped_feature_columns[group_name] = cols
            samples: list[dict[str, str]] = []
            for col in cols:
                sample_value = _first_non_null_display_value(preview_df, col)
                samples.append({"key": col, "label": feature_display_name(col), "value": sample_value})
            grouped_feature_samples[group_name] = samples
            if cols:
                family_table_columns = ["date", "symbol"] + cols
                family_preview_df = preview_df[family_table_columns].head(10)
                grouped_feature_tables[group_name] = {
                    "columns": family_table_columns,
                    "labels": [feature_display_name(col) for col in family_table_columns],
                    "rows": [
                        ["" if pd.isna(row.get(col)) else str(row.get(col)) for col in family_table_columns]
                        for row in family_preview_df.to_dict(orient="records")
                    ],
                }

        feature_sections = [
            {
                "key": key,
                "label": section_labels[key],
                "feature_count": len(grouped_feature_columns[key]),
                "samples": grouped_feature_samples[key],
                "table": grouped_feature_tables[key],
            }
            for key in section_order
            if key in selected_sections and grouped_feature_columns[key]
        ]

        summary = {
            "symbol": symbol,
            "computed_start_date": effective_start,
            "computed_end_date": effective_end,
            "price_rows": int(len(df_prices)),
            "preview_rows": int(len(preview_df)),
            "feature_count": int(len(feature_columns)),
            "source_counts": source_counts,
        }
        coverage_rows = [
            _build_feature_family_coverage_row(section_labels[key], merged, grouped_feature_columns[key])
            for key in section_order
            if key in selected_sections
        ]
    except Exception as exc:
        error = str(exc)

    return {
        "error": error,
        "summary": summary,
        "feature_columns": feature_columns,
        "grouped_feature_columns": grouped_feature_columns,
        "grouped_feature_samples": grouped_feature_samples,
        "grouped_feature_tables": grouped_feature_tables,
        "coverage_rows": coverage_rows,
        "feature_sections": feature_sections,
    }


@xframe_options_exempt
def feature_preview_form(request):
    section_order, section_labels = _feature_section_metadata()
    artifact_choices = universe_artifact_choices()
    selected_artifact_id = selected_universe_artifact_id(request)
    if selected_artifact_id <= 0:
        selected_artifact_id = latest_universe_artifact_id()

    preferred_symbols = universe_symbols_from_artifact_id(selected_artifact_id)
    if not preferred_symbols:
        preferred_symbols = workflow_symbols_from_request(request)
    initial_symbol = preferred_symbols[0] if preferred_symbols else default_feature_symbol(request)
    default_universe_name = universe_artifact_name(selected_artifact_id)
    initial_data = {
        **_default_feature_form_data(),
        "job_name": f"{default_universe_name} + Features" if default_universe_name else "",
        "universe_artifact_id": str(selected_artifact_id) if selected_artifact_id > 0 else "",
    }
    selected_universe_symbols = preferred_symbols[:200]
    symbol_table_data = _feature_form_toggle_data(initial_data)

    if request.method == "POST":
        form = FeaturePreviewForm(
            request.POST,
            universe_artifact_choices=artifact_choices,
        )
        results = _empty_feature_preview_result(section_order)
        normalized = ""
        if form.is_valid():
            cleaned_data = dict(form.cleaned_data)
            selected_symbols = universe_symbols_from_artifact_id(int(cleaned_data["universe_artifact_id"]))
            preview_symbol = selected_symbols[0] if selected_symbols else default_feature_symbol(request)
            symbol_obj = get_object_or_404(Symbol, symbol__iexact=preview_symbol)
            cleaned_data["preview_symbol"] = preview_symbol
            symbol_table_data = _feature_form_toggle_data(cleaned_data)
            results = _build_feature_preview_result(
                symbol_obj=symbol_obj,
                data=cleaned_data,
                section_order=section_order,
                section_labels=section_labels,
            )
            normalized = json.dumps(cleaned_data, indent=2, sort_keys=True, default=str)
    else:
        form = FeaturePreviewForm(
            initial=initial_data,
            universe_artifact_choices=artifact_choices,
        )
        normalized = json.dumps(initial_data, indent=2, sort_keys=True, default=str)
        symbol_obj = Symbol.objects.filter(symbol__iexact=initial_symbol).first()
        results = (
            _build_feature_preview_result(
                symbol_obj=symbol_obj,
                data=initial_data,
                section_order=section_order,
                section_labels=section_labels,
            )
            if symbol_obj is not None
            else _empty_feature_preview_result(section_order)
        )

    symbol_table_columns, symbol_table_rows = _build_symbol_table_rows(
        symbols=preferred_symbols,
        data=symbol_table_data,
        section_order=section_order,
        section_labels=section_labels,
    )

    return _render_feature_template(
        request,
        {
            "form": form,
            "normalized": normalized,
            "is_symbol_detail": False,
            "current_feature_job_id": 0,
            "universe_missing": not artifact_choices,
            "universe_job_name_defaults_json": json.dumps(_universe_job_name_defaults(artifact_choices), sort_keys=True),
            "selected_universe_symbols": selected_universe_symbols,
            "symbol_table_columns": symbol_table_columns,
            "symbol_table_rows": symbol_table_rows,
            "recent_feature_jobs": _recent_feature_job_rows(),
            **results,
        },
    )


@xframe_options_exempt
def feature_preview_symbol(request, symbol: str, feature_run_id: int | None = None):
    symbol_obj = get_object_or_404(Symbol, symbol__iexact=symbol)
    section_order, section_labels = _feature_section_metadata()
    artifact_choices = universe_artifact_choices()
    run = None
    if feature_run_id is not None:
        run = PipelineRun.objects.filter(pk=int(feature_run_id), requested_job="features").first()

    selected_artifact_id = 0
    initial_data = _default_feature_form_data()
    if run is not None:
        initial_data.update(dict(run.config or {}))
        selected_artifact_id = int(str((run.config or {}).get("universe_artifact_id") or "0").strip() or 0)
        if selected_artifact_id <= 0:
            rows = _recent_feature_job_rows(limit=100, include_statistics=False)
            matched = next((row for row in rows if int(row["pipeline_run_id"]) == int(run.id)), None)
            selected_artifact_id = int((matched or {}).get("source_universe_artifact_id") or 0)
    if selected_artifact_id <= 0:
        selected_artifact_id = selected_universe_artifact_id(request)
    if selected_artifact_id <= 0:
        selected_artifact_id = latest_universe_artifact_id()

    initial_data["universe_artifact_id"] = str(selected_artifact_id) if selected_artifact_id > 0 else ""
    results = _build_feature_preview_result(
        symbol_obj=symbol_obj,
        data=initial_data,
        section_order=section_order,
        section_labels=section_labels,
    )

    return _render_feature_template(
        request,
        {
            "form": FeaturePreviewForm(
                initial=initial_data,
                universe_artifact_choices=artifact_choices,
            ),
            "normalized": json.dumps(initial_data, indent=2, sort_keys=True, default=str),
            "is_symbol_detail": True,
            "symbol_detail_title": str(symbol_obj.symbol).strip().upper(),
            "current_feature_job_id": int(run.id) if run is not None else 0,
            "universe_missing": not artifact_choices,
            "universe_job_name_defaults_json": json.dumps(_universe_job_name_defaults(artifact_choices), sort_keys=True),
            "selected_universe_symbols": [],
            "symbol_table_columns": [],
            "symbol_table_rows": [],
            "recent_feature_jobs": [],
            **results,
        },
    )


def _first_non_null_display_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "-"
    series = df[column]
    non_null = series[series.notna()]
    if non_null.empty:
        return "-"
    value = non_null.iloc[0]
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:.6g}"
    return str(value)


def _build_feature_family_coverage_row(section_label: str, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    if df.empty or not feature_cols:
        return {
            "section_label": section_label,
            "min_date": None,
            "max_date": None,
            "count": 0,
        }

    usable_cols = [col for col in feature_cols if col in df.columns]
    if not usable_cols:
        return {
            "section_label": section_label,
            "min_date": None,
            "max_date": None,
            "count": 0,
        }

    mask = df[usable_cols].notna().any(axis=1)
    if not mask.any():
        return {
            "section_label": section_label,
            "min_date": None,
            "max_date": None,
            "count": 0,
        }

    valid_index = df.index[mask]
    if isinstance(valid_index, pd.MultiIndex):
        dates = pd.to_datetime(valid_index.get_level_values("date"))
    else:
        dates = pd.to_datetime(valid_index)

    return {
        "section_label": section_label,
        "min_date": dates.min().date() if len(dates) else None,
        "max_date": dates.max().date() if len(dates) else None,
        "count": int(mask.sum()),
    }
