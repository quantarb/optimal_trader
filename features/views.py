from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.template import engines

from fmp.models import EconomicIndicatorSeries, Symbol, SymbolSectionHistorical, TreasuryRateSeries
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.naming import feature_display_name


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
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_event_features": True,
        "include_ownership_features": True,
        "include_economic_indicators": True,
        "include_treasury_rates": True,
        "preview_rows": 100,
    }


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


def feature_preview_symbol(request, symbol: str):
    symbol_obj = get_object_or_404(Symbol, symbol__iexact=symbol)
    section_order = [
        "prices_div_adj",
        "key_metrics",
        "ratios",
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
        "key_metrics": "Key Metrics",
        "ratios": "Ratios",
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
    results = _build_feature_preview_result(
        symbol_obj=symbol_obj,
        data=_default_feature_preview_data(symbol_obj.symbol),
        section_order=section_order,
        section_labels=section_labels,
    )

    return _render_feature_template(
        request,
        {
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
