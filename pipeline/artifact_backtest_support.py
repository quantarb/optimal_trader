from __future__ import annotations

import json
from math import sqrt
from typing import Any, Callable

from .contracts import build_backtest_daily_rows_from_trade_rows, build_equity_curve_from_daily_rows
from .models import Artifact


BACKTEST_PREVIEW_ROWS_LIMIT = 5000
BENCHMARK_PREVIEW_ROWS_LIMIT = 50000
TRADING_DAYS_PER_YEAR = 252.0
SUMMARY_DECIMALS = 8
RATE_DECIMALS = 4
PERCENT_MULTIPLIER = 100.0
BENCHMARK_START_EQUITY = 1.0


PreviewLoader = Callable[[Artifact, int], tuple[list[dict[str, Any]], dict[str, Any]]]
StringNormalizer = Callable[[Any], str]
FloatCoercer = Callable[[Any], float | None]
IntCoercer = Callable[[Any], int]


def _round_or_none(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def _percent_or_none(value: float | None, digits: int) -> float | None:
    return round(value * PERCENT_MULTIPLIER, digits) if value is not None else None


def _resolve_backtest_rows(
    backtest_artifact: Artifact,
    *,
    load_artifact_preview_rows: PreviewLoader,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    preview_rows, content_payload = load_artifact_preview_rows(backtest_artifact, BACKTEST_PREVIEW_ROWS_LIMIT)
    metadata = dict(backtest_artifact.metadata or {})
    daily_rows = list(content_payload.get("daily_rows") or [])
    if not daily_rows and preview_rows:
        daily_rows = build_backtest_daily_rows_from_trade_rows(preview_rows)
    curve_rows = list(metadata.get("equity_curve") or [])
    if not curve_rows and daily_rows:
        curve_rows = build_equity_curve_from_daily_rows(daily_rows)
    if not daily_rows and curve_rows:
        daily_rows = [
            {
                "date": row.get("date"),
                "equity": row.get("equity"),
                "net_daily_return": row.get("net_daily_return"),
            }
            for row in curve_rows
        ]
    return preview_rows, content_payload, metadata, daily_rows, curve_rows


def _curve_points(
    curve_rows: list[dict[str, Any]],
    *,
    normalized_date: StringNormalizer,
    safe_float: FloatCoercer,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in curve_rows:
        date_value = normalized_date(row.get("date"))
        equity = safe_float(row.get("equity"))
        if not date_value or equity is None:
            continue
        points.append(
            {
                "x": date_value,
                "equity": equity,
                "net_daily_return": safe_float(row.get("net_daily_return")),
            }
        )
    return points


def _series_rows(
    daily_rows: list[dict[str, Any]],
    *,
    source_key: str,
    output_key: str,
    cast_type,
    normalized_date: StringNormalizer,
    safe_float: FloatCoercer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in daily_rows:
        date_value = normalized_date(row.get("date"))
        if not date_value:
            continue
        numeric_value = safe_float(row.get(source_key)) or 0.0
        rows.append({"x": date_value, output_key: cast_type(numeric_value)})
    return rows


def _daily_row_statistics(
    daily_rows: list[dict[str, Any]],
    *,
    normalized_date: StringNormalizer,
    safe_float: FloatCoercer,
) -> dict[str, Any]:
    monthly_returns: dict[str, float] = {}
    avg_positions: list[int] = []
    turnover_values: list[float] = []
    daily_return_values: list[float] = []
    daily_sorted: list[dict[str, Any]] = []
    for row in daily_rows:
        date_value = normalized_date(row.get("date"))
        if not date_value:
            continue
        month_key = date_value[:7]
        net_daily_return = safe_float(row.get("net_daily_return")) or 0.0
        monthly_returns[month_key] = (1.0 + monthly_returns.get(month_key, 0.0)) * (1.0 + net_daily_return) - 1.0
        avg_positions.append(int(safe_float(row.get("positions")) or 0))
        turnover_values.append(safe_float(row.get("turnover")) or 0.0)
        daily_return_values.append(net_daily_return)
        daily_sorted.append({"date": date_value, "return": round(net_daily_return, 8)})
    monthly_return_rows = [
        {"month": month, "return": round(value, SUMMARY_DECIMALS)}
        for month, value in sorted(monthly_returns.items(), key=lambda item: item[0], reverse=True)
    ]
    return {
        "monthly_return_rows": monthly_return_rows,
        "avg_positions": avg_positions,
        "turnover_values": turnover_values,
        "daily_return_values": daily_return_values,
        "daily_sorted": daily_sorted,
        "positive_days": sum(1 for value in daily_return_values if value > 0),
        "negative_days": sum(1 for value in daily_return_values if value < 0),
    }


def _contribution_rows(
    preview_rows: list[dict[str, Any]],
    *,
    safe_float: FloatCoercer,
) -> list[dict[str, Any]]:
    contribution_by_symbol: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        contribution = contribution_by_symbol.setdefault(
            symbol,
            {"symbol": symbol, "realized_return": 0.0, "rows": 0},
        )
        contribution["realized_return"] += safe_float(row.get("realized_return")) or 0.0
        contribution["rows"] += 1
    return sorted(
        [
            {
                "symbol": symbol,
                "realized_return": round(values["realized_return"], SUMMARY_DECIMALS),
                "rows": int(values["rows"]),
            }
            for symbol, values in contribution_by_symbol.items()
        ],
        key=lambda item: item["realized_return"],
        reverse=True,
    )


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return variance ** 0.5


def _downside_volatility(values: list[float]) -> float:
    downside_values = [value for value in values if value < 0]
    if not downside_values:
        return 0.0
    return (sum(value ** 2 for value in downside_values) / float(len(downside_values))) ** 0.5


def _annualized_return(points: list[dict[str, Any]], *, safe_float: FloatCoercer) -> float:
    if not points:
        return 0.0
    start_equity = safe_float(points[0].get("equity"))
    end_equity = safe_float(points[-1].get("equity"))
    if start_equity is None or end_equity is None or start_equity <= 0.0 or end_equity <= 0.0:
        return 0.0
    years = max(float(len(points)) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    return (end_equity / start_equity) ** (1.0 / years) - 1.0


def _profit_factor(preview_rows: list[dict[str, Any]], *, safe_float: FloatCoercer) -> float | None:
    gross_profit = sum(max(0.0, safe_float(row.get("realized_return")) or 0.0) for row in preview_rows)
    gross_loss = sum(abs(min(0.0, safe_float(row.get("realized_return")) or 0.0)) for row in preview_rows)
    return (gross_profit / gross_loss) if gross_loss > 0 else None


def _benchmark_curve_context(
    strategy_artifact_id: int,
    *,
    load_artifact_preview_rows: PreviewLoader,
    normalized_date: StringNormalizer,
    safe_float: FloatCoercer,
) -> dict[str, Any]:
    if strategy_artifact_id <= 0:
        return {"points": [], "final_equity": None, "cumulative_return": None}
    strategy_artifact = Artifact.objects.filter(pk=strategy_artifact_id, artifact_type="STRATEGY_DATASET").first()
    if strategy_artifact is None:
        return {"points": [], "final_equity": None, "cumulative_return": None}
    strategy_rows, _ = load_artifact_preview_rows(strategy_artifact, BENCHMARK_PREVIEW_ROWS_LIMIT)
    benchmark_daily: dict[str, list[float]] = {}
    for row in strategy_rows:
        date_value = normalized_date(row.get("date"))
        ret_1 = safe_float(row.get("ret_1"))
        if not date_value or ret_1 is None:
            continue
        benchmark_daily.setdefault(date_value, []).append(ret_1)
    benchmark_equity = BENCHMARK_START_EQUITY
    points: list[dict[str, Any]] = []
    for date_value in sorted(benchmark_daily):
        values = benchmark_daily[date_value]
        daily_bench = sum(values) / float(len(values)) if values else 0.0
        benchmark_equity *= 1.0 + float(daily_bench)
        points.append(
            {
                "x": date_value,
                "equity": round(float(benchmark_equity), SUMMARY_DECIMALS),
                "daily_return": round(float(daily_bench), SUMMARY_DECIMALS),
            }
        )
    final_equity = points[-1]["equity"] if points else None
    cumulative_return = round(float(final_equity) - 1.0, SUMMARY_DECIMALS) if final_equity is not None else None
    return {
        "points": points,
        "final_equity": final_equity,
        "cumulative_return": cumulative_return,
    }


def _performance_ratio_summary(
    *,
    content_payload: dict[str, Any],
    points: list[dict[str, Any]],
    preview_rows: list[dict[str, Any]],
    daily_return_values: list[float],
    safe_float: FloatCoercer,
) -> dict[str, Any]:
    avg_daily_return = sum(daily_return_values) / len(daily_return_values) if daily_return_values else 0.0
    daily_volatility = _sample_std(daily_return_values)
    downside_volatility = _downside_volatility(daily_return_values)
    annual_return = _annualized_return(points, safe_float=safe_float)
    annual_volatility = daily_volatility * sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (avg_daily_return / daily_volatility) * sqrt(TRADING_DAYS_PER_YEAR) if daily_volatility > 0 else None
    sortino = (avg_daily_return / downside_volatility) * sqrt(TRADING_DAYS_PER_YEAR) if downside_volatility > 0 else None
    max_drawdown = safe_float(content_payload.get("max_drawdown"))
    calmar = (annual_return / abs(max_drawdown)) if max_drawdown not in (None, 0.0) else None
    profit_factor = _profit_factor(preview_rows, safe_float=safe_float)
    return {
        "avg_daily_return": round(avg_daily_return, SUMMARY_DECIMALS),
        "annual_return": round(annual_return, SUMMARY_DECIMALS),
        "annual_return_pct": round(annual_return * PERCENT_MULTIPLIER, RATE_DECIMALS),
        "annual_volatility": round(annual_volatility, SUMMARY_DECIMALS),
        "annual_volatility_pct": round(annual_volatility * PERCENT_MULTIPLIER, RATE_DECIMALS),
        "sharpe": _round_or_none(sharpe, SUMMARY_DECIMALS),
        "sortino": _round_or_none(sortino, SUMMARY_DECIMALS),
        "calmar": _round_or_none(calmar, SUMMARY_DECIMALS),
        "profit_factor": _round_or_none(profit_factor, SUMMARY_DECIMALS),
    }


def _activity_summary(
    *,
    daily_stats: dict[str, Any],
    daily_return_values: list[float],
    contribution_rows: list[dict[str, Any]],
    benchmark_context: dict[str, Any],
) -> dict[str, Any]:
    monthly_return_rows = list(daily_stats["monthly_return_rows"])
    benchmark_cumulative_return = benchmark_context["cumulative_return"]
    return {
        "avg_positions": round(sum(daily_stats["avg_positions"]) / len(daily_stats["avg_positions"]), RATE_DECIMALS) if daily_stats["avg_positions"] else 0.0,
        "avg_turnover": round(sum(daily_stats["turnover_values"]) / len(daily_stats["turnover_values"]), SUMMARY_DECIMALS) if daily_stats["turnover_values"] else 0.0,
        "positive_day_rate": round(daily_stats["positive_days"] / len(daily_return_values), SUMMARY_DECIMALS) if daily_return_values else 0.0,
        "positive_day_rate_pct": round((daily_stats["positive_days"] / len(daily_return_values)) * PERCENT_MULTIPLIER, RATE_DECIMALS) if daily_return_values else 0.0,
        "negative_day_rate": round(daily_stats["negative_days"] / len(daily_return_values), SUMMARY_DECIMALS) if daily_return_values else 0.0,
        "negative_day_rate_pct": round((daily_stats["negative_days"] / len(daily_return_values)) * PERCENT_MULTIPLIER, RATE_DECIMALS) if daily_return_values else 0.0,
        "best_month": monthly_return_rows[0] if monthly_return_rows else None,
        "worst_month": monthly_return_rows[-1] if monthly_return_rows else None,
        "best_day": max(daily_stats["daily_sorted"], key=lambda item: item["return"], default=None),
        "worst_day": min(daily_stats["daily_sorted"], key=lambda item: item["return"], default=None),
        "benchmark_final_equity": benchmark_context["final_equity"],
        "benchmark_cumulative_return": benchmark_cumulative_return,
        "benchmark_cumulative_return_pct": _percent_or_none(benchmark_cumulative_return, RATE_DECIMALS),
        "top_contributor": contribution_rows[0] if contribution_rows else None,
        "bottom_contributor": contribution_rows[-1] if contribution_rows else None,
    }


def _report_summary(
    *,
    content_payload: dict[str, Any],
    points: list[dict[str, Any]],
    daily_stats: dict[str, Any],
    preview_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    benchmark_context: dict[str, Any],
    safe_float: FloatCoercer,
) -> dict[str, Any]:
    daily_return_values = list(daily_stats["daily_return_values"])
    return {
        **_performance_ratio_summary(
            content_payload=content_payload,
            points=points,
            preview_rows=preview_rows,
            daily_return_values=daily_return_values,
            safe_float=safe_float,
        ),
        **_activity_summary(
            daily_stats=daily_stats,
            daily_return_values=daily_return_values,
            contribution_rows=contribution_rows,
            benchmark_context=benchmark_context,
        ),
    }


def build_equity_curve_context(
    backtest_artifact: Artifact,
    *,
    load_artifact_preview_rows: PreviewLoader,
    normalized_date: StringNormalizer,
    safe_float: FloatCoercer,
    to_int: IntCoercer,
) -> dict[str, Any]:
    preview_rows, content_payload, metadata, daily_rows, curve_rows = _resolve_backtest_rows(
        backtest_artifact,
        load_artifact_preview_rows=load_artifact_preview_rows,
    )
    points = _curve_points(
        curve_rows,
        normalized_date=normalized_date,
        safe_float=safe_float,
    )
    daily_stats = _daily_row_statistics(
        daily_rows,
        normalized_date=normalized_date,
        safe_float=safe_float,
    )
    contribution_rows = _contribution_rows(preview_rows, safe_float=safe_float)
    benchmark_context = _benchmark_curve_context(
        to_int(metadata.get("source_strategy_dataset_artifact_id")),
        load_artifact_preview_rows=load_artifact_preview_rows,
        normalized_date=normalized_date,
        safe_float=safe_float,
    )
    report_summary = _report_summary(
        content_payload=content_payload,
        points=points,
        daily_stats=daily_stats,
        preview_rows=preview_rows,
        contribution_rows=contribution_rows,
        benchmark_context=benchmark_context,
        safe_float=safe_float,
    )
    return {
        "equity_curve_points_json": json.dumps(points),
        "benchmark_curve_points_json": json.dumps(benchmark_context["points"]),
        "equity_curve_count": int(len(points)),
        "backtest_daily_rows_full": daily_rows,
        "turnover_series_json": json.dumps(
            _series_rows(
                daily_rows,
                source_key="turnover",
                output_key="turnover",
                cast_type=float,
                normalized_date=normalized_date,
                safe_float=safe_float,
            )
        ),
        "positions_series_json": json.dumps(
            _series_rows(
                daily_rows,
                source_key="positions",
                output_key="positions",
                cast_type=int,
                normalized_date=normalized_date,
                safe_float=safe_float,
            )
        ),
        "monthly_return_rows": daily_stats["monthly_return_rows"],
        "contribution_rows": contribution_rows,
        "report_summary": report_summary,
    }
