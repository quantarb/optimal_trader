from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from fmp.models import Symbol


DEFAULT_TSMOM_PROXY_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "VNQ",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "UNG",
    "FXE",
    "FXY",
]


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "n/a"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _build_yearly_folds(start_year: int, end_year: int) -> list[dict[str, str]]:
    folds: list[dict[str, str]] = []
    for year in range(int(start_year), int(end_year) + 1):
        folds.append(
            {
                "name": f"wf_{year}",
                "train_end_date": f"{year - 1}-12-31",
                "backtest_start_date": f"{year}-01-01",
                "backtest_end_date": f"{year}-12-31",
            }
        )
    return folds


def _resolve_available_symbols(raw_symbols: Sequence[str]) -> tuple[list[str], list[str]]:
    normalized = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    available = {
        str(symbol).strip().upper()
        for symbol in Symbol.objects.filter(symbol__in=normalized).values_list("symbol", flat=True)
    }
    ordered_available = [symbol for symbol in normalized if symbol in available]
    missing = [symbol for symbol in normalized if symbol not in available]
    return ordered_available, missing


def _write_report(
    *,
    report_path: Path,
    payload: dict[str, object],
    available_symbols: list[str],
    missing_symbols: list[str],
    strategy_config: dict[str, object],
    backtest_config: dict[str, object],
) -> None:
    aggregate = dict((payload.get("aggregate_rows") or [{}])[0] if payload.get("aggregate_rows") else {})
    summary_rows = [dict(row) for row in list(payload.get("summary_rows") or [])]
    folds = [dict(row) for row in list(payload.get("folds") or [])]
    walk_forward_metrics = dict(payload.get("walk_forward_metrics") or {})
    positive_folds = sum(1 for row in summary_rows if _float(row.get("cumulative_return")) > 0.0)
    negative_folds = sum(1 for row in summary_rows if _float(row.get("cumulative_return")) < 0.0)
    sharpe_values = [_float(row.get("sharpe")) for row in summary_rows]
    mean_sharpe = sum(sharpe_values) / float(len(sharpe_values)) if sharpe_values else 0.0
    median_sharpe = _median(sharpe_values)
    best_fold = max(summary_rows, key=lambda row: _float(row.get("cumulative_return")), default={})
    worst_fold = min(summary_rows, key=lambda row: _float(row.get("cumulative_return")), default={})
    walk_forward_total_return = _float(
        walk_forward_metrics.get("total_return"),
        _float(aggregate.get("walk_forward_cumulative_return")),
    )
    walk_forward_final_equity = _float(
        walk_forward_metrics.get("final_equity"),
        1.0 + _float(aggregate.get("walk_forward_cumulative_return")),
    )
    walk_forward_sharpe = _float(walk_forward_metrics.get("sharpe"))
    walk_forward_drawdown = _float(
        walk_forward_metrics.get("max_drawdown"),
        _float(aggregate.get("walk_forward_max_drawdown")),
    )
    walk_forward_avg_turnover = _float(walk_forward_metrics.get("avg_turnover"))
    walk_forward_total_turnover = _float(walk_forward_metrics.get("total_turnover"))
    walk_forward_trade_count = int(_float(walk_forward_metrics.get("trade_count")))
    walk_forward_start_date = str(
        walk_forward_metrics.get("start_date")
        or (folds[0].get("backtest_start_date") if folds else "")
        or ""
    )
    walk_forward_end_date = str(
        walk_forward_metrics.get("end_date")
        or (folds[-1].get("backtest_end_date") if folds else "")
        or ""
    )
    lines = [
        "# Time Series Momentum Research Report",
        "",
        "## 1. Strategy implementation",
        "",
        "- Paper reference: Moskowitz, Ooi, and Pedersen (2012), \"Time Series Momentum.\"",
        "- Paper universe: 58 liquid futures across equity indexes, government bonds, currencies, and commodities.",
        "- Platform implementation: monthly-rebalanced direct-signal strategy on liquid multi-asset ETF proxies available in the local FMP-backed database.",
        "- Trading universe used here: " + ", ".join(available_symbols),
        "- Missing requested proxy symbols not present locally: " + (", ".join(missing_symbols) if missing_symbols else "none"),
        "- Signal: 12-month return excluding the most recent month, implemented as `(1 + px__ret_252_d) / (1 + px__ret_21_d) - 1`.",
        "- Position rule: sign transform applied to the direct signal so positive signals are equally weighted longs and negative signals are equally weighted shorts.",
        "- Rebalance frequency: " + str(strategy_config.get("rebalance_freq") or "M"),
        "- Gross exposure target: " + str(strategy_config.get("gross_exposure") or 1.0),
        "- Train/test split: yearly walk-forward validation; each fold trains on data through the prior December 31 and tests the next calendar year.",
        "- Evaluation metrics: Sharpe ratio, total return, max drawdown, turnover, trade count, plus fold-level stability statistics.",
        "",
        "## 2. Experiment results",
        "",
        f"- Fold count: {int(aggregate.get('fold_count') or len(summary_rows))}",
        f"- Walk-forward test window: {walk_forward_start_date} to {walk_forward_end_date}",
        f"- Positive folds: {positive_folds}",
        f"- Negative folds: {negative_folds}",
        f"- Walk-forward Sharpe ratio: {walk_forward_sharpe:.3f}",
        f"- Walk-forward total return: {_pct(walk_forward_total_return)}",
        f"- Walk-forward final equity: {walk_forward_final_equity:.4f}",
        f"- Walk-forward max drawdown: {_pct(walk_forward_drawdown)}",
        f"- Walk-forward excess cumulative return vs equal-weight benchmark: {_pct(aggregate.get('walk_forward_excess_cumulative_return'))}",
        f"- Mean fold Sharpe: {mean_sharpe:.3f}",
        f"- Median fold Sharpe: {median_sharpe:.3f}",
        f"- Mean fold excess return: {_pct(aggregate.get('mean_fold_excess_cumulative_return'))}",
        f"- Avg daily turnover: {walk_forward_avg_turnover:.4f}",
        f"- Total turnover: {walk_forward_total_turnover:.4f}",
        f"- Trade count: {walk_forward_trade_count}",
        f"- Best fold: {best_fold.get('fold_name', 'n/a')} ({_pct(best_fold.get('cumulative_return'))}, Sharpe {(_float(best_fold.get('sharpe'))):.3f})",
        f"- Worst fold: {worst_fold.get('fold_name', 'n/a')} ({_pct(worst_fold.get('cumulative_return'))}, Sharpe {(_float(worst_fold.get('sharpe'))):.3f})",
        "",
        "## 3. Differences from the paper",
        "",
        "- The paper studies 58 liquid futures across equity indexes, bonds, currencies, and commodities; this implementation uses liquid ETF proxies because the platform does not yet have a native futures dataset.",
        "- The paper works with excess returns and volatility-scaled positions; this implementation uses simple total returns from adjusted prices and equal-weight long/short sign positions.",
        "- The paper sample spans 1965 to 2009 for futures; this backtest uses the locally available proxy sample and a 2011-2025 yearly walk-forward evaluation window.",
        "- The paper reports pooled t-statistics, factor regressions, and decomposition across horizons; this run focuses on platform-native backtest metrics and walk-forward stability.",
        "",
        "## Backtest config",
        "",
        "- Fee bps: " + str(backtest_config.get("fee_bps") or 0.0),
        "- Slippage bps: " + str(backtest_config.get("slippage_bps") or 0.0),
        "- Short borrow bps annual: " + str(backtest_config.get("short_borrow_bps_annual") or 0.0),
        "- Execution delay days: " + str(backtest_config.get("execution_delay_days") or 0),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_time_series_momentum_research(
    *,
    symbols: Sequence[str] | None = None,
    test_start_year: int = 2011,
    test_end_year: int = 2025,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_basename: str = "time_series_momentum_research",
    resume_existing: bool = False,
    report_path: str | Path = Path("docs") / "research" / "time_series_momentum_report.md",
) -> dict[str, Any]:
    from .direct_strategy_runner import run_walk_forward_direct_strategy_backtests

    start_year = int(test_start_year)
    end_year = int(test_end_year)
    if start_year > end_year:
        raise ValueError("test_start_year must be <= test_end_year.")

    requested_symbols = list(symbols or DEFAULT_TSMOM_PROXY_SYMBOLS)
    available_symbols, missing_symbols = _resolve_available_symbols(requested_symbols)
    if not available_symbols:
        raise ValueError("None of the requested symbols were found locally.")

    strategy_config: dict[str, Any] = {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "combined_score_expr": "(1.0 + px__ret_252_d) / (1.0 + px__ret_21_d) - 1.0",
        "action_transform": "sign",
        "action_threshold": 0.0,
    }
    backtest_config: dict[str, Any] = {
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "short_borrow_bps_annual": float(short_borrow_bps_annual),
        "execution_delay_days": int(execution_delay_days),
        "turnover_half_l1": True,
        "use_lagged_weights": True,
    }
    validation_config = {
        "min_trained_rows": 252,
        "min_rows_scored": 50,
        "min_selected_rows": 10,
        "min_trades": 10,
        "min_benchmark_days": 50,
        "min_valid_fold_rate": 0.6,
        "max_fold_excess_std": 0.5,
    }
    feature_config = {
        "include_price_technicals": True,
        "include_fundamental_change": False,
        "include_statement_quality": False,
        "include_event_features": False,
        "include_ownership_features": False,
        "include_economic_indicators": False,
        "include_treasury_rates": False,
    }
    payload = run_walk_forward_direct_strategy_backtests(
        symbols=available_symbols,
        folds=_build_yearly_folds(start_year, end_year),
        feature_config=feature_config,
        strategy_definition_slug="time-series-momentum-12-1",
        strategy_definition_name="Time Series Momentum 12-1",
        strategy_config=strategy_config,
        validation_config=validation_config,
        backtest_config=backtest_config,
        output_basename=str(output_basename).strip(),
        resume_existing=bool(resume_existing),
    )

    resolved_report_path = Path(report_path)
    _write_report(
        report_path=resolved_report_path,
        payload=payload,
        available_symbols=available_symbols,
        missing_symbols=missing_symbols,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )

    output = {
        "mode": "time_series_momentum_research",
        "paper_reference": 'Moskowitz, Ooi, and Pedersen (2012), "Time Series Momentum."',
        "requested_symbols": [str(symbol).strip().upper() for symbol in requested_symbols if str(symbol).strip()],
        "symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "strategy_config": strategy_config,
        "backtest_config": backtest_config,
        "validation_config": validation_config,
        "feature_config": feature_config,
        "report_path": str(resolved_report_path),
        **dict(payload),
    }
    return json.loads(json.dumps(output))


__all__ = [
    "DEFAULT_TSMOM_PROXY_SYMBOLS",
    "run_time_series_momentum_research",
]
