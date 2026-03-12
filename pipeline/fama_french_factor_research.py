from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .cohort_runner import _load_cached_payload, _resolve_or_build_feature_artifact, _resolve_or_build_universe_artifact
from .cross_sectional_rank_labels import first_available_column
from .direct_strategy_runner import run_walk_forward_direct_strategy_backtests
from .experiments import available_feature_families
from .factor_analysis import (
    combine_named_daily_return_frames,
    compute_factor_correlation_rows,
    compute_strategy_factor_exposure_rows,
    load_daily_return_frame,
)
from .models import Artifact
from .universe_selection import filter_symbols_by_price_history, resolve_symbol_universe, summarize_symbol_price_history


FAMA_FRENCH_FACTOR_SCHEMA_VERSION = 1
DEFAULT_EXCLUDED_SYMBOL_PREFIXES: tuple[str, ...] = ("TIER",)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in list(values or []))
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in list(rows or [])]
    if not items:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in items:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def _summary_output_paths(output_basename: str) -> dict[str, Path]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "summary_json": output_dir / f"{output_basename}.json",
        "summary_csv": output_dir / f"{output_basename}.csv",
        "factor_returns_csv": output_dir / f"{output_basename}__factor_returns.csv",
        "factor_metrics_csv": output_dir / f"{output_basename}__factor_metrics.csv",
        "factor_metrics_json": output_dir / f"{output_basename}__factor_metrics.json",
        "factor_correlations_csv": output_dir / f"{output_basename}__factor_correlations.csv",
        "strategy_metrics_csv": output_dir / f"{output_basename}__strategy_metrics.csv",
        "strategy_factor_exposures_csv": output_dir / f"{output_basename}__strategy_factor_exposures.csv",
        "coverage_csv": output_dir / f"{output_basename}__coverage.csv",
    }


def build_yearly_folds(start_year: int, end_year: int) -> list[dict[str, str]]:
    return [
        {
            "name": f"wf_{year}",
            "train_end_date": f"{year - 1}-12-31",
            "backtest_start_date": f"{year}-01-01",
            "backtest_end_date": f"{year}-12-31",
        }
        for year in range(int(start_year), int(end_year) + 1)
    ]


def default_feature_config(*, start_year: int, end_year: int) -> dict[str, Any]:
    return {
        "feature_start_date": f"{int(start_year)}-01-01",
        "feature_end_date": f"{int(end_year)}-12-31",
        "include_price_technicals": True,
        "include_fundamental_change": True,
        "include_statement_quality": True,
        "include_event_features": False,
        "include_ownership_features": False,
        "include_economic_indicators": False,
        "include_treasury_rates": False,
        "include_representation_embedding": False,
    }


def default_backtest_config(
    *,
    fee_bps: float,
    slippage_bps: float,
    short_borrow_bps_annual: float,
    execution_delay_days: int,
) -> dict[str, Any]:
    return {
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "short_borrow_bps_annual": float(short_borrow_bps_annual),
        "execution_delay_days": int(execution_delay_days),
        "turnover_half_l1": True,
        "use_lagged_weights": True,
        "min_price": 5.0,
        "min_dollar_volume": 10_000_000.0,
    }


def _required_history_window(
    *,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int,
) -> tuple[str, str, int]:
    first_test_date = pd.Timestamp(f"{int(test_start_year)}-01-01")
    required_start = first_test_date - pd.offsets.BDay(int(lookback_days) + 10)
    required_end = pd.Timestamp(f"{int(test_end_year)}-12-31")
    expected_days = len(pd.bdate_range(start=required_start, end=required_end))
    min_history_days = max(int(expected_days * 0.75), int(lookback_days) + 10)
    return required_start.strftime("%Y-%m-%d"), required_end.strftime("%Y-%m-%d"), int(min_history_days)


def resolve_research_symbols(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int,
    candidate_limit: int,
    min_market_cap: float,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int = 252,
    exclude_symbol_prefixes: Sequence[str] = DEFAULT_EXCLUDED_SYMBOL_PREFIXES,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if requested_symbols:
        candidate_symbols = [
            str(symbol).strip().upper()
            for symbol in list(requested_symbols or [])
            if str(symbol).strip()
        ]
    else:
        candidate_symbols = resolve_symbol_universe(
            min_market_cap=float(min_market_cap),
            country="US",
            exchanges=["NASDAQ", "NYSE", "AMEX"],
            limit=max(int(candidate_limit), int(symbol_limit)),
            exclude_pooled_vehicles=True,
            exclude_symbol_prefixes=list(exclude_symbol_prefixes),
        )
    required_start, required_end, min_history_days = _required_history_window(
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        lookback_days=int(lookback_days),
    )
    filtered_symbols = filter_symbols_by_price_history(
        candidate_symbols,
        start_date=required_start,
        end_date=required_end,
        required_start_date=required_start,
        required_end_date=required_end,
        min_history_days=min_history_days,
    )
    selected_symbols = filtered_symbols[: max(int(symbol_limit), 1)]
    coverage_rows = summarize_symbol_price_history(
        selected_symbols,
        start_date=required_start,
        end_date=required_end,
        required_start_date=required_start,
        required_end_date=required_end,
        min_history_days=min_history_days,
    )
    missing_requested = [symbol for symbol in candidate_symbols if symbol not in filtered_symbols]
    return selected_symbols, coverage_rows, missing_requested


def load_feature_columns_frame(feature_artifact: Artifact) -> pd.DataFrame:
    if feature_artifact is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(str(feature_artifact.uri or ""), nrows=1)
    except Exception:
        return pd.DataFrame()


def resolve_factor_signal_specs(feature_artifact: Artifact) -> dict[str, dict[str, Any]]:
    columns = list(load_feature_columns_frame(feature_artifact).columns)
    momentum_long = first_available_column(
        columns,
        ("px__ret_252_d", "px__ret_252d", "ret_252_d", "ret_252d"),
    )
    momentum_short = first_available_column(
        columns,
        ("px__ret_21_d", "px__ret_21d", "ret_21_d", "ret_21d"),
    )
    if momentum_long and momentum_short:
        momentum_expression = f"(1.0 + {momentum_long}) / (1.0 + {momentum_short}) - 1.0"
        momentum_columns = [momentum_long, momentum_short]
    else:
        momentum_fallback = first_available_column(
            columns,
            ("px__ret_252_d", "px__ret_252d", "px__ret_189_d", "px__ret_189d", "ret_1"),
        )
        if not momentum_fallback:
            raise ValueError("Could not resolve a momentum signal from the existing feature artifact.")
        momentum_expression = str(momentum_fallback)
        momentum_columns = [momentum_fallback]

    size_field = first_available_column(columns, ("km__marketcap", "marketcap", "market_cap"))
    if not size_field:
        raise ValueError("Could not resolve a size signal from the existing feature artifact.")

    value_field = first_available_column(
        columns,
        (
            "rt__pricetobookratio",
            "rt__pricetofairvalue",
            "km__earningsyield",
            "km__freecashflowyield",
            "rt__dividendyield",
        ),
    )
    if not value_field:
        raise ValueError("Could not resolve a value signal from the existing feature artifact.")
    value_higher_is_better = value_field in {"km__earningsyield", "km__freecashflowyield", "rt__dividendyield"}

    return {
        "momentum": {
            "label": "Momentum",
            "expression": momentum_expression,
            "used_columns": momentum_columns,
            "higher_score_is_better": True,
            "description": "12-month return excluding the most recent month.",
        },
        "size": {
            "label": "Size",
            "expression": str(size_field),
            "used_columns": [str(size_field)],
            "higher_score_is_better": False,
            "description": "Smaller market capitalization ranks better.",
        },
        "value": {
            "label": "Value",
            "expression": str(value_field),
            "used_columns": [str(value_field)],
            "higher_score_is_better": bool(value_higher_is_better),
            "description": (
                "Higher valuation yield ranks better."
                if value_higher_is_better
                else "Lower price-to-book style ratios rank better."
            ),
        },
    }


def market_factor_strategy_config() -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_only",
        "signal_combination": "direct",
        "combined_score_expr": "1.0",
    }


def long_short_factor_strategy_config(
    *,
    factor_signal: str,
    higher_score_is_better: bool,
    long_quantile: float,
    short_quantile: float,
) -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "portfolio_construction": "long_short_factor",
        "factor_signal": str(factor_signal),
        "long_quantile": float(long_quantile),
        "short_quantile": float(short_quantile),
        "holding_period_rebalances": 1,
        "ranking_lag_days": 0,
        "higher_score_is_better": bool(higher_score_is_better),
    }


def ranking_strategy_config(
    *,
    bucket_count: int,
    score_expression: str = "",
    factor_components: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    config = {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "portfolio_construction": "cross_sectional_quantiles",
        "cross_sectional_score_field": "strategy_score",
        "cross_sectional_bucket_count": int(bucket_count),
        "long_bucket": "top",
        "short_bucket": "bottom",
        "holding_period_rebalances": 1,
        "ranking_lag_days": 0,
        "higher_score_is_better": True,
    }
    if factor_components:
        config["factor_components"] = [dict(component) for component in list(factor_components or [])]
    else:
        config["combined_score_expr"] = str(score_expression)
    return config


def _summary_row(
    payload: Mapping[str, Any],
    *,
    series_name: str,
    series_label: str,
    series_kind: str,
    signal_source: str,
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in list(payload.get("summary_rows") or [])]
    aggregate = dict((payload.get("aggregate_rows") or [{}])[0] if payload.get("aggregate_rows") else {})
    walk_forward = dict(payload.get("walk_forward_metrics") or {})
    sharpe_values = [_safe_float(row.get("sharpe")) for row in summary_rows]
    cumulative_values = [_safe_float(row.get("cumulative_return")) for row in summary_rows]
    positive_folds = sum(1 for value in cumulative_values if value > 0.0)
    return {
        "series_name": str(series_name),
        "series_label": str(series_label),
        "series_kind": str(series_kind),
        "signal_source": str(signal_source),
        "fold_count": int(aggregate.get("fold_count") or len(summary_rows)),
        "positive_folds": int(positive_folds),
        "positive_fold_rate": round(float(positive_folds) / float(len(summary_rows)), 8) if summary_rows else 0.0,
        "sharpe": round(_safe_float(walk_forward.get("sharpe")), 8),
        "total_return": round(_safe_float(walk_forward.get("total_return")), 8),
        "final_equity": round(_safe_float(walk_forward.get("final_equity"), 1.0), 8),
        "max_drawdown": round(_safe_float(walk_forward.get("max_drawdown")), 8),
        "avg_turnover": round(_safe_float(walk_forward.get("avg_turnover")), 8),
        "total_turnover": round(_safe_float(walk_forward.get("total_turnover")), 8),
        "trade_count": int(_safe_float(walk_forward.get("trade_count"))),
        "mean_fold_sharpe": round(sum(sharpe_values) / float(len(sharpe_values)), 8) if sharpe_values else 0.0,
        "median_fold_sharpe": round(_median(sharpe_values), 8),
        "fold_return_std": round(_safe_float(aggregate.get("fold_cumulative_return_std")), 8),
        "fold_drawdown_abs_std": round(_safe_float(aggregate.get("fold_drawdown_abs_std")), 8),
        "passed_stability_gates": bool(aggregate.get("passed_stability_gates", True)),
        "stability_gate_reasons": list(aggregate.get("stability_gate_reasons") or []),
        "summary_json_path": str(payload.get("summary_json_path") or ""),
        "summary_csv_path": str(payload.get("summary_csv_path") or ""),
    }


def load_walk_forward_daily_frame(payload: Mapping[str, Any], *, series_name: str, series_kind: str) -> pd.DataFrame:
    ordered_rows = sorted(
        [dict(row) for row in list(payload.get("summary_rows") or [])],
        key=lambda row: (
            str(row.get("backtest_start_date") or ""),
            str(row.get("fold_name") or ""),
            int(row.get("backtest_artifact_id") or 0),
        ),
    )
    artifact_ids = [
        int(row.get("backtest_artifact_id") or 0)
        for row in ordered_rows
        if int(row.get("backtest_artifact_id") or 0) > 0
    ]
    artifact_lookup = Artifact.objects.in_bulk(artifact_ids)
    daily_rows: list[dict[str, Any]] = []
    for row in ordered_rows:
        artifact = artifact_lookup.get(int(row.get("backtest_artifact_id") or 0))
        if artifact is None:
            continue
        fold_name = str(row.get("fold_name") or "")
        for daily_row in list((artifact.content or {}).get("daily_rows") or []):
            daily_rows.append(
                {
                    **dict(daily_row),
                    "fold_name": fold_name,
                    "series_name": str(series_name),
                    "series_kind": str(series_kind),
                }
            )
    return load_daily_return_frame(
        pd.DataFrame(daily_rows),
        series_name=series_name,
        series_kind=series_kind,
    )


def write_fama_french_factor_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    factor_rows = [dict(row) for row in list(payload.get("factor_rows") or [])]
    strategy_rows = [dict(row) for row in list(payload.get("strategy_rows") or [])]
    exposure_rows = [dict(row) for row in list(payload.get("strategy_factor_exposure_rows") or [])]
    correlation_rows = [dict(row) for row in list(payload.get("factor_correlation_rows") or [])]
    coverage_rows = [dict(row) for row in list(payload.get("coverage_rows") or [])]
    if not factor_rows or not strategy_rows:
        raise ValueError("Expected factor_rows and strategy_rows to write the report.")

    best_strategy = max(
        strategy_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
    )
    baseline_strategy = next((row for row in strategy_rows if str(row.get("series_name") or "") == "momentum_baseline"), strategy_rows[0])
    top_factors = sorted(
        factor_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
        ),
        reverse=True,
    )
    matrix_rows = [row for row in correlation_rows if row.get("left_factor") == row.get("right_factor")]
    correlation_pairs = [
        row
        for row in correlation_rows
        if str(row.get("left_factor") or "") < str(row.get("right_factor") or "")
    ]
    coverage_summary = ", ".join(
        f"{row['symbol']} ({row['history_start_date']} to {row['history_end_date']})"
        for row in coverage_rows[:10]
    )
    lines = [
        "# Fama-French Style Factor Research",
        "",
        "Reference framing:",
        "- Fama and French (1993): [Common Risk Factors in the Returns on Stocks and Bonds](https://www.jstor.org/stable/2329112)",
        "- Ken French Data Library overview: [Description of Fama/French factors](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html)",
        "",
        "## 1. Implemented factors",
        "",
        "- `MKT`: equal-weight market proxy built from the full selected universe and run through the shared direct-strategy backtest path.",
        "- `SMB`: long the smaller-cap sleeve and short the larger-cap sleeve using the existing `km__marketcap` feature.",
        "- `HML`: long the value sleeve and short the growth sleeve using the platform's existing valuation feature set.",
        "- Multi-factor ranking strategy: equal-weight blend of cross-sectional momentum, value, and size component ranks, then fed into `portfolio_construction=cross_sectional_quantiles`.",
        f"- Universe used here: {len(list(payload.get('symbols') or []))} US large-cap equities with sufficient history and platform liquidity filters.",
        "- Selected symbols: " + ", ".join(str(symbol) for symbol in list(payload.get("symbols") or [])),
        "- Coverage snapshot: " + (coverage_summary if coverage_summary else "n/a"),
        "",
        "## 2. Factor results",
        "",
    ]
    for row in top_factors:
        lines.append(
            f"- {row['series_name']}: Sharpe {(_safe_float(row.get('sharpe'))):.3f}, total return {_pct(row.get('total_return'))}, drawdown {_pct(row.get('max_drawdown'))}, positive folds {int(row.get('positive_folds') or 0)}/{int(row.get('fold_count') or 0)}"
        )
    lines.extend(
        [
            "",
            "Factor correlations:",
        ]
    )
    for row in correlation_pairs:
        lines.append(
            f"- {row['left_factor']} vs {row['right_factor']}: correlation {(_safe_float(row.get('correlation'))):.3f}"
        )
    lines.extend(
        [
            "",
            "## 3. Strategy comparison",
            "",
            f"- Momentum baseline: Sharpe {(_safe_float(baseline_strategy.get('sharpe'))):.3f}, total return {_pct(baseline_strategy.get('total_return'))}, drawdown {_pct(baseline_strategy.get('max_drawdown'))}.",
            f"- Best strategy: {best_strategy.get('series_name')} | Sharpe {(_safe_float(best_strategy.get('sharpe'))):.3f}, total return {_pct(best_strategy.get('total_return'))}, drawdown {_pct(best_strategy.get('max_drawdown'))}.",
            "- Interpretation: this experiment asks whether reusable factor primitives improve cross-sectional ranking versus a simple momentum sort, not whether the local universe reproduces the exact original paper coefficients.",
            "",
            "## 4. Factor exposure diagnostics",
            "",
        ]
    )
    for row in exposure_rows:
        beta_fields = sorted(field for field in row.keys() if field.startswith("beta_"))
        beta_text = ", ".join(f"{field.replace('beta_', '').upper()} {(_safe_float(row.get(field))):.3f}" for field in beta_fields)
        lines.append(
            f"- {row['strategy_name']}: alpha {(_safe_float(row.get('alpha'))):.5f}, R^2 {(_safe_float(row.get('r_squared'))):.3f}, residual return {_pct(row.get('residual_total_return'))}, betas [{beta_text}]"
        )
    lines.extend(
        [
            "",
            "## 5. Platform capabilities added",
            "",
            "- Added a reusable `portfolio_construction=long_short_factor` mode so any existing score or feature expression can become a factor portfolio.",
            "- Added reusable cross-sectional factor-component scoring so weighted factor blends can drive the existing ranking portfolio constructor.",
            "- Added reusable factor analytics for return-series metrics, correlation analysis, and strategy exposure regression.",
            "",
            "## 6. Differences from the original paper",
            "",
            "- The original Fama-French construction uses value-weighted portfolios, NYSE breakpoints, and a market-minus-risk-free series; this implementation uses a local equal-weight proxy universe and practical backtest costs.",
            "- The research stack uses existing platform features such as `km__marketcap` and valuation ratios, rather than rebuilding CRSP/Compustat style accounting pipelines.",
            "- The main goal here is reusable platform capability for future multi-factor and ML ranking work, so the report emphasizes factor returns, cross-strategy exposures, and ranking portability.",
            "",
            "## 7. Implications for future ML ranking models",
            "",
            "- The new factor-component path means future ML ranking models can blend model scores with reusable value/size/momentum context inside the same ranking workflow.",
            "- Strategy exposure regression now makes it easier to tell whether a future ML model is producing genuine alpha or just repackaging factor loads.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fama_french_factor_research(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int = 25,
    candidate_limit: int = 80,
    min_market_cap: float = 10_000_000_000.0,
    test_start_year: int = 2021,
    test_end_year: int = 2025,
    bucket_count: int = 5,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_basename: str = "fama_french_factor_research",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_paths = _summary_output_paths(output_basename)
    if resume_existing:
        cached_payload = _load_cached_payload(
            output_paths["summary_json"],
            required_keys=("factor_rows", "strategy_rows", "strategy_factor_exposure_rows"),
            schema_version=FAMA_FRENCH_FACTOR_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            return cached_payload

    selected_symbols, coverage_rows, missing_requested = resolve_research_symbols(
        requested_symbols=requested_symbols,
        symbol_limit=int(symbol_limit),
        candidate_limit=int(candidate_limit),
        min_market_cap=float(min_market_cap),
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
    )
    if not selected_symbols:
        raise ValueError("No symbols passed the coverage-aware universe filter for the requested factor research window.")

    feature_config = default_feature_config(
        start_year=int(test_start_year) - 1,
        end_year=int(test_end_year),
    )
    backtest_config = default_backtest_config(
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        short_borrow_bps_annual=float(short_borrow_bps_annual),
        execution_delay_days=int(execution_delay_days),
    )
    folds = build_yearly_folds(int(test_start_year), int(test_end_year))
    universe_artifact = _resolve_or_build_universe_artifact(
        symbols=selected_symbols,
        output_basename=output_basename,
    )
    feature_artifact = _resolve_or_build_feature_artifact(
        universe_artifact=universe_artifact,
        symbols=selected_symbols,
        feature_config=feature_config,
        output_basename=output_basename,
    )
    signal_specs = resolve_factor_signal_specs(feature_artifact)

    factor_runs = {
        "MKT": {
            "label": "Market Proxy",
            "signal_source": "equal_weight_universe",
            "payload": run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=f"{output_basename}__mkt",
                strategy_definition_name="Fama-French Market Proxy",
                strategy_config=market_factor_strategy_config(),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__mkt",
                resume_existing=resume_existing,
            ),
        },
        "SMB": {
            "label": "Small Minus Big",
            "signal_source": str(signal_specs["size"]["expression"]),
            "payload": run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=f"{output_basename}__smb",
                strategy_definition_name="Fama-French SMB",
                strategy_config=long_short_factor_strategy_config(
                    factor_signal=str(signal_specs["size"]["expression"]),
                    higher_score_is_better=bool(signal_specs["size"]["higher_score_is_better"]),
                    long_quantile=0.5,
                    short_quantile=0.5,
                ),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__smb",
                resume_existing=resume_existing,
            ),
        },
        "HML": {
            "label": "High Minus Low",
            "signal_source": str(signal_specs["value"]["expression"]),
            "payload": run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=f"{output_basename}__hml",
                strategy_definition_name="Fama-French HML",
                strategy_config=long_short_factor_strategy_config(
                    factor_signal=str(signal_specs["value"]["expression"]),
                    higher_score_is_better=bool(signal_specs["value"]["higher_score_is_better"]),
                    long_quantile=0.3,
                    short_quantile=0.3,
                ),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__hml",
                resume_existing=resume_existing,
            ),
        },
    }
    factor_rows = [
        _summary_row(
            payload=factor_run["payload"],
            series_name=factor_name,
            series_label=str(factor_run["label"]),
            series_kind="factor",
            signal_source=str(factor_run["signal_source"]),
        )
        for factor_name, factor_run in factor_runs.items()
    ]
    factor_daily_lookup = {
        factor_name: load_walk_forward_daily_frame(
            factor_run["payload"],
            series_name=factor_name,
            series_kind="factor",
        )
        for factor_name, factor_run in factor_runs.items()
    }
    factor_return_rows = combine_named_daily_return_frames(factor_daily_lookup, series_kind="factor").to_dict(orient="records")
    factor_correlation_rows = compute_factor_correlation_rows(factor_daily_lookup)

    strategy_runs = {
        "momentum_baseline": {
            "label": "Momentum Baseline",
            "signal_source": str(signal_specs["momentum"]["expression"]),
            "payload": run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=f"{output_basename}__momentum_baseline",
                strategy_definition_name="Fama-French Momentum Baseline",
                strategy_config=ranking_strategy_config(
                    bucket_count=int(bucket_count),
                    score_expression=str(signal_specs["momentum"]["expression"]),
                ),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__momentum_baseline",
                resume_existing=resume_existing,
            ),
        },
        "multi_factor_rank": {
            "label": "Multi-Factor Rank",
            "signal_source": "factor_components",
            "payload": run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=f"{output_basename}__multi_factor_rank",
                strategy_definition_name="Fama-French Multi-Factor Rank",
                strategy_config=ranking_strategy_config(
                    bucket_count=int(bucket_count),
                    factor_components=[
                        {
                            "name": "momentum",
                            "expression": str(signal_specs["momentum"]["expression"]),
                            "weight": 1.0,
                            "transform": "rank_pct",
                            "higher_is_better": bool(signal_specs["momentum"]["higher_score_is_better"]),
                        },
                        {
                            "name": "value",
                            "expression": str(signal_specs["value"]["expression"]),
                            "weight": 1.0,
                            "transform": "rank_pct",
                            "higher_is_better": bool(signal_specs["value"]["higher_score_is_better"]),
                        },
                        {
                            "name": "size",
                            "expression": str(signal_specs["size"]["expression"]),
                            "weight": 1.0,
                            "transform": "rank_pct",
                            "higher_is_better": bool(signal_specs["size"]["higher_score_is_better"]),
                        },
                    ],
                ),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__multi_factor_rank",
                resume_existing=resume_existing,
            ),
        },
    }
    strategy_rows = [
        _summary_row(
            payload=strategy_run["payload"],
            series_name=strategy_name,
            series_label=str(strategy_run["label"]),
            series_kind="strategy",
            signal_source=str(strategy_run["signal_source"]),
        )
        for strategy_name, strategy_run in strategy_runs.items()
    ]
    strategy_daily_lookup = {
        strategy_name: load_walk_forward_daily_frame(
            strategy_run["payload"],
            series_name=strategy_name,
            series_kind="strategy",
        )
        for strategy_name, strategy_run in strategy_runs.items()
    }
    strategy_factor_exposure_rows = compute_strategy_factor_exposure_rows(strategy_daily_lookup, factor_daily_lookup)

    factor_rows = sorted(
        factor_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
        reverse=True,
    )
    strategy_rows = sorted(
        strategy_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
        reverse=True,
    )

    _write_rows_csv(output_paths["summary_csv"], factor_rows + strategy_rows)
    _write_rows_csv(output_paths["factor_returns_csv"], factor_return_rows)
    _write_rows_csv(output_paths["factor_metrics_csv"], factor_rows)
    _write_rows_csv(output_paths["factor_correlations_csv"], factor_correlation_rows)
    _write_rows_csv(output_paths["strategy_metrics_csv"], strategy_rows)
    _write_rows_csv(output_paths["strategy_factor_exposures_csv"], strategy_factor_exposure_rows)
    _write_rows_csv(output_paths["coverage_csv"], coverage_rows)
    output_paths["factor_metrics_json"].write_text(
        json.dumps(
            {
                "schema_version": FAMA_FRENCH_FACTOR_SCHEMA_VERSION,
                "factor_rows": factor_rows,
                "factor_correlation_rows": factor_correlation_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = {
        "schema_version": FAMA_FRENCH_FACTOR_SCHEMA_VERSION,
        "mode": "fama_french_factor_research",
        "paper": {
            "title": "Common Risk Factors in the Returns on Stocks and Bonds",
            "authors": ["Eugene F. Fama", "Kenneth R. French"],
            "year": 1993,
        },
        "symbols": selected_symbols,
        "missing_requested_symbols": missing_requested,
        "coverage_rows": coverage_rows,
        "folds": folds,
        "feature_config": feature_config,
        "backtest_config": backtest_config,
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
        },
        "feature_families": available_feature_families(feature_artifact),
        "signal_specs": signal_specs,
        "factor_rows": factor_rows,
        "strategy_rows": strategy_rows,
        "factor_correlation_rows": factor_correlation_rows,
        "strategy_factor_exposure_rows": strategy_factor_exposure_rows,
        "factor_payload_rows": [
            {
                "series_name": factor_name,
                "summary_json_path": str(run_data["payload"].get("summary_json_path") or ""),
                "summary_csv_path": str(run_data["payload"].get("summary_csv_path") or ""),
            }
            for factor_name, run_data in factor_runs.items()
        ],
        "strategy_payload_rows": [
            {
                "series_name": strategy_name,
                "summary_json_path": str(run_data["payload"].get("summary_json_path") or ""),
                "summary_csv_path": str(run_data["payload"].get("summary_csv_path") or ""),
            }
            for strategy_name, run_data in strategy_runs.items()
        ],
        "summary_json_path": str(output_paths["summary_json"]),
        "summary_csv_path": str(output_paths["summary_csv"]),
        "factor_returns_csv_path": str(output_paths["factor_returns_csv"]),
        "factor_metrics_csv_path": str(output_paths["factor_metrics_csv"]),
        "factor_metrics_json_path": str(output_paths["factor_metrics_json"]),
        "factor_correlations_csv_path": str(output_paths["factor_correlations_csv"]),
        "strategy_metrics_csv_path": str(output_paths["strategy_metrics_csv"]),
        "strategy_factor_exposures_csv_path": str(output_paths["strategy_factor_exposures_csv"]),
        "coverage_csv_path": str(output_paths["coverage_csv"]),
    }
    output_paths["summary_json"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = [
    "build_yearly_folds",
    "default_backtest_config",
    "default_feature_config",
    "load_walk_forward_daily_frame",
    "resolve_factor_signal_specs",
    "resolve_research_symbols",
    "run_fama_french_factor_research",
    "write_fama_french_factor_report",
]
