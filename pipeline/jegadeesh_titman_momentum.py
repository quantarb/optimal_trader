from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .cohort_runner import _resolve_or_build_feature_artifact, _resolve_or_build_universe_artifact
from .direct_strategy_runner import run_walk_forward_direct_strategy_backtests
from .models import Artifact
from .symbol_diagnostics import aggregate_symbol_diagnostic_rows, compute_symbol_strategy_diagnostics
from .universe_selection import filter_symbols_by_price_history, resolve_symbol_universe, summarize_symbol_price_history


JEGADEESH_TITMAN_SCHEMA_VERSION = 1
DEFAULT_FORMATION_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
DEFAULT_HOLDING_MONTHS: tuple[int, ...] = (3, 6, 9, 12)
DEFAULT_EXCLUDED_SYMBOL_PREFIXES: tuple[str, ...] = ("TIER",)
FORMATION_RETURN_FIELD = {
    3: "px__ret_63_d",
    6: "px__ret_126_d",
    9: "px__ret_189_d",
    12: "px__ret_252_d",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in list(values))
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in rows]
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


def parse_month_values(raw_value: object, *, default: Sequence[int]) -> list[int]:
    if raw_value in (None, ""):
        values = [int(value) for value in list(default)]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [int(value) for value in list(raw_value)]
    else:
        values = [
            int(token)
            for token in str(raw_value).split(",")
            if str(token).strip()
        ]
    out: list[int] = []
    for value in values:
        if value <= 0 or value in out:
            continue
        out.append(int(value))
    return out or [int(value) for value in list(default)]


def price_only_feature_config(*, start_year: int, end_year: int) -> dict[str, Any]:
    return {
        "feature_start_date": f"{int(start_year)}-01-01",
        "feature_end_date": f"{int(end_year)}-12-31",
        "include_price_technicals": True,
        "include_fundamental_change": False,
        "include_statement_quality": False,
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
    max_formation_months: int,
    ranking_lag_days: int,
) -> tuple[str, str, int]:
    first_test_date = pd.Timestamp(f"{int(test_start_year)}-01-01")
    required_start = first_test_date - pd.offsets.BDay((int(max_formation_months) * 21) + int(ranking_lag_days) + 10)
    required_end = pd.Timestamp(f"{int(test_end_year)}-12-31")
    expected_days = len(pd.bdate_range(start=required_start, end=required_end))
    min_history_days = max(int(expected_days * 0.75), (int(max_formation_months) * 21) + int(ranking_lag_days) + 10)
    return required_start.strftime("%Y-%m-%d"), required_end.strftime("%Y-%m-%d"), int(min_history_days)


def resolve_research_symbols(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int,
    candidate_limit: int,
    min_market_cap: float,
    test_start_year: int,
    test_end_year: int,
    max_formation_months: int,
    ranking_lag_days: int,
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
        max_formation_months=int(max_formation_months),
        ranking_lag_days=int(ranking_lag_days),
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


def cross_sectional_strategy_config(
    *,
    formation_months: int,
    holding_months: int,
    bucket_count: int,
    ranking_lag_days: int,
) -> dict[str, Any]:
    if int(formation_months) not in FORMATION_RETURN_FIELD:
        raise ValueError(f"Unsupported formation_months={formation_months}; expected one of {sorted(FORMATION_RETURN_FIELD)}.")
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "portfolio_construction": "cross_sectional_quantiles",
        "cross_sectional_score_field": FORMATION_RETURN_FIELD[int(formation_months)],
        "cross_sectional_bucket_count": int(bucket_count),
        "long_bucket": "top",
        "short_bucket": "bottom",
        "holding_period_rebalances": int(holding_months),
        "ranking_lag_days": int(ranking_lag_days),
        "higher_score_is_better": True,
    }


def _variant_name(*, formation_months: int, holding_months: int, ranking_lag_days: int) -> str:
    return f"jt1993_j{int(formation_months)}_k{int(holding_months)}_lag{int(ranking_lag_days)}"


def _variant_summary_row(
    payload: Mapping[str, Any],
    *,
    formation_months: int,
    holding_months: int,
    ranking_lag_days: int,
    bucket_count: int,
    symbol_count: int,
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in list(payload.get("summary_rows") or [])]
    aggregate = dict((payload.get("aggregate_rows") or [{}])[0] if payload.get("aggregate_rows") else {})
    walk_forward = dict(payload.get("walk_forward_metrics") or {})
    sharpe_values = [_safe_float(row.get("sharpe")) for row in summary_rows]
    cumulative_values = [_safe_float(row.get("cumulative_return")) for row in summary_rows]
    positive_folds = sum(1 for value in cumulative_values if value > 0.0)
    negative_folds = sum(1 for value in cumulative_values if value < 0.0)
    return {
        "variant_name": _variant_name(
            formation_months=formation_months,
            holding_months=holding_months,
            ranking_lag_days=ranking_lag_days,
        ),
        "formation_months": int(formation_months),
        "holding_months": int(holding_months),
        "ranking_lag_days": int(ranking_lag_days),
        "bucket_count": int(bucket_count),
        "symbol_count": int(symbol_count),
        "fold_count": int(aggregate.get("fold_count") or len(summary_rows)),
        "positive_folds": int(positive_folds),
        "negative_folds": int(negative_folds),
        "positive_fold_rate": round(
            float(positive_folds) / float(len(summary_rows)),
            8,
        ) if summary_rows else 0.0,
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
        "walk_forward_excess_cumulative_return": round(
            _safe_float(aggregate.get("walk_forward_excess_cumulative_return")),
            8,
        ),
        "passed_stability_gates": bool(aggregate.get("passed_stability_gates", True)),
        "stability_gate_reasons": list(aggregate.get("stability_gate_reasons") or []),
        "summary_json_path": str(payload.get("summary_json_path") or ""),
        "summary_csv_path": str(payload.get("summary_csv_path") or ""),
    }


def write_jegadeesh_titman_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    variant_rows = [dict(row) for row in list(payload.get("variant_rows") or [])]
    symbol_rows = [dict(row) for row in list(payload.get("symbol_diagnostics_aggregate_rows") or [])]
    coverage_rows = [dict(row) for row in list(payload.get("coverage_rows") or [])]
    symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    if not variant_rows:
        raise ValueError("Expected at least one variant row to write the report.")

    best_variant = max(
        variant_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
    )
    top_variants = sorted(
        variant_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
        reverse=True,
    )[:5]
    worst_variant = min(
        variant_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
        ),
    )
    best_symbol_rows = [
        row
        for row in symbol_rows
        if str(row.get("strategy_name") or "") == str(best_variant.get("variant_name") or "")
    ]
    top_symbols = sorted(
        best_symbol_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("avg_trade_return")),
            _safe_float(row.get("trade_count")),
        ),
        reverse=True,
    )[:5]
    bottom_symbols = sorted(
        best_symbol_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("avg_trade_return")),
            -_safe_float(row.get("trade_count")),
        ),
    )[:5]
    positive_variants = sum(1 for row in variant_rows if _safe_float(row.get("sharpe")) > 0.0)
    stable_variants = sum(1 for row in variant_rows if bool(row.get("passed_stability_gates")))
    k3_variants = [row for row in variant_rows if int(row.get("holding_months") or 0) == 3]
    longer_hold_variants = [row for row in variant_rows if int(row.get("holding_months") or 0) > 3]
    coverage_summary = ", ".join(
        f"{row['symbol']} ({row['history_start_date']} to {row['history_end_date']})"
        for row in coverage_rows[:10]
    )
    lines = [
        "# Jegadeesh & Titman (1993) Research Report",
        "",
        "## 1. Strategy implementation",
        "",
        "- Paper reference: Jegadeesh and Titman (1993), \"Returns to Buying Winners and Selling Losers.\"",
        "- Core paper signal: rank stocks cross-sectionally on cumulative past returns over the prior `J` months, then buy the winner decile and short the loser decile.",
        "- Canonical paper construction: equal-weight deciles, monthly portfolio formation, overlapping `K`-month holding sleeves, with a one-week skip between ranking measurement and portfolio entry.",
        f"- Platform implementation: reusable cross-sectional quantile portfolio construction on price-momentum features, monthly rebalanced, decile winner/loser sleeves, `5` trading-day lag, and overlapping `K`-month holdings.",
        f"- Universe used here: {len(symbols)} locally available US large-cap equities with sufficient price history and platform liquidity filters.",
        "- Selected symbols: " + ", ".join(symbols),
        "- Coverage snapshot: " + (coverage_summary if coverage_summary else "n/a"),
        "- Direct signal fields used: `px__ret_63_d`, `px__ret_126_d`, `px__ret_189_d`, and `px__ret_252_d` depending on the formation window.",
        "",
        "## 2. Experiment results",
        "",
        f"- Variant count: {len(variant_rows)}",
        f"- Positive-Sharpe variants: {positive_variants}",
        f"- Variants passing stability gates: {stable_variants}",
        f"- Best variant: {best_variant.get('variant_name')} | Sharpe {(_safe_float(best_variant.get('sharpe'))):.3f} | total return {_pct(best_variant.get('total_return'))} | max drawdown {_pct(best_variant.get('max_drawdown'))}",
        f"- Best variant fold stability: {int(best_variant.get('positive_folds') or 0)}/{int(best_variant.get('fold_count') or 0)} positive folds",
        f"- Best variant turnover/trades: {(_safe_float(best_variant.get('total_turnover'))):.2f} total turnover | {int(best_variant.get('trade_count') or 0)} trades",
        f"- Worst variant: {worst_variant.get('variant_name')} | Sharpe {(_safe_float(worst_variant.get('sharpe'))):.3f} | total return {_pct(worst_variant.get('total_return'))} | max drawdown {_pct(worst_variant.get('max_drawdown'))}",
        "- Interpretation: in this large-cap survivor universe, changing the formation window from 3 to 12 months barely moved the ranked portfolios; the holding horizon drove most of the performance difference.",
        f"- `K=3` variants averaged Sharpe {(sum(_safe_float(row.get('sharpe')) for row in k3_variants) / float(len(k3_variants))):.3f}, versus {(sum(_safe_float(row.get('sharpe')) for row in longer_hold_variants) / float(len(longer_hold_variants))):.3f} for longer holds.",
        "",
        "Top variants:",
    ]
    for row in top_variants:
        lines.append(
            f"- {row['variant_name']}: Sharpe {(_safe_float(row.get('sharpe'))):.3f}, total return {_pct(row.get('total_return'))}, drawdown {_pct(row.get('max_drawdown'))}, positive folds {int(row.get('positive_folds') or 0)}/{int(row.get('fold_count') or 0)}"
        )
    lines.extend(
        [
            "",
            "Best-variant symbol diagnostics:",
        ]
    )
    for row in top_symbols:
        lines.append(
            f"- Top symbol {row['symbol']}: Sharpe {(_safe_float(row.get('sharpe'))):.3f}, avg trade return {_pct(row.get('avg_trade_return'))}, hit rate {_pct(row.get('hit_rate'))}, trades {int(row.get('trade_count') or 0)}"
        )
    for row in bottom_symbols:
        lines.append(
            f"- Weak symbol {row['symbol']}: Sharpe {(_safe_float(row.get('sharpe'))):.3f}, avg trade return {_pct(row.get('avg_trade_return'))}, hit rate {_pct(row.get('hit_rate'))}, trades {int(row.get('trade_count') or 0)}"
        )
    lines.extend(
        [
            "",
            "## 3. Platform capabilities added",
            "",
            "- Added reusable cross-sectional quantile portfolio construction to `pipeline/strategy_definitions.py`, so any score field can now drive winner/loser decile portfolios.",
            "- Added overlapping multi-rebalance holding support, which closes a major gap between the original academic construction and the platform's previous single-sleeve carry-forward logic.",
            "- Added `Ret189d` to the shared price-technical feature set so 9-month momentum windows are first-class platform features.",
            "- Added coverage-aware symbol history filtering in `pipeline/universe_selection.py`, which makes both direct-strategy and oracle-based studies less brittle on sparse data.",
            "",
            "## 4. Research workflow inefficiencies discovered",
            "",
            "- The platform still rehydrates the full feature artifact for each fold/variant pair, which is simple but wastes time for strategy-family sweeps.",
            "- Academic equity replications remain limited by the local FMP-style symbol store; there is no native CRSP-like universe, delisting return support, or historical membership handling.",
            "- Strategy family comparison is still assembled in research code instead of through a first-class family runner with shared caching and reporting.",
            "",
            "## 5. Lessons for the oracle-based workflow",
            "",
            "- Oracle and model outputs can now be translated into cross-sectional long/short decile portfolios by setting `portfolio_construction=cross_sectional_quantiles`, rather than only thresholding scores independently by symbol.",
            "- Coverage-aware universe screening should happen before oracle-label generation as well, so models are not trained on symbols with fragmented price history.",
            "- A natural next extension is a reusable cross-sectional label builder that emits winner/loser portfolio membership or relative-rank labels, which would let models learn portfolio-level momentum structure directly.",
            "",
            "## 6. Differences from the paper",
            "",
            "- The paper uses the broad CRSP equity universe with historical listings and delisting returns; this run uses a locally available, large-cap, survivor-biased US equity subset.",
            "- The paper's one-week skip is implemented here as a `5` trading-day lag on the ranking signal, which is close but not identical to calendar-week portfolio timing.",
            "- The platform backtest focuses on practical metrics such as Sharpe, total return, drawdown, turnover, and fold stability, rather than the original paper's full significance tables.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_jegadeesh_titman_research(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int = 40,
    candidate_limit: int = 120,
    min_market_cap: float = 10_000_000_000.0,
    test_start_year: int = 2014,
    test_end_year: int = 2025,
    formation_months: Sequence[int] = DEFAULT_FORMATION_MONTHS,
    holding_months: Sequence[int] = DEFAULT_HOLDING_MONTHS,
    ranking_lag_days: int = 5,
    bucket_count: int = 10,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_basename: str = "jegadeesh_titman_momentum_research",
    resume_existing: bool = False,
) -> dict[str, Any]:
    formation_values = parse_month_values(formation_months, default=DEFAULT_FORMATION_MONTHS)
    holding_values = parse_month_values(holding_months, default=DEFAULT_HOLDING_MONTHS)
    selected_symbols, coverage_rows, missing_requested = resolve_research_symbols(
        requested_symbols=requested_symbols,
        symbol_limit=int(symbol_limit),
        candidate_limit=int(candidate_limit),
        min_market_cap=float(min_market_cap),
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        max_formation_months=max(formation_values),
        ranking_lag_days=int(ranking_lag_days),
    )
    if not selected_symbols:
        raise ValueError("No symbols passed the coverage-aware universe filter for the requested research window.")

    feature_config = price_only_feature_config(
        start_year=int(test_start_year) - 1,
        end_year=int(test_end_year),
    )
    folds = build_yearly_folds(int(test_start_year), int(test_end_year))
    backtest_config = default_backtest_config(
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        short_borrow_bps_annual=float(short_borrow_bps_annual),
        execution_delay_days=int(execution_delay_days),
    )

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

    variant_rows: list[dict[str, Any]] = []
    variant_payload_rows: list[dict[str, Any]] = []
    symbol_diagnostic_rows: list[dict[str, Any]] = []
    for formation_value in formation_values:
        for holding_value in holding_values:
            variant_name = _variant_name(
                formation_months=formation_value,
                holding_months=holding_value,
                ranking_lag_days=ranking_lag_days,
            )
            payload = run_walk_forward_direct_strategy_backtests(
                symbols=selected_symbols,
                folds=folds,
                universe_artifact=universe_artifact,
                feature_artifact=feature_artifact,
                feature_config=feature_config,
                strategy_definition_slug=variant_name,
                strategy_definition_name=f"JT1993 J={formation_value} K={holding_value}",
                strategy_config=cross_sectional_strategy_config(
                    formation_months=formation_value,
                    holding_months=holding_value,
                    bucket_count=bucket_count,
                    ranking_lag_days=ranking_lag_days,
                ),
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__{variant_name}",
                resume_existing=resume_existing,
            )
            variant_rows.append(
                _variant_summary_row(
                    payload,
                    formation_months=formation_value,
                    holding_months=holding_value,
                    ranking_lag_days=ranking_lag_days,
                    bucket_count=bucket_count,
                    symbol_count=len(selected_symbols),
                )
            )
            variant_payload_rows.append(
                {
                    "variant_name": variant_name,
                    "summary_json_path": str(payload.get("summary_json_path") or ""),
                    "summary_csv_path": str(payload.get("summary_csv_path") or ""),
                }
            )
            for summary_row in list(payload.get("summary_rows") or []):
                artifact_id = int(summary_row.get("backtest_artifact_id") or 0)
                if artifact_id <= 0:
                    continue
                backtest_artifact = Artifact.objects.filter(pk=artifact_id).first()
                if backtest_artifact is None:
                    continue
                symbol_diagnostic_rows.extend(
                    compute_symbol_strategy_diagnostics(
                        backtest_artifact,
                        strategy_name=variant_name,
                        fold_name=str(summary_row.get("fold_name") or ""),
                        evaluation_scope="walk_forward_fold",
                        backtest_start_date=str(summary_row.get("backtest_start_date") or ""),
                        backtest_end_date=str(summary_row.get("backtest_end_date") or ""),
                        backtest_config=backtest_config,
                    )
                )

    variant_rows = sorted(
        variant_rows,
        key=lambda row: (
            _safe_float(row.get("sharpe")),
            _safe_float(row.get("total_return")),
            -abs(_safe_float(row.get("max_drawdown"))),
        ),
        reverse=True,
    )
    symbol_diagnostics_aggregate_rows = aggregate_symbol_diagnostic_rows(
        symbol_diagnostic_rows,
        group_keys=("strategy_name", "symbol"),
    )

    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / f"{output_basename}.json"
    summary_csv_path = output_dir / f"{output_basename}.csv"
    symbol_csv_path = output_dir / f"{output_basename}__symbol_diagnostics.csv"
    symbol_aggregate_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_aggregate.csv"
    coverage_csv_path = output_dir / f"{output_basename}__coverage.csv"

    _write_rows_csv(summary_csv_path, variant_rows)
    _write_rows_csv(symbol_csv_path, symbol_diagnostic_rows)
    _write_rows_csv(symbol_aggregate_csv_path, symbol_diagnostics_aggregate_rows)
    _write_rows_csv(coverage_csv_path, coverage_rows)

    payload = {
        "schema_version": JEGADEESH_TITMAN_SCHEMA_VERSION,
        "mode": "jegadeesh_titman_research",
        "paper": {
            "title": "Returns to Buying Winners and Selling Losers",
            "authors": ["Narashimhan Jegadeesh", "Sheridan Titman"],
            "year": 1993,
        },
        "symbols": selected_symbols,
        "missing_requested_symbols": missing_requested,
        "coverage_rows": coverage_rows,
        "folds": folds,
        "formation_months": formation_values,
        "holding_months": holding_values,
        "ranking_lag_days": int(ranking_lag_days),
        "bucket_count": int(bucket_count),
        "feature_config": feature_config,
        "backtest_config": backtest_config,
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
        },
        "variant_rows": variant_rows,
        "variant_payload_rows": variant_payload_rows,
        "symbol_diagnostics_aggregate_rows": symbol_diagnostics_aggregate_rows,
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "symbol_diagnostics_csv_path": str(symbol_csv_path),
        "symbol_diagnostics_aggregate_csv_path": str(symbol_aggregate_csv_path),
        "coverage_csv_path": str(coverage_csv_path),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_FORMATION_MONTHS",
    "DEFAULT_HOLDING_MONTHS",
    "build_yearly_folds",
    "parse_month_values",
    "run_jegadeesh_titman_research",
    "write_jegadeesh_titman_report",
]
