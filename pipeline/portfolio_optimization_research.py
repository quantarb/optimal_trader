from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .factor_analysis import summarize_return_frame
from .models import Artifact, StrategyDefinition
from .service_runtime import read_frame_artifact
from .strategy_definitions import upsert_strategy_definition
from .universe_selection import filter_symbols_by_price_history, resolve_symbol_universe, summarize_symbol_price_history


PORTFOLIO_OPTIMIZATION_RESEARCH_SCHEMA_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


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


def _load_cached_payload(path: Path, required_keys: Sequence[str], *, schema_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if any(key not in payload for key in required_keys):
        return None
    if int(payload.get("schema_version") or 0) != int(schema_version):
        return None
    return payload


def _default_validation_config() -> dict[str, Any]:
    return {
        "min_trained_rows": 50,
        "min_rows_scored": 20,
        "min_selected_rows": 10,
        "min_trades": 5,
        "min_benchmark_days": 20,
    }


def _default_backtest_config(
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


def _required_history_window(
    *,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int,
    forward_horizon_days: int,
    start_offset_days: int,
) -> tuple[str, str, int]:
    first_test_date = pd.Timestamp(f"{int(test_start_year)}-01-01")
    buffer_days = int(lookback_days) + int(forward_horizon_days) + int(start_offset_days) + 10
    required_start = first_test_date - pd.offsets.BDay(buffer_days)
    required_end = pd.Timestamp(f"{int(test_end_year)}-12-31")
    expected_days = len(pd.bdate_range(start=required_start, end=required_end))
    min_history_days = max(int(expected_days * 0.75), buffer_days)
    return required_start.strftime("%Y-%m-%d"), required_end.strftime("%Y-%m-%d"), int(min_history_days)


def resolve_research_symbols(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int,
    candidate_limit: int,
    min_market_cap: float,
    test_start_year: int,
    test_end_year: int,
    lookback_days: int,
    forward_horizon_days: int,
    start_offset_days: int,
    exclude_symbol_prefixes: Sequence[str] = ("TIER",),
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
        forward_horizon_days=int(forward_horizon_days),
        start_offset_days=int(start_offset_days),
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


def _resolve_momentum_signal_spec(feature_artifact: Artifact) -> dict[str, Any]:
    feature_df = read_frame_artifact(feature_artifact, parse_dates=False, normalize_symbols=False).head(1)
    columns = list(feature_df.columns)
    long_candidates = ("px__ret_252d", "px__ret_252_d", "ret_252d", "ret_252_d")
    short_candidates = ("px__ret_21d", "px__ret_21_d", "ret_21d", "ret_21_d")
    long_col = next((column for column in long_candidates if column in columns), "")
    short_col = next((column for column in short_candidates if column in columns), "")
    if long_col and short_col:
        return {
            "signal_name": "twelve_minus_one_momentum",
            "expression": f"(1.0 + {long_col}) / (1.0 + {short_col}) - 1.0",
            "used_columns": [long_col, short_col],
        }
    for candidate in ("px__ret_252d", "px__ret_252_d", "px__ret_189d", "px__ret_189_d", "px__ret_126d", "px__ret_126_d", "ret_1"):
        if candidate in columns:
            return {
                "signal_name": "trailing_return_momentum",
                "expression": candidate,
                "used_columns": [candidate],
            }
    raise ValueError("Could not resolve a baseline momentum feature from the current feature artifact.")


def _quantile_strategy_config(
    *,
    bucket_count: int,
    score_expression: str = "",
    action_source_field: str = "",
) -> dict[str, Any]:
    out = {
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
    if score_expression:
        out["combined_score_expr"] = str(score_expression)
    if action_source_field:
        out["action_source_field"] = str(action_source_field)
    return out


def _optimized_strategy_config(
    *,
    score_expression: str = "",
    action_source_field: str = "",
    expected_return_input: str,
    risk_model_type: str,
    risk_lookback_days: int,
    risk_shrinkage: float,
    risk_factor_count: int,
    risk_aversion: float,
    turnover_penalty: float,
    turnover_cap: float | None,
    max_name_weight: float,
    net_exposure_target: float,
    sector_neutral: bool,
    alpha_quantile: float,
    alpha_scale: float,
) -> dict[str, Any]:
    out = {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "portfolio_construction": "optimized_mean_variance",
        "cross_sectional_score_field": "strategy_score",
        "holding_period_rebalances": 1,
        "ranking_lag_days": 0,
        "higher_score_is_better": True,
        "portfolio_optimization": {
            "expected_return_input": str(expected_return_input),
            "alpha_quantile": float(alpha_quantile),
            "alpha_scale": float(alpha_scale),
            "risk_aversion": float(risk_aversion),
            "turnover_penalty": float(turnover_penalty),
            "turnover_cap": turnover_cap,
            "bucket_count": 10,
            "constraints": {
                "gross_exposure_limit": 1.0,
                "net_exposure_target": float(net_exposure_target),
                "max_name_weight": float(max_name_weight),
                "sector_neutral": bool(sector_neutral),
            },
            "risk_model": {
                "model_type": str(risk_model_type),
                "lookback_days": int(risk_lookback_days),
                "min_observations": max(10, int(risk_lookback_days // 2)),
                "shrinkage": float(risk_shrinkage),
                "factor_count": int(risk_factor_count),
            },
        },
    }
    if score_expression:
        out["combined_score_expr"] = str(score_expression)
    if action_source_field:
        out["action_source_field"] = str(action_source_field)
    return out


def _resolve_artifact(artifact_id: Any, *, label: str) -> Artifact:
    artifact = Artifact.objects.filter(pk=int(artifact_id or 0)).first()
    if artifact is None:
        raise ValueError(f"{label} artifact was not found.")
    return artifact


def _single_summary_row(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(payload.get("summary_rows") or [])]
    if rows:
        return rows[0]
    raise ValueError(f"{label} did not produce a usable summary row.")


def _annotate_summary_row(
    row: Mapping[str, Any],
    *,
    signal_source: str,
    signal_label: str,
    construction: str,
    construction_label: str,
    fold_name: str,
    train_end_date: str,
    backtest_start_date: str,
    backtest_end_date: str,
) -> dict[str, Any]:
    out = dict(row)
    out["signal_source"] = str(signal_source)
    out["signal_label"] = str(signal_label)
    out["construction"] = str(construction)
    out["construction_label"] = str(construction_label)
    out["variant_name"] = f"{signal_source}__{construction}"
    out["fold_name"] = str(fold_name)
    out["train_end_date"] = str(train_end_date)
    out["backtest_start_date"] = str(backtest_start_date)
    out["backtest_end_date"] = str(backtest_end_date)
    out["trade_count"] = int(_safe_float(out.get("trade_count") or out.get("trades")))
    out["sharpe"] = _safe_float(out.get("sharpe"))
    out["total_return"] = _safe_float(out.get("total_return") or out.get("cumulative_return"))
    out["max_drawdown"] = _safe_float(out.get("max_drawdown"))
    out["avg_turnover"] = _safe_float(out.get("avg_turnover"))
    out["total_turnover"] = _safe_float(out.get("total_turnover"))
    return out


def _strategy_frame(artifact_id: Any) -> pd.DataFrame:
    artifact = _resolve_artifact(artifact_id, label="strategy")
    return read_frame_artifact(artifact, parse_dates=False, normalize_symbols=False)


def _backtest_daily_rows(artifact_id: Any) -> list[dict[str, Any]]:
    artifact = _resolve_artifact(artifact_id, label="backtest")
    return [dict(row) for row in list((artifact.content or {}).get("daily_rows") or [])]


def _aggregate_performance_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in list(summary_rows or []):
        grouped.setdefault(str(row.get("variant_name") or ""), []).append(dict(row))
    aggregate_rows: list[dict[str, Any]] = []
    for variant_name, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda item: (str(item.get("backtest_start_date") or ""), str(item.get("fold_name") or "")))
        combined_daily_rows: list[dict[str, Any]] = []
        for row in sorted_rows:
            combined_daily_rows.extend(_backtest_daily_rows(row.get("backtest_artifact_id")))
        summary = summarize_return_frame(combined_daily_rows, series_name=variant_name, series_kind="strategy")
        positive_folds = sum(1 for row in sorted_rows if _safe_float(row.get("total_return")) > 0.0)
        sharpe_values = [_safe_float(row.get("sharpe")) for row in sorted_rows]
        trade_count = sum(int(_safe_float(row.get("trade_count") or row.get("trades"))) for row in sorted_rows)
        first = sorted_rows[0]
        aggregate_rows.append(
            {
                "variant_name": variant_name,
                "signal_source": str(first.get("signal_source") or ""),
                "signal_label": str(first.get("signal_label") or ""),
                "construction": str(first.get("construction") or ""),
                "construction_label": str(first.get("construction_label") or ""),
                "fold_count": int(len(sorted_rows)),
                "sharpe": _safe_float(summary.get("sharpe")),
                "total_return": _safe_float(summary.get("total_return")),
                "final_equity": _safe_float(summary.get("final_equity"), 1.0),
                "max_drawdown": _safe_float(summary.get("max_drawdown")),
                "avg_turnover": _safe_float(summary.get("avg_turnover")),
                "total_turnover": _safe_float(summary.get("total_turnover")),
                "trade_count": int(trade_count),
                "positive_fold_count": int(positive_folds),
                "positive_fold_rate": round(float(positive_folds) / float(len(sorted_rows)) if sorted_rows else 0.0, 8),
                "mean_fold_sharpe": round(float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0, 8),
                "fold_sharpe_std": round(float(pd.Series(sharpe_values).std(ddof=0)) if sharpe_values else 0.0, 8),
            }
        )
    aggregate_rows.sort(
        key=lambda item: (
            str(item.get("signal_source") or ""),
            0 if str(item.get("construction") or "") == "optimized_mean_variance" else 1,
        )
    )
    return aggregate_rows


def _comparison_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_signal: dict[str, dict[str, dict[str, Any]]] = {}
    for row in list(aggregate_rows or []):
        signal_source = str(row.get("signal_source") or "")
        construction = str(row.get("construction") or "")
        by_signal.setdefault(signal_source, {})[construction] = dict(row)
    rows: list[dict[str, Any]] = []
    for signal_source, construction_rows in sorted(by_signal.items()):
        baseline = construction_rows.get("equal_weight_quantiles", {})
        optimized = construction_rows.get("optimized_mean_variance", {})
        rows.append(
            {
                "signal_source": signal_source,
                "signal_label": str((optimized or baseline).get("signal_label") or signal_source),
                "baseline_sharpe": _safe_float(baseline.get("sharpe")),
                "optimized_sharpe": _safe_float(optimized.get("sharpe")),
                "sharpe_delta": round(_safe_float(optimized.get("sharpe")) - _safe_float(baseline.get("sharpe")), 8),
                "baseline_total_return": _safe_float(baseline.get("total_return")),
                "optimized_total_return": _safe_float(optimized.get("total_return")),
                "total_return_delta": round(_safe_float(optimized.get("total_return")) - _safe_float(baseline.get("total_return")), 8),
                "baseline_max_drawdown": _safe_float(baseline.get("max_drawdown")),
                "optimized_max_drawdown": _safe_float(optimized.get("max_drawdown")),
                "drawdown_delta": round(_safe_float(optimized.get("max_drawdown")) - _safe_float(baseline.get("max_drawdown")), 8),
                "baseline_total_turnover": _safe_float(baseline.get("total_turnover")),
                "optimized_total_turnover": _safe_float(optimized.get("total_turnover")),
                "turnover_delta": round(_safe_float(optimized.get("total_turnover")) - _safe_float(baseline.get("total_turnover")), 8),
            }
        )
    return rows


def _collect_weight_and_diagnostic_rows(summary_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    optimized_weight_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for row in list(summary_rows or []):
        signal_source = str(row.get("signal_source") or "")
        signal_label = str(row.get("signal_label") or signal_source)
        construction = str(row.get("construction") or "")
        construction_label = str(row.get("construction_label") or construction)
        fold_name = str(row.get("fold_name") or "")
        strategy_df = _strategy_frame(row.get("strategy_artifact_id"))
        if not strategy_df.empty:
            strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            strategy_df["symbol"] = strategy_df["symbol"].astype(str).str.strip().str.upper()
            active_rows = strategy_df[pd.to_numeric(strategy_df.get("target_weight"), errors="coerce").abs() > 1e-12].copy()
            if construction == "optimized_mean_variance":
                for item in active_rows.to_dict(orient="records"):
                    optimized_weight_rows.append(
                        {
                            "signal_source": signal_source,
                            "signal_label": signal_label,
                            "construction": construction,
                            "construction_label": construction_label,
                            "fold_name": fold_name,
                            **dict(item),
                        }
                    )
                if "optimization_status" in strategy_df.columns:
                    diagnostics_df = strategy_df[strategy_df["rebalance_date"].astype(int) == 1].copy()
                    if not diagnostics_df.empty:
                        deduped = diagnostics_df.sort_values(["date", "symbol"]).drop_duplicates(subset=["date"], keep="first")
                        for item in deduped.to_dict(orient="records"):
                            covariance_rows.append(
                                {
                                    "signal_source": signal_source,
                                    "signal_label": signal_label,
                                    "construction": construction,
                                    "construction_label": construction_label,
                                    "fold_name": fold_name,
                                    "date": item.get("date"),
                                    "optimization_status": item.get("optimization_status"),
                                    "optimization_success": item.get("optimization_success"),
                                    "optimization_objective": item.get("optimization_objective"),
                                    "optimization_variance": item.get("optimization_variance"),
                                    "optimization_expected_portfolio_return": item.get("optimization_expected_portfolio_return"),
                                    "optimization_turnover": item.get("optimization_turnover"),
                                    "optimization_gross_exposure": item.get("optimization_gross_exposure"),
                                    "optimization_net_exposure": item.get("optimization_net_exposure"),
                                    "optimization_max_abs_weight": item.get("optimization_max_abs_weight"),
                                    "optimization_constraint_violation": item.get("optimization_constraint_violation"),
                                    "optimization_iterations": item.get("optimization_iterations"),
                                    "risk_model_type": item.get("risk_model_type"),
                                    "risk_model_observations": item.get("risk_model_observations"),
                                    "risk_model_condition_number": item.get("risk_model_condition_number"),
                                    "risk_model_min_eigenvalue": item.get("risk_model_min_eigenvalue"),
                                    "risk_model_max_eigenvalue": item.get("risk_model_max_eigenvalue"),
                                    "risk_model_shrinkage": item.get("risk_model_shrinkage"),
                                    "risk_model_variance_floor": item.get("risk_model_variance_floor"),
                                    "neutrality_exposure_summary": item.get("neutrality_exposure_summary"),
                                }
                            )
            grouped = strategy_df.groupby("date", sort=True)
            for date_value, group in grouped:
                weights = pd.to_numeric(group.get("target_weight"), errors="coerce").fillna(0.0)
                active = weights[weights.abs() > 1e-12]
                exposure_row = {
                    "signal_source": signal_source,
                    "signal_label": signal_label,
                    "construction": construction,
                    "construction_label": construction_label,
                    "fold_name": fold_name,
                    "date": str(date_value),
                    "positions": int(active.shape[0]),
                    "gross_exposure": round(float(weights.abs().sum()), 8),
                    "net_exposure": round(float(weights.sum()), 8),
                    "long_exposure": round(float(weights.clip(lower=0.0).sum()), 8),
                    "short_exposure": round(float(weights.clip(upper=0.0).abs().sum()), 8),
                    "max_abs_weight": round(float(weights.abs().max()) if len(weights) else 0.0, 8),
                    "avg_abs_weight": round(float(weights.abs().mean()) if len(weights) else 0.0, 8),
                }
                if "sector" in group.columns:
                    sector_net = (
                        group.assign(_weight=weights)
                        .groupby(group["sector"].astype(str).fillna("Unknown"))["_weight"]
                        .sum()
                        .abs()
                        .sort_values(ascending=False)
                    )
                    if not sector_net.empty:
                        exposure_row["largest_sector_abs_net"] = round(float(sector_net.iloc[0]), 8)
                        exposure_row["largest_sector_name"] = str(sector_net.index[0])
                exposure_rows.append(exposure_row)

        for daily_row in _backtest_daily_rows(row.get("backtest_artifact_id")):
            turnover_rows.append(
                {
                    "signal_source": signal_source,
                    "signal_label": signal_label,
                    "construction": construction,
                    "construction_label": construction_label,
                    "fold_name": fold_name,
                    **dict(daily_row),
                }
            )
    return optimized_weight_rows, covariance_rows, turnover_rows, exposure_rows


def write_portfolio_optimization_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate_rows = [dict(row) for row in list(payload.get("aggregate_rows") or [])]
    comparison_rows = [dict(row) for row in list(payload.get("comparison_rows") or [])]
    symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    optimizer_settings = dict(payload.get("optimizer_settings") or {})
    if not aggregate_rows:
        raise ValueError("Expected aggregate rows to write the portfolio optimization report.")

    lines = [
        "# Portfolio Optimization Research Report",
        "",
        "## 1. Experiment setup",
        "",
        "- Objective: convert existing ranking and prediction signals into constrained, risk-aware portfolio weights using reusable portfolio-construction infrastructure.",
        "- Research framing: inspired by the active-portfolio-management idea that signal quality and portfolio construction should be separated into expected returns, risk, constraints, and an optimizer.",
        f"- Universe: {len(symbols)} symbols.",
        "- Signals tested: " + ", ".join(str(row.get("signal_label") or row.get("signal_source") or "") for row in comparison_rows) if comparison_rows else "- Signals tested: n/a",
        "- Symbols: " + ", ".join(symbols),
        "",
        "## 2. Optimizer specification",
        "",
        "- Objective: maximize expected return minus risk aversion times portfolio variance minus turnover penalty times portfolio change.",
        f"- Expected-return input: `{optimizer_settings.get('expected_return_input') or ''}`.",
        f"- Risk model: `{optimizer_settings.get('risk_model_type') or ''}` with {int(optimizer_settings.get('risk_lookback_days') or 0)}-day lookback.",
        f"- Constraints: gross <= 1.0, net target {float(optimizer_settings.get('net_exposure_target') or 0.0):.2f}, max single-name weight {float(optimizer_settings.get('max_name_weight') or 0.0):.2f}.",
        f"- Turnover controls: penalty {float(optimizer_settings.get('turnover_penalty') or 0.0):.3f}, cap {optimizer_settings.get('turnover_cap') if optimizer_settings.get('turnover_cap') not in (None, '') else 'n/a'}.",
        "",
        "## 3. Performance comparison",
        "",
        "| Signal | Construction | Sharpe | Total Return | Max DD | Turnover | Trades | Positive Fold Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row.get('signal_label', '')} | "
            + f"{row.get('construction_label', '')} | "
            + f"{_safe_float(row.get('sharpe')):.3f} | "
            + f"{_pct(row.get('total_return'))} | "
            + f"{_pct(row.get('max_drawdown'))} | "
            + f"{_safe_float(row.get('total_turnover')):.2f} | "
            + f"{int(_safe_float(row.get('trade_count')))} | "
            + f"{_safe_float(row.get('positive_fold_rate')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Signal-by-signal deltas",
            "",
            "| Signal | Sharpe Delta | Return Delta | Drawdown Delta | Turnover Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            + f"{row.get('signal_label', '')} | "
            + f"{_safe_float(row.get('sharpe_delta')):.3f} | "
            + f"{_pct(row.get('total_return_delta'))} | "
            + f"{_pct(row.get('drawdown_delta'))} | "
            + f"{_safe_float(row.get('turnover_delta')):.2f} |"
        )
    observations: list[str] = []
    for row in comparison_rows:
        label = str(row.get("signal_label") or row.get("signal_source") or "")
        sharpe_delta = _safe_float(row.get("sharpe_delta"))
        return_delta = _safe_float(row.get("total_return_delta"))
        turnover_delta = _safe_float(row.get("turnover_delta"))
        if sharpe_delta > 0.0 and return_delta > 0.0:
            observations.append(f"- {label}: optimized construction improved both Sharpe and total return.")
        elif sharpe_delta > 0.0 and return_delta <= 0.0:
            observations.append(f"- {label}: optimization improved Sharpe but not raw total return, so the gain looks risk-adjusted rather than purely directional.")
        elif sharpe_delta <= 0.0 and return_delta > 0.0:
            observations.append(f"- {label}: optimization raised total return but not Sharpe, so the added complexity may mainly be levering the same signal differently.")
        else:
            observations.append(f"- {label}: optimization did not improve the headline return metrics in this pilot.")
        if turnover_delta < 0.0:
            observations.append(f"- {label}: turnover fell versus equal-weight quantiles, which suggests the optimizer is smoothing rebalances rather than just re-ranking names.")
    if not observations:
        observations.append("- No comparison rows were available.")
    lines.extend(
        [
            "",
            "## 5. Key observations",
            "",
            *observations,
            "",
            "## 6. Limitations and next steps",
            "",
            "- Phase 1 uses a simple covariance model and linear constraints; richer alpha calibration, factor neutrality, and transaction-cost modeling are natural follow-ons.",
            "- The current optimizer works at the cross-sectional rebalance level; future work can add portfolio-level forecasts for borrow, liquidity, and regime-dependent risk aversion.",
            "- If optimization does not improve a signal, that is still informative: it suggests the current equal-weight quantile construction is already extracting most of the available cross-sectional edge.",
            "",
            "## 7. Output artifacts",
            "",
            f"- Portfolio metrics JSON: `{payload.get('summary_json_path') or ''}`",
            f"- Aggregate metrics CSV: `{payload.get('summary_csv_path') or ''}`",
            f"- Optimized weights CSV: `{payload.get('optimized_weights_csv_path') or ''}`",
            f"- Covariance diagnostics CSV: `{payload.get('covariance_diagnostics_csv_path') or ''}`",
            f"- Turnover diagnostics CSV: `{payload.get('turnover_diagnostics_csv_path') or ''}`",
            f"- Exposure diagnostics CSV: `{payload.get('exposure_diagnostics_csv_path') or ''}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_portfolio_optimization_research(
    *,
    requested_symbols: Sequence[str] | None = None,
    symbol_limit: int = 20,
    candidate_limit: int = 60,
    min_market_cap: float = 25_000_000_000.0,
    test_start_year: int = 2022,
    test_end_year: int = 2025,
    bucket_count: int = 10,
    fee_bps: float = 2.0,
    slippage_bps: float = 8.0,
    short_borrow_bps_annual: float = 25.0,
    execution_delay_days: int = 1,
    output_basename: str = "portfolio_optimization_research",
    resume_existing: bool = False,
    include_characteristics_factor: bool = True,
    prediction_artifact_ids: Sequence[int] = (),
    expected_return_input: str = "ranking_score",
    risk_model_type: str = "sample_covariance",
    risk_lookback_days: int = 63,
    risk_shrinkage: float = 0.15,
    risk_factor_count: int = 3,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.0,
    turnover_cap: float | None = None,
    max_name_weight: float = 0.10,
    net_exposure_target: float = 0.0,
    sector_neutral: bool = False,
    alpha_quantile: float = 0.2,
    alpha_scale: float = 0.05,
    n_factors: int = 3,
    exposure_lookback_days: int = 63,
    minimum_exposure_observations: int = 30,
    random_state: int = 1337,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}__portfolio_metrics.json"
    csv_path = output_dir / f"{output_basename}.csv"
    fold_csv_path = output_dir / f"{output_basename}__fold_rows.csv"
    optimized_weights_csv_path = output_dir / f"{output_basename}__optimized_weights.csv"
    covariance_csv_path = output_dir / f"{output_basename}__covariance_diagnostics.csv"
    turnover_csv_path = output_dir / f"{output_basename}__turnover_diagnostics.csv"
    exposure_csv_path = output_dir / f"{output_basename}__exposure_diagnostics.csv"
    comparison_csv_path = output_dir / f"{output_basename}__comparison.csv"
    if resume_existing:
        cached = _load_cached_payload(
            json_path,
            required_keys=("summary_rows", "aggregate_rows", "comparison_rows", "symbols"),
            schema_version=PORTFOLIO_OPTIMIZATION_RESEARCH_SCHEMA_VERSION,
        )
        if cached is not None:
            cached["summary_json_path"] = str(json_path)
            cached["summary_csv_path"] = str(csv_path)
            return cached

    folds = build_yearly_folds(int(test_start_year), int(test_end_year))
    symbols, coverage_rows, missing_symbols = resolve_research_symbols(
        requested_symbols=requested_symbols,
        symbol_limit=int(symbol_limit),
        candidate_limit=int(candidate_limit),
        min_market_cap=float(min_market_cap),
        test_start_year=int(test_start_year),
        test_end_year=int(test_end_year),
        lookback_days=int(risk_lookback_days),
        forward_horizon_days=21,
        start_offset_days=1,
    )
    if not symbols:
        raise ValueError("No symbols were available after applying the history screen.")

    from .cohort_runner import _resolve_or_build_feature_artifact, _resolve_or_build_universe_artifact
    from .direct_strategy_runner import run_direct_feature_strategy_backtests
    from .prediction_strategy_runner import run_prediction_artifact_strategy_backtest

    universe_artifact = _resolve_or_build_universe_artifact(symbols=symbols, output_basename=output_basename)
    feature_artifact = _resolve_or_build_feature_artifact(
        universe_artifact=universe_artifact,
        symbols=symbols,
        feature_config={},
        output_basename=output_basename,
    )
    momentum_signal = _resolve_momentum_signal_spec(feature_artifact)
    validation_config = _default_validation_config()
    backtest_config = _default_backtest_config(
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        short_borrow_bps_annual=float(short_borrow_bps_annual),
        execution_delay_days=int(execution_delay_days),
    )
    optimizer_settings = {
        "expected_return_input": str(expected_return_input),
        "risk_model_type": str(risk_model_type),
        "risk_lookback_days": int(risk_lookback_days),
        "risk_shrinkage": float(risk_shrinkage),
        "risk_factor_count": int(risk_factor_count),
        "risk_aversion": float(risk_aversion),
        "turnover_penalty": float(turnover_penalty),
        "turnover_cap": turnover_cap,
        "max_name_weight": float(max_name_weight),
        "net_exposure_target": float(net_exposure_target),
        "sector_neutral": bool(sector_neutral),
        "alpha_quantile": float(alpha_quantile),
        "alpha_scale": float(alpha_scale),
    }

    baseline_quantile_config = _quantile_strategy_config(
        bucket_count=int(bucket_count),
        score_expression=str(momentum_signal.get("expression") or ""),
    )
    optimized_momentum_config = _optimized_strategy_config(
        score_expression=str(momentum_signal.get("expression") or ""),
        expected_return_input=str(expected_return_input),
        risk_model_type=str(risk_model_type),
        risk_lookback_days=int(risk_lookback_days),
        risk_shrinkage=float(risk_shrinkage),
        risk_factor_count=int(risk_factor_count),
        risk_aversion=float(risk_aversion),
        turnover_penalty=float(turnover_penalty),
        turnover_cap=turnover_cap,
        max_name_weight=float(max_name_weight),
        net_exposure_target=float(net_exposure_target),
        sector_neutral=bool(sector_neutral),
        alpha_quantile=float(alpha_quantile),
        alpha_scale=float(alpha_scale),
    )
    baseline_prediction_config = _quantile_strategy_config(
        bucket_count=int(bucket_count),
        action_source_field="ranking",
    )
    optimized_prediction_config = _optimized_strategy_config(
        action_source_field="ranking",
        expected_return_input=str(expected_return_input),
        risk_model_type=str(risk_model_type),
        risk_lookback_days=int(risk_lookback_days),
        risk_shrinkage=float(risk_shrinkage),
        risk_factor_count=int(risk_factor_count),
        risk_aversion=float(risk_aversion),
        turnover_penalty=float(turnover_penalty),
        turnover_cap=turnover_cap,
        max_name_weight=float(max_name_weight),
        net_exposure_target=float(net_exposure_target),
        sector_neutral=bool(sector_neutral),
        alpha_quantile=float(alpha_quantile),
        alpha_scale=float(alpha_scale),
    )
    characteristics_quantile_strategy = upsert_strategy_definition(
        slug="portfolio-opt-characteristics-quantiles",
        name="Portfolio Optimization Characteristics Quantiles",
        strategy_type="notebook_topk_v1",
        description="Equal-weight quantile baseline for portfolio optimization research.",
        config=dict(baseline_prediction_config),
    )
    characteristics_label_artifact = None
    factor_spec = None
    if include_characteristics_factor:
        from .cross_sectional_rank_labels import CrossSectionalRankLabelSpec, resolve_or_build_cross_sectional_rank_label_artifact
        from .characteristics_factor_model import LatentFactorSpec, resolve_feature_scope_variants

        factor_spec = LatentFactorSpec(
            n_factors=int(n_factors),
            exposure_lookback_days=int(exposure_lookback_days),
            minimum_exposure_observations=int(minimum_exposure_observations),
        )
        label_spec = CrossSectionalRankLabelSpec(
            horizon_days=21,
            rebalance_freq="M",
            start_offset_days=1,
            minimum_cross_section=max(2, int(min(len(symbols), max(10, bucket_count * 2)))),
            target_col="future_rank_pct",
            forward_return_col="trade_return",
        )
        characteristics_label_artifact = resolve_or_build_cross_sectional_rank_label_artifact(
            feature_artifact=feature_artifact,
            spec=label_spec,
            output_basename=f"{output_basename}__rank_labels",
        )
        characteristic_variants = resolve_feature_scope_variants(
            feature_artifact,
            include_prices_only=False,
            include_context_only=False,
        )
        characteristic_variant = next(
            (
                {
                    "variant_name": f"characteristics_factor_rf_{variant['variant_name']}",
                    "variant_label": f"Characteristics Factor {variant['variant_label']}",
                    "feature_scope": str(variant["feature_scope"]),
                    "feature_families": list(variant.get("feature_families") or []),
                }
                for variant in characteristic_variants
                if str(variant.get("variant_name") or "") == "all_features"
            ),
            {
                "variant_name": "characteristics_factor_rf_all_features",
                "variant_label": "Characteristics Factor All Features",
                "feature_scope": "all_features",
                "feature_families": [],
            },
        )
    else:
        characteristic_variant = {
            "variant_name": "characteristics_factor_rf_all_features",
            "variant_label": "Characteristics Factor All Features",
            "feature_scope": "all_features",
            "feature_families": [],
        }

    summary_rows: list[dict[str, Any]] = []
    prediction_source_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_name = str(fold.get("name") or "")
        train_end_date = str(fold.get("train_end_date") or "")
        backtest_start_date = str(fold.get("backtest_start_date") or "")
        backtest_end_date = str(fold.get("backtest_end_date") or "")

        momentum_baseline = run_direct_feature_strategy_backtests(
            symbols=symbols,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            universe_artifact=universe_artifact,
            feature_artifact=feature_artifact,
            feature_config={},
            strategy_definition_slug="portfolio-opt-momentum-baseline",
            strategy_definition_name="Portfolio Optimization Baseline Momentum",
            strategy_config=baseline_quantile_config,
            validation_config=validation_config,
            backtest_config=backtest_config,
            output_basename=f"{output_basename}__baseline_momentum__equal_weight_quantiles__{fold_name}",
            resume_existing=resume_existing,
        )
        summary_rows.append(
            _annotate_summary_row(
                _single_summary_row(momentum_baseline, label=f"{fold_name} baseline momentum"),
                signal_source="baseline_momentum",
                signal_label="Baseline Momentum",
                construction="equal_weight_quantiles",
                construction_label="Equal-Weight Quantiles",
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
        )
        momentum_optimized = run_direct_feature_strategy_backtests(
            symbols=symbols,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            universe_artifact=universe_artifact,
            feature_artifact=feature_artifact,
            feature_config={},
            strategy_definition_slug="portfolio-opt-momentum-optimized",
            strategy_definition_name="Portfolio Optimization Optimized Momentum",
            strategy_config=optimized_momentum_config,
            validation_config=validation_config,
            backtest_config=backtest_config,
            output_basename=f"{output_basename}__baseline_momentum__optimized_mean_variance__{fold_name}",
            resume_existing=resume_existing,
        )
        summary_rows.append(
            _annotate_summary_row(
                _single_summary_row(momentum_optimized, label=f"{fold_name} optimized momentum"),
                signal_source="baseline_momentum",
                signal_label="Baseline Momentum",
                construction="optimized_mean_variance",
                construction_label="Optimized Mean-Variance",
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
        )

        if include_characteristics_factor and characteristics_label_artifact is not None:
            from .characteristics_factor_model import build_characteristic_factor_targets, run_characteristic_factor_variant

            _factor_return_df, factor_target_df, factor_cols, _factor_meta = build_characteristic_factor_targets(
                feature_artifact,
                characteristics_label_artifact,
                train_end_date=train_end_date,
                score_end_date=backtest_end_date,
                spec=factor_spec,
            )
            characteristic_result = run_characteristic_factor_variant(
                variant_name=str(characteristic_variant.get("variant_name") or ""),
                variant_label=str(characteristic_variant.get("variant_label") or ""),
                feature_scope=str(characteristic_variant.get("feature_scope") or ""),
                feature_families=list(characteristic_variant.get("feature_families") or []),
                feature_artifact=feature_artifact,
                label_artifact=characteristics_label_artifact,
                strategy_definition=characteristics_quantile_strategy,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                factor_target_df=factor_target_df,
                factor_cols=factor_cols,
                backtest_config=backtest_config,
                validation_config=validation_config,
                output_basename=f"{output_basename}__characteristics_factor__equal_weight_quantiles__{fold_name}",
                random_state=int(random_state),
            )
            baseline_row = _annotate_summary_row(
                dict(characteristic_result["summary_row"]),
                signal_source="characteristics_factor",
                signal_label="Characteristics Factor",
                construction="equal_weight_quantiles",
                construction_label="Equal-Weight Quantiles",
                fold_name=fold_name,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
            )
            summary_rows.append(baseline_row)
            prediction_artifact = characteristic_result["prediction_artifact"]
            prediction_source_rows.extend(
                [
                    {
                        "signal_source": "characteristics_factor",
                        "fold_name": fold_name,
                        **dict(item),
                    }
                    for item in list(characteristic_result.get("prediction_rows") or [])
                ]
            )
            optimized_characteristics = run_prediction_artifact_strategy_backtest(
                feature_artifact=feature_artifact,
                prediction_artifact=prediction_artifact,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                strategy_definition_slug="portfolio-opt-characteristics-optimized",
                strategy_definition_name="Portfolio Optimization Optimized Characteristics",
                strategy_config=optimized_prediction_config,
                label_artifact=characteristics_label_artifact,
                validation_config=validation_config,
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__characteristics_factor__optimized_mean_variance__{fold_name}",
            )
            summary_rows.append(
                _annotate_summary_row(
                    _single_summary_row(optimized_characteristics, label=f"{fold_name} optimized characteristics"),
                    signal_source="characteristics_factor",
                    signal_label="Characteristics Factor",
                    construction="optimized_mean_variance",
                    construction_label="Optimized Mean-Variance",
                    fold_name=fold_name,
                    train_end_date=train_end_date,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                )
            )

        for artifact_id in list(prediction_artifact_ids or []):
            prediction_artifact = _resolve_artifact(artifact_id, label="prediction")
            signal_source = f"prediction_artifact_{int(prediction_artifact.id)}"
            signal_label = str((prediction_artifact.metadata or {}).get("variant_label") or (prediction_artifact.metadata or {}).get("variant_name") or signal_source)
            baseline_prediction = run_prediction_artifact_strategy_backtest(
                feature_artifact=feature_artifact,
                prediction_artifact=prediction_artifact,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                strategy_definition_slug=f"{signal_source}-baseline",
                strategy_definition_name=f"{signal_label} Baseline",
                strategy_config=baseline_prediction_config,
                validation_config=validation_config,
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__{signal_source}__equal_weight_quantiles__{fold_name}",
            )
            summary_rows.append(
                _annotate_summary_row(
                    _single_summary_row(baseline_prediction, label=f"{fold_name} {signal_source} baseline"),
                    signal_source=signal_source,
                    signal_label=signal_label,
                    construction="equal_weight_quantiles",
                    construction_label="Equal-Weight Quantiles",
                    fold_name=fold_name,
                    train_end_date=train_end_date,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                )
            )
            optimized_prediction = run_prediction_artifact_strategy_backtest(
                feature_artifact=feature_artifact,
                prediction_artifact=prediction_artifact,
                train_end_date=train_end_date,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                strategy_definition_slug=f"{signal_source}-optimized",
                strategy_definition_name=f"{signal_label} Optimized",
                strategy_config=optimized_prediction_config,
                validation_config=validation_config,
                backtest_config=backtest_config,
                output_basename=f"{output_basename}__{signal_source}__optimized_mean_variance__{fold_name}",
            )
            summary_rows.append(
                _annotate_summary_row(
                    _single_summary_row(optimized_prediction, label=f"{fold_name} {signal_source} optimized"),
                    signal_source=signal_source,
                    signal_label=signal_label,
                    construction="optimized_mean_variance",
                    construction_label="Optimized Mean-Variance",
                    fold_name=fold_name,
                    train_end_date=train_end_date,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                )
            )

    aggregate_rows = _aggregate_performance_rows(summary_rows)
    comparison_rows = _comparison_rows(aggregate_rows)
    optimized_weight_rows, covariance_rows, turnover_rows, exposure_rows = _collect_weight_and_diagnostic_rows(summary_rows)
    payload = {
        "schema_version": PORTFOLIO_OPTIMIZATION_RESEARCH_SCHEMA_VERSION,
        "mode": "portfolio_optimization_research",
        "symbols": symbols,
        "missing_requested_symbols": missing_symbols,
        "coverage_rows": coverage_rows,
        "folds": [dict(fold) for fold in folds],
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
            "labels": int(characteristics_label_artifact.id) if characteristics_label_artifact is not None else 0,
        },
        "optimizer_settings": optimizer_settings,
        "momentum_signal": momentum_signal,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "comparison_rows": comparison_rows,
        "prediction_source_rows": prediction_source_rows,
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "fold_summary_csv_path": str(fold_csv_path),
        "comparison_csv_path": str(comparison_csv_path),
        "optimized_weights_csv_path": str(optimized_weights_csv_path),
        "covariance_diagnostics_csv_path": str(covariance_csv_path),
        "turnover_diagnostics_csv_path": str(turnover_csv_path),
        "exposure_diagnostics_csv_path": str(exposure_csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, aggregate_rows)
    _write_rows_csv(fold_csv_path, summary_rows)
    _write_rows_csv(comparison_csv_path, comparison_rows)
    _write_rows_csv(optimized_weights_csv_path, optimized_weight_rows)
    _write_rows_csv(covariance_csv_path, covariance_rows)
    _write_rows_csv(turnover_csv_path, turnover_rows)
    _write_rows_csv(exposure_csv_path, exposure_rows)
    return payload


__all__ = [
    "PORTFOLIO_OPTIMIZATION_RESEARCH_SCHEMA_VERSION",
    "run_portfolio_optimization_research",
    "write_portfolio_optimization_report",
]
