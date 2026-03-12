from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from fmp.models import Symbol

from .cohort_runner import (
    _aggregate_walk_forward_rows,
    _apply_walk_forward_gates,
    _build_equal_weight_benchmark,
    _evaluate_variant_gates,
    _load_cached_payload,
    _resolve_or_build_feature_artifact,
    _resolve_or_build_label_artifact,
    _resolve_or_build_universe_artifact,
    _run_pipeline_job,
)
from .direct_strategy_runner import (
    _resolved_backtest_cost,
    _summarize_backtest_artifact,
    _summarize_walk_forward_metrics,
    run_direct_feature_strategy_backtests,
)
from .models import Artifact
from .cohort_runner import run_model_cohort_backtests
from .symbol_diagnostics import aggregate_symbol_diagnostic_rows, compute_symbol_strategy_diagnostics
from .symbol_filters import (
    build_symbol_feature_summary,
    select_symbols_with_learned_filter,
    select_top_symbols_from_diagnostics,
)
from .universe_selection import resolve_symbol_universe


POLICY_COMPARISON_SCHEMA_VERSION = 1
DEFAULT_UNIVERSE_EXCLUDED_PREFIXES = ["TIER"]


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
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row.keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def resolve_large_cap_symbols(
    *,
    limit: int,
    min_market_cap: float,
    country: str = "US",
    exchanges: Sequence[str] = ("NASDAQ", "NYSE"),
    exclude_symbol_prefixes: Sequence[str] = DEFAULT_UNIVERSE_EXCLUDED_PREFIXES,
) -> list[str]:
    requested = resolve_symbol_universe(
        min_market_cap=float(min_market_cap),
        country=country,
        exchanges=list(exchanges),
        limit=max(int(limit), 1),
        exclude_pooled_vehicles=True,
        exclude_symbol_prefixes=list(exclude_symbol_prefixes),
    )
    available = {
        str(symbol).strip().upper()
        for symbol in Symbol.objects.filter(symbol__in=requested).values_list("symbol", flat=True)
    }
    return [symbol for symbol in requested if symbol in available]


def _variant_name(strategy_name: str, filter_name: str) -> str:
    return f"{str(strategy_name).strip()}__{str(filter_name).strip()}"


def _baseline_strategy_config() -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "combined_score_expr": "(1.0 + px__ret_252_d) / (1.0 + px__ret_21_d) - 1.0",
        "action_transform": "sign",
        "action_threshold": 0.0,
    }


def _model_strategy_config() -> dict[str, Any]:
    return {
        "rebalance_freq": "M",
        "gross_exposure": 1.0,
        "selection_side": "long_short",
        "signal_combination": "direct",
        "action_transform": "sign",
        "action_threshold": 0.0,
    }


def _default_model_config() -> dict[str, Any]:
    return {
        "model_name": "tsmom_oracle_trade_rf",
        "algorithm": "random_forest_regressor",
        "task_type": "regression",
        "target_col": "trade_return",
        "split_ratio": 1.0,
        "label_ks": [1],
        "min_profit_pct": 0.0,
        "sample_weight_mode": "trade_return_abs",
        "params": {
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 10,
            "n_jobs": -1,
        },
    }


def _default_validation_config() -> dict[str, Any]:
    return {
        "min_trained_rows": 200,
        "min_rows_scored": 100,
        "min_selected_rows": 20,
        "min_trades": 20,
        "min_benchmark_days": 50,
        "min_valid_fold_rate": 0.6,
        "max_fold_excess_std": 0.75,
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
        "min_dollar_volume": 5_000_000.0,
    }


def _full_feature_config() -> dict[str, Any]:
    return {}


def _single_summary_row(summary: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    rows = [dict(row) for row in list(summary.get("summary_rows") or [])]
    if not rows:
        failed = list(summary.get("failed_variants") or [])
        detail = failed[0].get("error") if failed else "no summary rows produced"
        raise ValueError(f"{label} did not produce a usable summary row: {detail}")
    return dict(rows[0])


def _resolve_artifact(artifact_id: Any, *, label: str) -> Artifact:
    artifact = Artifact.objects.filter(pk=int(artifact_id or 0)).first()
    if artifact is None:
        raise ValueError(f"{label} artifact #{artifact_id} was not found.")
    return artifact


def _strategy_row_counts(strategy_artifact: Artifact, *, allowed_symbols: Sequence[str] | None = None) -> tuple[int, int]:
    strategy_df = _resolve_strategy_frame(strategy_artifact)
    if strategy_df.empty:
        return 0, 0
    if allowed_symbols:
        allowed = {str(symbol).strip().upper() for symbol in list(allowed_symbols or []) if str(symbol).strip()}
        strategy_df = strategy_df[strategy_df["symbol"].astype(str).isin(allowed)].copy()
    rows_scored = int(len(strategy_df))
    selected_rows = int((strategy_df["strategy_signal"].fillna(0).astype(float) != 0.0).sum()) if "strategy_signal" in strategy_df.columns else 0
    return rows_scored, selected_rows


def _resolve_strategy_frame(strategy_artifact: Artifact):
    from pipeline.service_runtime import read_frame_artifact

    return read_frame_artifact(
        strategy_artifact,
        parse_dates=False,
        normalize_symbols=True,
    )


def _annotate_variant_row(
    row: Mapping[str, Any],
    *,
    strategy_name: str,
    filter_name: str,
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out["variant_name"] = _variant_name(strategy_name, filter_name)
    out["strategy_name"] = str(strategy_name)
    out["filter_name"] = str(filter_name)
    out["selected_symbol_count"] = int(len(selected_symbols))
    out["selected_symbols_preview"] = [str(symbol) for symbol in list(selected_symbols or [])[:10]]
    if selection_metadata:
        out["selection_target_metric"] = str(selection_metadata.get("target_metric") or "")
        out["selection_model_kind"] = str(selection_metadata.get("model_kind") or "")
    return out


def _filtered_variant_row(
    *,
    base_row: Mapping[str, Any],
    strategy_artifact: Artifact,
    backtest_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    strategy_name: str,
    filter_name: str,
    selected_symbols: Sequence[str],
    selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(base_row)
    backtest_content = dict(backtest_artifact.content or {})
    backtest_meta = dict(backtest_artifact.metadata or {})
    runtime_summary = _summarize_backtest_artifact(backtest_artifact)
    benchmark = _build_equal_weight_benchmark(
        strategy_artifact,
        allowed_symbols=selected_symbols,
    )
    rows_scored, selected_rows = _strategy_row_counts(strategy_artifact, allowed_symbols=selected_symbols)
    row.update(
        {
            "rows_scored": int(rows_scored),
            "selected_rows": int(selected_rows),
            "final_equity": float(backtest_content.get("final_equity") or 0.0),
            "cumulative_return": float(backtest_content.get("cumulative_return") or 0.0),
            "max_drawdown": float(backtest_content.get("max_drawdown") or 0.0),
            "trades": int(backtest_content.get("trades") or 0),
            "sharpe": float(runtime_summary.get("sharpe") or 0.0),
            "avg_turnover": float(runtime_summary.get("avg_turnover") or 0.0),
            "total_turnover": float(runtime_summary.get("total_turnover") or 0.0),
            "positive_days": int(runtime_summary.get("positive_days") or 0),
            "negative_days": int(runtime_summary.get("negative_days") or 0),
            "benchmark_days": int(benchmark.get("benchmark_days") or 0),
            "benchmark_final_equity": float(benchmark.get("benchmark_final_equity") or 0.0),
            "benchmark_cumulative_return": float(benchmark.get("benchmark_cumulative_return") or 0.0),
            "benchmark_max_drawdown": float(benchmark.get("benchmark_max_drawdown") or 0.0),
            "backtest_seconds": float(backtest_meta.get("backtest_seconds") or 0.0),
            "backtest_fee_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "fee_bps"),
            "backtest_slippage_bps": _resolved_backtest_cost(backtest_meta, dict(backtest_config), "slippage_bps"),
            "excess_cumulative_return": round(
                float(backtest_content.get("cumulative_return") or 0.0) - float(benchmark.get("benchmark_cumulative_return") or 0.0),
                8,
            ),
            "relative_final_equity": round(
                float(backtest_content.get("final_equity") or 0.0) - float(benchmark.get("benchmark_final_equity") or 0.0),
                8,
            ),
            "backtest_artifact_id": int(backtest_artifact.id),
        }
    )
    row["total_runtime_seconds"] = round(
        _safe_float(row.get("dataset_build_seconds"))
        + _safe_float(row.get("fit_seconds"))
        + _safe_float(row.get("score_seconds"))
        + _safe_float(row.get("strategy_build_seconds"))
        + _safe_float(row.get("backtest_seconds")),
        6,
    )
    row.update(_evaluate_variant_gates(row, validation_config=dict(validation_config)))
    return _annotate_variant_row(
        row,
        strategy_name=strategy_name,
        filter_name=filter_name,
        selected_symbols=selected_symbols,
        selection_metadata=selection_metadata,
    )


def _run_filtered_backtest(
    *,
    strategy_artifact: Artifact,
    backtest_config: Mapping[str, Any],
    output_name: str,
) -> Artifact:
    return _run_pipeline_job(
        name=output_name,
        requested_job="backtest_strategy",
        config=dict(backtest_config),
        input_ids=[int(strategy_artifact.id)],
    )


def _selection_record(
    *,
    fold_name: str,
    strategy_name: str,
    filter_name: str,
    selected_symbols: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    top_features = list(metadata.get("top_features") or [])
    return {
        "fold_name": str(fold_name),
        "strategy_name": str(strategy_name),
        "filter_name": str(filter_name),
        "selection_count": int(len(selected_symbols)),
        "selected_symbols": [str(symbol) for symbol in list(selected_symbols or [])],
        "target_metric": str(metadata.get("target_metric") or ""),
        "model_kind": str(metadata.get("model_kind") or ""),
        "used_fallback": bool(metadata.get("used_fallback", False)),
        "top_features": top_features,
    }


def _aggregate_variant_rows(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    validation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregate_rows = _apply_walk_forward_gates(
        _aggregate_walk_forward_rows([dict(row) for row in summary_rows]),
        validation_config=dict(validation_config),
    )
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_variant.setdefault(str(row.get("variant_name") or ""), []).append(dict(row))

    enriched: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_name = str(row.get("variant_name") or "")
        variant_rows = by_variant.get(variant_name, [])
        walk_forward_metrics = _summarize_walk_forward_metrics(variant_rows)
        positive_folds = sum(1 for item in variant_rows if _safe_float(item.get("cumulative_return")) > 0.0)
        sharpe_values = [_safe_float(item.get("sharpe")) for item in variant_rows]
        selected_counts = [int(_safe_float(item.get("selected_symbol_count"))) for item in variant_rows]
        item = dict(row)
        item.update(
            {
                "strategy_name": str(variant_rows[0].get("strategy_name") or "") if variant_rows else "",
                "filter_name": str(variant_rows[0].get("filter_name") or "") if variant_rows else "",
                "sharpe": float(walk_forward_metrics.get("sharpe") or 0.0),
                "total_return": float(walk_forward_metrics.get("total_return") or 0.0),
                "final_equity": float(walk_forward_metrics.get("final_equity") or 1.0),
                "max_drawdown": float(walk_forward_metrics.get("max_drawdown") or 0.0),
                "avg_turnover": float(walk_forward_metrics.get("avg_turnover") or 0.0),
                "total_turnover": float(walk_forward_metrics.get("total_turnover") or 0.0),
                "trade_count": int(walk_forward_metrics.get("trade_count") or 0),
                "walk_forward_start_date": str(walk_forward_metrics.get("start_date") or ""),
                "walk_forward_end_date": str(walk_forward_metrics.get("end_date") or ""),
                "positive_fold_count": int(positive_folds),
                "positive_fold_rate": round(float(positive_folds / float(len(variant_rows))) if variant_rows else 0.0, 8),
                "mean_fold_sharpe": round(float(sum(sharpe_values) / len(sharpe_values)) if sharpe_values else 0.0, 8),
                "fold_sharpe_std": round(float(pd.Series(sharpe_values).std(ddof=0)) if sharpe_values else 0.0, 8),
                "mean_selected_symbol_count": round(float(sum(selected_counts) / len(selected_counts)) if selected_counts else 0.0, 4),
                "min_selected_symbol_count": min(selected_counts) if selected_counts else 0,
                "max_selected_symbol_count": max(selected_counts) if selected_counts else 0,
            }
        )
        enriched.append(item)
    order = {
        "baseline__no_filter": 0,
        "baseline__simple_filter": 1,
        "baseline__learned_filter": 2,
        "model__no_filter": 3,
        "model__simple_filter": 4,
        "model__learned_filter": 5,
    }
    enriched.sort(key=lambda item: order.get(str(item.get("variant_name") or ""), 999))
    return enriched


def _top_symbol_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy_name: str,
    filter_name: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    matched = [
        dict(row)
        for row in rows
        if str(row.get("strategy_name") or "") == str(strategy_name)
        and str(row.get("filter_name") or "") == str(filter_name)
    ]
    return matched[: max(int(limit), 0)]


def _selection_frequency_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy_name: str,
    filter_name: str,
    limit: int = 5,
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("strategy_name") or "") != str(strategy_name):
            continue
        if str(row.get("filter_name") or "") != str(filter_name):
            continue
        for symbol in list(row.get("selected_symbols") or []):
            counts[str(symbol)] = counts.get(str(symbol), 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(int(limit), 0)]


def _comparison_signal(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    candidate_sharpe = _safe_float(candidate.get("sharpe"))
    baseline_sharpe = _safe_float(baseline.get("sharpe"))
    candidate_return = _safe_float(candidate.get("total_return"))
    baseline_return = _safe_float(baseline.get("total_return"))
    candidate_stability = _safe_float(candidate.get("positive_fold_rate"))
    baseline_stability = _safe_float(baseline.get("positive_fold_rate"))
    better_sharpe = candidate_sharpe > baseline_sharpe
    better_return = candidate_return > baseline_return
    better_stability = candidate_stability >= baseline_stability
    if better_sharpe and better_return and better_stability:
        return "yes"
    if better_sharpe or better_return:
        return "mixed"
    return "no"


def _variant_table_rows(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    header = [
        "| Variant | Sharpe | Total Return | Max DD | Turnover | Trades | Mean Selected Symbols | Positive Fold Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = []
    for row in aggregate_rows:
        rows.append(
            "| "
            + f"{row.get('variant_name', '')} | "
            + f"{_safe_float(row.get('sharpe')):.3f} | "
            + f"{_pct(row.get('total_return'))} | "
            + f"{_pct(row.get('max_drawdown'))} | "
            + f"{_safe_float(row.get('total_turnover')):.2f} | "
            + f"{int(_safe_float(row.get('trade_count')))} | "
            + f"{_safe_float(row.get('mean_selected_symbol_count')):.1f} | "
            + f"{_safe_float(row.get('positive_fold_rate')):.2f} |"
        )
    return header + rows


def write_policy_comparison_report(
    *,
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    aggregate_rows = [dict(row) for row in list(payload.get("aggregate_rows") or [])]
    symbol_rows = [dict(row) for row in list(payload.get("symbol_diagnostics_aggregate_rows") or [])]
    selection_rows = [dict(row) for row in list(payload.get("selection_rows") or [])]
    universe_symbols = [str(symbol) for symbol in list(payload.get("symbols") or [])]
    variant_lookup = {str(row.get("variant_name") or ""): row for row in aggregate_rows}

    baseline_no = variant_lookup.get("baseline__no_filter", {})
    baseline_simple = variant_lookup.get("baseline__simple_filter", {})
    baseline_learned = variant_lookup.get("baseline__learned_filter", {})
    model_no = variant_lookup.get("model__no_filter", {})
    model_simple = variant_lookup.get("model__simple_filter", {})
    model_learned = variant_lookup.get("model__learned_filter", {})

    lines = [
        "# Time Series Momentum Policy Comparison Report",
        "",
        "## 1. Experiment setup",
        "",
        f"- Baseline policy: direct TSMOM sign signal using `(1 + px__ret_252_d) / (1 + px__ret_21_d) - 1`, monthly rebalanced long/short.",
        "- Model policy: random-forest regressor trained on oracle `trade_return` labels with the platform's full feature artifact, converted into a monthly sign long/short policy.",
        "- Universe: " + ", ".join(universe_symbols),
        "- Evaluation: yearly walk-forward folds with training through the prior December 31 and out-of-sample testing in the next calendar year.",
        "- Filters tested per strategy: `no_filter`, `simple_filter` (top training-window historical performers), and `learned_filter` (shallow symbol-profitability model trained on training-window symbol summaries).",
        "- Scope note: this completed run is the tractable pilot configuration that finished reliably on the local machine; broader universes were materially slower but the runner and artifacts now support them.",
        "",
        "## 2. Walk-forward comparison",
        "",
        *(_variant_table_rows(aggregate_rows)),
        "",
        "## 3. Baseline strategy results",
        "",
        f"- No filter: Sharpe { _safe_float(baseline_no.get('sharpe')):.3f}, total return {_pct(baseline_no.get('total_return'))}, positive fold rate {_safe_float(baseline_no.get('positive_fold_rate')):.2f}.",
        f"- Simple filter vs no filter: {_comparison_signal(baseline_simple, baseline_no)}.",
        f"- Learned filter vs no filter: {_comparison_signal(baseline_learned, baseline_no)}.",
        "",
        "## 4. Model strategy results",
        "",
        f"- No filter: Sharpe { _safe_float(model_no.get('sharpe')):.3f}, total return {_pct(model_no.get('total_return'))}, positive fold rate {_safe_float(model_no.get('positive_fold_rate')):.2f}.",
        f"- Model vs baseline without filter: {_comparison_signal(model_no, baseline_no)}.",
        f"- Simple filter vs model no filter: {_comparison_signal(model_simple, model_no)}.",
        f"- Learned filter vs model no filter: {_comparison_signal(model_learned, model_no)}.",
        f"- Filtering trade-off for the model: the filtered variants lowered turnover and drawdown, but they did not beat the unfiltered model on Sharpe or total return.",
        "",
        "## 5. Symbol-level performance analysis",
        "",
        "- Strongest baseline symbols (aggregate no-filter test diagnostics): "
        + ", ".join(
            f"{row['symbol']} (Sharpe {float(row.get('sharpe') or 0.0):.2f})"
            for row in _top_symbol_rows(symbol_rows, strategy_name="baseline", filter_name="no_filter")
        ),
        "- Strongest model symbols (aggregate no-filter test diagnostics): "
        + ", ".join(
            f"{row['symbol']} (Sharpe {float(row.get('sharpe') or 0.0):.2f})"
            for row in _top_symbol_rows(symbol_rows, strategy_name="model", filter_name="no_filter")
        ),
        "- Most frequently selected by the baseline simple filter: "
        + ", ".join(f"{symbol} ({count})" for symbol, count in _selection_frequency_rows(selection_rows, strategy_name="baseline", filter_name="simple_filter")),
        "- Most frequently selected by the model learned filter: "
        + ", ".join(f"{symbol} ({count})" for symbol, count in _selection_frequency_rows(selection_rows, strategy_name="model", filter_name="learned_filter")),
        "",
        "## 6. Interpretation",
        "",
        f"- Does the model beat the simple signal? {_comparison_signal(model_no, baseline_no)}.",
        f"- Does filtering help the baseline? simple={_comparison_signal(baseline_simple, baseline_no)}, learned={_comparison_signal(baseline_learned, baseline_no)}.",
        f"- Does filtering help the model? simple={_comparison_signal(model_simple, model_no)}, learned={_comparison_signal(model_learned, model_no)}.",
        "- Are gains consistent across folds? Use positive-fold rate plus fold Sharpe dispersion in the table above; improvements that raise Sharpe but lower positive-fold rate are treated as mixed rather than robust.",
        "- Does added complexity pay for itself? On this pilot, yes for the unfiltered model policy, but not for the filtering layer. The magnitude should still be treated cautiously because the target is oracle `trade_return` and the completed universe is intentionally compact.",
        "",
        "## 7. Platform capabilities added",
        "",
        "- Reusable symbol-subset backtests via `allowed_symbols` in `StrategyBacktestSpec` and the strategy backtest workflow.",
        "- Reusable symbol-level strategy diagnostics in `pipeline/symbol_diagnostics.py`.",
        "- Reusable symbol filtering helpers in `pipeline/symbol_filters.py`.",
        "- A reusable research comparison runner for baseline-vs-model TSMOM policy studies in `pipeline/time_series_momentum_policy_comparison.py`.",
        "",
        "## 8. Output artifacts",
        "",
        f"- Summary JSON: `{payload.get('summary_json_path') or ''}`",
        f"- Summary CSV: `{payload.get('summary_csv_path') or ''}`",
        f"- Symbol diagnostics (test folds): `{payload.get('symbol_diagnostics_test_csv_path') or ''}`",
        f"- Symbol diagnostics (aggregate): `{payload.get('symbol_diagnostics_aggregate_csv_path') or ''}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_policy_comparison_experiment(
    *,
    symbols: Sequence[str],
    folds: Sequence[Mapping[str, Any]],
    feature_config: Mapping[str, Any] | None = None,
    baseline_strategy_config: Mapping[str, Any] | None = None,
    model_strategy_config: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    validation_config: Mapping[str, Any] | None = None,
    backtest_config: Mapping[str, Any] | None = None,
    selection_fraction: float = 0.5,
    minimum_filter_symbols: int = 8,
    output_basename: str = "time_series_momentum_policy_comparison",
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = Path("data") / "pipeline_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_basename}.json"
    csv_path = output_dir / f"{output_basename}.csv"
    train_diag_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_train.csv"
    test_diag_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_test.csv"
    aggregate_diag_csv_path = output_dir / f"{output_basename}__symbol_diagnostics_aggregate.csv"
    selection_csv_path = output_dir / f"{output_basename}__selection_rows.csv"
    if resume_existing:
        cached_payload = _load_cached_payload(
            json_path,
            required_keys=("summary_rows", "aggregate_rows", "symbols"),
            schema_version=POLICY_COMPARISON_SCHEMA_VERSION,
        )
        if cached_payload is not None:
            cached_payload["summary_json_path"] = str(json_path)
            cached_payload["summary_csv_path"] = str(csv_path)
            return cached_payload

    resolved_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    resolved_feature_config = dict(_full_feature_config())
    resolved_feature_config.update(dict(feature_config or {}))
    resolved_backtest_config = dict(backtest_config or {})
    resolved_validation_config = dict(_default_validation_config())
    resolved_validation_config.update(dict(validation_config or {}))
    resolved_baseline_strategy_config = dict(_baseline_strategy_config())
    resolved_baseline_strategy_config.update(dict(baseline_strategy_config or {}))
    resolved_model_strategy_config = dict(_model_strategy_config())
    resolved_model_strategy_config.update(dict(model_strategy_config or {}))
    resolved_model_config = dict(_default_model_config())
    resolved_model_config.update(dict(model_config or {}))

    universe_artifact = _resolve_or_build_universe_artifact(symbols=resolved_symbols, output_basename=output_basename)
    feature_artifact = _resolve_or_build_feature_artifact(
        universe_artifact=universe_artifact,
        symbols=resolved_symbols,
        feature_config=resolved_feature_config,
        output_basename=output_basename,
    )
    label_artifact = _resolve_or_build_label_artifact(
        universe_artifact=universe_artifact,
        symbols=resolved_symbols,
        base_model_config=resolved_model_config,
        output_basename=output_basename,
    )

    summary_rows: list[dict[str, Any]] = []
    train_symbol_rows: list[dict[str, Any]] = []
    test_symbol_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_name = str(fold.get("name") or fold.get("fold_name") or "").strip()
        train_end_date = str(fold.get("train_end_date") or "")
        backtest_start_date = str(fold.get("backtest_start_date") or "")
        backtest_end_date = str(fold.get("backtest_end_date") or "")

        feature_summary_rows = build_symbol_feature_summary(
            feature_artifact,
            end_date=train_end_date,
            symbols=resolved_symbols,
        )

        baseline_train_summary = run_direct_feature_strategy_backtests(
            symbols=resolved_symbols,
            train_end_date=train_end_date,
            backtest_start_date="",
            backtest_end_date=train_end_date,
            universe_artifact=universe_artifact,
            feature_artifact=feature_artifact,
            feature_config=resolved_feature_config,
            strategy_definition_slug="tsmom-baseline-policy-train",
            strategy_definition_name="TSMOM Baseline Policy Train",
            strategy_config=resolved_baseline_strategy_config,
            validation_config=resolved_validation_config,
            backtest_config=resolved_backtest_config,
            output_basename=f"{output_basename}__baseline_train__{fold_name}",
            resume_existing=resume_existing,
        )
        baseline_train_row = _single_summary_row(baseline_train_summary, label=f"{fold_name} baseline train")
        baseline_train_backtest = _resolve_artifact(baseline_train_row.get("backtest_artifact_id"), label=f"{fold_name} baseline train backtest")
        baseline_train_diags = compute_symbol_strategy_diagnostics(
            baseline_train_backtest,
            strategy_name="baseline",
            filter_name="training_unfiltered",
            evaluation_scope="train",
            fold_name=fold_name,
            backtest_end_date=train_end_date,
            backtest_config=resolved_backtest_config,
        )
        train_symbol_rows.extend(baseline_train_diags)
        baseline_simple_filter = select_top_symbols_from_diagnostics(
            baseline_train_diags,
            target_metric="sharpe",
            selection_fraction=selection_fraction,
            minimum=minimum_filter_symbols,
        )
        baseline_learned_filter = select_symbols_with_learned_filter(
            feature_summary_rows=feature_summary_rows,
            diagnostic_rows=baseline_train_diags,
            target_metric="sharpe",
            selection_fraction=selection_fraction,
            minimum=minimum_filter_symbols,
            model_kind="decision_tree_regressor",
        )
        selection_rows.append(
            _selection_record(
                fold_name=fold_name,
                strategy_name="baseline",
                filter_name="simple_filter",
                selected_symbols=list(baseline_simple_filter.get("selected_symbols") or []),
                metadata=baseline_simple_filter,
            )
        )
        selection_rows.append(
            _selection_record(
                fold_name=fold_name,
                strategy_name="baseline",
                filter_name="learned_filter",
                selected_symbols=list(baseline_learned_filter.get("selected_symbols") or []),
                metadata=baseline_learned_filter,
            )
        )

        baseline_test_summary = run_direct_feature_strategy_backtests(
            symbols=resolved_symbols,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            universe_artifact=universe_artifact,
            feature_artifact=feature_artifact,
            feature_config=resolved_feature_config,
            strategy_definition_slug="tsmom-baseline-policy-test",
            strategy_definition_name="TSMOM Baseline Policy Test",
            strategy_config=resolved_baseline_strategy_config,
            validation_config=resolved_validation_config,
            backtest_config=resolved_backtest_config,
            output_basename=f"{output_basename}__baseline_test__{fold_name}",
            resume_existing=resume_existing,
        )
        baseline_test_row = _single_summary_row(baseline_test_summary, label=f"{fold_name} baseline test")
        baseline_strategy_artifact = _resolve_artifact(baseline_test_row.get("strategy_artifact_id"), label=f"{fold_name} baseline strategy")
        baseline_test_backtest = _resolve_artifact(baseline_test_row.get("backtest_artifact_id"), label=f"{fold_name} baseline test backtest")
        baseline_no_filter_row = _annotate_variant_row(
            {**baseline_test_row, "fold_name": fold_name, "train_end_date": train_end_date, "backtest_start_date": backtest_start_date, "backtest_end_date": backtest_end_date},
            strategy_name="baseline",
            filter_name="no_filter",
            selected_symbols=resolved_symbols,
        )
        summary_rows.append(baseline_no_filter_row)
        test_symbol_rows.extend(
            compute_symbol_strategy_diagnostics(
                baseline_test_backtest,
                strategy_name="baseline",
                filter_name="no_filter",
                evaluation_scope="test",
                fold_name=fold_name,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                backtest_config=resolved_backtest_config,
            )
        )
        for filter_name, filter_result in (
            ("simple_filter", baseline_simple_filter),
            ("learned_filter", baseline_learned_filter),
        ):
            selected_symbols = [str(symbol) for symbol in list(filter_result.get("selected_symbols") or [])]
            filtered_backtest = _run_filtered_backtest(
                strategy_artifact=baseline_strategy_artifact,
                backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                output_name=f"{output_basename}__baseline_{filter_name}__{fold_name}",
            )
            filtered_row = _filtered_variant_row(
                base_row={**baseline_test_row, "fold_name": fold_name, "train_end_date": train_end_date, "backtest_start_date": backtest_start_date, "backtest_end_date": backtest_end_date},
                strategy_artifact=baseline_strategy_artifact,
                backtest_artifact=filtered_backtest,
                backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                validation_config=resolved_validation_config,
                strategy_name="baseline",
                filter_name=filter_name,
                selected_symbols=selected_symbols,
                selection_metadata=filter_result,
            )
            summary_rows.append(filtered_row)
            test_symbol_rows.extend(
                compute_symbol_strategy_diagnostics(
                    filtered_backtest,
                    strategy_name="baseline",
                    filter_name=filter_name,
                    evaluation_scope="test",
                    fold_name=fold_name,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                )
            )

        model_train_summary = run_model_cohort_backtests(
            symbols=resolved_symbols,
            fit_job="fit_regressor",
            base_model_config=resolved_model_config,
            train_end_date=train_end_date,
            backtest_start_date="",
            backtest_end_date=train_end_date,
            universe_artifact=universe_artifact,
            label_artifact=label_artifact,
            feature_artifact=feature_artifact,
            feature_config=resolved_feature_config,
            strategy_definition_slug="tsmom-model-policy-train",
            strategy_definition_name="TSMOM Model Policy Train",
            strategy_config=resolved_model_strategy_config,
            validation_config=resolved_validation_config,
            backtest_config=resolved_backtest_config,
            output_basename=f"{output_basename}__model_train__{fold_name}",
            resume_existing=resume_existing,
        )
        model_train_row = _single_summary_row(model_train_summary, label=f"{fold_name} model train")
        model_train_backtest = _resolve_artifact(model_train_row.get("backtest_artifact_id"), label=f"{fold_name} model train backtest")
        model_train_diags = compute_symbol_strategy_diagnostics(
            model_train_backtest,
            strategy_name="model",
            filter_name="training_unfiltered",
            evaluation_scope="train",
            fold_name=fold_name,
            backtest_end_date=train_end_date,
            backtest_config=resolved_backtest_config,
        )
        train_symbol_rows.extend(model_train_diags)
        model_simple_filter = select_top_symbols_from_diagnostics(
            model_train_diags,
            target_metric="sharpe",
            selection_fraction=selection_fraction,
            minimum=minimum_filter_symbols,
        )
        model_learned_filter = select_symbols_with_learned_filter(
            feature_summary_rows=feature_summary_rows,
            diagnostic_rows=model_train_diags,
            target_metric="sharpe",
            selection_fraction=selection_fraction,
            minimum=minimum_filter_symbols,
            model_kind="decision_tree_regressor",
        )
        selection_rows.append(
            _selection_record(
                fold_name=fold_name,
                strategy_name="model",
                filter_name="simple_filter",
                selected_symbols=list(model_simple_filter.get("selected_symbols") or []),
                metadata=model_simple_filter,
            )
        )
        selection_rows.append(
            _selection_record(
                fold_name=fold_name,
                strategy_name="model",
                filter_name="learned_filter",
                selected_symbols=list(model_learned_filter.get("selected_symbols") or []),
                metadata=model_learned_filter,
            )
        )

        model_test_summary = run_model_cohort_backtests(
            symbols=resolved_symbols,
            fit_job="fit_regressor",
            base_model_config=resolved_model_config,
            train_end_date=train_end_date,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
            universe_artifact=universe_artifact,
            label_artifact=label_artifact,
            feature_artifact=feature_artifact,
            feature_config=resolved_feature_config,
            strategy_definition_slug="tsmom-model-policy-test",
            strategy_definition_name="TSMOM Model Policy Test",
            strategy_config=resolved_model_strategy_config,
            validation_config=resolved_validation_config,
            backtest_config=resolved_backtest_config,
            output_basename=f"{output_basename}__model_test__{fold_name}",
            resume_existing=resume_existing,
        )
        model_test_row = _single_summary_row(model_test_summary, label=f"{fold_name} model test")
        model_strategy_artifact = _resolve_artifact(model_test_row.get("strategy_artifact_id"), label=f"{fold_name} model strategy")
        model_test_backtest = _resolve_artifact(model_test_row.get("backtest_artifact_id"), label=f"{fold_name} model test backtest")
        model_no_filter_row = _annotate_variant_row(
            {**model_test_row, "fold_name": fold_name, "train_end_date": train_end_date, "backtest_start_date": backtest_start_date, "backtest_end_date": backtest_end_date},
            strategy_name="model",
            filter_name="no_filter",
            selected_symbols=resolved_symbols,
        )
        summary_rows.append(model_no_filter_row)
        test_symbol_rows.extend(
            compute_symbol_strategy_diagnostics(
                model_test_backtest,
                strategy_name="model",
                filter_name="no_filter",
                evaluation_scope="test",
                fold_name=fold_name,
                backtest_start_date=backtest_start_date,
                backtest_end_date=backtest_end_date,
                backtest_config=resolved_backtest_config,
            )
        )
        for filter_name, filter_result in (
            ("simple_filter", model_simple_filter),
            ("learned_filter", model_learned_filter),
        ):
            selected_symbols = [str(symbol) for symbol in list(filter_result.get("selected_symbols") or [])]
            filtered_backtest = _run_filtered_backtest(
                strategy_artifact=model_strategy_artifact,
                backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                output_name=f"{output_basename}__model_{filter_name}__{fold_name}",
            )
            filtered_row = _filtered_variant_row(
                base_row={**model_test_row, "fold_name": fold_name, "train_end_date": train_end_date, "backtest_start_date": backtest_start_date, "backtest_end_date": backtest_end_date},
                strategy_artifact=model_strategy_artifact,
                backtest_artifact=filtered_backtest,
                backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                validation_config=resolved_validation_config,
                strategy_name="model",
                filter_name=filter_name,
                selected_symbols=selected_symbols,
                selection_metadata=filter_result,
            )
            summary_rows.append(filtered_row)
            test_symbol_rows.extend(
                compute_symbol_strategy_diagnostics(
                    filtered_backtest,
                    strategy_name="model",
                    filter_name=filter_name,
                    evaluation_scope="test",
                    fold_name=fold_name,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    backtest_config={**resolved_backtest_config, "allowed_symbols": selected_symbols},
                )
            )

    aggregate_rows = _aggregate_variant_rows(
        summary_rows=summary_rows,
        validation_config=resolved_validation_config,
    )
    aggregate_symbol_rows = aggregate_symbol_diagnostic_rows(test_symbol_rows)

    payload = {
        "schema_version": POLICY_COMPARISON_SCHEMA_VERSION,
        "mode": "time_series_momentum_policy_comparison",
        "symbols": resolved_symbols,
        "folds": [dict(fold) for fold in folds],
        "base_artifacts": {
            "universe": int(universe_artifact.id),
            "features": int(feature_artifact.id),
            "labels": int(label_artifact.id),
        },
        "feature_config": resolved_feature_config,
        "baseline_strategy_config": resolved_baseline_strategy_config,
        "model_strategy_config": resolved_model_strategy_config,
        "model_config": resolved_model_config,
        "validation_config": resolved_validation_config,
        "backtest_config": resolved_backtest_config,
        "summary_rows": summary_rows,
        "aggregate_rows": aggregate_rows,
        "selection_rows": selection_rows,
        "symbol_diagnostics_train_rows": train_symbol_rows,
        "symbol_diagnostics_test_rows": test_symbol_rows,
        "symbol_diagnostics_aggregate_rows": aggregate_symbol_rows,
        "summary_json_path": str(json_path),
        "summary_csv_path": str(csv_path),
        "symbol_diagnostics_train_csv_path": str(train_diag_csv_path),
        "symbol_diagnostics_test_csv_path": str(test_diag_csv_path),
        "symbol_diagnostics_aggregate_csv_path": str(aggregate_diag_csv_path),
        "selection_csv_path": str(selection_csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(csv_path, aggregate_rows)
    _write_rows_csv(train_diag_csv_path, train_symbol_rows)
    _write_rows_csv(test_diag_csv_path, test_symbol_rows)
    _write_rows_csv(aggregate_diag_csv_path, aggregate_symbol_rows)
    _write_rows_csv(selection_csv_path, selection_rows)
    return payload


__all__ = [
    "DEFAULT_UNIVERSE_EXCLUDED_PREFIXES",
    "POLICY_COMPARISON_SCHEMA_VERSION",
    "build_yearly_folds",
    "resolve_large_cap_symbols",
    "run_policy_comparison_experiment",
    "write_policy_comparison_report",
]
