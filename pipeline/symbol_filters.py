from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from domain.models.datasets import feature_columns_from_frame, filter_frame_by_date
from pipeline.service_runtime import read_frame_artifact


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def resolve_symbol_selection_count(
    *,
    total_symbols: int,
    selection_fraction: float = 0.5,
    minimum: int = 5,
    maximum: int | None = None,
) -> int:
    total = max(int(total_symbols), 0)
    if total <= 0:
        return 0
    fraction = min(max(float(selection_fraction), 0.0), 1.0)
    requested = int(ceil(total * fraction)) if fraction > 0.0 else total
    requested = max(int(minimum), requested)
    requested = min(total, requested)
    if maximum not in (None, ""):
        requested = min(requested, max(int(maximum), 1))
    return max(1, requested)


def build_symbol_feature_summary(
    feature_artifact,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std", "last"),
) -> list[dict[str, Any]]:
    feature_df = read_frame_artifact(
        feature_artifact,
        parse_dates=False,
        normalize_symbols=True,
    )
    if feature_df.empty:
        return []
    feature_df = filter_frame_by_date(feature_df, start_date=start_date, end_date=end_date)
    if feature_df.empty:
        return []
    if symbols:
        allowed = {str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()}
        feature_df = feature_df[feature_df["symbol"].astype(str).isin(allowed)].copy()
        if feature_df.empty:
            return []

    numeric_map: dict[str, pd.Series] = {}
    for column in feature_columns_from_frame(feature_df):
        numeric = pd.to_numeric(feature_df[column], errors="coerce")
        if numeric.notna().any():
            numeric_map[column] = numeric
    numeric_cols = list(numeric_map.keys())
    if not numeric_cols:
        return []

    numeric_frame = pd.DataFrame(numeric_map, index=feature_df.index)
    summary_input = pd.concat(
        [
            feature_df[["symbol"]].reset_index(drop=True),
            numeric_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    grouped = summary_input.groupby("symbol", observed=True)
    summary_frames: list[pd.DataFrame] = []
    for aggregation in list(aggregations):
        if aggregation == "last":
            frame = grouped[numeric_cols].last().add_suffix("__last")
        elif aggregation == "mean":
            frame = grouped[numeric_cols].mean().add_suffix("__mean")
        elif aggregation == "std":
            frame = grouped[numeric_cols].std(ddof=0).add_suffix("__std")
        else:
            continue
        summary_frames.append(frame)
    if not summary_frames:
        return []
    summary = pd.concat(summary_frames, axis=1).copy().reset_index()

    coverage_grouped = feature_df.groupby("symbol", observed=True)
    coverage = (
        coverage_grouped["date"]
        .agg(row_count="size", coverage_start_date="min", coverage_end_date="max")
        .reset_index()
    )
    coverage["coverage_start_date"] = pd.to_datetime(coverage["coverage_start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    coverage["coverage_end_date"] = pd.to_datetime(coverage["coverage_end_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out = coverage.merge(summary, on="symbol", how="left")
    numeric_summary_cols = [column for column in out.columns if column not in {"symbol", "coverage_start_date", "coverage_end_date"}]
    out[numeric_summary_cols] = out[numeric_summary_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return list(out.to_dict(orient="records"))


def select_top_symbols_from_diagnostics(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    *,
    target_metric: str = "sharpe",
    selection_fraction: float = 0.5,
    minimum: int = 5,
    maximum: int | None = None,
    min_trade_count: int = 1,
) -> dict[str, Any]:
    df = pd.DataFrame([dict(row) for row in diagnostic_rows])
    if df.empty or "symbol" not in df.columns:
        return {
            "selected_symbols": [],
            "selection_count": 0,
            "target_metric": str(target_metric),
            "score_rows": [],
        }

    for column in ("sharpe", "avg_trade_return", "hit_rate", "trade_count"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    eligible = df[df["trade_count"].fillna(0.0) >= max(int(min_trade_count), 0)].copy()
    if eligible.empty:
        eligible = df.copy()
    metric = str(target_metric or "sharpe").strip() or "sharpe"
    if metric not in eligible.columns:
        metric = "sharpe" if "sharpe" in eligible.columns else "avg_trade_return"
    selection_count = resolve_symbol_selection_count(
        total_symbols=int(df["symbol"].nunique()),
        selection_fraction=selection_fraction,
        minimum=minimum,
        maximum=maximum,
    )
    ranked = eligible.sort_values(
        [metric, "avg_trade_return", "hit_rate", "trade_count", "symbol"],
        ascending=[False, False, False, False, True],
    ).head(selection_count)
    score_rows = [
        {
            "symbol": str(row["symbol"]),
            "score": _round_float(_safe_float(row.get(metric))),
            "target_metric": metric,
            "trade_count": int(_safe_float(row.get("trade_count"))),
            "sharpe": _round_float(_safe_float(row.get("sharpe"))),
            "avg_trade_return": _round_float(_safe_float(row.get("avg_trade_return"))),
            "hit_rate": _round_float(_safe_float(row.get("hit_rate")), 6),
        }
        for row in ranked.to_dict(orient="records")
    ]
    return {
        "selected_symbols": [str(symbol) for symbol in ranked["symbol"].astype(str).tolist()],
        "selection_count": int(len(ranked)),
        "target_metric": metric,
        "score_rows": score_rows,
    }


def select_symbols_with_learned_filter(
    *,
    feature_summary_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    target_metric: str = "sharpe",
    selection_fraction: float = 0.5,
    minimum: int = 5,
    maximum: int | None = None,
    min_trade_count: int = 1,
    model_kind: str = "decision_tree_regressor",
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    n_estimators: int = 100,
    random_state: int = 1337,
) -> dict[str, Any]:
    feature_df = pd.DataFrame([dict(row) for row in feature_summary_rows])
    diagnostic_df = pd.DataFrame([dict(row) for row in diagnostic_rows])
    if feature_df.empty or diagnostic_df.empty or "symbol" not in feature_df.columns or "symbol" not in diagnostic_df.columns:
        fallback = select_top_symbols_from_diagnostics(
            diagnostic_rows,
            target_metric=target_metric,
            selection_fraction=selection_fraction,
            minimum=minimum,
            maximum=maximum,
            min_trade_count=min_trade_count,
        )
        fallback["model_kind"] = str(model_kind)
        fallback["used_fallback"] = True
        return fallback

    diagnostic_df["trade_count"] = pd.to_numeric(diagnostic_df.get("trade_count"), errors="coerce").fillna(0.0)
    metric = str(target_metric or "sharpe").strip() or "sharpe"
    if metric not in diagnostic_df.columns:
        metric = "sharpe" if "sharpe" in diagnostic_df.columns else "avg_trade_return"
    diagnostic_df[metric] = pd.to_numeric(diagnostic_df.get(metric), errors="coerce")
    train_df = diagnostic_df[diagnostic_df["trade_count"] >= max(int(min_trade_count), 0)].copy()
    if train_df.empty:
        train_df = diagnostic_df.copy()
    train_df = train_df[["symbol", metric]].dropna(subset=["symbol", metric]).copy()
    if train_df.empty:
        fallback = select_top_symbols_from_diagnostics(
            diagnostic_rows,
            target_metric=metric,
            selection_fraction=selection_fraction,
            minimum=minimum,
            maximum=maximum,
            min_trade_count=min_trade_count,
        )
        fallback["model_kind"] = str(model_kind)
        fallback["used_fallback"] = True
        return fallback

    merged_train = train_df.merge(feature_df, on="symbol", how="inner")
    if merged_train.empty:
        fallback = select_top_symbols_from_diagnostics(
            diagnostic_rows,
            target_metric=metric,
            selection_fraction=selection_fraction,
            minimum=minimum,
            maximum=maximum,
            min_trade_count=min_trade_count,
        )
        fallback["model_kind"] = str(model_kind)
        fallback["used_fallback"] = True
        return fallback

    feature_cols: list[str] = []
    for column in feature_df.columns:
        if column in {"symbol", "coverage_start_date", "coverage_end_date"}:
            continue
        numeric = pd.to_numeric(feature_df[column], errors="coerce")
        if numeric.notna().any():
            feature_df[column] = numeric
            if column in merged_train.columns:
                merged_train[column] = pd.to_numeric(merged_train[column], errors="coerce")
            feature_cols.append(column)
    if not feature_cols:
        fallback = select_top_symbols_from_diagnostics(
            diagnostic_rows,
            target_metric=metric,
            selection_fraction=selection_fraction,
            minimum=minimum,
            maximum=maximum,
            min_trade_count=min_trade_count,
        )
        fallback["model_kind"] = str(model_kind)
        fallback["used_fallback"] = True
        return fallback

    x_train = merged_train[feature_cols].fillna(0.0)
    y_train = pd.to_numeric(merged_train[metric], errors="coerce").fillna(0.0)
    if len(x_train) < max(int(minimum), 4):
        fallback = select_top_symbols_from_diagnostics(
            diagnostic_rows,
            target_metric=metric,
            selection_fraction=selection_fraction,
            minimum=minimum,
            maximum=maximum,
            min_trade_count=min_trade_count,
        )
        fallback["model_kind"] = str(model_kind)
        fallback["used_fallback"] = True
        return fallback

    model_kind_value = str(model_kind or "decision_tree_regressor").strip().lower()
    if model_kind_value == "random_forest_regressor":
        model = RandomForestRegressor(
            n_estimators=max(int(n_estimators), 10),
            max_depth=max(int(max_depth), 1),
            min_samples_leaf=max(int(min_samples_leaf), 1),
            random_state=int(random_state),
        )
    else:
        model_kind_value = "decision_tree_regressor"
        model = DecisionTreeRegressor(
            max_depth=max(int(max_depth), 1),
            min_samples_leaf=max(int(min_samples_leaf), 1),
            random_state=int(random_state),
        )
    model.fit(x_train, y_train)

    score_df = feature_df.copy()
    x_score = score_df[feature_cols].fillna(0.0)
    score_df["predicted_profitability"] = model.predict(x_score)
    selection_count = resolve_symbol_selection_count(
        total_symbols=int(score_df["symbol"].nunique()),
        selection_fraction=selection_fraction,
        minimum=minimum,
        maximum=maximum,
    )
    ranked = score_df.sort_values(
        ["predicted_profitability", "row_count", "symbol"],
        ascending=[False, False, True],
    ).head(selection_count)
    importances = getattr(model, "feature_importances_", None)
    top_features: list[tuple[str, float]] = []
    if importances is not None:
        top_features = sorted(
            (
                (str(column), float(value))
                for column, value in zip(feature_cols, list(importances))
                if float(value) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

    return {
        "selected_symbols": [str(symbol) for symbol in ranked["symbol"].astype(str).tolist()],
        "selection_count": int(len(ranked)),
        "target_metric": metric,
        "model_kind": model_kind_value,
        "used_fallback": False,
        "feature_count": int(len(feature_cols)),
        "trained_symbols": int(len(merged_train)),
        "top_features": [(name, _round_float(value, 6)) for name, value in top_features],
        "score_rows": [
            {
                "symbol": str(row["symbol"]),
                "predicted_profitability": _round_float(_safe_float(row.get("predicted_profitability"))),
                "row_count": int(_safe_float(row.get("row_count"))),
            }
            for row in ranked.to_dict(orient="records")
        ],
    }


__all__ = [
    "build_symbol_feature_summary",
    "resolve_symbol_selection_count",
    "select_symbols_with_learned_filter",
    "select_top_symbols_from_diagnostics",
]
