from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from .cross_sectional_rank_labels import first_available_column
from domain.models.datasets import feature_columns_from_frame, filter_frame_by_date
from pipeline.service_runtime import read_frame_artifact
from .ranking_diagnostics import build_symbol_metadata_lookup


MARKET_CAP_METADATA_COL_CANDIDATES: tuple[str, ...] = (
    "km__marketcap",
    "marketcap",
    "market_cap",
    "km__market_cap",
    "own__market_cap",
)


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


def build_symbol_metadata_filter_summary(
    feature_frame_or_artifact,
    *,
    end_date: str | None = None,
    symbols: Sequence[str] | None = None,
    market_cap_col_candidates: Sequence[str] = MARKET_CAP_METADATA_COL_CANDIDATES,
) -> list[dict[str, Any]]:
    if isinstance(feature_frame_or_artifact, pd.DataFrame):
        feature_df = pd.DataFrame(feature_frame_or_artifact).copy()
    elif feature_frame_or_artifact is None:
        feature_df = pd.DataFrame()
    else:
        feature_df = read_frame_artifact(
            feature_frame_or_artifact,
            parse_dates=False,
            normalize_symbols=True,
        )

    normalized_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    market_cap_map: dict[str, float] = {}
    market_cap_source_symbols: set[str] = set()
    if not feature_df.empty:
        feature_df = feature_df.copy()
        if "symbol" in feature_df.columns:
            feature_df["symbol"] = feature_df["symbol"].astype(str).str.strip().str.upper()
        feature_df = filter_frame_by_date(feature_df, end_date=end_date)
        if normalized_symbols:
            allowed = set(normalized_symbols)
            feature_df = feature_df[feature_df["symbol"].astype(str).isin(allowed)].copy()
        market_cap_col = first_available_column(feature_df.columns, market_cap_col_candidates)
        if market_cap_col:
            market_cap_series = pd.to_numeric(feature_df[market_cap_col], errors="coerce")
            feature_df[market_cap_col] = market_cap_series
            grouped = (
                feature_df.dropna(subset=["symbol"])
                .groupby("symbol", observed=True)[market_cap_col]
                .mean()
                .dropna()
            )
            market_cap_map = {
                str(symbol).strip().upper(): _round_float(value)
                for symbol, value in grouped.items()
            }
            market_cap_source_symbols = set(market_cap_map.keys())
        if not normalized_symbols and "symbol" in feature_df.columns:
            normalized_symbols = sorted(
                {
                    str(symbol).strip().upper()
                    for symbol in feature_df["symbol"].astype(str).tolist()
                    if str(symbol).strip()
                }
            )

    lookup = build_symbol_metadata_lookup(normalized_symbols or market_cap_map.keys())
    ordered_symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in [*(normalized_symbols or []), *list(market_cap_map.keys()), *list(lookup.keys())]:
        symbol = str(raw_symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered_symbols.append(symbol)

    rows: list[dict[str, Any]] = []
    for symbol in ordered_symbols:
        metadata = dict(lookup.get(symbol) or {})
        snapshot_cap = _safe_float(metadata.get("market_cap"))
        avg_market_cap = _safe_float(market_cap_map.get(symbol), default=snapshot_cap)
        rows.append(
            {
                "symbol": symbol,
                "sector": str(metadata.get("sector") or "Unknown"),
                "industry": str(metadata.get("industry") or "Unknown"),
                "exchange": str(metadata.get("exchange") or "Unknown"),
                "country": str(metadata.get("country") or "Unknown"),
                "company_name": str(metadata.get("company_name") or ""),
                "avg_market_cap": _round_float(avg_market_cap),
                "snapshot_market_cap": _round_float(snapshot_cap),
                "market_cap_source": "feature_mean" if symbol in market_cap_source_symbols else "snapshot",
            }
        )
    return rows


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


def _value_counts_map(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
    )
    return {str(index): int(value) for index, value in counts.items()}


def _metadata_design_matrix(
    frame: pd.DataFrame,
    *,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
    feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    for column in categorical_cols:
        if column not in work.columns:
            work[column] = "Unknown"
        work[column] = work[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    for column in numeric_cols:
        work[column] = pd.to_numeric(work.get(column), errors="coerce").fillna(0.0)

    parts: list[pd.DataFrame] = []
    if categorical_cols:
        parts.append(pd.get_dummies(work[list(categorical_cols)], prefix=list(categorical_cols), dtype=float))
    if numeric_cols:
        parts.append(work[list(numeric_cols)].astype(float))
    design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=work.index)
    if feature_columns:
        design = design.reindex(columns=list(feature_columns), fill_value=0.0)
        return design, list(feature_columns)
    return design, design.columns.astype(str).tolist()


def select_symbols_with_metadata_filter(
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    target_col: str,
    minimum_selected_symbols: int = 5,
    maximum_selected_symbols: int | None = None,
    categorical_cols: Sequence[str] = ("sector", "industry", "exchange"),
    numeric_cols: Sequence[str] = ("avg_market_cap",),
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    random_state: int = 1337,
) -> dict[str, Any]:
    metadata_df = pd.DataFrame([dict(row) for row in metadata_rows])
    target_df = pd.DataFrame([dict(row) for row in target_rows])
    if metadata_df.empty or target_df.empty or "symbol" not in metadata_df.columns or "symbol" not in target_df.columns:
        return {
            "selected_symbols": [],
            "selection_count": 0,
            "target_col": str(target_col),
            "model_kind": "decision_tree_classifier",
            "used_fallback": True,
            "fallback_reason": "missing_metadata_or_targets",
            "score_rows": [],
            "top_features": [],
        }

    metadata_df = metadata_df.copy()
    metadata_df["symbol"] = metadata_df["symbol"].astype(str).str.strip().str.upper()
    target_df = target_df.copy()
    target_df["symbol"] = target_df["symbol"].astype(str).str.strip().str.upper()
    if target_col not in target_df.columns:
        return {
            "selected_symbols": [],
            "selection_count": 0,
            "target_col": str(target_col),
            "model_kind": "decision_tree_classifier",
            "used_fallback": True,
            "fallback_reason": "missing_target_col",
            "score_rows": [],
            "top_features": [],
        }

    target_df[target_col] = pd.to_numeric(target_df[target_col], errors="coerce").fillna(0.0).astype(int).clip(lower=0, upper=1)
    train_df = metadata_df.merge(target_df[["symbol", target_col]], on="symbol", how="inner")
    if train_df.empty:
        return {
            "selected_symbols": [],
            "selection_count": 0,
            "target_col": str(target_col),
            "model_kind": "decision_tree_classifier",
            "used_fallback": True,
            "fallback_reason": "no_training_overlap",
            "score_rows": [],
            "top_features": [],
        }

    x_train, feature_columns = _metadata_design_matrix(
        train_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    if x_train.empty:
        return {
            "selected_symbols": [],
            "selection_count": 0,
            "target_col": str(target_col),
            "model_kind": "decision_tree_classifier",
            "used_fallback": True,
            "fallback_reason": "empty_design_matrix",
            "score_rows": [],
            "top_features": [],
        }
    y_train = train_df[target_col].astype(int)
    resolved_min_samples_leaf = min(max(int(min_samples_leaf), 1), max(int(len(train_df)), 1))
    model = DecisionTreeClassifier(
        max_depth=max(int(max_depth), 1),
        min_samples_leaf=resolved_min_samples_leaf,
        random_state=int(random_state),
    )
    model.fit(x_train, y_train)

    score_df = metadata_df.copy()
    x_score, _feature_columns = _metadata_design_matrix(
        score_df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        feature_columns=feature_columns,
    )
    predicted_label = pd.Series(model.predict(x_score), index=score_df.index).astype(int)
    positive_probability = pd.Series(0.0, index=score_df.index, dtype=float)
    predict_proba = getattr(model, "predict_proba", None)
    classes = list(getattr(model, "classes_", []))
    if callable(predict_proba) and classes:
        proba = predict_proba(x_score)
        if 1 in classes:
            class_index = classes.index(1)
            positive_probability = pd.Series(proba[:, class_index], index=score_df.index).astype(float)
        elif len(classes) == 1 and int(classes[0]) == 1:
            positive_probability = pd.Series(1.0, index=score_df.index, dtype=float)
    score_df["predicted_label"] = predicted_label
    score_df["positive_probability"] = positive_probability
    score_df["avg_market_cap"] = pd.to_numeric(score_df.get("avg_market_cap"), errors="coerce").fillna(0.0)
    ranked = score_df.sort_values(
        ["positive_probability", "predicted_label", "avg_market_cap", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    selected = ranked[ranked["predicted_label"] > 0].copy()
    minimum_requested = max(int(minimum_selected_symbols), 0)
    if maximum_selected_symbols not in (None, ""):
        maximum_requested = max(int(maximum_selected_symbols), 1)
    else:
        maximum_requested = None
    used_fallback = False
    fallback_reason = ""
    if minimum_requested > 0 and len(selected) < minimum_requested:
        used_fallback = True
        fallback_reason = "minimum_selected_symbols"
        selected = ranked.head(minimum_requested).copy()
    if maximum_requested is not None and len(selected) > maximum_requested:
        selected = selected.head(maximum_requested).copy()

    importances = getattr(model, "feature_importances_", None)
    top_features: list[tuple[str, float]] = []
    if importances is not None:
        top_features = sorted(
            (
                (str(column), float(value))
                for column, value in zip(feature_columns, list(importances))
                if float(value) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]

    rules = export_text(model, feature_names=list(feature_columns)).strip()
    selected_preview = selected.head(25)
    return {
        "selected_symbols": [str(symbol) for symbol in selected["symbol"].astype(str).tolist()],
        "selection_count": int(len(selected)),
        "target_col": str(target_col),
        "model_kind": "decision_tree_classifier",
        "used_fallback": bool(used_fallback),
        "fallback_reason": str(fallback_reason),
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "trained_symbols": int(len(train_df)),
        "positive_target_count": int(y_train.sum()),
        "positive_target_rate": _round_float(float(y_train.mean()) if len(y_train) else 0.0, 6),
        "tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "top_features": [(name, _round_float(value, 6)) for name, value in top_features],
        "tree_rules": rules,
        "selected_sector_counts": _value_counts_map(selected.get("sector", pd.Series(dtype=object))),
        "selected_industry_counts": _value_counts_map(selected.get("industry", pd.Series(dtype=object))),
        "selected_exchange_counts": _value_counts_map(selected.get("exchange", pd.Series(dtype=object))),
        "score_rows": [
            {
                "symbol": str(row["symbol"]),
                "predicted_label": int(_safe_float(row.get("predicted_label"))),
                "positive_probability": _round_float(_safe_float(row.get("positive_probability")), 6),
                "sector": str(row.get("sector") or "Unknown"),
                "industry": str(row.get("industry") or "Unknown"),
                "exchange": str(row.get("exchange") or "Unknown"),
                "avg_market_cap": _round_float(_safe_float(row.get("avg_market_cap"))),
            }
            for row in selected_preview.to_dict(orient="records")
        ],
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
    "build_symbol_metadata_filter_summary",
    "MARKET_CAP_METADATA_COL_CANDIDATES",
    "resolve_symbol_selection_count",
    "select_symbols_with_metadata_filter",
    "select_symbols_with_learned_filter",
    "select_top_symbols_from_diagnostics",
]
