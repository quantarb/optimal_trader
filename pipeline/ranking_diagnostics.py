from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from fmp.models import Symbol
from ml.execution import load_artifact_csv_frame

from .cross_sectional_rank_labels import first_available_column


DEFAULT_SCORE_COL_CANDIDATES: tuple[str, ...] = (
    "score",
    "signal_score",
    "prediction_score",
    "prediction",
    "strategy_score",
    "ranking",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _round_float(value: Any, digits: int = 8) -> float:
    return round(_safe_float(value), digits)


def _is_etf_symbol(row: Symbol) -> bool:
    payload = row.payload if isinstance(row.payload, dict) else {}
    name = str(row.company_name or "").strip().lower()
    return bool(payload.get("isEtf")) or bool(payload.get("isETF")) or bool(payload.get("isFund")) or (" etf" in f" {name}")


def build_symbol_metadata_lookup(symbols: Iterable[str]) -> dict[str, dict[str, Any]]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if not normalized:
        return {}
    rows = Symbol.objects.filter(symbol__in=normalized).only(
        "symbol",
        "company_name",
        "exchange",
        "country",
        "sector",
        "industry",
        "market_cap",
        "payload",
    )
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.symbol).strip().upper()
        is_etf = _is_etf_symbol(row)
        lookup[symbol] = {
            "sector": str(row.sector or "").strip() or "Unknown",
            "industry": str(row.industry or "").strip() or "Unknown",
            "exchange": str(row.exchange or "").strip() or "Unknown",
            "country": str(row.country or "").strip() or "Unknown",
            "company_name": str(row.company_name or "").strip(),
            "market_cap": _round_float(row.market_cap) if row.market_cap not in (None, "") else 0.0,
            "is_etf": bool(is_etf),
            "instrument_type": "etf" if is_etf else "stock",
        }
    return lookup


def build_expression_score_frame(
    feature_frame_or_artifact,
    *,
    score_expression: str = "",
    score_col_candidates: Sequence[str] = (),
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    feature_df = (
        load_artifact_csv_frame(feature_frame_or_artifact)
        if hasattr(feature_frame_or_artifact, "uri")
        else pd.DataFrame(feature_frame_or_artifact).copy()
    )
    if feature_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "score"])
    score_expr = str(score_expression or "").strip()
    candidate_col = first_available_column(feature_df.columns, score_col_candidates)
    score_series = pd.Series(index=feature_df.index, dtype=float)
    if score_expr:
        if score_expr in feature_df.columns:
            score_series = pd.to_numeric(feature_df[score_expr], errors="coerce")
        else:
            score_series = pd.to_numeric(feature_df.eval(score_expr, engine="python"), errors="coerce")
    elif candidate_col:
        score_series = pd.to_numeric(feature_df[candidate_col], errors="coerce")
    out = feature_df[["date", "symbol"]].copy()
    out["score"] = score_series
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if start_date:
        out = out[out["date"] >= pd.Timestamp(str(start_date))].copy()
    if end_date:
        out = out[out["date"] <= pd.Timestamp(str(end_date))].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    return out.dropna(subset=["date", "symbol", "score"]).sort_values(["date", "symbol"]).reset_index(drop=True)


def build_signal_ranking_panel(
    score_frame_or_artifact,
    label_frame_or_artifact,
    *,
    score_col_candidates: Sequence[str] = DEFAULT_SCORE_COL_CANDIDATES,
    target_col: str = "future_rank_pct",
    forward_return_col: str = "trade_return",
    start_date: str | None = None,
    end_date: str | None = None,
    variant_name: str = "",
    fold_name: str = "",
    feature_scope: str = "",
    variant_kind: str = "",
    variant_label: str = "",
    symbol_metadata_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    score_df = (
        load_artifact_csv_frame(score_frame_or_artifact)
        if hasattr(score_frame_or_artifact, "uri")
        else pd.DataFrame(score_frame_or_artifact).copy()
    )
    label_df = (
        load_artifact_csv_frame(label_frame_or_artifact)
        if hasattr(label_frame_or_artifact, "uri")
        else pd.DataFrame(label_frame_or_artifact).copy()
    )
    if score_df.empty or label_df.empty:
        return pd.DataFrame()

    score_col = "score" if "score" in score_df.columns else first_available_column(score_df.columns, score_col_candidates)
    if not score_col:
        raise ValueError(
            "Score frame does not contain a usable score column. "
            f"Tried: {', '.join(score_col_candidates)}."
        )
    if target_col not in label_df.columns or forward_return_col not in label_df.columns:
        raise ValueError(
            f"Label frame must contain {target_col!r} and {forward_return_col!r}."
        )

    score_panel = score_df[["date", "symbol", score_col]].copy().rename(columns={score_col: "score"})
    score_panel["date"] = pd.to_datetime(score_panel["date"], errors="coerce")
    score_panel["symbol"] = score_panel["symbol"].astype(str).str.strip().str.upper()
    score_panel["score"] = pd.to_numeric(score_panel["score"], errors="coerce")
    score_panel = score_panel.dropna(subset=["date", "symbol", "score"])

    label_panel = label_df[["date", "symbol", target_col, forward_return_col]].copy()
    label_panel["date"] = pd.to_datetime(label_panel["date"], errors="coerce")
    label_panel["symbol"] = label_panel["symbol"].astype(str).str.strip().str.upper()
    label_panel[target_col] = pd.to_numeric(label_panel[target_col], errors="coerce")
    label_panel[forward_return_col] = pd.to_numeric(label_panel[forward_return_col], errors="coerce")
    label_panel = label_panel.dropna(subset=["date", "symbol", target_col, forward_return_col])

    if start_date:
        start_ts = pd.Timestamp(str(start_date))
        score_panel = score_panel[score_panel["date"] >= start_ts].copy()
        label_panel = label_panel[label_panel["date"] >= start_ts].copy()
    if end_date:
        end_ts = pd.Timestamp(str(end_date))
        score_panel = score_panel[score_panel["date"] <= end_ts].copy()
        label_panel = label_panel[label_panel["date"] <= end_ts].copy()

    out = score_panel.merge(label_panel, on=["date", "symbol"], how="inner")
    if out.empty:
        return out
    out["variant_name"] = str(variant_name or "")
    out["fold_name"] = str(fold_name or "")
    out["feature_scope"] = str(feature_scope or "")
    out["variant_kind"] = str(variant_kind or "")
    out["variant_label"] = str(variant_label or "")
    metadata_lookup = (
        {str(key).strip().upper(): dict(value) for key, value in dict(symbol_metadata_lookup or {}).items()}
        if symbol_metadata_lookup is not None
        else build_symbol_metadata_lookup(out["symbol"].unique().tolist())
    )
    out["sector"] = out["symbol"].map(lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("sector") or "Unknown"))
    out["industry"] = out["symbol"].map(lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("industry") or "Unknown"))
    out["exchange"] = out["symbol"].map(lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("exchange") or "Unknown"))
    out["country"] = out["symbol"].map(lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("country") or "Unknown"))
    out["instrument_type"] = out["symbol"].map(
        lambda symbol: str(metadata_lookup.get(str(symbol), {}).get("instrument_type") or "stock")
    )
    out["is_etf"] = out["instrument_type"].eq("etf").astype(int)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def assign_cross_sectional_buckets(
    panel_df: pd.DataFrame,
    *,
    bucket_count: int = 10,
    higher_score_is_better: bool = True,
) -> pd.DataFrame:
    if panel_df.empty:
        return panel_df.copy()
    resolved_bucket_count = max(int(bucket_count), 2)
    out = panel_df.copy()
    out["bucket"] = 0
    out["score_rank"] = 0
    group_cols = [column for column in ["variant_name", "fold_name", "date"] if column in out.columns]
    if "date" not in group_cols:
        group_cols.append("date")
    for _keys, group in out.groupby(group_cols, sort=True):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna()
        if scores.empty:
            continue
        rank_source = scores if higher_score_is_better else (-1.0 * scores)
        rank_positions = rank_source.rank(method="first", ascending=True)
        display_rank = rank_source.rank(method="first", ascending=False).astype(int)
        bucket_labels = (((rank_positions - 1.0) * float(resolved_bucket_count)) // float(len(rank_positions))).astype(int) + 1
        out.loc[bucket_labels.index, "bucket"] = bucket_labels.astype(int)
        out.loc[display_rank.index, "score_rank"] = display_rank.astype(int)
    return out


def _group_columns(df: pd.DataFrame, include_fold: bool = True) -> list[str]:
    columns = ["variant_name"]
    if include_fold and "fold_name" in df.columns:
        columns.append("fold_name")
    for optional in ["variant_kind", "variant_label", "feature_scope"]:
        if optional in df.columns:
            columns.append(optional)
    return [column for column in columns if column in df.columns]


def _spearman(series_a: pd.Series, series_b: pd.Series) -> float | None:
    valid = pd.concat([series_a, series_b], axis=1).dropna()
    if len(valid) < 2:
        return None
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    value = valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())
    return None if pd.isna(value) else float(value)


def compute_ranking_summary_rows(
    bucketed_df: pd.DataFrame,
    *,
    bucket_count: int = 10,
    target_col: str = "future_rank_pct",
    forward_return_col: str = "trade_return",
) -> list[dict[str, Any]]:
    if bucketed_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_cols = _group_columns(bucketed_df, include_fold=True)
    for keys, group in bucketed_df.groupby(group_cols, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = {column: key_values[idx] for idx, column in enumerate(group_cols)}
        daily_ic: list[float] = []
        daily_target_ic: list[float] = []
        daily_long_short_spread: list[float] = []
        daily_top_bucket_return: list[float] = []
        daily_bottom_bucket_return: list[float] = []
        for _date_value, date_group in group.groupby("date", sort=True):
            ic_value = _spearman(date_group["score"], date_group[forward_return_col])
            if ic_value is not None:
                daily_ic.append(ic_value)
            target_ic_value = _spearman(date_group["score"], date_group[target_col])
            if target_ic_value is not None:
                daily_target_ic.append(target_ic_value)
            top_mean = pd.to_numeric(
                date_group.loc[date_group["bucket"] == int(bucket_count), forward_return_col],
                errors="coerce",
            ).mean()
            bottom_mean = pd.to_numeric(
                date_group.loc[date_group["bucket"] == 1, forward_return_col],
                errors="coerce",
            ).mean()
            if pd.notna(top_mean):
                daily_top_bucket_return.append(float(top_mean))
            if pd.notna(bottom_mean):
                daily_bottom_bucket_return.append(float(bottom_mean))
            if pd.notna(top_mean) and pd.notna(bottom_mean):
                daily_long_short_spread.append(float(top_mean) - float(bottom_mean))
        rows.append(
            {
                **key_map,
                "rebalance_dates": int(group["date"].nunique()),
                "scored_rows": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "mean_spearman_ic": _round_float(sum(daily_ic) / len(daily_ic) if daily_ic else 0.0),
                "mean_target_spearman_ic": _round_float(sum(daily_target_ic) / len(daily_target_ic) if daily_target_ic else 0.0),
                "mean_long_short_spread": _round_float(
                    sum(daily_long_short_spread) / len(daily_long_short_spread) if daily_long_short_spread else 0.0
                ),
                "mean_top_bucket_return": _round_float(
                    sum(daily_top_bucket_return) / len(daily_top_bucket_return) if daily_top_bucket_return else 0.0
                ),
                "mean_bottom_bucket_return": _round_float(
                    sum(daily_bottom_bucket_return) / len(daily_bottom_bucket_return) if daily_bottom_bucket_return else 0.0
                ),
                "top_bucket_selection_count": int((group["bucket"] == int(bucket_count)).sum()),
                "bottom_bucket_selection_count": int((group["bucket"] == 1).sum()),
            }
        )
    return rows


def aggregate_ranking_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("variant_name", "variant_kind", "variant_label", "feature_scope"),
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return []
    keys = [column for column in list(group_keys) if column in df.columns]
    if not keys:
        keys = ["variant_name"]
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            fold_count=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
            rebalance_dates=("rebalance_dates", "sum"),
            scored_rows=("scored_rows", "sum"),
            symbols=("symbols", "max"),
            mean_spearman_ic=("mean_spearman_ic", "mean"),
            mean_target_spearman_ic=("mean_target_spearman_ic", "mean"),
            mean_long_short_spread=("mean_long_short_spread", "mean"),
            mean_top_bucket_return=("mean_top_bucket_return", "mean"),
            mean_bottom_bucket_return=("mean_bottom_bucket_return", "mean"),
            top_bucket_selection_count=("top_bucket_selection_count", "sum"),
            bottom_bucket_selection_count=("bottom_bucket_selection_count", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["mean_spearman_ic", "mean_long_short_spread", "variant_name"], ascending=[False, False, True])
    return [
        {
            **dict(row),
            "fold_count": int(row["fold_count"]),
            "rebalance_dates": int(row["rebalance_dates"]),
            "scored_rows": int(row["scored_rows"]),
            "symbols": int(row["symbols"]),
            "top_bucket_selection_count": int(row["top_bucket_selection_count"]),
            "bottom_bucket_selection_count": int(row["bottom_bucket_selection_count"]),
            "mean_spearman_ic": _round_float(row["mean_spearman_ic"]),
            "mean_target_spearman_ic": _round_float(row["mean_target_spearman_ic"]),
            "mean_long_short_spread": _round_float(row["mean_long_short_spread"]),
            "mean_top_bucket_return": _round_float(row["mean_top_bucket_return"]),
            "mean_bottom_bucket_return": _round_float(row["mean_bottom_bucket_return"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


def compute_bucket_return_rows(
    bucketed_df: pd.DataFrame,
    *,
    target_col: str = "future_rank_pct",
    forward_return_col: str = "trade_return",
) -> list[dict[str, Any]]:
    if bucketed_df.empty:
        return []
    group_cols = _group_columns(bucketed_df, include_fold=True)
    grouped = (
        bucketed_df.groupby(group_cols + ["bucket"], dropna=False)
        .agg(
            avg_forward_return=(forward_return_col, "mean"),
            avg_target_rank=(target_col, "mean"),
            avg_score=("score", "mean"),
            selection_count=("symbol", "size"),
            rebalance_dates=("date", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(group_cols + ["bucket"])
    return [
        {
            **dict(row),
            "bucket": int(row["bucket"]),
            "selection_count": int(row["selection_count"]),
            "rebalance_dates": int(row["rebalance_dates"]),
            "avg_forward_return": _round_float(row["avg_forward_return"]),
            "avg_target_rank": _round_float(row["avg_target_rank"]),
            "avg_score": _round_float(row["avg_score"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


def aggregate_bucket_return_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("variant_name", "variant_kind", "variant_label", "feature_scope", "bucket"),
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return []
    keys = [column for column in list(group_keys) if column in df.columns]
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            fold_count=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
            avg_forward_return=("avg_forward_return", "mean"),
            avg_target_rank=("avg_target_rank", "mean"),
            avg_score=("avg_score", "mean"),
            selection_count=("selection_count", "sum"),
            rebalance_dates=("rebalance_dates", "sum"),
        )
        .reset_index()
        .sort_values(["variant_name", "bucket"])
    )
    return [
        {
            **dict(row),
            "bucket": int(row["bucket"]),
            "fold_count": int(row["fold_count"]),
            "selection_count": int(row["selection_count"]),
            "rebalance_dates": int(row["rebalance_dates"]),
            "avg_forward_return": _round_float(row["avg_forward_return"]),
            "avg_target_rank": _round_float(row["avg_target_rank"]),
            "avg_score": _round_float(row["avg_score"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


def top_bucket_rows(bucketed_df: pd.DataFrame, *, bucket_count: int = 10) -> list[dict[str, Any]]:
    if bucketed_df.empty:
        return []
    top_df = bucketed_df[bucketed_df["bucket"] == int(bucket_count)].copy()
    if top_df.empty:
        return []
    top_df = top_df.sort_values(["variant_name", "fold_name", "date", "symbol"])
    return top_df.to_dict(orient="records")


def compute_top_bucket_cohort_rows(
    bucketed_df: pd.DataFrame,
    *,
    bucket_count: int = 10,
) -> list[dict[str, Any]]:
    if bucketed_df.empty:
        return []
    top_df = bucketed_df[bucketed_df["bucket"] == int(bucket_count)].copy()
    if top_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_cols = _group_columns(top_df, include_fold=True)
    for cohort_kind, cohort_col in (("sector", "sector"), ("instrument_type", "instrument_type")):
        grouped = (
            top_df.groupby(group_cols + [cohort_col], dropna=False)
            .agg(selection_count=("symbol", "size"), rebalance_dates=("date", "nunique"))
            .reset_index()
        )
        totals = grouped.groupby(group_cols, dropna=False)["selection_count"].sum().rename("total_selection_count").reset_index()
        grouped = grouped.merge(totals, on=group_cols, how="left")
        for row in grouped.to_dict(orient="records"):
            rows.append(
                {
                    **{column: row[column] for column in group_cols},
                    "cohort_kind": cohort_kind,
                    "cohort_value": str(row[cohort_col] or "Unknown"),
                    "selection_count": int(row["selection_count"]),
                    "rebalance_dates": int(row["rebalance_dates"]),
                    "selection_share": _round_float(
                        float(row["selection_count"]) / float(row["total_selection_count"])
                        if float(row["total_selection_count"]) > 0
                        else 0.0
                    ),
                }
            )
    return sorted(rows, key=lambda row: (str(row.get("variant_name") or ""), str(row.get("cohort_kind") or ""), -int(row.get("selection_count") or 0), str(row.get("cohort_value") or "")))


def aggregate_top_bucket_cohort_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("variant_name", "variant_kind", "variant_label", "feature_scope", "cohort_kind", "cohort_value"),
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return []
    keys = [column for column in list(group_keys) if column in df.columns]
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            fold_count=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
            selection_count=("selection_count", "sum"),
            rebalance_dates=("rebalance_dates", "sum"),
            selection_share=("selection_share", "mean"),
        )
        .reset_index()
        .sort_values(["variant_name", "cohort_kind", "selection_share", "selection_count", "cohort_value"], ascending=[True, True, False, False, True])
    )
    return [
        {
            **dict(row),
            "fold_count": int(row["fold_count"]),
            "selection_count": int(row["selection_count"]),
            "rebalance_dates": int(row["rebalance_dates"]),
            "selection_share": _round_float(row["selection_share"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


def compute_bucket_overlap_rows(
    left_bucketed_df: pd.DataFrame,
    right_bucketed_df: pd.DataFrame,
    *,
    bucket_count: int = 10,
    left_variant_name: str = "",
    right_variant_name: str = "",
) -> list[dict[str, Any]]:
    if left_bucketed_df.empty or right_bucketed_df.empty:
        return []
    left_top = left_bucketed_df[left_bucketed_df["bucket"] == int(bucket_count)].copy()
    right_top = right_bucketed_df[right_bucketed_df["bucket"] == int(bucket_count)].copy()
    if left_top.empty or right_top.empty:
        return []
    join_keys = [column for column in ["fold_name", "date"] if column in left_top.columns and column in right_top.columns]
    if "date" not in join_keys:
        join_keys.append("date")
    left_sets = (
        left_top.groupby(join_keys, dropna=False)["symbol"]
        .apply(lambda values: {str(value) for value in values})
        .to_dict()
    )
    right_sets = (
        right_top.groupby(join_keys, dropna=False)["symbol"]
        .apply(lambda values: {str(value) for value in values})
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    common_keys = sorted(set(left_sets.keys()) & set(right_sets.keys()))
    for key in common_keys:
        key_values = key if isinstance(key, tuple) else (key,)
        key_map = {join_keys[idx]: key_values[idx] for idx in range(len(join_keys))}
        left_symbols = left_sets.get(key, set())
        right_symbols = right_sets.get(key, set())
        union = left_symbols | right_symbols
        overlap = left_symbols & right_symbols
        rows.append(
            {
                **key_map,
                "left_variant_name": str(left_variant_name or ""),
                "right_variant_name": str(right_variant_name or ""),
                "left_count": int(len(left_symbols)),
                "right_count": int(len(right_symbols)),
                "overlap_count": int(len(overlap)),
                "union_count": int(len(union)),
                "jaccard": _round_float(float(len(overlap)) / float(len(union)) if union else 0.0),
            }
        )
    return rows


def aggregate_bucket_overlap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("left_variant_name", "right_variant_name"),
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return []
    keys = [column for column in list(group_keys) if column in df.columns]
    grouped = (
        df.groupby(keys, dropna=False)
        .agg(
            fold_count=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
            observations=("date", "size"),
            left_count=("left_count", "mean"),
            right_count=("right_count", "mean"),
            overlap_count=("overlap_count", "mean"),
            jaccard=("jaccard", "mean"),
        )
        .reset_index()
        .sort_values(["jaccard", "right_variant_name"], ascending=[False, True])
    )
    return [
        {
            **dict(row),
            "fold_count": int(row["fold_count"]),
            "observations": int(row["observations"]),
            "left_count": _round_float(row["left_count"]),
            "right_count": _round_float(row["right_count"]),
            "overlap_count": _round_float(row["overlap_count"]),
            "jaccard": _round_float(row["jaccard"]),
        }
        for row in grouped.to_dict(orient="records")
    ]


def compute_top_bucket_stability_rows(
    bucketed_df: pd.DataFrame,
    *,
    bucket_count: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if bucketed_df.empty:
        return [], []
    top_df = bucketed_df[bucketed_df["bucket"] == int(bucket_count)].copy()
    if top_df.empty:
        return [], []
    stability_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for variant_name, variant_group in top_df.groupby("variant_name", sort=True):
        fold_sets: dict[str, set[str]] = {}
        for fold_name, fold_group in variant_group.groupby("fold_name", sort=True):
            fold_sets[str(fold_name or "all")] = {str(symbol) for symbol in fold_group["symbol"].astype(str).tolist()}
        pairwise = []
        fold_items = sorted(fold_sets.items())
        for (_left_name, left_set), (_right_name, right_set) in combinations(fold_items, 2):
            union = left_set | right_set
            pairwise.append(float(len(left_set & right_set)) / float(len(union)) if union else 0.0)
        stability_rows.append(
            {
                "variant_name": str(variant_name or ""),
                "fold_count": int(len(fold_sets)),
                "unique_symbols_total": int(variant_group["symbol"].nunique()),
                "avg_symbols_per_fold": _round_float(
                    sum(len(symbols) for symbols in fold_sets.values()) / float(len(fold_sets)) if fold_sets else 0.0
                ),
                "mean_pairwise_jaccard": _round_float(sum(pairwise) / len(pairwise) if pairwise else 1.0),
            }
        )
        grouped_symbols = (
            variant_group.groupby("symbol", dropna=False)
            .agg(
                folds_selected=("fold_name", lambda values: len({str(value) for value in values if str(value).strip()})),
                top_bucket_dates=("date", "nunique"),
                sector=("sector", "first"),
                instrument_type=("instrument_type", "first"),
            )
            .reset_index()
            .sort_values(["folds_selected", "top_bucket_dates", "symbol"], ascending=[False, False, True])
        )
        for row in grouped_symbols.to_dict(orient="records"):
            symbol_rows.append(
                {
                    "variant_name": str(variant_name or ""),
                    "symbol": str(row["symbol"]),
                    "folds_selected": int(row["folds_selected"]),
                    "top_bucket_dates": int(row["top_bucket_dates"]),
                    "sector": str(row["sector"] or "Unknown"),
                    "instrument_type": str(row["instrument_type"] or "stock"),
                }
            )
    return stability_rows, symbol_rows


__all__ = [
    "DEFAULT_SCORE_COL_CANDIDATES",
    "aggregate_bucket_overlap_rows",
    "aggregate_bucket_return_rows",
    "aggregate_ranking_summary_rows",
    "aggregate_top_bucket_cohort_rows",
    "assign_cross_sectional_buckets",
    "build_expression_score_frame",
    "build_signal_ranking_panel",
    "build_symbol_metadata_lookup",
    "compute_bucket_overlap_rows",
    "compute_bucket_return_rows",
    "compute_ranking_summary_rows",
    "compute_top_bucket_cohort_rows",
    "compute_top_bucket_stability_rows",
    "top_bucket_rows",
]
