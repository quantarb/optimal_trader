from __future__ import annotations

from math import sqrt
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _round_float(value: Any, digits: int = 8) -> float:
    return round(_safe_float(value), digits)


def load_daily_return_frame(
    source: Any,
    *,
    series_name: str = "",
    series_kind: str = "",
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif hasattr(source, "content"):
        frame = pd.DataFrame(list((getattr(source, "content", {}) or {}).get("daily_rows") or []))
    else:
        frame = pd.DataFrame(list(source or []))
    if frame.empty:
        return pd.DataFrame(columns=["date", "net_daily_return", "turnover", "series_name", "series_kind"])
    out = frame.copy()
    if "net_daily_return" not in out.columns and "daily_return" in out.columns:
        out["net_daily_return"] = out["daily_return"]
    if "net_daily_return" not in out.columns:
        out["net_daily_return"] = 0.0
    if "turnover" not in out.columns:
        out["turnover"] = 0.0
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["net_daily_return"] = pd.to_numeric(out["net_daily_return"], errors="coerce").fillna(0.0)
    out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    resolved_name = str(series_name or (out["series_name"].iloc[0] if "series_name" in out.columns and len(out) else ""))
    resolved_kind = str(series_kind or (out["series_kind"].iloc[0] if "series_kind" in out.columns and len(out) else ""))
    out["series_name"] = resolved_name
    out["series_kind"] = resolved_kind
    return out


def combine_named_daily_return_frames(
    sources: Mapping[str, Any],
    *,
    series_kind: str = "",
) -> pd.DataFrame:
    frames = [
        load_daily_return_frame(source, series_name=str(name), series_kind=series_kind)
        for name, source in dict(sources or {}).items()
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "net_daily_return", "turnover", "series_name", "series_kind"])
    return pd.concat(frames, ignore_index=True).sort_values(["series_name", "date"]).reset_index(drop=True)


def summarize_return_frame(
    frame_or_source: Any,
    *,
    series_name: str = "",
    series_kind: str = "",
) -> dict[str, Any]:
    frame = load_daily_return_frame(frame_or_source, series_name=series_name, series_kind=series_kind)
    if frame.empty:
        return {
            "series_name": str(series_name or ""),
            "series_kind": str(series_kind or ""),
            "days": 0,
            "start_date": "",
            "end_date": "",
            "sharpe": 0.0,
            "total_return": 0.0,
            "final_equity": 1.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "total_turnover": 0.0,
            "positive_days": 0,
            "negative_days": 0,
        }
    returns = frame["net_daily_return"].astype(float)
    turnover = frame["turnover"].astype(float)
    equity_curve = (1.0 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    volatility = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    sharpe = ((float(returns.mean()) / volatility) * sqrt(252.0)) if volatility > 1e-12 else 0.0
    resolved_name = str(series_name or (frame["series_name"].iloc[0] if "series_name" in frame.columns and len(frame) else ""))
    resolved_kind = str(series_kind or (frame["series_kind"].iloc[0] if "series_kind" in frame.columns and len(frame) else ""))
    return {
        "series_name": resolved_name,
        "series_kind": resolved_kind,
        "days": int(len(frame)),
        "start_date": str(frame["date"].min().strftime("%Y-%m-%d")),
        "end_date": str(frame["date"].max().strftime("%Y-%m-%d")),
        "sharpe": _round_float(sharpe),
        "total_return": _round_float(float(equity_curve.iloc[-1] - 1.0)),
        "final_equity": _round_float(float(equity_curve.iloc[-1])),
        "max_drawdown": _round_float(float(drawdown.min()) if len(drawdown) else 0.0),
        "avg_turnover": _round_float(float(turnover.mean()) if len(turnover) else 0.0),
        "total_turnover": _round_float(float(turnover.sum()) if len(turnover) else 0.0),
        "positive_days": int((returns > 0.0).sum()),
        "negative_days": int((returns < 0.0).sum()),
    }


def compute_factor_correlation_rows(factor_frame_or_sources: Any) -> list[dict[str, Any]]:
    if isinstance(factor_frame_or_sources, pd.DataFrame):
        factor_frame = factor_frame_or_sources.copy()
    else:
        factor_frame = combine_named_daily_return_frames(factor_frame_or_sources, series_kind="factor")
    if factor_frame.empty:
        return []
    pivot = (
        factor_frame.pivot_table(
            index="date",
            columns="series_name",
            values="net_daily_return",
            aggfunc="last",
        )
        .sort_index()
    )
    if pivot.empty:
        return []
    corr = pivot.corr()
    rows: list[dict[str, Any]] = []
    for left_factor in corr.index.tolist():
        for right_factor in corr.columns.tolist():
            rows.append(
                {
                    "left_factor": str(left_factor),
                    "right_factor": str(right_factor),
                    "correlation": _round_float(corr.loc[left_factor, right_factor]),
                }
            )
    return rows


def compute_strategy_factor_exposure_rows(
    strategy_sources: Mapping[str, Any],
    factor_sources: Mapping[str, Any],
) -> list[dict[str, Any]]:
    factor_frame = combine_named_daily_return_frames(factor_sources, series_kind="factor")
    if factor_frame.empty:
        return []
    factor_pivot = (
        factor_frame.pivot_table(
            index="date",
            columns="series_name",
            values="net_daily_return",
            aggfunc="last",
        )
        .sort_index()
    )
    factor_names = [str(column) for column in factor_pivot.columns.tolist()]
    rows: list[dict[str, Any]] = []
    for strategy_name, strategy_source in dict(strategy_sources or {}).items():
        strategy_frame = load_daily_return_frame(strategy_source, series_name=str(strategy_name), series_kind="strategy")
        if strategy_frame.empty:
            continue
        merged = factor_pivot.join(
            strategy_frame.set_index("date")["net_daily_return"].rename("strategy_return"),
            how="inner",
        ).dropna()
        if merged.empty:
            continue

        y = merged["strategy_return"].astype(float).to_numpy()
        X = merged[factor_names].astype(float).to_numpy()
        X_design = np.column_stack([np.ones(len(merged)), X])
        coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        fitted = X_design @ coeffs
        residual = y - fitted
        y_mean = float(y.mean()) if len(y) else 0.0
        ss_tot = float(((y - y_mean) ** 2).sum())
        ss_res = float((residual ** 2).sum())
        residual_series = pd.Series(residual, index=merged.index, dtype=float)
        residual_vol = float(residual_series.std(ddof=0)) if len(residual_series) > 1 else 0.0
        residual_sharpe = (
            (float(residual_series.mean()) / residual_vol) * sqrt(252.0)
            if residual_vol > 1e-12
            else 0.0
        )
        row = {
            "strategy_name": str(strategy_name),
            "observations": int(len(merged)),
            "start_date": str(merged.index.min().strftime("%Y-%m-%d")),
            "end_date": str(merged.index.max().strftime("%Y-%m-%d")),
            "alpha": _round_float(coeffs[0]),
            "r_squared": _round_float(1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0),
            "residual_total_return": _round_float(float((1.0 + residual_series).cumprod().iloc[-1] - 1.0)),
            "residual_sharpe": _round_float(residual_sharpe),
            "residual_mean_daily_return": _round_float(float(residual_series.mean())),
        }
        for factor_index, factor_name in enumerate(factor_names, start=1):
            row[f"beta_{str(factor_name).lower()}"] = _round_float(coeffs[factor_index])
        rows.append(row)
    return rows


__all__ = [
    "combine_named_daily_return_frames",
    "compute_factor_correlation_rows",
    "compute_strategy_factor_exposure_rows",
    "load_daily_return_frame",
    "summarize_return_frame",
]
