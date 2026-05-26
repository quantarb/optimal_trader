from __future__ import annotations

from datetime import date
from io import StringIO
from typing import Any

import pandas as pd
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.urls import reverse

from app.live_trade_leaderboard import (
    build_robinhood_trade_sheet,
    default_live_trade_config,
    latest_scored_staleness_reason,
    load_saved_leaderboard,
    load_saved_latest_scored,
    run_live_trade_leaderboard_build,
)
from app.optimal_trade_lookup import OptimalTradeQuery, find_nearest_optimal_trades
from trading.robinhood import (
    build_robinhood_option_trade_plan,
    load_robinhood_open_option_orders,
    load_robinhood_option_positions,
    robinhood_login,
    submit_robinhood_option_orders,
)


LEADERBOARD_PAGE_SIZE = 50


def trading_leaderboard(request: HttpRequest) -> HttpResponse:
    default_cfg = default_live_trade_config()
    build_log_lines: list[str] = []
    build_form = _leaderboard_build_form(request, default_cfg)
    trade_sheet_form = _trade_sheet_form(request)
    robinhood_form = _robinhood_option_form(request, default_cfg)
    robinhood_plan: dict[str, Any] | None = None
    robinhood_order_results_table = None
    robinhood_preview_table = None

    if request.method == "POST" and request.POST.get("action") == "build_leaderboard":
        cfg = _leaderboard_config_from_form(build_form)

        def _progress_logger(message: str) -> None:
            build_log_lines.append(str(message))

        try:
            result = run_live_trade_leaderboard_build(config=cfg, progress_logger=_progress_logger)
        except Exception as exc:
            messages.error(request, f"Leaderboard build failed: {type(exc).__name__}: {exc}")
        else:
            messages.success(
                request,
                f"Leaderboard build complete for {pd.Timestamp(result.latest_date).date().isoformat()} "
                f"with {len(result.leaderboard):,} scored symbols.",
            )

    saved = load_saved_leaderboard()
    leaderboard = pd.DataFrame()
    meta: dict[str, Any] = {}
    if saved is not None:
        leaderboard, meta = saved
        leaderboard = _recompute_eligibility(leaderboard, threshold=float(trade_sheet_form["threshold"]))
        leaderboard = _rank_leaderboard_for_display(leaderboard)

    page = _coerce_int(request.GET.get("page"), default=1, minimum=1)
    page_rows, page_meta = _paginate_frame(leaderboard, page=page, page_size=LEADERBOARD_PAGE_SIZE)
    display_columns = [
        "Rank",
        "Scored Date",
        "Symbol",
        "Direction",
        "Eligible",
        "Classifier Score",
        "Regressor Score",
        "Autoencoder Score",
        "Combined Score",
        "Similar Trades",
    ]
    table = _frame_table(page_rows, display_columns, link_columns={"Similar Trades": "Similar Trades"})

    trade_sheet_table = None
    trade_sheet_summary: dict[str, Any] | None = None
    if request.method == "POST" and request.POST.get("action") == "build_trade_sheet" and not leaderboard.empty:
        sheet, trade_sheet_summary = build_robinhood_trade_sheet(
            leaderboard=leaderboard,
            top_k=int(trade_sheet_form["top_k"]),
            account_size=float(trade_sheet_form["account_size"]),
            eligible_only=bool(trade_sheet_form["eligible_only"]),
            include_shorts=bool(trade_sheet_form["include_shorts"]),
        )
        trade_sheet_table = _frame_table(
            sheet,
            [
                "Rank",
                "Symbol",
                "Scored Date",
                "Direction",
                "Robinhood Action",
                "Price",
                "Target Weight",
                "Target Dollars",
                "Estimated Shares",
                "Robinhood URL",
            ],
            link_columns={"Robinhood URL": "Open"},
        )

    artifact_dir = (
        (meta.get("config") or {}).get("runtime", {}).get("artifact_dir")
        or meta.get("artifact_dir")
        or default_cfg["runtime"]["artifact_dir"]
    )

    if request.method == "POST" and str(request.POST.get("action") or "").startswith("robinhood_"):
        if not leaderboard.empty:
            if request.POST.get("action") == "robinhood_build_option_plan":
                robinhood_plan = _build_robinhood_option_plan(request, robinhood_form, artifact_dir=str(artifact_dir))
            elif request.POST.get("action") == "robinhood_preview_option_orders":
                orders = _load_robinhood_session_orders(request)
                if orders.empty:
                    messages.warning(request, "No Robinhood option order preview is saved yet. Generate a plan first.")
                else:
                    robinhood_preview_table = _frame_table(orders)
            elif request.POST.get("action") == "robinhood_submit_option_orders":
                orders = _load_robinhood_session_orders(request)
                if orders.empty:
                    messages.error(request, "No Robinhood option orders are saved for submission. Generate a plan first.")
                elif not _coerce_bool(request.POST.get("confirm_submit"), default=False):
                    messages.error(request, "Check the live-order confirmation box before submitting Robinhood orders.")
                    robinhood_preview_table = _frame_table(orders)
                else:
                    try:
                        result = submit_robinhood_option_orders(
                            orders_df=orders,
                            account_number=str(robinhood_form["account_number"] or "") or None,
                            time_in_force=str(robinhood_form["time_in_force"] or "gtc"),
                        )
                    except Exception as exc:
                        messages.error(request, f"Robinhood option order submit failed: {type(exc).__name__}: {exc}")
                        robinhood_preview_table = _frame_table(orders)
                    else:
                        submitted = int(pd.Series(result.get("submitted"), dtype="boolean").fillna(False).sum()) if not result.empty and "submitted" in result.columns else 0
                        messages.success(request, f"Robinhood option submit complete: {submitted:,}/{len(result):,} accepted by the submit call.")
                        robinhood_order_results_table = _frame_table(result.drop(columns=["response"], errors="ignore"))
        else:
            messages.error(request, "Build or load a leaderboard before generating Robinhood option orders.")

    stale_reason = latest_scored_staleness_reason(artifact_dir=artifact_dir)
    summary = _leaderboard_summary(leaderboard, meta)
    return render(
        request,
        "trading/leaderboard.html",
        {
            "build_form": build_form,
            "trade_sheet_form": trade_sheet_form,
            "robinhood_form": robinhood_form,
            "build_log_lines": build_log_lines,
            "leaderboard_table": table,
            "leaderboard_summary": summary,
            "page_meta": page_meta,
            "meta": meta,
            "artifact_dir": artifact_dir,
            "stale_reason": stale_reason,
            "trade_sheet_table": trade_sheet_table,
            "trade_sheet_summary": trade_sheet_summary,
            "robinhood_plan": _robinhood_plan_context(robinhood_plan),
            "robinhood_has_saved_orders": not _load_robinhood_session_orders(request).empty,
            "robinhood_preview_table": robinhood_preview_table,
            "robinhood_order_results_table": robinhood_order_results_table,
            "has_leaderboard": not leaderboard.empty,
        },
    )


def similar_trades(request: HttpRequest) -> HttpResponse:
    form = _similar_trades_form(request)
    result = None
    error = ""
    if request.method == "POST" or str(request.GET.get("symbol") or "").strip():
        try:
            result = find_nearest_optimal_trades(
                OptimalTradeQuery(
                    symbol=str(form["symbol"]).strip().upper(),
                    as_of_date=str(form["as_of_date"] or "") or None,
                    top_k=int(form["top_k"]),
                    query_lookback_years=int(form["query_lookback_years"]),
                    reference_start_date=str(form["reference_start_date"]),
                    min_profit_pct_points=float(form["min_profit_pct_points"]),
                    download_missing_prices=False,
                    artifact_dir=str(form["artifact_dir"]),
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            messages.error(request, f"Similar-trades lookup failed: {error}")

    context: dict[str, Any] = {"form": form, "error": error, "result": result}
    if result is not None:
        context.update(
            {
                "query_summary_table": _frame_table(result.query_summary),
                "indicator_table": _frame_table(result.indicator_summary),
                "nearest_trades_table": _frame_table(result.nearest_trades),
                "feature_attribution_table": _frame_table(result.feature_attribution),
                "model_predictions": result.model_predictions,
                "metadata": result.metadata,
            }
        )
    return render(request, "trading/similar_trades.html", context)


def _leaderboard_build_form(request: HttpRequest, default_cfg: dict[str, Any]) -> dict[str, Any]:
    source = request.POST if request.method == "POST" else request.GET
    return {
        "data_start": str(source.get("data_start") or default_cfg["dates"]["data_start"]),
        "data_end": str(source.get("data_end") or pd.Timestamp.today().strftime("%Y-%m-%d")),
        "min_market_cap_b": _coerce_float(
            source.get("min_market_cap_b"),
            default=float(default_cfg["universe"]["min_market_cap"]) / 1_000_000_000.0,
            minimum=0.0,
        ),
        "refresh_fmp": _coerce_bool(source.get("refresh_fmp"), default=True),
    }


def _leaderboard_config_from_form(form: dict[str, Any]) -> dict[str, Any]:
    refresh_fmp = bool(form["refresh_fmp"])
    return {
        "dates": {
            "data_start": str(pd.Timestamp(form["data_start"]).date()),
            "data_end": str(pd.Timestamp(form["data_end"]).date()),
        },
        "universe": {
            "min_market_cap": float(form["min_market_cap_b"]) * 1_000_000_000.0,
        },
        "fmp_refresh": {
            "enabled": refresh_fmp,
            "refresh_symbol_sections_before_build": refresh_fmp,
            "refresh_macro_before_build": refresh_fmp,
            "existing_historical_sections_only": True,
            "verbose": False,
        },
    }


def _trade_sheet_form(request: HttpRequest) -> dict[str, Any]:
    source = request.POST if request.method == "POST" else request.GET
    return {
        "top_k": _coerce_int(source.get("top_k"), default=20, minimum=1),
        "account_size": _coerce_float(source.get("account_size"), default=10_000.0, minimum=0.0),
        "threshold": _coerce_float(source.get("threshold"), default=0.50, minimum=0.0),
        "eligible_only": _coerce_bool(source.get("eligible_only"), default=True),
        "include_shorts": _coerce_bool(source.get("include_shorts"), default=True),
    }


def _robinhood_option_form(request: HttpRequest, default_cfg: dict[str, Any]) -> dict[str, Any]:
    source = request.POST if request.method == "POST" else request.GET
    strategy_cfg = dict(default_cfg.get("strategy") or {})
    return {
        "top_k": _coerce_int(source.get("rh_top_k"), default=20, minimum=1),
        "account_equity": _coerce_float(source.get("rh_account_equity"), default=10_000.0, minimum=0.0),
        "strategy_allocation": _coerce_float(source.get("rh_strategy_allocation"), default=10_000.0, minimum=0.0),
        "component_threshold": _coerce_float(
            source.get("rh_component_threshold"),
            default=float(strategy_cfg.get("component_threshold", 0.50)),
            minimum=0.0,
        ),
        "score_col": str(source.get("rh_score_col") or strategy_cfg.get("score_col") or "buy_score_mean_raw_pct6"),
        "option_bucket": str(source.get("rh_option_bucket") or "otm_option"),
        "tenor_days": _coerce_int(source.get("rh_tenor_days"), default=90, minimum=1),
        "max_contracts_per_position": None,
        "account_number": str(source.get("rh_account_number") or "").strip(),
        "mfa_code": str(source.get("rh_mfa_code") or "").strip(),
        "store_session": _coerce_bool(source.get("rh_store_session"), default=True),
        "login_first": _coerce_bool(source.get("rh_login_first"), default=True),
        "time_in_force": str(source.get("rh_time_in_force") or "gtc").strip().lower(),
    }


def _build_robinhood_option_plan(
    request: HttpRequest,
    form: dict[str, Any],
    *,
    artifact_dir: str,
) -> dict[str, Any] | None:
    latest_scored = load_saved_latest_scored(artifact_dir=artifact_dir, require_fresh=True)
    if latest_scored is None or latest_scored.empty:
        messages.error(request, "No fresh latest-scored artifact is available. Rebuild the leaderboard before generating Robinhood orders.")
        return None
    latest_scored = _normalize_latest_scored_for_robinhood(latest_scored)
    account_number = str(form["account_number"] or "") or None
    try:
        if bool(form["login_first"]):
            robinhood_login(
                mfa_code=str(form["mfa_code"] or "") or None,
                store_session=bool(form["store_session"]),
            )
        current_options = load_robinhood_option_positions(account_number=account_number)
        pending_orders = load_robinhood_open_option_orders(account_number=account_number)
        plan = build_robinhood_option_trade_plan(
            latest_scored_df=latest_scored,
            current_option_positions=current_options,
            pending_option_orders=pending_orders,
            top_k=int(form["top_k"]),
            score_col=str(form["score_col"]),
            component_threshold=float(form["component_threshold"]),
            account_equity=float(form["account_equity"]),
            strategy_allocation=float(form["strategy_allocation"]),
            as_of_date=date.today().isoformat(),
            option_bucket=str(form["option_bucket"]),
            tenor_days=int(form["tenor_days"]),
            max_contracts_per_position=form["max_contracts_per_position"],
        )
    except Exception as exc:
        messages.error(request, f"Robinhood option plan failed: {type(exc).__name__}: {exc}")
        return None

    actionable = plan.get("actionable_orders")
    if isinstance(actionable, pd.DataFrame) and not actionable.empty:
        if "skip_submit" in actionable.columns:
            skip_submit = pd.Series(actionable["skip_submit"], index=actionable.index, dtype="boolean").fillna(False)
            actionable = actionable.loc[~skip_submit].copy()
        else:
            actionable = actionable.copy()
    else:
        actionable = pd.DataFrame()
    _save_robinhood_session_orders(request, actionable)
    messages.success(request, f"Robinhood option plan generated with {len(actionable):,} actionable order(s) saved for preview.")
    return plan


def _normalize_latest_scored_for_robinhood(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "symbol" in out.columns:
        out.index = pd.Index(out["symbol"].astype(str).str.strip().str.upper(), name="symbol")
    elif isinstance(out.index, pd.MultiIndex) and "symbol" in out.index.names:
        out = out.reset_index("symbol")
        out.index = pd.Index(out["symbol"].astype(str).str.strip().str.upper(), name="symbol")
    else:
        out.index = pd.Index([str(value).strip().upper() for value in out.index], name="symbol")
    return out


def _save_robinhood_session_orders(request: HttpRequest, orders: pd.DataFrame) -> None:
    serializable = orders.drop(columns=["raw", "response"], errors="ignore").copy() if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    request.session["robinhood_option_orders_json"] = serializable.to_json(orient="split", date_format="iso", default_handler=str)
    request.session.modified = True


def _load_robinhood_session_orders(request: HttpRequest) -> pd.DataFrame:
    payload = str(request.session.get("robinhood_option_orders_json") or "")
    if not payload:
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(payload), orient="split")
    except Exception:
        return pd.DataFrame()


def _robinhood_plan_context(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {}
    return {
        "summary_table": _frame_table(plan.get("summary")),
        "target_portfolio_table": _frame_table(plan.get("target_portfolio")),
        "desired_contracts_table": _frame_table(plan.get("desired_contracts")),
        "pending_orders_table": _frame_table(plan.get("pending_option_orders")),
        "actions_table": _frame_table(plan.get("actions")),
        "actionable_orders_table": _frame_table(plan.get("actionable_orders")),
        "skipped_symbols_table": _frame_table(plan.get("skipped_symbols")),
        "log_lines": list(plan.get("plan_log_lines") or []),
    }


def _similar_trades_form(request: HttpRequest) -> dict[str, Any]:
    default_cfg = default_live_trade_config()
    source = request.POST if request.method == "POST" else request.GET
    return {
        "symbol": str(source.get("symbol") or "AAPL").strip().upper(),
        "as_of_date": str(source.get("as_of_date") or date.today().isoformat()),
        "top_k": _coerce_int(source.get("top_k"), default=10, minimum=1),
        "query_lookback_years": _coerce_int(source.get("query_lookback_years"), default=5, minimum=1),
        "reference_start_date": str(source.get("reference_start_date") or "2010-01-01"),
        "min_profit_pct_points": _coerce_float(source.get("min_profit_pct_points"), default=5.0, minimum=0.0),
        "artifact_dir": str(source.get("artifact_dir") or default_cfg["runtime"]["artifact_dir"]),
    }


def _recompute_eligibility(frame: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    component_cols = [column for column in out.columns if str(column).startswith("__component__")]
    if component_cols:
        eligible = pd.Series(True, index=out.index, dtype=bool)
        for component_col in component_cols:
            eligible &= pd.to_numeric(out[component_col], errors="coerce").gt(float(threshold)).fillna(False)
        out["Eligible"] = eligible & pd.to_numeric(out.get("Combined Score"), errors="coerce").notna()
        return out
    required_cols = ["Classifier Score", "Regressor Score", "Autoencoder Score"]
    if all(column in out.columns for column in required_cols):
        out["Eligible"] = (
            pd.to_numeric(out["Classifier Score"], errors="coerce").gt(float(threshold))
            & pd.to_numeric(out["Regressor Score"], errors="coerce").gt(float(threshold))
            & pd.to_numeric(out["Autoencoder Score"], errors="coerce").gt(float(threshold))
        )
    return out


def _rank_leaderboard_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["Combined Score"] = pd.to_numeric(out.get("Combined Score"), errors="coerce")
    if "Eligible" in out.columns:
        out = out.sort_values(["Eligible", "Combined Score"], ascending=[False, False], kind="stable")
    else:
        out = out.sort_values(["Combined Score"], ascending=[False], kind="stable")
    out = out.reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    out["Similar Trades"] = out["Symbol"].map(
        lambda symbol: reverse("trading-similar-trades") + f"?symbol={str(symbol).strip().upper()}"
    )
    return out


def _leaderboard_summary(frame: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "eligible": 0, "ineligible": 0}
    eligible = int(pd.Series(frame.get("Eligible"), dtype="boolean").fillna(False).sum()) if "Eligible" in frame.columns else 0
    return {
        "rows": int(len(frame)),
        "eligible": eligible,
        "ineligible": int(max(len(frame) - eligible, 0)),
        "latest_date": str(meta.get("latest_date") or ""),
        "universe_size": meta.get("universe_size"),
        "reference_trade_count": meta.get("reference_trade_count"),
        "vector_backend": (meta.get("vector_metadata") or {}).get("backend") or meta.get("vector_backend"),
    }


def _paginate_frame(frame: pd.DataFrame, *, page: int, page_size: int) -> tuple[pd.DataFrame, dict[str, int]]:
    total = int(len(frame))
    total_pages = max((total - 1) // page_size + 1, 1)
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * page_size
    end = start + page_size
    return frame.iloc[start:end].copy(), {
        "page": safe_page,
        "page_size": int(page_size),
        "total_rows": total,
        "total_pages": total_pages,
        "prev_page": max(1, safe_page - 1),
        "next_page": min(total_pages, safe_page + 1),
    }


def _frame_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    link_columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"columns": [], "rows": []}
    display = frame.copy()
    if columns is None:
        columns = [str(column) for column in display.columns if not str(column).startswith("__")]
    else:
        columns = [column for column in columns if column in display.columns]
    rows: list[dict[str, Any]] = []
    link_columns = dict(link_columns or {})
    for _, raw_row in display[columns].iterrows():
        row = {"cells": [], "is_eligible": bool(raw_row.get("Eligible", False))}
        for column in columns:
            value = raw_row.get(column)
            cell = {"column": column, "value": _format_cell(column, value), "href": ""}
            if column in link_columns:
                cell["href"] = str(value or "")
                cell["value"] = link_columns[column]
            row["cells"].append(cell)
        rows.append(row)
    return {"columns": columns, "rows": rows}


def _format_cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if column in {"Classifier Score", "Regressor Score", "Autoencoder Score", "Combined Score"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"{float(numeric):.4f}"
    if column in {"Price", "__price", "Target Dollars"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"${float(numeric):,.2f}"
    if column in {"Target Weight"}:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "" if pd.isna(numeric) else f"{float(numeric) * 100.0:.2f}%"
    if column == "Eligible":
        return "Yes" if bool(value) else "No"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, *, default: int, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _coerce_float(value: Any, *, default: float, minimum: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if minimum is not None:
        out = max(float(minimum), out)
    return out
