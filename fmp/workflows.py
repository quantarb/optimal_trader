from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from features.macro import MacroFeatureConfig
from trading.live_trade import (
    REQUIRED_SCORING_HISTORICAL_SECTIONS,
    plan_symbol_price_refresh_from_fmp,
    plan_symbol_section_refresh_from_fmp,
    refresh_macro_series_from_fmp,
    refresh_universe_price_history_from_fmp,
    refresh_universe_symbol_sections_from_fmp,
    resolve_fmp_api_key,
)


def run_scoring_data_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    refresh_mode: str = "scoring_ready",
    refresh_symbol_sections_before_build: bool = True,
    refresh_macro_before_build: bool = False,
    max_symbols=None,
    existing_historical_sections_only: bool = True,
    required_historical_sections: Sequence[str] | None = None,
    macro_config: MacroFeatureConfig | None = None,
    verbose: bool = False,
    progress_logger=None,
) -> dict[str, Any]:
    refresh_mode = str(refresh_mode or "scoring_ready").strip().lower()
    log = progress_logger if callable(progress_logger) else None
    results: dict[str, Any] = {
        "refresh_mode": refresh_mode,
        "price_plan": pd.DataFrame(),
        "price_refresh_results": pd.DataFrame(),
        "fundamental_plan": pd.DataFrame(),
        "fundamental_refresh_results": pd.DataFrame(),
        "symbol_refresh_plan": pd.DataFrame(),
        "symbol_refresh_results": pd.DataFrame(),
        "macro_refresh_results": pd.DataFrame(),
    }
    if not refresh_symbol_sections_before_build and not refresh_macro_before_build:
        return results

    _ = resolve_fmp_api_key(required=True)
    scoring_historical_sections = tuple(
        str(section).strip()
        for section in (required_historical_sections or REQUIRED_SCORING_HISTORICAL_SECTIONS)
        if str(section).strip()
    )
    scoring_fundamental_sections = tuple(
        section for section in scoring_historical_sections if section != "prices_div_adj"
    )

    if refresh_symbol_sections_before_build:
        if refresh_mode == "prices_only":
            price_plan = plan_symbol_price_refresh_from_fmp(
                symbols=symbols,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
            )
            results["price_plan"] = price_plan
            price_symbols = tuple(
                price_plan.loc[price_plan["needs_refresh"].fillna(False), "symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            if log is not None:
                log(
                    "Refreshing missing latest price history from FMP before feature build"
                    f" | total symbols {len(price_plan):,}"
                    f" | targeted symbols {len(price_symbols):,}"
                    f" | needs refresh {len(price_symbols):,}"
                    f" | already fresh {int(len(price_plan) - len(price_symbols)):,}"
                )
                if not price_plan.empty and len(price_symbols):
                    top_reasons = (
                        price_plan.loc[price_plan["needs_refresh"].fillna(False), "refresh_reason"]
                        .astype(str)
                        .value_counts()
                        .head(5)
                    )
                    if len(top_reasons):
                        reason_text = " | ".join(f"{reason}={count:,}" for reason, count in top_reasons.items())
                        log(f"Top price refresh reasons | {reason_text}")
            price_refresh_results = refresh_universe_price_history_from_fmp(
                symbols=price_symbols,
                target_start_date=None,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
                verbose=bool(verbose),
                progress_logger=progress_logger,
            )
            results["price_refresh_results"] = price_refresh_results
            if log is not None:
                refreshed_count = int((price_refresh_results.get("fetch_mode") != "skip").sum()) if not price_refresh_results.empty and "fetch_mode" in price_refresh_results.columns else 0
                error_count = int((price_refresh_results.get("status") == "error").sum()) if not price_refresh_results.empty and "status" in price_refresh_results.columns else 0
                log(f"FMP price refresh complete | refreshed {refreshed_count:,} | errors {error_count:,}")
        elif refresh_mode == "scoring_ready":
            price_plan = plan_symbol_price_refresh_from_fmp(
                symbols=symbols,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
            )
            results["price_plan"] = price_plan
            price_symbols = tuple(
                price_plan.loc[price_plan["needs_refresh"].fillna(False), "symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            if log is not None:
                log(
                    "Refreshing latest price data from FMP before feature build"
                    f" | total symbols {len(price_plan):,}"
                    f" | targeted symbols {len(price_symbols):,}"
                    f" | needs refresh {len(price_symbols):,}"
                    f" | already fresh {int(len(price_plan) - len(price_symbols)):,}"
                )
                if not price_plan.empty and len(price_symbols):
                    top_reasons = (
                        price_plan.loc[price_plan["needs_refresh"].fillna(False), "refresh_reason"]
                        .astype(str)
                        .value_counts()
                        .head(5)
                    )
                    if len(top_reasons):
                        reason_text = " | ".join(f"{reason}={count:,}" for reason, count in top_reasons.items())
                        log(f"Top price refresh reasons | {reason_text}")
            price_refresh_results = refresh_universe_price_history_from_fmp(
                symbols=price_symbols,
                target_start_date=None,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
                verbose=bool(verbose),
                progress_logger=progress_logger,
            )
            results["price_refresh_results"] = price_refresh_results
            if log is not None:
                price_refreshed_count = int((price_refresh_results.get("fetch_mode") != "skip").sum()) if not price_refresh_results.empty and "fetch_mode" in price_refresh_results.columns else 0
                price_error_count = int((price_refresh_results.get("status") == "error").sum()) if not price_refresh_results.empty and "status" in price_refresh_results.columns else 0
                log(f"FMP price refresh complete | refreshed {price_refreshed_count:,} | errors {price_error_count:,}")

            fundamental_plan_frames = []
            fundamental_refresh_frames = []
            for section_key in scoring_fundamental_sections:
                section_plan = plan_symbol_section_refresh_from_fmp(
                    symbols=symbols,
                    target_start_date=target_start_date,
                    target_end_date=target_end_date,
                    max_symbols=max_symbols,
                    include_snapshot_sections=False,
                    existing_historical_sections_only=False,
                    required_historical_sections=(section_key,),
                    allowed_historical_sections=(section_key,),
                )
                if not section_plan.empty:
                    section_plan = section_plan.copy()
                    section_plan.insert(0, "section_key", section_key)
                fundamental_plan_frames.append(section_plan)

                section_symbols = tuple(
                    section_plan.loc[section_plan["needs_refresh"].fillna(False), "symbol"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .tolist()
                ) if not section_plan.empty else ()
                if log is not None:
                    log(
                        f"Refreshing FMP section {section_key} before feature build"
                        f" | total symbols {len(section_plan):,}"
                        f" | targeted symbols {len(section_symbols):,}"
                        f" | needs refresh {len(section_symbols):,}"
                        f" | already fresh {int(len(section_plan) - len(section_symbols)) if not section_plan.empty else 0:,}"
                    )
                    if not section_plan.empty and len(section_symbols):
                        top_reasons = (
                            section_plan.loc[section_plan["needs_refresh"].fillna(False), "refresh_reason"]
                            .astype(str)
                            .value_counts()
                            .head(5)
                        )
                        if len(top_reasons):
                            reason_text = " | ".join(f"{reason}={count:,}" for reason, count in top_reasons.items())
                            log(f"Top {section_key} refresh reasons | {reason_text}")

                section_progress_logger = (
                    (lambda message, section_key=section_key: progress_logger(f"FMP section {section_key} | {message}"))
                    if callable(progress_logger)
                    else None
                )
                section_refresh_results = (
                    refresh_universe_symbol_sections_from_fmp(
                        symbols=section_symbols,
                        target_start_date=target_start_date,
                        target_end_date=target_end_date,
                        max_symbols=max_symbols,
                        include_snapshot_sections=False,
                        existing_historical_sections_only=False,
                        required_historical_sections=(section_key,),
                        allowed_historical_sections=(section_key,),
                        verbose=bool(verbose),
                        progress_logger=section_progress_logger,
                    )
                    if section_symbols
                    else pd.DataFrame()
                )
                if not section_refresh_results.empty:
                    section_refresh_results = section_refresh_results.copy()
                    section_refresh_results.insert(0, "section_key", section_key)
                fundamental_refresh_frames.append(section_refresh_results)
                if log is not None:
                    refreshed_count = int((section_refresh_results.get("status") != "skipped_fresh").sum()) if not section_refresh_results.empty and "status" in section_refresh_results.columns else 0
                    error_count = int((section_refresh_results.get("status") == "error").sum()) if not section_refresh_results.empty and "status" in section_refresh_results.columns else 0
                    log(f"FMP section {section_key} refresh complete | refreshed {refreshed_count:,} | errors {error_count:,}")

            results["fundamental_plan"] = (
                pd.concat([frame for frame in fundamental_plan_frames if frame is not None and not frame.empty], ignore_index=True)
                if any(frame is not None and not frame.empty for frame in fundamental_plan_frames)
                else pd.DataFrame()
            )
            results["fundamental_refresh_results"] = (
                pd.concat([frame for frame in fundamental_refresh_frames if frame is not None and not frame.empty], ignore_index=True)
                if any(frame is not None and not frame.empty for frame in fundamental_refresh_frames)
                else pd.DataFrame()
            )
        elif refresh_mode == "historical_only":
            refresh_plan = plan_symbol_section_refresh_from_fmp(
                symbols=symbols,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
                include_snapshot_sections=False,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=scoring_historical_sections,
            )
            refresh_symbols = tuple(
                refresh_plan.loc[refresh_plan["needs_refresh"].fillna(False), "symbol"]
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            results["symbol_refresh_plan"] = refresh_plan
            if log is not None:
                log(
                    "Refreshing historical-only symbol data from FMP before feature build"
                    f" | total symbols {len(refresh_plan):,}"
                    f" | targeted symbols {len(refresh_symbols):,}"
                    f" | needs refresh {len(refresh_symbols):,}"
                    f" | already fresh {int(len(refresh_plan) - len(refresh_symbols)):,}"
                )
                if not refresh_plan.empty and len(refresh_symbols):
                    top_reasons = (
                        refresh_plan.loc[refresh_plan["needs_refresh"].fillna(False), "refresh_reason"]
                        .astype(str)
                        .value_counts()
                        .head(5)
                    )
                    if len(top_reasons):
                        reason_text = " | ".join(f"{reason}={count:,}" for reason, count in top_reasons.items())
                        log(f"Top refresh reasons | {reason_text}")
            symbol_refresh_results = refresh_universe_symbol_sections_from_fmp(
                symbols=refresh_symbols,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
                include_snapshot_sections=False,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=scoring_historical_sections,
                verbose=bool(verbose),
                progress_logger=progress_logger,
            )
            results["symbol_refresh_results"] = symbol_refresh_results
            if log is not None:
                refreshed_count = int((symbol_refresh_results.get("status") != "skipped_fresh").sum()) if not symbol_refresh_results.empty and "status" in symbol_refresh_results.columns else 0
                error_count = int((symbol_refresh_results.get("status") == "error").sum()) if not symbol_refresh_results.empty and "status" in symbol_refresh_results.columns else 0
                log(f"FMP symbol refresh complete | refreshed {refreshed_count:,} | errors {error_count:,}")
        else:
            if log is not None:
                log("Refreshing stale or missing symbol history from FMP before feature build")
            symbol_refresh_results = refresh_universe_symbol_sections_from_fmp(
                symbols=symbols,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                max_symbols=max_symbols,
                verbose=bool(verbose),
                progress_logger=progress_logger,
            )
            results["symbol_refresh_results"] = symbol_refresh_results
            if log is not None:
                refreshed_count = int((symbol_refresh_results.get("status") != "skipped_fresh").sum()) if not symbol_refresh_results.empty and "status" in symbol_refresh_results.columns else 0
                error_count = int((symbol_refresh_results.get("status") == "error").sum()) if not symbol_refresh_results.empty and "status" in symbol_refresh_results.columns else 0
                log(f"FMP symbol refresh complete | refreshed {refreshed_count:,} | errors {error_count:,}")

    if refresh_macro_before_build:
        if log is not None:
            log(
                "Refreshing macro feature series from FMP before feature build"
                f" | start {pd.Timestamp(target_start_date).date().isoformat() if target_start_date is not None else '<default>'}"
                f" | end {pd.Timestamp(target_end_date).date().isoformat() if target_end_date is not None else '<default>'}"
            )
        macro_refresh_results = refresh_macro_series_from_fmp(
            start_date=target_start_date,
            end_date=target_end_date,
            macro_config=macro_config,
            verbose=bool(verbose),
        )
        results["macro_refresh_results"] = macro_refresh_results
        if log is not None:
            if macro_refresh_results.empty:
                log("FMP macro refresh complete | no datasets returned")
            else:
                parts = []
                for row in macro_refresh_results.itertuples(index=False):
                    dataset = str(getattr(row, "dataset", "macro"))
                    status = str(getattr(row, "status", ""))
                    rows = int(getattr(row, "rows", 0) or 0)
                    max_date = str(getattr(row, "max_date", "") or "")
                    parts.append(f"{dataset}:status={status},rows={rows:,},max={max_date or '<none>'}")
                log("FMP macro refresh complete | " + " | ".join(parts))

    return results
