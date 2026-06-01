from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from data import FMPClient
from fmp.models import (
    EconomicIndicatorObservation,
    EconomicIndicatorSeries,
    Symbol,
    TreasuryRateObservation,
    TreasuryRateSeries,
)
from fmp.refresh import refresh_symbol_price_history
from fmp.views import (
    _fetch_economic_indicators_from_api,
    _fetch_treasury_rates_from_api,
    _is_historical_section_stale_from_coverage,
    _refresh_all_symbol_sections,
    _store_series_dataframe,
    historical_symbol_refresh_needed,
)
from features.feature_builders import build_price_technical_features
from features.macro import MacroFeatureConfig
from features.views import _load_adjusted_prices

REQUIRED_FUNDAMENTAL_SECTION_KEYS = (
    "key_metrics",
    "ratios",
    "income_statement",
    "income_statement_growth",
    "cash_flow",
    "cash_flow_growth",
    "balance_sheet",
    "balance_sheet_growth",
    "financial_growth",
    "earnings",
)
REQUIRED_SCORING_HISTORICAL_SECTIONS = (
    "prices_div_adj",
    *REQUIRED_FUNDAMENTAL_SECTION_KEYS,
)

_REPO_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_repo_env() -> None:
    if load_dotenv is not None:
        try:
            load_dotenv(dotenv_path=_REPO_DOTENV_PATH, override=True)
            return
        except Exception:
            pass
    if not _REPO_DOTENV_PATH.exists():
        return
    try:
        for raw_line in _REPO_DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = str(key).strip()
            if not key:
                continue
            cleaned = str(value).strip().strip('"').strip("'")
            os.environ[key] = cleaned
    except Exception:
        return


def resolve_fmp_api_key(*, required: bool = False) -> str:
    _load_repo_env()
    api_key = str(os.getenv("FMP_API_KEY") or "").strip()
    if required and not api_key:
        raise ValueError("Missing FMP_API_KEY in environment/.env.")
    return api_key


def build_technical_dataframe_from_django(
    *,
    symbols: Sequence[str],
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, list[str]]:
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] = []

    for sym in symbols:
        code = str(sym).strip().upper()
        if not code:
            continue

        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol").first()
        if symbol_obj is None:
            continue

        df_prices = _load_adjusted_prices(
            symbol_obj,
            start_ts.date() if start_ts is not None else None,
            end_ts.date() if end_ts is not None else None,
        )
        if df_prices.empty:
            continue

        built = build_price_technical_features(code, df_prices)
        if built.df.empty:
            continue

        px = df_prices[["open", "high", "low", "close", "volume"]].copy()
        px["symbol"] = code
        px = px.reset_index().set_index(["date", "symbol"]).sort_index()

        panel = px.join(built.df[built.feature_cols], how="left")
        frames.append(panel)
        for col in built.feature_cols:
            if col not in feature_cols:
                feature_cols.append(col)

    if not frames:
        empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["date", "symbol"])
        return pd.DataFrame(index=empty_index), feature_cols

    technical_df = pd.concat(frames, axis=0).sort_index()
    if technical_df.index.has_duplicates:
        technical_df = technical_df[~technical_df.index.duplicated(keep="last")]
    return technical_df, feature_cols


def expected_latest_price_date_from_market_clock() -> Any:
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.weekday() < 5 and now_et.hour >= 17:
        return now_et.date()
    return (now_et.normalize() - pd.offsets.BDay(1)).date()


def _symbol_is_explicitly_inactive(symbol_obj: Symbol) -> bool:
    payload = dict(symbol_obj.payload or {})
    active_value = payload.get("isActivelyTrading")
    if active_value is None:
        active_value = payload.get("activelyTrading")
    if active_value is None:
        active_value = payload.get("is_active")
    if active_value is not None:
        return not bool(active_value)
    company_name = str(symbol_obj.company_name or "").strip().lower()
    return "(delisted)" in company_name


def _historical_section_max_date(symbol_obj: Symbol, section_key: str):
    ranges = dict(symbol_obj.historical_date_ranges or {})
    payload = ranges.get(section_key) if isinstance(ranges, dict) else None
    raw_value = payload.get("max_date") if isinstance(payload, dict) else None
    if not raw_value:
        return None
    return pd.Timestamp(raw_value).date()


def _latest_fundamental_anchor_date(symbol_obj: Symbol):
    anchor_keys = ("income_statement", "balance_sheet", "cash_flow")
    dates = [_historical_section_max_date(symbol_obj, key) for key in anchor_keys]
    dates = [value for value in dates if value is not None]
    return max(dates) if dates else None


def symbol_needs_price_refresh(symbol_obj: Symbol, *, target_end_date=None) -> tuple[bool, str]:
    symbol_obj.refresh_from_db(fields=["historical_date_ranges", "payload", "company_name"])
    if _symbol_is_explicitly_inactive(symbol_obj):
        return False, "inactive_symbol"
    expected_price_date = (
        pd.Timestamp(target_end_date).date()
        if target_end_date is not None
        else expected_latest_price_date_from_market_clock()
    )
    price_max_date = _historical_section_max_date(symbol_obj, "prices_div_adj")
    if price_max_date is None:
        return True, "missing_prices_div_adj"
    if price_max_date < expected_price_date:
        return True, f"prices_div_adj_max_date_lt_{expected_price_date.isoformat()}"
    return False, "fresh_prices_div_adj"


def symbol_needs_fundamental_refresh(
    symbol_obj: Symbol,
    *,
    target_end_date=None,
    required_section_keys: Sequence[str] | None = None,
) -> tuple[bool, str]:
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
    anchor_date = _latest_fundamental_anchor_date(symbol_obj)
    if anchor_date is None:
        return True, "missing_fundamental_anchor"

    required_keys = tuple(required_section_keys or REQUIRED_FUNDAMENTAL_SECTION_KEYS)
    for section_key in required_keys:
        section_max_date = _historical_section_max_date(symbol_obj, section_key)
        if section_max_date is None:
            return True, f"missing_{section_key}"
        if section_max_date < anchor_date:
            return True, f"{section_key}_lt_anchor_{anchor_date.isoformat()}"
    return False, "fresh_fundamental_sections"


def symbol_needs_required_refresh(symbol_obj: Symbol, *, target_start_date=None, target_end_date=None) -> tuple[bool, str]:
    del target_start_date
    symbol_obj.refresh_from_db(fields=["historical_date_ranges"])
    ranges = dict(symbol_obj.historical_date_ranges or {})

    price_range = dict(ranges.get("prices_div_adj") or {})
    price_max_date = pd.Timestamp(price_range.get("max_date")).date() if price_range.get("max_date") else None
    expected_price_date = (
        pd.Timestamp(target_end_date).date()
        if target_end_date is not None
        else expected_latest_price_date_from_market_clock()
    )

    if price_max_date is None:
        return True, "missing_prices_div_adj"
    if price_max_date < expected_price_date:
        return True, f"prices_div_adj_max_date_lt_{expected_price_date.isoformat()}"

    if bool(
        _is_historical_section_stale_from_coverage(
            symbol_obj,
            "key_metrics",
            threshold_days=7,
            target_min_date=None,
        )
    ):
        return True, "stale_key_metrics"

    if bool(
        _is_historical_section_stale_from_coverage(
            symbol_obj,
            "ratios",
            threshold_days=7,
            target_min_date=None,
        )
    ):
        return True, "stale_ratios"

    return False, "fresh_required_inputs"


def refresh_universe_symbol_sections_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    include_snapshot_sections: bool = True,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Sequence[str] | None = None,
    allowed_historical_sections: Sequence[str] | None = None,
    verbose: bool = True,
    progress_logger=None,
) -> pd.DataFrame:
    api_key = resolve_fmp_api_key(required=True)
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    total = len(selected)
    refreshed_count = 0
    for idx, sym in enumerate(selected, start=1):
        code = str(sym).strip().upper()
        if not code:
            continue
        if callable(progress_logger):
            progress_logger(f"FMP historical refresh start [{idx:,}/{total:,}] {code}")
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            symbol_obj = Symbol.objects.create(symbol=code)
        if include_snapshot_sections:
            needs_refresh, refresh_reason = symbol_needs_required_refresh(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
            )
        else:
            needs_refresh, refresh_reason = historical_symbol_refresh_needed(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
        if not needs_refresh:
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | status=skipped_fresh | reason={refresh_reason}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "skipped_fresh",
                    "refresh_reason": refresh_reason,
                    "error_count": 0,
                    "error": "",
                    "sections_total": 0,
                    "sections_fetched": 0,
                    "sections_skipped": 3,
                    "partial_sections": 0,
                    "retry_attempts": 0,
                    "duration_s": 0.0,
                }
            )
            continue
        try:
            section_errors, section_stats = _refresh_all_symbol_sections(
                symbol_obj,
                client,
                include_snapshot_sections=bool(include_snapshot_sections),
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
            refreshed_count += 1
            rows.append(
                {
                    "symbol": code,
                    "status": "ok" if not section_errors else "completed_with_errors",
                    "refresh_reason": refresh_reason,
                    "error_count": int(len(section_errors)),
                    "error": "; ".join(f"{k}={v}" for k, v in section_errors.items()),
                    "sections_total": int(section_stats.get("sections_total", 0)),
                    "sections_fetched": int(section_stats.get("sections_fetched", 0)),
                    "sections_skipped": int(section_stats.get("sections_skipped", 0)),
                    "partial_sections": int(section_stats.get("partial_sections", 0)),
                    "retry_attempts": int(section_stats.get("retry_attempts", 0)),
                    "duration_s": float(section_stats.get("duration_s", 0.0)),
                }
            )
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | "
                    f"status={'ok' if not section_errors else 'completed_with_errors'} | "
                    f"reason={refresh_reason} | fetched={int(section_stats.get('sections_fetched', 0))} | "
                    f"skipped={int(section_stats.get('sections_skipped', 0))} | "
                    f"errors={int(len(section_errors))}"
                )
        except Exception as exc:
            if callable(progress_logger):
                progress_logger(
                    f"FMP historical refresh done  [{idx:,}/{total:,}] {code} | status=error | reason={refresh_reason} | error={exc}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "error",
                    "refresh_reason": refresh_reason,
                    "error_count": 1,
                    "error": str(exc),
                    "sections_total": 0,
                    "sections_fetched": 0,
                    "sections_skipped": 0,
                    "partial_sections": 0,
                    "retry_attempts": 0,
                    "duration_s": np.nan,
                }
            )
        if callable(progress_logger) and (idx == 1 or idx % 25 == 0 or idx == total):
            progress_logger(f"FMP historical refresh progress: {idx:,}/{total:,} symbols processed | {refreshed_count:,} refreshed so far")
        if verbose and refreshed_count > 0 and (refreshed_count % 25 == 0 or idx == total):
            print(f"FMP symbol refresh progress: refreshed {refreshed_count}/{total}")
    return pd.DataFrame(rows)


def plan_symbol_section_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    include_snapshot_sections: bool = True,
    existing_historical_sections_only: bool = False,
    required_historical_sections: Sequence[str] | None = None,
    allowed_historical_sections: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            rows.append(
                {
                    "symbol": code,
                    "needs_refresh": True,
                    "refresh_reason": "missing_symbol_record",
                }
            )
            continue
        if include_snapshot_sections:
            needs_refresh, refresh_reason = symbol_needs_required_refresh(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
            )
        else:
            needs_refresh, refresh_reason = historical_symbol_refresh_needed(
                symbol_obj,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                existing_historical_sections_only=bool(existing_historical_sections_only),
                required_historical_sections=required_historical_sections,
                allowed_historical_sections=allowed_historical_sections,
            )
        rows.append(
            {
                "symbol": code,
                "needs_refresh": bool(needs_refresh),
                "refresh_reason": str(refresh_reason),
            }
        )
    return pd.DataFrame(rows)


def plan_symbol_price_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_end_date=None,
    max_symbols=None,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges", "payload", "company_name").first()
        if symbol_obj is None:
            rows.append({"symbol": code, "needs_refresh": True, "refresh_reason": "missing_symbol_record"})
            continue
        needs_refresh, refresh_reason = symbol_needs_price_refresh(
            symbol_obj,
            target_end_date=target_end_date,
        )
        rows.append({"symbol": code, "needs_refresh": bool(needs_refresh), "refresh_reason": str(refresh_reason)})
    return pd.DataFrame(rows)


def plan_symbol_fundamental_refresh_from_fmp(
    *,
    symbols: Sequence[str],
    target_end_date=None,
    max_symbols=None,
    required_section_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    for sym in selected:
        code = str(sym).strip().upper()
        if not code:
            continue
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges").first()
        if symbol_obj is None:
            rows.append({"symbol": code, "needs_refresh": True, "refresh_reason": "missing_symbol_record"})
            continue
        needs_refresh, refresh_reason = symbol_needs_fundamental_refresh(
            symbol_obj,
            target_end_date=target_end_date,
            required_section_keys=required_section_keys,
        )
        rows.append({"symbol": code, "needs_refresh": bool(needs_refresh), "refresh_reason": str(refresh_reason)})
    return pd.DataFrame(rows)


def refresh_universe_price_history_from_fmp(
    *,
    symbols: Sequence[str],
    target_start_date=None,
    target_end_date=None,
    max_symbols=None,
    verbose: bool = True,
    progress_logger=None,
) -> pd.DataFrame:
    api_key = resolve_fmp_api_key(required=True)
    client = FMPClient(api_key=api_key, timeout_s=30.0, max_retries=2)
    selected = list(symbols or [])
    if max_symbols is not None:
        selected = selected[: max(0, int(max_symbols))]

    rows: list[dict[str, Any]] = []
    total = len(selected)
    refreshed_count = 0
    for idx, sym in enumerate(selected, start=1):
        code = str(sym).strip().upper()
        if not code:
            continue
        if callable(progress_logger):
            progress_logger(f"FMP price refresh start [{idx:,}/{total:,}] {code}")
        symbol_obj = Symbol.objects.filter(symbol__iexact=code).only("id", "symbol", "historical_date_ranges", "payload", "company_name").first()
        if symbol_obj is None:
            symbol_obj = Symbol.objects.create(symbol=code)
        elif _symbol_is_explicitly_inactive(symbol_obj):
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | status=skipped_inactive | reason=inactive_symbol"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "skipped_inactive",
                    "fetch_mode": "skip",
                    "records_fetched": 0,
                    "retries_used": 0,
                    "range_after": {},
                }
            )
            continue
        try:
            result = refresh_symbol_price_history(
                symbol_obj,
                client,
                target_start_date=pd.Timestamp(target_start_date).date() if target_start_date else None,
                target_end_date=pd.Timestamp(target_end_date).date() if target_end_date else None,
            )
            fetch_mode = str(result.get("fetch_mode") or "skip")
            if fetch_mode != "skip":
                refreshed_count += 1
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | "
                    f"status=ok | mode={fetch_mode} | records={int(result.get('records_fetched') or 0)} | "
                    f"retries={int(result.get('retries_used') or 0)}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "ok",
                    "fetch_mode": fetch_mode,
                    "records_fetched": int(result.get("records_fetched") or 0),
                    "retries_used": int(result.get("retries_used") or 0),
                    "range_after": result.get("range_after") or {},
                }
            )
        except Exception as exc:
            if callable(progress_logger):
                progress_logger(
                    f"FMP price refresh done  [{idx:,}/{total:,}] {code} | status=error | error={exc}"
                )
            rows.append(
                {
                    "symbol": code,
                    "status": "error",
                    "fetch_mode": "error",
                    "records_fetched": 0,
                    "retries_used": 0,
                    "range_after": {},
                    "error": str(exc),
                }
            )
        if callable(progress_logger) and (idx == 1 or idx % 25 == 0 or idx == total):
            progress_logger(f"FMP price refresh progress: {idx:,}/{total:,} symbols processed | {refreshed_count:,} refreshed so far")
        if verbose and (idx % 25 == 0 or idx == total):
            print(f"FMP price refresh progress: processed {idx}/{total} | refreshed {refreshed_count}")
    return pd.DataFrame(rows)


def refresh_macro_series_from_fmp(
    *,
    start_date,
    end_date,
    macro_config: MacroFeatureConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    api_key = resolve_fmp_api_key(required=True)
    cfg = macro_config or MacroFeatureConfig()
    rows: list[dict[str, Any]] = []

    economic_series = tuple(str(raw).strip() for raw in tuple(cfg.economic_indicator_series or ()) if str(raw).strip())
    economic_df = _fetch_economic_indicators_from_api(
        api_key=api_key,
        start_date=str(start_date),
        end_date=str(end_date),
        series=economic_series,
    )
    if not economic_df.empty:
        _store_series_dataframe(
            economic_df,
            series_model=EconomicIndicatorSeries,
            observation_model=EconomicIndicatorObservation,
        )
    rows.append(
        {
            "dataset": "economic_indicators",
            "status": "ok" if not economic_df.empty else "empty",
            "series_count": int(economic_df.shape[1]) if not economic_df.empty else 0,
            "rows": int(len(economic_df)),
            "min_date": str(economic_df.index.min().date()) if not economic_df.empty else "",
            "max_date": str(economic_df.index.max().date()) if not economic_df.empty else "",
        }
    )

    if bool(cfg.include_treasury_rates):
        treasury_df = _fetch_treasury_rates_from_api(
            api_key=api_key,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        if not treasury_df.empty:
            _store_series_dataframe(
                treasury_df,
                series_model=TreasuryRateSeries,
                observation_model=TreasuryRateObservation,
            )
        rows.append(
            {
                "dataset": "treasury_rates",
                "status": "ok" if not treasury_df.empty else "empty",
                "series_count": int(treasury_df.shape[1]) if not treasury_df.empty else 0,
                "rows": int(len(treasury_df)),
                "min_date": str(treasury_df.index.min().date()) if not treasury_df.empty else "",
                "max_date": str(treasury_df.index.max().date()) if not treasury_df.empty else "",
            }
        )

    if verbose:
        print("FMP macro refresh complete")
    return pd.DataFrame(rows)


def normalize_holdings(raw_holdings: Sequence[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(raw_holdings or []):
        code = str(raw).strip().upper()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def component_cols_for_score(score_col: str) -> list[str]:
    mapping = {
        "buy_score_mean_raw3": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score_mean_raw_pct6": [
            "prob_buy",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_buy_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "buy_score_pct_mean": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_pct_product": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_raw": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
    }
    if score_col not in mapping:
        raise KeyError(f"No component mapping configured for score column: {score_col}")
    return list(mapping[score_col])


def build_live_trade_plan(
    *,
    latest_scored_df: pd.DataFrame,
    current_holdings: Sequence[str] | None,
    top_k: int,
    score_col: str,
    component_cols: Sequence[str],
    component_threshold: float,
    price_col: str = "close",
) -> dict[str, Any]:
    work = latest_scored_df.copy()
    work.index = pd.Index([str(idx).strip().upper() for idx in work.index], name="symbol")

    required_cols = [score_col, price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", *component_cols]
    missing_cols = [col for col in required_cols if col not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing required latest-score columns: {missing_cols}")

    numeric_cols = list(dict.fromkeys(required_cols))
    work.loc[:, numeric_cols] = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
    work["entry_ok"] = work[score_col].notna() & np.isfinite(work[score_col]) & work[price_col].gt(0.0)
    for col in component_cols:
        work["entry_ok"] &= work[col].notna() & np.isfinite(work[col]) & work[col].gt(float(component_threshold))

    work["classifier_long"] = (work["prob_buy"] > work["prob_short"]).fillna(False)
    work["classifier_short"] = (work["prob_short"] > work["prob_buy"]).fillna(False)
    work["component_min"] = work[list(component_cols)].min(axis=1, skipna=True)
    work["score_rank"] = work[score_col].rank(ascending=False, method="first")

    current = normalize_holdings(current_holdings)
    current_set = set(current)
    exits: list[dict[str, Any]] = []
    retained: list[str] = []

    for sym in current:
        if sym not in work.index:
            exits.append({"symbol": sym, "action": "sell", "reason": "missing_from_latest_panel"})
            continue

        row = work.loc[sym]
        if (not np.isfinite(row[price_col])) or float(row[price_col]) <= 0.0:
            exits.append({"symbol": sym, "action": "sell", "reason": "invalid_price"})
        elif (not np.isfinite(row["prob_buy"])) or (not np.isfinite(row["prob_short"])):
            exits.append({"symbol": sym, "action": "sell", "reason": "invalid_probability"})
        elif bool(row["classifier_short"]):
            exits.append({"symbol": sym, "action": "sell", "reason": "classifier_flipped_short"})
        else:
            retained.append(sym)

    slots_left = max(0, int(top_k) - len(retained))
    candidates = work.loc[work["entry_ok"]].copy()
    if current_set:
        candidates = candidates.drop(index=[sym for sym in current_set if sym in candidates.index], errors="ignore")
    candidates = candidates.sort_values(
        [score_col, "prob_buy", "pred_rf_reg", "ae_familiarity"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    buys = candidates.head(slots_left).index.tolist()
    target_symbols = retained + buys
    target_weight = (1.0 / float(top_k)) if int(top_k) > 0 and target_symbols else 0.0
    cash_weight = max(0.0, 1.0 - (float(len(target_symbols)) * target_weight))

    portfolio_cols = [score_col, "score_rank", price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", "component_min"]
    if target_symbols:
        target_portfolio = work.loc[target_symbols, portfolio_cols].copy()
        target_portfolio.insert(0, "target_weight", target_weight)
        target_portfolio.insert(1, "status", ["hold" if sym in retained else "buy" for sym in target_portfolio.index])
        target_portfolio = target_portfolio.sort_values(["status", score_col], ascending=[True, False], kind="stable")
    else:
        target_portfolio = pd.DataFrame(columns=["target_weight", "status", *portfolio_cols])

    action_rows: list[dict[str, Any]] = []
    for row in exits:
        sym = row["symbol"]
        if sym in work.index:
            live_row = work.loc[sym]
            action_rows.append(
                {
                    "symbol": sym,
                    "action": "sell",
                    "reason": row["reason"],
                    "target_weight": 0.0,
                    price_col: live_row.get(price_col, np.nan),
                    score_col: live_row.get(score_col, np.nan),
                    "prob_buy": live_row.get("prob_buy", np.nan),
                    "prob_short": live_row.get("prob_short", np.nan),
                }
            )
        else:
            action_rows.append(
                {
                    "symbol": sym,
                    "action": "sell",
                    "reason": row["reason"],
                    "target_weight": 0.0,
                    price_col: np.nan,
                    score_col: np.nan,
                    "prob_buy": np.nan,
                    "prob_short": np.nan,
                }
            )

    for sym in retained:
        row = work.loc[sym]
        action_rows.append(
            {
                "symbol": sym,
                "action": "hold",
                "reason": "still_held_not_exited",
                "target_weight": target_weight,
                price_col: row[price_col],
                score_col: row[score_col],
                "prob_buy": row["prob_buy"],
                "prob_short": row["prob_short"],
            }
        )

    for sym in buys:
        row = work.loc[sym]
        action_rows.append(
            {
                "symbol": sym,
                "action": "buy",
                "reason": "eligible_with_open_slot",
                "target_weight": target_weight,
                price_col: row[price_col],
                score_col: row[score_col],
                "prob_buy": row["prob_buy"],
                "prob_short": row["prob_short"],
            }
        )

    actions = pd.DataFrame(action_rows)
    if not actions.empty:
        action_order = {"sell": 0, "buy": 1, "hold": 2}
        actions["_action_order"] = actions["action"].map(action_order).fillna(99)
        actions = actions.sort_values(["_action_order", score_col], ascending=[True, False], kind="stable").drop(columns=["_action_order"])

    watchlist_cols = [score_col, "score_rank", price_col, "prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity", "component_min"]
    watchlist = candidates.loc[:, watchlist_cols].head(max(20, int(top_k) * 3)).copy()

    summary = pd.DataFrame(
        [
            {
                "current_holdings": len(current),
                "positions_kept": len(retained),
                "positions_sold": len(exits),
                "slots_open_after_sells": slots_left,
                "new_buys": len(buys),
                "target_positions": len(target_symbols),
                "target_weight_per_position": target_weight,
                "target_cash_weight": cash_weight,
                "component_threshold": float(component_threshold),
                "top_k": int(top_k),
                "score_col": score_col,
            }
        ]
    )

    return {
        "summary": summary,
        "target_portfolio": target_portfolio,
        "actions": actions,
        "watchlist": watchlist,
        "latest_scored": work.sort_values(score_col, ascending=False, kind="stable"),
        "retained": retained,
        "buys": buys,
        "exits": exits,
    }
