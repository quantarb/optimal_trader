from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


APPROVE_RATINGS = {"buy", "overweight"}
REJECT_RATINGS = {"hold", "underweight", "sell"}
_REPO_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True)
class TradingAgentsReviewConfig:
    repo_path: Path | None = None
    selected_analysts: tuple[str, ...] = ("market",)
    asset_type: str = "stock"
    llm_provider: str | None = "deepseek"
    deep_think_llm: str | None = "deepseek-v4-flash"
    quick_think_llm: str | None = "deepseek-v4-flash"
    backend_url: str | None = None
    temperature: float | None = None
    llm_max_retries: int | None = None
    max_debate_rounds: int | None = 1
    max_risk_discuss_rounds: int | None = 1
    checkpoint_enabled: bool | None = None
    debug: bool = False
    max_workers: int = 4
    fast_symbol_date_only: bool = True
    request_timeout_seconds: float = 30.0
    data_vendors: Mapping[str, str] | None = field(
        default_factory=lambda: {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        }
    )


def default_trading_agents_repo(repo_root: Path | None = None) -> Path:
    root = Path(repo_root or Path.cwd()).resolve()
    for candidate in (root, *root.parents):
        sibling = candidate.parent / "TradingAgents"
        if (sibling / "tradingagents").is_dir():
            return sibling
    return root.parent / "TradingAgents"


def ensure_tradingagents_importable(repo_path: Path | None = None) -> None:
    load_repo_env()
    if "tradingagents" in sys.modules:
        return
    if importlib.util.find_spec("tradingagents") is not None:
        return
    candidate = Path(repo_path or os.getenv("TRADINGAGENTS_REPO") or default_trading_agents_repo()).expanduser().resolve()
    if (candidate / "tradingagents").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def load_repo_env(path: Path | None = None) -> None:
    dotenv_path = Path(path or _REPO_DOTENV_PATH)
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def review_trade_candidates(
    candidates: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    as_of_date: str | None = None,
    config: TradingAgentsReviewConfig | None = None,
) -> pd.DataFrame:
    frame = _candidate_frame(candidates)
    if frame.empty:
        return _empty_review_frame()

    review_config = config or TradingAgentsReviewConfig()
    if review_config.fast_symbol_date_only:
        return _review_candidates_fast(frame, as_of_date=as_of_date, config=review_config)
    ensure_tradingagents_importable(review_config.repo_path)

    try:
        graph_module = importlib.import_module("tradingagents.graph.trading_graph")
        config_module = importlib.import_module("tradingagents.default_config")
    except Exception as exc:
        return _mark_unavailable(frame, exc)

    graph_cls = getattr(graph_module, "TradingAgentsGraph")
    agent_config = _build_agent_config(getattr(config_module, "DEFAULT_CONFIG", {}) or {}, review_config)
    def review_record(record: dict[str, Any]) -> dict[str, Any]:
        graph = graph_cls(
            selected_analysts=list(review_config.selected_analysts),
            debug=bool(review_config.debug),
            config=dict(agent_config),
        )
        symbol = str(record.get("symbol") or "").strip().upper()
        trade_date = _review_date(record, as_of_date)
        row = dict(record)
        row.update({"llm_provider": "tradingagents", "llm_review_date": trade_date})
        try:
            final_state, rating = _propagate(graph, symbol, trade_date, asset_type=review_config.asset_type)
        except Exception as exc:
            row.update(
                {
                    "llm_decision": "error",
                    "llm_rating": "",
                    "llm_reason": f"tradingagents error: {type(exc).__name__}: {exc}",
                    "llm_raw_decision": "",
                }
            )
        else:
            normalized_rating = str(rating or "").strip()
            decision = "approved" if normalized_rating.lower() in APPROVE_RATINGS else "rejected"
            row.update(
                {
                    "llm_decision": decision,
                    "llm_rating": normalized_rating,
                    "llm_reason": f"TradingAgents rating: {normalized_rating or 'unknown'}",
                    "llm_raw_decision": _raw_final_decision(final_state),
                }
            )
        return row

    records = frame.to_dict(orient="records")
    worker_count = max(1, min(int(review_config.max_workers), len(records)))
    if worker_count == 1:
        rows = [review_record(record) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tradingagents") as executor:
            rows = list(executor.map(review_record, records))
    return pd.DataFrame(rows)


def _review_candidates_fast(
    frame: pd.DataFrame,
    *,
    as_of_date: str | None,
    config: TradingAgentsReviewConfig,
) -> pd.DataFrame:
    load_repo_env()
    records = frame.to_dict(orient="records")

    def review(record: dict[str, Any]) -> dict[str, Any]:
        symbol = str(record.get("symbol") or "").strip().upper()
        trade_date = _review_date(record, as_of_date)
        row = dict(record)
        row.update({"llm_provider": "deepseek_fast", "llm_review_date": trade_date})
        try:
            decision, reason = _deepseek_symbol_date_decision(symbol, trade_date, config=config)
        except Exception as exc:
            row.update(
                {
                    "llm_decision": "error",
                    "llm_rating": "",
                    "llm_reason": f"DeepSeek fast review error: {type(exc).__name__}: {exc}",
                    "llm_raw_decision": "",
                }
            )
            return row
        rating = {"buy": "Buy", "sell": "Sell", "hold": "Hold"}[decision]
        row.update(
            {
                "llm_decision": "approved" if decision == "buy" else "rejected",
                "llm_rating": rating,
                "llm_reason": reason,
                "llm_raw_decision": decision.upper(),
            }
        )
        return row

    workers = max(1, min(int(config.max_workers), len(records)))
    if workers == 1:
        rows = [review(record) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepseek-fast") as executor:
            rows = list(executor.map(review, records))
    return pd.DataFrame(rows)


def _deepseek_symbol_date_decision(
    symbol: str,
    trade_date: str,
    *,
    config: TradingAgentsReviewConfig,
) -> tuple[str, str]:
    api_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    payload = {
        "model": config.quick_think_llm or "deepseek-v4-flash",
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {
                "role": "system",
                "content": "Return one trading decision using exactly this format: DECISION: BUY, SELL, or HOLD; REASON: one short sentence.",
            },
            {"role": "user", "content": f"Symbol: {symbol}\nAs-of date: {trade_date}"},
        ],
    }
    request = urllib.request.Request(
        (config.backend_url or "https://api.deepseek.com") .rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(config.request_timeout_seconds)) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = dict(body["choices"][0]["message"])
    content = str(message.get("content") or message.get("reasoning_content") or "").strip()
    upper = content.upper()
    decision = next((value for value in ("BUY", "SELL", "HOLD") if f"DECISION: {value}" in upper), "")
    if not decision:
        raise ValueError(f"Unparseable DeepSeek decision: {content[:160]!r}")
    reason = content.split("REASON:", 1)[-1].strip() if "REASON:" in upper else content
    return decision.lower(), reason


def approved_symbols(reviewed: pd.DataFrame) -> set[str]:
    if reviewed is None or reviewed.empty or "symbol" not in reviewed.columns:
        return set()
    decision = reviewed.get("llm_decision", pd.Series(index=reviewed.index, dtype=str)).astype(str).str.lower()
    rating = reviewed.get("llm_rating", pd.Series(index=reviewed.index, dtype=str)).astype(str).str.lower()
    approved = decision.eq("approved") | rating.isin(APPROVE_RATINGS)
    return set(reviewed.loc[approved, "symbol"].astype(str).str.upper())


def _build_agent_config(default_config: Mapping[str, Any], review_config: TradingAgentsReviewConfig) -> dict[str, Any]:
    config = dict(default_config or {})
    override_map = {
        "llm_provider": review_config.llm_provider,
        "deep_think_llm": review_config.deep_think_llm,
        "quick_think_llm": review_config.quick_think_llm,
        "backend_url": review_config.backend_url,
        "temperature": review_config.temperature,
        "llm_max_retries": review_config.llm_max_retries,
        "max_debate_rounds": review_config.max_debate_rounds,
        "max_risk_discuss_rounds": review_config.max_risk_discuss_rounds,
        "checkpoint_enabled": review_config.checkpoint_enabled,
    }
    for key, value in override_map.items():
        if value is not None:
            config[key] = value
    if review_config.data_vendors is not None:
        merged_vendors = dict(config.get("data_vendors") or {})
        merged_vendors.update(dict(review_config.data_vendors))
        config["data_vendors"] = merged_vendors
    return config


def _propagate(graph: Any, symbol: str, trade_date: str, *, asset_type: str) -> tuple[Any, str]:
    try:
        return graph.propagate(symbol, trade_date, asset_type=asset_type)
    except TypeError as exc:
        if "asset_type" not in str(exc):
            raise
        return graph.propagate(symbol, trade_date)


def _candidate_frame(candidates: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame(list(candidates))
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])
    if "symbol" not in frame.columns:
        raise KeyError("TradingAgents candidates require a symbol column")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    return out.loc[out["symbol"].ne("")].reset_index(drop=True)


def _review_date(record: Mapping[str, Any], as_of_date: str | None) -> str:
    if as_of_date:
        return str(pd.Timestamp(as_of_date).date())
    for key in ("score_date", "date", "Scored Date"):
        value = record.get(key)
        if value not in (None, ""):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return str(parsed.date())
    return str(pd.Timestamp.today().date())


def _raw_final_decision(final_state: Any) -> str:
    if isinstance(final_state, Mapping):
        return str(final_state.get("final_trade_decision") or "")
    return str(final_state or "")


def _mark_unavailable(frame: pd.DataFrame, exc: Exception) -> pd.DataFrame:
    out = frame.copy()
    out["llm_provider"] = "tradingagents"
    out["llm_decision"] = "unavailable"
    out["llm_rating"] = ""
    out["llm_reason"] = f"tradingagents unavailable: {type(exc).__name__}: {exc}"
    out["llm_raw_decision"] = ""
    return out


def _empty_review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "llm_provider",
            "llm_decision",
            "llm_rating",
            "llm_reason",
            "llm_raw_decision",
            "llm_review_date",
        ]
    )
