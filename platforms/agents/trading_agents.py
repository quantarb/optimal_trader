from __future__ import annotations

import json
import os
import subprocess
import urllib.request
import uuid
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
    conda_env: str = "tradingagents"
    worker_command: tuple[str, ...] | None = None
    worker_env_file: Path | None = None
    worker_timeout_seconds: float = 900.0
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
    return _review_candidates_worker(frame, as_of_date=as_of_date, config=review_config)


def _review_candidates_worker(
    frame: pd.DataFrame,
    *,
    as_of_date: str | None,
    config: TradingAgentsReviewConfig,
) -> pd.DataFrame:
    load_repo_env()
    trade_date = _review_date(frame.iloc[0].to_dict(), as_of_date)
    symbols = list(dict.fromkeys(frame["symbol"].astype(str).str.upper()))
    if len(symbols) > 20:
        return _mark_unavailable(frame, ValueError("TradingAgents accepts at most 20 symbols"), trade_date)
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "as_of_date": trade_date,
        "candidates": [{"symbol": symbol} for symbol in symbols],
        "options": {
            "selected_analysts": list(config.selected_analysts),
            "asset_type": config.asset_type,
            "debug": bool(config.debug),
            "max_workers": int(config.max_workers),
            "agent_config": _build_agent_config({}, config),
        },
    }
    command = list(
        config.worker_command
        or (
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            config.conda_env,
            "python",
            "-m",
            "tradingagents.batch_worker",
        )
    )
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=float(config.worker_timeout_seconds),
            check=False,
            env=_worker_environment(config.worker_env_file),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"worker exited {completed.returncode}: {detail[-500:]}")
        response = json.loads(completed.stdout)
        if response.get("request_id") != request_id or not isinstance(response.get("decisions"), list):
            raise ValueError("worker response has an invalid request ID or decisions list")
        decisions = _validated_worker_decisions(response["decisions"], expected_symbols=set(symbols))
    except Exception as exc:
        return _mark_unavailable(frame, exc, trade_date)

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        symbol = str(record["symbol"]).upper()
        result = decisions.get(symbol)
        row = dict(record)
        row.update({"llm_provider": "tradingagents_worker", "llm_review_date": trade_date})
        if result is None:
            row.update(_hold_fields("TradingAgents worker omitted this symbol"))
        else:
            decision = result["decision"]
            row.update(
                {
                    "llm_decision": "approved" if decision == "BUY" else "rejected",
                    "llm_rating": result.get("rating") or decision.title(),
                    "llm_reason": result.get("reason") or "TradingAgents worker decision",
                    "llm_raw_decision": result.get("raw_decision") or decision,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _validated_worker_decisions(
    rows: Sequence[Mapping[str, Any]], *, expected_symbols: set[str]
) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for value in rows:
        symbol = str(value.get("symbol") or "").strip().upper()
        decision = str(value.get("decision") or "").strip().upper()
        if symbol not in expected_symbols or symbol in decisions:
            raise ValueError(f"worker returned an unknown or duplicate symbol: {symbol!r}")
        if decision not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(f"worker returned an invalid decision for {symbol}: {decision!r}")
        decisions[symbol] = {key: str(value.get(key) or "") for key in ("decision", "rating", "reason", "raw_decision")}
    return decisions


def _worker_environment(env_file: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        upper = key.upper()
        if any(token in upper for token in ("ALPACA", "ROBINHOOD", "BROKER")) or upper.startswith(
            ("APCA_", "IBKR_", "IB_")
        ):
            env.pop(key, None)
    env.pop("DEEPSEEK_API_KEY", None)
    configured = env_file or os.getenv("TRADINGAGENTS_ENV_FILE")
    if configured:
        env["TRADINGAGENTS_ENV_FILE"] = str(Path(configured).expanduser().resolve())
    else:
        repo_root = Path(__file__).resolve().parents[2]
        sibling_env = repo_root.parent / "TradingAgents" / ".env"
        env["TRADINGAGENTS_ENV_FILE"] = str(sibling_env.resolve())
    return env


def _hold_fields(reason: str) -> dict[str, str]:
    return {
        "llm_decision": "rejected",
        "llm_rating": "Hold",
        "llm_reason": reason,
        "llm_raw_decision": "HOLD",
    }


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


def _mark_unavailable(frame: pd.DataFrame, exc: Exception, trade_date: str = "") -> pd.DataFrame:
    out = frame.copy()
    out["llm_provider"] = "tradingagents_worker"
    out["llm_decision"] = "rejected"
    out["llm_rating"] = "Hold"
    out["llm_reason"] = f"tradingagents unavailable: {type(exc).__name__}: {exc}"
    out["llm_raw_decision"] = "HOLD"
    out["llm_review_date"] = trade_date
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
