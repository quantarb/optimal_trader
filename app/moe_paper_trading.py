from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from app.live_app_shared import inspect_saved_artifacts


ARTIFACT_DIR_NAME = "moe_paper_trading"
LATEST_SCORED_FILE = "latest_scored.pkl"
METADATA_FILE = "metadata.json"
MODEL_ARTIFACT_FILES = ("classifier_families.pkl", "classifier_families_meta.json")


@dataclass(frozen=True)
class MoePaperArtifacts:
    artifact_dir: Path
    latest_scored: pd.DataFrame
    metadata: dict[str, Any]


def default_artifact_dir(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "artifacts" / ARTIFACT_DIR_NAME


def save_moe_paper_artifacts(
    *,
    artifact_dir: Path,
    latest_scored: pd.DataFrame,
    metadata: dict[str, Any],
) -> MoePaperArtifacts:
    target_dir = Path(artifact_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    scored = latest_scored.copy()
    scored.index = pd.Index(
        [str(value).strip().upper() for value in scored.index],
        name="symbol",
    )
    scored.to_pickle(target_dir / LATEST_SCORED_FILE)
    (target_dir / METADATA_FILE).write_text(
        json.dumps(dict(metadata), indent=2, default=str),
        encoding="utf-8",
    )
    return MoePaperArtifacts(target_dir, scored, dict(metadata))


def load_moe_paper_artifacts(artifact_dir: Path) -> MoePaperArtifacts:
    target_dir = Path(artifact_dir).resolve()
    scored = pd.read_pickle(target_dir / LATEST_SCORED_FILE)
    metadata = json.loads((target_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return MoePaperArtifacts(target_dir, scored, dict(metadata))


def load_recent_moe_paper_build(
    *,
    artifact_dir: Path,
    model_artifact_dir: Path,
    expected_score_date: str | date | pd.Timestamp,
    max_age: pd.Timedelta = pd.Timedelta(days=1),
) -> tuple[MoePaperArtifacts | None, dict[str, Any]]:
    score_dir = Path(artifact_dir).resolve()
    model_dir = Path(model_artifact_dir).resolve()
    required_paths = [
        score_dir / LATEST_SCORED_FILE,
        score_dir / METADATA_FILE,
        *(model_dir / filename for filename in MODEL_ARTIFACT_FILES),
    ]
    preliminary_status = inspect_saved_artifacts(
        required_paths=required_paths,
        artifact_dir=score_dir,
        max_age=max_age,
        extra_status={"model_artifact_dir": str(model_dir)},
    )
    if not preliminary_status["reusable"]:
        return None, preliminary_status

    try:
        artifacts = load_moe_paper_artifacts(score_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, {
            **preliminary_status,
            "reason": f"artifact_load_failed={type(exc).__name__}",
            "reusable": False,
        }

    saved_date = pd.Timestamp(artifacts.metadata.get("strategy_date")).normalize()
    expected_date = pd.Timestamp(expected_score_date).normalize()
    status = inspect_saved_artifacts(
        required_paths=required_paths,
        artifact_dir=score_dir,
        max_age=max_age,
        saved_score_date=saved_date,
        expected_score_date=expected_date,
        extra_status={"model_artifact_dir": str(model_dir)},
    )
    return (artifacts if status["reusable"] else None), status


def build_latest_moe_scored(
    scored_panel: pd.DataFrame,
    *,
    scoring_date: str | date | pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    scoring_ts = pd.Timestamp(scoring_date).normalize()
    work = scored_panel.reset_index().copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work = work.loc[
        work["date"].notna()
        & work["symbol"].ne("")
        & work["date"].le(scoring_ts)
    ].copy()
    if work.empty:
        raise RuntimeError(
            f"No MoE score rows were available on or before {scoring_ts.date().isoformat()}."
        )

    latest = (
        work.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    latest["feature_as_of_date"] = latest["date"]
    exact_date_mask = latest["feature_as_of_date"].eq(scoring_ts)
    inactive_count = int((~exact_date_mask).sum())
    latest = latest.loc[exact_date_mask].copy()
    if latest.empty:
        raise RuntimeError(
            f"No MoE score rows were available exactly on {scoring_ts.date().isoformat()}."
        )
    latest["date"] = scoring_ts
    latest = latest.set_index("symbol").sort_index()
    return latest, {
        "symbol_count": int(len(latest)),
        "exact_date_count": int(exact_date_mask.sum()),
        "carry_forward_count": 0,
        "inactive_count": inactive_count,
    }


def build_moe_ranked_scores(
    latest_scored: pd.DataFrame,
    *,
    top_k: int,
    threshold: float = 0.50,
) -> pd.DataFrame:
    frame = latest_scored.copy()
    frame.index = pd.Index(
        [str(value).strip().upper() for value in frame.index],
        name="symbol",
    )
    frame["prob_buy"] = pd.to_numeric(frame.get("prob_buy"), errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame = frame.loc[frame["prob_buy"].notna() & frame["close"].gt(0.0)].copy()
    frame = frame.sort_values(["prob_buy"], ascending=False, kind="stable")
    frame["eligible"] = frame["prob_buy"].gt(float(threshold))
    selected_index = frame.loc[frame["eligible"]].head(max(int(top_k), 0)).index
    frame["selected"] = frame.index.isin(selected_index)
    return frame


def _number(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def _option_quote(snapshot: Mapping[str, Any]) -> tuple[float, float, float]:
    quote = dict(snapshot.get("latestQuote") or snapshot.get("latest_quote") or {})
    trade = dict(snapshot.get("latestTrade") or snapshot.get("latest_trade") or {})
    bid = _number(quote.get("bp", quote.get("bid_price")))
    ask = _number(quote.get("ap", quote.get("ask_price")))
    trade_price = _number(trade.get("p", trade.get("price")))
    if bid > 0 and ask > 0:
        mark = (bid + ask) / 2.0
    else:
        mark = ask or bid or trade_price
    return bid, ask, mark


def select_alpaca_option_contract(
    contracts: Sequence[Mapping[str, Any]],
    *,
    underlying_price: float,
    target_expiration: date,
    option_bucket: str,
) -> dict[str, Any] | None:
    strike_multiplier = {
        "atm_option": 1.0,
        "otm_option": 1.05,
        "ditm_option": 0.90,
    }.get(str(option_bucket), 1.05)
    target_strike = float(underlying_price) * strike_multiplier
    candidates: list[tuple[int, float, str, dict[str, Any]]] = []
    for raw_contract in contracts:
        contract = dict(raw_contract)
        expiration = pd.to_datetime(contract.get("expiration_date"), errors="coerce")
        strike = _number(contract.get("strike_price"), default=float("nan"))
        symbol = str(contract.get("symbol") or "").strip().upper()
        if pd.isna(expiration) or not math.isfinite(strike) or strike <= 0 or not symbol:
            continue
        expiration_distance = abs((expiration.date() - target_expiration).days)
        candidates.append((expiration_distance, abs(strike - target_strike), symbol, contract))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def build_alpaca_option_trade_plan(
    *,
    ranked_scores: pd.DataFrame,
    current_option_positions: Sequence[Mapping[str, Any]],
    open_orders: Sequence[Mapping[str, Any]],
    option_contracts: Mapping[str, Sequence[Mapping[str, Any]]],
    option_snapshots: Mapping[str, Mapping[str, Any]],
    strategy_allocation: float,
    as_of_date: str | date,
    option_bucket: str,
    tenor_days: int,
    max_contracts_per_position: int | None = None,
) -> dict[str, pd.DataFrame]:
    selected = ranked_scores.loc[ranked_scores["selected"]].copy()
    target_symbols = set(selected.index.astype(str))
    slot_budget = float(strategy_allocation) / len(target_symbols) if target_symbols else 0.0
    target_date = pd.Timestamp(as_of_date).date() + timedelta(days=int(tenor_days))

    position_rows: list[dict[str, Any]] = []
    held_underlyings: set[str] = set()
    action_rows: list[dict[str, Any]] = []
    contracts_to_close = 0
    for raw_position in current_option_positions:
        position = dict(raw_position)
        symbol = str(position.get("symbol") or "").strip().upper()
        underlying = str(position.get("underlying_symbol") or "").strip().upper()
        option_type = str(position.get("option_type") or position.get("type") or "").lower()
        qty = int(abs(_number(position.get("qty"))))
        if not symbol or not underlying or qty <= 0:
            continue
        position_rows.append(position)
        if option_type in {"", "call"}:
            held_underlyings.add(underlying)
        if underlying not in target_symbols or option_type == "put":
            contracts_to_close += qty
            action_rows.append({
                "symbol": symbol,
                "underlying_symbol": underlying,
                "action": "sell_to_close_put" if option_type == "put" else "sell_to_close_call",
                "side": "sell",
                "qty": qty,
                "quantity": qty,
                "order_type": "market",
                "time_in_force": "day",
                "reason": (
                    "The MoE option variant holds calls, not puts."
                    if option_type == "put"
                    else "Underlying is no longer in the current top-K MoE scores."
                ),
            })

    normalized_orders: list[dict[str, Any]] = []
    pending_buy_underlyings: set[str] = set()
    for raw_order in open_orders:
        order = dict(raw_order)
        symbol = str(order.get("symbol") or "").strip().upper()
        underlying = str(order.get("underlying_symbol") or "").strip().upper()
        option_type = str(order.get("option_type") or order.get("type") or "").lower()
        side = str(order.get("side") or "").strip().lower()
        qty = _number(order.get("qty", order.get("quantity")))
        filled_qty = _number(order.get("filled_qty", order.get("filled_quantity")))
        remaining = max(qty - filled_qty, 0.0)
        normalized = {
            "order_id": str(order.get("id") or order.get("order_id") or ""),
            "symbol": symbol,
            "underlying_symbol": underlying,
            "option_type": option_type,
            "side": side,
            "remaining_qty": remaining,
            "limit_price": _number(order.get("limit_price"), default=float("nan")),
            "status": str(order.get("status") or ""),
        }
        normalized_orders.append(normalized)
        if side == "buy" and remaining > 0 and underlying and option_type in {"", "call"}:
            pending_buy_underlyings.add(underlying)
        if side == "buy" and remaining > 0 and underlying:
            if underlying not in target_symbols or option_type == "put":
                action_rows.append({
                    **normalized,
                    "action": "cancel_buy_to_open_put" if option_type == "put" else "cancel_buy_to_open_call",
                    "quantity": remaining,
                    "reason": (
                        "The MoE option variant opens calls, not puts."
                        if option_type == "put"
                        else "Underlying is no longer in the current top-K MoE scores."
                    ),
                })

    desired_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for underlying in selected.index.astype(str):
        if underlying in held_underlyings or underlying in pending_buy_underlyings:
            continue
        contract = select_alpaca_option_contract(
            option_contracts.get(underlying, []),
            underlying_price=float(selected.loc[underlying, "close"]),
            target_expiration=target_date,
            option_bucket=option_bucket,
        )
        if contract is None:
            skipped_rows.append({"symbol": underlying, "reason": "No matching active Alpaca call contract."})
            continue
        contract_symbol = str(contract.get("symbol") or "").strip().upper()
        bid, ask, mark = _option_quote(option_snapshots.get(contract_symbol, {}))
        limit_price = bid or mark
        contract_value = limit_price * 100.0
        quantity = int(slot_budget // contract_value) if contract_value > 0 else 0
        if max_contracts_per_position is not None:
            quantity = min(quantity, max(int(max_contracts_per_position), 0))
        desired = {
            "symbol": underlying,
            "option_symbol": contract_symbol,
            "option_type": "call",
            "expiry_date": contract.get("expiration_date"),
            "strike_price": _number(contract.get("strike_price")),
            "underlying_price": float(selected.loc[underlying, "close"]),
            "bid_price": bid,
            "ask_price": ask,
            "mark_price": mark,
            "limit_price": limit_price,
            "contract_value": contract_value,
            "target_dollars": slot_budget,
            "quantity": quantity,
            "combined_score": float(selected.loc[underlying, "prob_buy"]),
        }
        desired_rows.append(desired)
        if quantity <= 0:
            skipped_rows.append({**desired, "reason": "One contract exceeds the per-position option budget."})
            continue
        action_rows.append({
            **desired,
            "symbol": contract_symbol,
            "underlying_symbol": underlying,
            "action": "buy_to_open_call",
            "side": "buy",
            "qty": quantity,
            "order_type": "limit",
            "time_in_force": "day",
            "reason": "New current top-K MoE option position.",
        })

    actions = pd.DataFrame(action_rows)
    calls_to_open = int(actions.get("action", pd.Series(dtype=str)).eq("buy_to_open_call").sum())
    summary = pd.DataFrame([{
        "target_positions": len(target_symbols),
        "calls_to_open": calls_to_open,
        "puts_to_open": 0,
        "contracts_to_close": contracts_to_close,
        "strategy_allocation": float(strategy_allocation),
        "occupied_slots": len(held_underlyings & target_symbols),
        "pending_buy_underlyings": len(pending_buy_underlyings & target_symbols),
        "remaining_buy_slots": calls_to_open,
    }])
    return {
        "summary": summary,
        "desired_contracts": pd.DataFrame(desired_rows),
        "current_option_positions": pd.DataFrame(position_rows),
        "pending_option_orders": pd.DataFrame(normalized_orders),
        "actions": actions,
        "actionable_orders": actions.copy(),
        "skipped_symbols": pd.DataFrame(skipped_rows),
    }


__all__ = [
    "MoePaperArtifacts",
    "build_alpaca_option_trade_plan",
    "build_latest_moe_scored",
    "build_moe_ranked_scores",
    "default_artifact_dir",
    "load_moe_paper_artifacts",
    "load_recent_moe_paper_build",
    "save_moe_paper_artifacts",
    "select_alpaca_option_contract",
]
