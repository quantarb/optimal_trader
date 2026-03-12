from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _as_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value) if value not in (None, "") else int(default)
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_ids(values: list[Any] | tuple[Any, ...] | None) -> tuple[int, ...]:
    out: list[int] = []
    for value in list(values or []):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    return tuple(out)


def _normalize_symbols(values: Any) -> tuple[str, ...]:
    if values in (None, ""):
        return ()
    raw_values: list[Any]
    if isinstance(values, str):
        raw_values = values.split(",")
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    out: list[str] = []
    for value in raw_values:
        normalized = str(value or "").strip().upper()
        if normalized and normalized not in out:
            out.append(normalized)
    return tuple(out)


@dataclass(frozen=True)
class StrategyDatasetSpec:
    """Typed config for building a strategy dataset from features and predictions."""

    strategy_definition_id: int | None = None
    prediction_artifact_ids: tuple[int, ...] = ()
    label_artifact_id: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> "StrategyDatasetSpec":
        raw = dict(config or {})
        strategy_definition_id = _as_int(raw.get("strategy_definition_id"), 0, minimum=0)
        label_artifact_id = _as_int(raw.get("label_artifact_id"), 0, minimum=0)
        return cls(
            strategy_definition_id=strategy_definition_id if strategy_definition_id > 0 else None,
            prediction_artifact_ids=_normalize_ids(raw.get("prediction_artifact_ids")),
            label_artifact_id=label_artifact_id if label_artifact_id > 0 else None,
            start_date=str(raw.get("strategy_start_date") or raw.get("start_date") or "").strip() or None,
            end_date=str(raw.get("strategy_end_date") or raw.get("end_date") or "").strip() or None,
        )


@dataclass(frozen=True)
class StrategyBacktestSpec:
    """Typed config for executing a portfolio backtest on a strategy dataset."""

    start_date: str | None = None
    end_date: str | None = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    transaction_cost_bps: float = 0.0
    max_position_weight: float = 0.0
    min_price: float = 0.0
    min_dollar_volume: float = 0.0
    short_borrow_bps_annual: float = 0.0
    execution_delay_days: int = 1
    turnover_half_l1: bool = True
    use_lagged_weights: bool = True
    allowed_symbols: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> "StrategyBacktestSpec":
        raw = dict(config or {})
        return cls(
            start_date=str(raw.get("backtest_start_date") or raw.get("start_date") or "").strip() or None,
            end_date=str(raw.get("backtest_end_date") or raw.get("end_date") or "").strip() or None,
            fee_bps=max(0.0, _as_float(raw.get("fee_bps"), 0.0)),
            slippage_bps=max(0.0, _as_float(raw.get("slippage_bps"), 0.0)),
            transaction_cost_bps=max(0.0, _as_float(raw.get("transaction_cost_bps"), 0.0)),
            max_position_weight=max(0.0, _as_float(raw.get("max_position_weight"), 0.0)),
            min_price=max(0.0, _as_float(raw.get("min_price"), 0.0)),
            min_dollar_volume=max(0.0, _as_float(raw.get("min_dollar_volume"), 0.0)),
            short_borrow_bps_annual=max(0.0, _as_float(raw.get("short_borrow_bps_annual"), 0.0)),
            execution_delay_days=_as_int(raw.get("execution_delay_days"), 1, minimum=0),
            turnover_half_l1=_as_bool(raw.get("turnover_half_l1"), True),
            use_lagged_weights=_as_bool(raw.get("use_lagged_weights"), True),
            allowed_symbols=_normalize_symbols(raw.get("allowed_symbols")),
        )

    def effective_slippage_bps(self) -> float:
        if self.fee_bps <= 0.0 and self.slippage_bps <= 0.0:
            return float(self.transaction_cost_bps)
        return float(self.slippage_bps)
