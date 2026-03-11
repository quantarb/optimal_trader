from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Candidate = tuple[str, dict[str, Any]]
ExtraParamsBuilder = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class EndpointDefinition:
    key: str
    title: str
    kind: str
    threshold_days: int
    max_rows: int
    candidates: list[Candidate]
    supported_periods: tuple[str, ...] = ()
    min_history_years: int | None = None
    filter_symbol: bool = False


def build_symbol_endpoint(
    *,
    key: str,
    title: str,
    kind: str,
    threshold_days: int,
    max_rows: int,
    candidate_path: str,
    supported_periods: tuple[str, ...] = (),
    min_history_years: int | None = None,
    filter_symbol: bool = False,
    symbol_param: str = "symbol",
    extra_params_builder: ExtraParamsBuilder | None = None,
) -> Callable[[Any], EndpointDefinition]:
    def build(symbol_obj) -> EndpointDefinition:
        params = {symbol_param: symbol_obj.symbol}
        if extra_params_builder is not None:
            params.update(dict(extra_params_builder(symbol_obj) or {}))
        return EndpointDefinition(
            key=key,
            title=title,
            kind=kind,
            threshold_days=threshold_days,
            max_rows=max_rows,
            candidates=[(candidate_path, params)],
            supported_periods=supported_periods,
            min_history_years=min_history_years,
            filter_symbol=filter_symbol,
        )

    return build
