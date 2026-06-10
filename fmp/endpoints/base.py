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
    pagination: str = "none"
    page_size: int = 10_000
    max_pages: int = 1_000
    supports_date_window: bool = False
    chunk_years: int | None = None
    dedupe_by_date: bool = False
    stability_mode: str = "auto"
    minimum_observations: int = 1


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
    pagination: str = "none",
    page_size: int = 10_000,
    max_pages: int = 1_000,
    supports_date_window: bool = False,
    chunk_years: int | None = None,
    dedupe_by_date: bool = False,
    stability_mode: str = "auto",
    minimum_observations: int = 1,
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
            pagination=pagination,
            page_size=page_size,
            max_pages=max_pages,
            supports_date_window=supports_date_window,
            chunk_years=chunk_years,
            dedupe_by_date=dedupe_by_date,
            stability_mode=stability_mode,
            minimum_observations=minimum_observations,
        )

    return build
