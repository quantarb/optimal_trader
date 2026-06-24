from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from data.warehouse import _symbol_is_etf, load_warehouse_price_frame
from fmp.models import EconomicIndicatorSeries, Symbol, SymbolSectionHistorical, TreasuryRateSeries


@dataclass
class DjangoSymbolRepository:
    """Django adapter for symbol metadata lookups."""

    def by_symbols(self, symbols: Sequence[str]) -> dict[str, Symbol]:
        normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
        return {
            str(symbol.symbol).strip().upper(): symbol
            for symbol in Symbol.objects.filter(symbol__in=normalized).only("id", "symbol")
        }

    def ordered_universe(self, *, limit: int) -> list[str]:
        qs = Symbol.objects.order_by("symbol").values_list("symbol", flat=True)
        return [str(symbol).strip().upper() for symbol in qs[: max(1, int(limit))]]


@dataclass
class DjangoSectionHistoryRepository:
    """Django adapter for historical section payloads."""

    def price_history(self, symbol: Symbol, *, start_date=None, end_date=None):
        df = load_warehouse_price_frame(symbol.symbol, start_date=start_date, end_date=end_date, is_etf=_symbol_is_etf(symbol))
        if df is None or df.empty:
            return iter(())
        records = []
        for idx, row in df.sort_index().iterrows():
            records.append(
                {
                    "symbol": str(symbol.symbol).strip().upper(),
                    "record_date": pd.Timestamp(idx).date() if pd.notna(idx) else None,
                    "payload": row.to_dict(),
                }
            )
        return iter(records)

    def sparse_sections(self, symbol_ids: Sequence[int], section_keys: Sequence[str]):
        return (
            SymbolSectionHistorical.objects.filter(symbol_id__in=list(symbol_ids), section_key__in=list(section_keys))
            .only("symbol_id", "section_key", "record_date", "payload")
            .order_by("symbol_id", "section_key", "record_date", "updated_at")
            .iterator()
        )

    def latest_generated_labels(self, symbols: Sequence[str]):
        return (
            SymbolSectionHistorical.objects.filter(symbol__symbol__in=list(symbols), section_key="labels_generated")
            .select_related("symbol")
            .only("symbol__symbol", "payload", "record_date")
            .order_by("-record_date", "-updated_at")
            .iterator()
        )


@dataclass
class DjangoMacroSeriesRepository:
    """Django adapter for configured macro series lists."""

    def economic_indicator_codes(self) -> tuple[str, ...]:
        return tuple(str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True))

    def treasury_rate_codes(self) -> tuple[str, ...]:
        return tuple(str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True))
