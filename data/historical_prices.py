from __future__ import annotations

from typing import Any

import pandas as pd

from fmp.models import SymbolSectionHistorical


def load_adjusted_price_frames(
    symbols: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    grouped_rows: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in normalized}
    if not normalized:
        return {}

    qs = (
        SymbolSectionHistorical.objects.filter(symbol__symbol__in=normalized, section_key="prices_div_adj")
        .select_related("symbol")
        .only("symbol__symbol", "record_date", "payload")
        .order_by("symbol__symbol", "record_date", "updated_at")
    )
    if start_date:
        qs = qs.filter(record_date__gte=pd.to_datetime(start_date).date())
    if end_date:
        qs = qs.filter(record_date__lte=pd.to_datetime(end_date).date())

    for item in qs.iterator():
        payload = item.payload if isinstance(item.payload, dict) else {}
        date_value = str(payload.get("date") or (item.record_date.isoformat() if item.record_date else ""))[:10]
        if not date_value:
            continue
        grouped_rows.setdefault(str(item.symbol.symbol).strip().upper(), []).append(
            {
                "date": date_value,
                "open": payload.get("adjOpen"),
                "high": payload.get("adjHigh"),
                "low": payload.get("adjLow"),
                "close": payload.get("adjClose"),
                "adj_open": payload.get("adjOpen"),
                "adj_high": payload.get("adjHigh"),
                "adj_low": payload.get("adjLow"),
                "adj_close": payload.get("adjClose"),
                "volume": payload.get("volume"),
            }
        )

    frames: dict[str, pd.DataFrame] = {}
    for symbol in normalized:
        rows = grouped_rows.get(symbol) or []
        if not rows:
            frames[symbol] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        frames[symbol] = df
    return frames


__all__ = ["load_adjusted_price_frames"]
