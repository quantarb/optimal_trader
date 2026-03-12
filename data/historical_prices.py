from __future__ import annotations

import pandas as pd

from fmp.models import SymbolSectionHistorical


def load_adjusted_price_frames(
    symbols: list[str],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    normalized = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    grouped_rows: dict[str, dict[str, tuple[str, object, object, object, object, object]]] = {symbol: {} for symbol in normalized}
    if not normalized:
        return {}

    qs = (
        SymbolSectionHistorical.objects.filter(symbol__symbol__in=normalized, section_key="prices_div_adj")
        .order_by("symbol__symbol", "record_date", "updated_at")
        .values_list("symbol__symbol", "record_date", "payload")
    )
    if start_date:
        qs = qs.filter(record_date__gte=pd.to_datetime(start_date).date())
    if end_date:
        qs = qs.filter(record_date__lte=pd.to_datetime(end_date).date())

    for symbol_value, record_date, payload_value in qs.iterator(chunk_size=5000):
        payload = payload_value if isinstance(payload_value, dict) else {}
        date_value = str(payload.get("date") or (record_date.isoformat() if record_date else ""))[:10]
        if not date_value:
            continue
        grouped_rows.setdefault(str(symbol_value).strip().upper(), {})[date_value] = (
            date_value,
            payload.get("adjOpen"),
            payload.get("adjHigh"),
            payload.get("adjLow"),
            payload.get("adjClose"),
            payload.get("volume"),
        )

    frames: dict[str, pd.DataFrame] = {}
    for symbol in normalized:
        rows = list((grouped_rows.get(symbol) or {}).values())
        if not rows:
            frames[symbol] = pd.DataFrame()
            continue
        df = pd.DataFrame.from_records(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["adj_open"] = df["open"]
        df["adj_high"] = df["high"]
        df["adj_low"] = df["low"]
        df["adj_close"] = df["close"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        frames[symbol] = df
    return frames


__all__ = ["load_adjusted_price_frames"]
