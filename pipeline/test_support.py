from __future__ import annotations

import csv
import gzip
import json
import tempfile
from functools import lru_cache
from pathlib import Path

from fmp.models import Symbol, SymbolSectionHistorical


MAG7_SYMBOLS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
]
REAL_MARKET_FIXTURE_DIR = Path(__file__).resolve().parent / "testdata" / "real_market"


def _parse_optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


@lru_cache(maxsize=None)
def _load_fixture_rows(name: str, *, compressed: bool = False) -> list[dict[str, str]]:
    path = REAL_MARKET_FIXTURE_DIR / name
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=None)
def _load_fixture_manifest(name: str) -> dict:
    return json.loads((REAL_MARKET_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def market_cap_tier_symbols(tier_name: str, *, limit: int | None = None) -> list[str]:
    manifest = _load_fixture_manifest("scalability_manifest.json")
    key = str(tier_name or "").strip().lower()
    symbols = list(manifest.get(key) or [])
    if limit is not None and int(limit) > 0:
        return symbols[: int(limit)]
    return symbols


def _ensure_symbol_rows(rows: list[dict[str, str]]) -> list[Symbol]:
    ordered_symbols = [str(row["symbol"]).strip().upper() for row in rows if str(row.get("symbol") or "").strip()]
    existing = set(Symbol.objects.filter(symbol__in=ordered_symbols).values_list("symbol", flat=True))
    create_rows: list[Symbol] = []
    for row in rows:
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or symbol in existing:
            continue
        create_rows.append(
            Symbol(
                symbol=symbol,
                company_name=str(row.get("company_name") or ""),
                exchange=str(row.get("exchange") or ""),
                country=str(row.get("country") or ""),
                sector=str(row.get("sector") or ""),
                industry=str(row.get("industry") or ""),
                market_cap=_parse_optional_float(row.get("market_cap")),
                price=_parse_optional_float(row.get("price")),
                beta=_parse_optional_float(row.get("beta")),
                volume=_parse_optional_float(row.get("volume")),
                dividend=_parse_optional_float(row.get("dividend")),
                dividend_yield=_parse_optional_float(row.get("dividend_yield")),
                payload=json.loads(str(row.get("payload") or "{}")),
            )
        )
        existing.add(symbol)
    if create_rows:
        Symbol.objects.bulk_create(create_rows, batch_size=1000, ignore_conflicts=True)
    symbol_map = {
        str(row.symbol): row
        for row in Symbol.objects.filter(symbol__in=ordered_symbols).only("id", "symbol")
    }
    return [symbol_map[symbol] for symbol in ordered_symbols if symbol in symbol_map]


def _ensure_price_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    symbol_map = {
        str(row.symbol): row
        for row in Symbol.objects.filter(symbol__in=sorted({str(item["symbol"]).strip().upper() for item in rows})).only("id", "symbol")
    }
    history_rows: list[SymbolSectionHistorical] = []
    for row in rows:
        symbol = str(row["symbol"]).strip().upper()
        symbol_obj = symbol_map.get(symbol)
        date_value = str(row.get("date") or "").strip()
        if symbol_obj is None or not date_value:
            continue
        history_rows.append(
            SymbolSectionHistorical(
                symbol=symbol_obj,
                section_key="prices_div_adj",
                record_key=f"{symbol}:{date_value}",
                record_date=date_value,
                payload={
                    "date": date_value,
                    "adjOpen": _parse_optional_float(row.get("adj_open")),
                    "adjHigh": _parse_optional_float(row.get("adj_high")),
                    "adjLow": _parse_optional_float(row.get("adj_low")),
                    "adjClose": _parse_optional_float(row.get("adj_close")),
                    "volume": int(float(row.get("volume") or 0.0)),
                },
            )
        )
    if history_rows:
        SymbolSectionHistorical.objects.bulk_create(history_rows, batch_size=5000, ignore_conflicts=True)


def _slice_price_rows(
    rows: list[dict[str, str]],
    *,
    start_date: str | None = None,
    days: int | None = None,
) -> list[dict[str, str]]:
    if not rows:
        return []
    filtered = [dict(row) for row in rows if not start_date or str(row.get("date") or "") >= str(start_date)]
    ordered_dates = sorted({str(row["date"]) for row in filtered})
    if days is not None and int(days) > 0:
        allowed_dates = set(ordered_dates[: int(days)])
        filtered = [row for row in filtered if str(row["date"]) in allowed_dates]
    return filtered


class ArtifactTestMixin:
    maxDiff = None

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def write_csv(self, name: str, fieldnames: list[str], rows: list[dict]) -> str:
        path = self.temp_path / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def write_json(self, name: str, payload: dict) -> str:
        path = self.temp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def assert_rows_have_columns(self, rows: list[dict], expected_columns: list[str]) -> None:
        self.assertGreater(len(rows), 0, "Expected at least one row to validate schema.")
        for col in expected_columns:
            self.assertIn(col, rows[0], f"Missing expected column {col!r} in first row.")


class MarketCapTierFixtureMixin(ArtifactTestMixin):
    def create_screened_symbols(self) -> list[Symbol]:
        return _ensure_symbol_rows(_load_fixture_rows("screened_symbols.csv"))

    def create_market_cap_tier_symbols(self, tier_name: str, *, limit: int | None = None) -> list[Symbol]:
        selected_symbols = market_cap_tier_symbols(tier_name, limit=limit)
        row_map = {
            str(row["symbol"]): row
            for row in _load_fixture_rows("scalability_symbols.csv.gz", compressed=True)
        }
        ordered_rows = [row_map[symbol] for symbol in selected_symbols if symbol in row_map]
        return _ensure_symbol_rows(ordered_rows)

    def seed_market_cap_tier_price_history(
        self,
        tier_name: str,
        *,
        start_date: str = "2024-01-02",
        business_days: int = 90,
        limit: int | None = None,
    ) -> None:
        selected_symbols = set(market_cap_tier_symbols(tier_name, limit=limit))
        self.create_market_cap_tier_symbols(tier_name, limit=limit)
        rows = [
            row
            for row in _load_fixture_rows("scalability_prices.csv.gz", compressed=True)
            if str(row.get("symbol") or "") in selected_symbols
        ]
        _ensure_price_rows(
            _slice_price_rows(
                rows,
                start_date=start_date,
                days=business_days,
            )
        )


class Mag7FixtureMixin(ArtifactTestMixin):
    def create_mag7_symbols(self) -> list[Symbol]:
        return _ensure_symbol_rows(_load_fixture_rows("mag7_symbols.csv"))

    def seed_mag7_price_history(self, start_date: str = "2024-01-01", days: int = 5) -> None:
        self.create_mag7_symbols()
        _ensure_price_rows(
            _slice_price_rows(
                _load_fixture_rows("mag7_prices.csv"),
                start_date=start_date,
                days=days,
            )
        )


class ScalabilityFixtureMixin:
    SCALABILITY_TIER_MARKET_CAPS = {
        "tier1": 1_500_000_000_000.0,
        "tier2": 150_000_000_000.0,
        "tier3": 12_000_000_000.0,
    }
    SCALABILITY_TIER_COUNTS = {
        "tier1": 10,
        "tier2": 100,
        "tier3": 1000,
    }

    @classmethod
    def seed_scalability_universe(
        cls,
        *,
        start_date: str = "2024-01-02",
        business_days: int = 90,
    ) -> dict[str, list[str]]:
        manifest = _load_fixture_manifest("scalability_manifest.json")
        tier_symbols = {
            tier_name: list(manifest.get(tier_name) or [])[: int(cls.SCALABILITY_TIER_COUNTS[tier_name])]
            for tier_name in cls.SCALABILITY_TIER_COUNTS
        }
        _ensure_symbol_rows(_load_fixture_rows("scalability_symbols.csv.gz", compressed=True))
        _ensure_price_rows(
            _slice_price_rows(
                _load_fixture_rows("scalability_prices.csv.gz", compressed=True),
                start_date=start_date,
                days=business_days,
            )
        )
        return tier_symbols
