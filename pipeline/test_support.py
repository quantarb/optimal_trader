from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pandas as pd

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


class Mag7FixtureMixin:
    maxDiff = None

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()
        super().tearDown()

    def create_mag7_symbols(self) -> list[Symbol]:
        rows: list[tuple[str, str]] = [
            ("AAPL", "Apple Inc."),
            ("AMZN", "Amazon.com, Inc."),
            ("GOOGL", "Alphabet Inc."),
            ("META", "Meta Platforms, Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("NVDA", "NVIDIA Corporation"),
            ("TSLA", "Tesla, Inc."),
        ]
        return [Symbol.objects.create(symbol=symbol, company_name=name) for symbol, name in rows]

    def create_screened_symbols(self) -> list[Symbol]:
        rows = [
            ("AAPL", "Apple Inc.", "NASDAQ", "US", 3_000_000_000_000.0),
            ("MSFT", "Microsoft Corporation", "NASDAQ", "US", 2_900_000_000_000.0),
            ("NVDA", "NVIDIA Corporation", "NASDAQ", "US", 1_500_000_000_000.0),
            ("ORCL", "Oracle Corporation", "NYSE", "US", 320_000_000_000.0),
            ("CRM", "Salesforce, Inc.", "NYSE", "US", 250_000_000_000.0),
            ("UBER", "Uber Technologies, Inc.", "NYSE", "US", 150_000_000_000.0),
            ("VOO", "Vanguard S&P 500 ETF", "AMEX", "US", 500_000_000_000.0),
            ("VTSAX", "Vanguard Total Stock Market Index Fund Admiral Shares", "NASDAQ", "US", 400_000_000_000.0),
            ("SHOP", "Shopify Inc.", "NYSE", "CA", 120_000_000_000.0),
            ("PLTR", "Palantir Technologies Inc.", "NASDAQ", "US", 60_000_000_000.0),
            ("SNOW", "Snowflake Inc.", "NYSE", "US", 55_000_000_000.0),
            ("DDOG", "Datadog, Inc.", "NASDAQ", "US", 42_000_000_000.0),
            ("NET", "Cloudflare, Inc.", "NYSE", "US", 35_000_000_000.0),
            ("DUOL", "Duolingo, Inc.", "NASDAQ", "US", 10_500_000_000.0),
            ("RKLB", "Rocket Lab USA, Inc.", "NASDAQ", "US", 9_500_000_000.0),
            ("NVO", "Novo Nordisk A/S", "NYSE", "DK", 500_000_000_000.0),
            ("RY", "Royal Bank of Canada", "NYSE", "CA", 140_000_000_000.0),
        ]
        return [
            Symbol.objects.create(
                symbol=symbol,
                company_name=name,
                exchange=exchange,
                country=country,
                market_cap=market_cap,
                payload={"isEtf": True} if "ETF" in name else {},
            )
            for symbol, name, exchange, country, market_cap in rows
        ]

    def seed_mag7_price_history(self, start_date: str = "2024-01-01", days: int = 5) -> None:
        symbols = {row.symbol: row for row in Symbol.objects.filter(symbol__in=MAG7_SYMBOLS)}
        for symbol_index, symbol in enumerate(MAG7_SYMBOLS, start=1):
            symbol_obj = symbols[symbol]
            base = 100.0 + (symbol_index * 25.0)
            for offset in range(days):
                date_value = f"2024-01-0{offset + 1}"
                open_px = base + offset
                close_px = open_px + (0.75 if offset % 2 == 0 else -0.25)
                high_px = max(open_px, close_px) + 1.0
                low_px = min(open_px, close_px) - 1.0
                SymbolSectionHistorical.objects.create(
                    symbol=symbol_obj,
                    section_key="prices_div_adj",
                    record_key=f"{symbol}:{date_value}",
                    record_date=date_value,
                    payload={
                        "date": date_value,
                        "adjOpen": round(open_px, 4),
                        "adjHigh": round(high_px, 4),
                        "adjLow": round(low_px, 4),
                        "adjClose": round(close_px, 4),
                        "volume": int(1_000_000 + symbol_index * 10_000 + offset * 1_000),
                    },
                )

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
        tier_symbols: dict[str, list[str]] = {key: [] for key in cls.SCALABILITY_TIER_COUNTS}
        symbols_to_create: list[Symbol] = []
        for tier_name, count in cls.SCALABILITY_TIER_COUNTS.items():
            base_market_cap = float(cls.SCALABILITY_TIER_MARKET_CAPS[tier_name])
            prefix = tier_name.upper()
            for index in range(count):
                symbol = f"{prefix}{index:04d}"
                tier_symbols[tier_name].append(symbol)
                symbols_to_create.append(
                    Symbol(
                        symbol=symbol,
                        company_name=f"{tier_name} synthetic {index:04d}",
                        exchange="NASDAQ",
                        country="US",
                        market_cap=base_market_cap - float(index * 1_000_000.0),
                        payload={},
                    )
                )
        Symbol.objects.bulk_create(symbols_to_create, batch_size=1000)

        dates = pd.bdate_range(start=start_date, periods=business_days)
        symbol_map = {
            str(row.symbol): row
            for row in Symbol.objects.filter(symbol__in=[symbol for symbols in tier_symbols.values() for symbol in symbols]).only("id", "symbol")
        }
        history_rows: list[SymbolSectionHistorical] = []
        for tier_name, symbols in tier_symbols.items():
            tier_bias = 0.002 if tier_name == "tier1" else (0.001 if tier_name == "tier2" else 0.0005)
            for symbol_index, symbol in enumerate(symbols):
                symbol_obj = symbol_map[symbol]
                base_price = 40.0 + (symbol_index % 50) * 1.5 + (10.0 if tier_name == "tier1" else 5.0)
                for day_index, date_value in enumerate(dates):
                    seasonal = ((day_index % 10) - 5) * 0.08
                    drift = tier_bias * day_index
                    open_px = base_price + seasonal + drift
                    close_px = open_px * (1.0 + tier_bias + ((symbol_index % 7) - 3) * 0.0007)
                    high_px = max(open_px, close_px) + 0.6
                    low_px = min(open_px, close_px) - 0.6
                    history_rows.append(
                        SymbolSectionHistorical(
                            symbol=symbol_obj,
                            section_key="prices_div_adj",
                            record_key=f"{symbol}:{date_value.date().isoformat()}",
                            record_date=date_value.date(),
                            payload={
                                "date": date_value.date().isoformat(),
                                "adjOpen": round(open_px, 6),
                                "adjHigh": round(high_px, 6),
                                "adjLow": round(low_px, 6),
                                "adjClose": round(close_px, 6),
                                "volume": int(1_000_000 + (symbol_index % 25) * 25_000 + day_index * 500),
                            },
                        )
                    )
        SymbolSectionHistorical.objects.bulk_create(history_rows, batch_size=5000)
        return tier_symbols
