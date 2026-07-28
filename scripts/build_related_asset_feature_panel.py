"""Materialize FMP related-security features for an optimal-trader universe.

Provider discovery, classification, adjusted OHLCV retrieval, and feature
construction live in quant-warehouse. This script owns only the tier-specific
universe and artifact path used by optimal-trader.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT.parent / "quant-warehouse"
sys.path.insert(0, str(WAREHOUSE))

from quant_warehouse.platforms.data_providers.fmp.related_assets import (  # noqa: E402
    build_related_asset_panel,
)


def key() -> str:
    if os.getenv("FMP_API_KEY"):
        return os.environ["FMP_API_KEY"]
    for env_path in (WAREHOUSE / ".env", ROOT / ".env", ROOT.parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            if line.startswith("FMP_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("FMP_API_KEY is required")


def symbols(tier: str) -> list[str]:
    cap = {"1T": 1_000_000_000_000, "100B": 100_000_000_000, "10B": 10_000_000_000}[tier]
    index = pd.read_csv(
        ROOT / "artifacts" / "trading_app_v2" / f"equity_meta_model_{tier.lower()}"
        / f"mcap_{cap}_train_2020-12-31_seed_20260707" / "feature_family_panels" / "index.csv"
    )
    return sorted({
        symbol
        for path in index.panel_path
        for symbol in pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str).str.upper()
    })


def build(tier: str = "100B") -> Path:
    tier = str(tier).strip().upper()
    panel, stats = build_related_asset_panel(symbols(tier), key())
    output = ROOT / "artifacts" / "graph_oracle_feature_family_gnn_improved_hits" / "cache" / f"related_asset_features_{tier.lower()}_adjusted.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    print({**stats, "tier": tier, "output": str(output)}, flush=True)
    return output


if __name__ == "__main__":
    build(os.getenv("RELATED_ASSET_TIER", "100B"))
