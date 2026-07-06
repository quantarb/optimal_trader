from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_QW_HOME = Path.home() / ".quant-warehouse"
DEFAULT_PROVIDER_KEYS = ("fmp", "thetadata")


@dataclass(frozen=True)
class QuantWarehouseStorage:
    home: Path
    arctic_uri: str
    catalog_path: Path
    provider_arctic_uris: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "home": str(self.home),
            "arctic_uri": self.arctic_uri,
            "catalog_path": str(self.catalog_path),
            "provider_arctic_uris": dict(self.provider_arctic_uris),
        }


def ensure_quant_warehouse_storage(
    *,
    provider_keys: Sequence[str] = DEFAULT_PROVIDER_KEYS,
) -> QuantWarehouseStorage:
    """Resolve the shared Quant Warehouse ArcticDB/catalog configuration.

    optimal_trader should not create its own warehouse database. It leaves any
    caller-provided QW_* settings intact and otherwise falls back to the shared
    quant-warehouse default under ~/.quant-warehouse.
    """

    os.environ.setdefault("QW_HOME", str(DEFAULT_QW_HOME))

    from quant_warehouse.config import WarehouseConfig
    from quant_warehouse.ingest.credentials import load_shared_env

    load_shared_env()
    config = WarehouseConfig.from_env()
    return QuantWarehouseStorage(
        home=config.home,
        arctic_uri=config.arctic_uri,
        catalog_path=config.catalog_path,
        provider_arctic_uris={
            str(provider): config.provider_arctic_uri(str(provider))
            for provider in provider_keys
        },
    )
