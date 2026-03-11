"""Backward-compatible re-export of the FMP infrastructure client."""

from infra.fmp.client import FMPClient, FMPClientConfig, FMPInvalidNameError, fundamentals_to_daily_panel

__all__ = [
    "FMPClient",
    "FMPClientConfig",
    "FMPInvalidNameError",
    "fundamentals_to_daily_panel",
]
