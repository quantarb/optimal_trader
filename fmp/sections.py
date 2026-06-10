"""Central definitions for FMP symbol sections and required section groups.

This module owns the canonical list of section keys used for historical and snapshot
data, the groups needed for scoring/backtesting/live, and will evolve into the
typed section registry (builders + stability metadata).
"""

from __future__ import annotations

from typing import Sequence

# Core required fundamentals for most modeling / labeling work.
REQUIRED_FUNDAMENTAL_SECTION_KEYS: tuple[str, ...] = (
    "key_metrics",
    "ratios",
    "income_statement",
    "income_statement_ttm",
    "income_statement_growth",
    "cash_flow",
    "cash_flow_ttm",
    "cash_flow_growth",
    "balance_sheet",
    "balance_sheet_ttm",
    "balance_sheet_growth",
    "financial_growth",
    "earnings",
    "dividends",
    "splits",
)

# Historical sections typically required before building features/labels for scoring.
REQUIRED_SCORING_HISTORICAL_SECTIONS: tuple[str, ...] = (
    "prices_div_adj",
    *REQUIRED_FUNDAMENTAL_SECTION_KEYS,
)

# Anchors used to decide if fundamentals are "fresh enough".
_FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS: tuple[str, ...] = (
    "income_statement",
    "balance_sheet",
    "cash_flow",
)
_FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS: tuple[str, ...] = ("earnings",)

# Sections that are sparse / event-like (not dense periodic).
_SPARSE_EVENT_HISTORICAL_SECTION_KEYS: frozenset[str] = frozenset({"dividends", "splits"})

# Sections whose freshness often depends on the fundamental statement anchor.
_FUNDAMENTAL_DEPENDENT_SECTION_KEYS: frozenset[str] = frozenset(
    {
        "key_metrics",
        "ratios",
        "income_statement_growth",
        "cash_flow_growth",
        "balance_sheet_growth",
        "financial_growth",
    }
)

# Default history target when doing full universe backfills.
_UNIVERSE_DOWNLOAD_TARGET_YEARS = 20
_UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD = 0.75

# Retry policy for bulk symbol section downloads (used by tasks + universe refresh).
_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS = 3
_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S = 1.5


def get_required_scoring_historical_sections() -> tuple[str, ...]:
    return REQUIRED_SCORING_HISTORICAL_SECTIONS


def get_required_fundamental_section_keys() -> tuple[str, ...]:
    return REQUIRED_FUNDAMENTAL_SECTION_KEYS


__all__ = [
    "REQUIRED_FUNDAMENTAL_SECTION_KEYS",
    "REQUIRED_SCORING_HISTORICAL_SECTIONS",
    "get_required_fundamental_section_keys",
    "get_required_scoring_historical_sections",
    # The leading-underscore ones are for internal use by refresh/stability but exported
    # so that trading/views/etc. that previously defined their own copies can migrate.
    "_FUNDAMENTAL_STATEMENT_ANCHOR_SECTION_KEYS",
    "_FUNDAMENTAL_FALLBACK_ANCHOR_SECTION_KEYS",
    "_SPARSE_EVENT_HISTORICAL_SECTION_KEYS",
    "_FUNDAMENTAL_DEPENDENT_SECTION_KEYS",
    "_UNIVERSE_DOWNLOAD_TARGET_YEARS",
    "_UNIVERSE_DOWNLOAD_COVERAGE_THRESHOLD",
    "_UNIVERSE_DOWNLOAD_RETRY_MAX_ATTEMPTS",
    "_UNIVERSE_DOWNLOAD_RETRY_BASE_DELAY_S",
]
