"""Run the multi-rate transformer through the canonical anchored-WFO workflow.

The established symbol-year runner already owns the data preparation, fold-safe
normalisation, training loop, shared-book backtest, and artifact schema.  This
entry point selects its issuer-encoder/instrument-decoder architecture instead
of maintaining a second, drifting WFO implementation.

All normal ``run_symbol_year_transformer_mtl.py`` environment variables remain
available.  In particular, ``TRANSFORMER_FIRST_TEST_YEAR``,
``TRANSFORMER_LAST_TEST_YEAR``, ``TRANSFORMER_TIERS``, ``TRANSFORMER_EPOCHS``,
and the backtest settings can be used unchanged.
"""

from __future__ import annotations

import os


def _enable_multirate_defaults() -> None:
    """Select the multi-rate architecture without overwriting user settings."""
    os.environ.setdefault("TRANSFORMER_ISSUER_ENCODER_INSTRUMENT_DECODER", "1")
    os.environ.setdefault("TRANSFORMER_MULTIRATE_ISSUER_STATE", "1")
    # This entry point is explicitly anchored WFO: each test year gets a fresh
    # fit using only observations before that year's test period.
    os.environ.setdefault("TRANSFORMER_SINGLE_FIT", "0")


def run() -> None:
    """Execute the canonical anchored WFO runner with multi-rate enabled."""
    _enable_multirate_defaults()

    # Import after setting defaults because the reference runner reads its
    # configuration at module import time.
    from run_symbol_year_transformer_mtl import main

    main()


if __name__ == "__main__":
    run()
