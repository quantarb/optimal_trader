from __future__ import annotations

import pandas as pd

from fmp.models import Symbol
from features.section_utils import BuiltFeatureSet, build_passthrough_section_features


def build_cash_flow_growth_features(
    symbol_obj: Symbol,
    target_index: pd.MultiIndex,
    filing_lag_days: int = 45,
) -> BuiltFeatureSet:
    return build_passthrough_section_features(
        symbol_obj,
        target_index,
        section_key="cash_flow_growth",
        prefix="cfg__",
        filing_lag_days=filing_lag_days,
    )
