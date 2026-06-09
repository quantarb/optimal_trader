from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from domain.features.specs import FeatureBuildSpec, FeatureToggleSpec, RepresentationEmbeddingSpec


SECTION_ORDER = [
    "prices_div_adj",
    "technical_candles",
    "technical_cycles",
    "technical_math",
    "technical_momentum",
    "technical_overlap",
    "technical_performance",
    "time_calendar",
    "key_metrics",
    "ratios",
    "key_metrics_ttm",
    "ratios_ttm",
    "income_statement_ttm",
    "cash_flow_ttm",
    "balance_sheet_ttm",
    "income_statement",
    "income_statement_growth",
    "cash_flow",
    "cash_flow_growth",
    "balance_sheet",
    "balance_sheet_growth",
    "financial_growth",
    "earnings",
    "analyst_estimates",
    "ratings_historical",
    "grades_historical",
    "insider_trading",
    "economic_indicators",
    "treasury_rates",
    "representation_embedding",
]

SECTION_LABELS = {
    "prices_div_adj": "Prices Div Adj",
    "technical_candles": "Technical Candles",
    "technical_cycles": "Technical Cycles",
    "technical_math": "Technical Math",
    "technical_momentum": "Technical Momentum",
    "technical_overlap": "Technical Overlap",
    "technical_performance": "Technical Performance",
    "time_calendar": "Time Calendar",
    "key_metrics": "Key Metrics",
    "ratios": "Ratios",
    "key_metrics_ttm": "Key Metrics TTM",
    "ratios_ttm": "Ratios TTM",
    "income_statement_ttm": "Income Statement TTM",
    "cash_flow_ttm": "Cash Flow TTM",
    "balance_sheet_ttm": "Balance Sheet TTM",
    "income_statement": "Income Statement",
    "income_statement_growth": "Income Statement Growth",
    "cash_flow": "Cash Flow",
    "cash_flow_growth": "Cash Flow Growth",
    "balance_sheet": "Balance Sheet",
    "balance_sheet_growth": "Balance Sheet Growth",
    "financial_growth": "Financial Growth",
    "earnings": "Earnings",
    "analyst_estimates": "Analyst Estimates",
    "ratings_historical": "Ratings Historical",
    "grades_historical": "Grades Historical",
    "insider_trading": "Insider Trading",
    "economic_indicators": "Economic Indicators",
    "treasury_rates": "Treasury Rates",
    "representation_embedding": "Representation Embedding",
}

REPRESENTATION_EMBEDDING_MODEL_VERSION = "semantic_grouped_v2"
REPRESENTATION_EMBEDDING_FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "price_technical": (
        "prices_div_adj",
        "technical_candles",
        "technical_cycles",
        "technical_math",
        "technical_momentum",
        "technical_overlap",
        "technical_performance",
    ),
    "time_calendar": ("time_calendar",),
    "valuation_quality": ("key_metrics", "ratios"),
    "ttm_financial_statements": (
        "key_metrics_ttm",
        "ratios_ttm",
        "income_statement_ttm",
        "cash_flow_ttm",
        "balance_sheet_ttm",
    ),
    "income_statement": ("income_statement", "income_statement_growth"),
    "cash_flow": ("cash_flow", "cash_flow_growth"),
    "balance_sheet": ("balance_sheet", "balance_sheet_growth"),
    "broad_fundamental_growth": ("financial_growth",),
    "earnings_analyst_sentiment": ("earnings", "analyst_estimates", "ratings_historical", "grades_historical"),
    "insider_ownership": ("insider_trading",),
    "macro_rates": ("economic_indicators", "treasury_rates"),
}


def feature_toggle_data(source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility helper returning toggle payloads as a plain dict."""

    return FeatureToggleSpec.from_mapping(source).to_dict()


def representation_embedding_config(source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compatibility helper returning embedding settings as a plain dict."""

    default_store_dir = Path.cwd() / "data" / "embedding_store"
    return RepresentationEmbeddingSpec.from_mapping(
        source,
        default_store_dir=str(default_store_dir),
        default_model_version=REPRESENTATION_EMBEDDING_MODEL_VERSION,
    ).to_dict()


def resolve_feature_date_window(config: FeatureBuildSpec | dict[str, Any] | None) -> tuple[str | None, str | None]:
    if isinstance(config, FeatureBuildSpec):
        return config.start_date, config.end_date
    spec = FeatureBuildSpec.from_mapping(
        config,
        default_store_dir=str(Path.cwd() / "data" / "embedding_store"),
        default_model_version=REPRESENTATION_EMBEDDING_MODEL_VERSION,
    )
    return spec.start_date, spec.end_date


def needed_sparse_sections(feature_flags: FeatureToggleSpec | dict[str, Any]) -> list[str]:
    toggles = feature_flags if isinstance(feature_flags, FeatureToggleSpec) else FeatureToggleSpec.from_mapping(feature_flags)
    sections: list[str] = []
    if toggles.include_fundamental_change:
        sections.extend(["key_metrics", "ratios"])
    if toggles.include_ttm_financial_statements:
        sections.extend(
            [
                "key_metrics_ttm",
                "ratios_ttm",
                "income_statement_ttm",
                "cash_flow_ttm",
                "balance_sheet_ttm",
            ]
        )
    if getattr(toggles, "include_time_calendar_features", False):
        sections.extend(["earnings", "dividends", "splits"])
    if toggles.include_statement_quality:
        sections.extend(
            [
                "income_statement",
                "income_statement_growth",
                "cash_flow",
                "cash_flow_growth",
                "balance_sheet",
                "balance_sheet_growth",
                "financial_growth",
            ]
        )
    if toggles.include_event_features:
        sections.extend(["earnings", "analyst_estimates", "ratings_historical", "grades_historical"])
    if toggles.include_ownership_features:
        sections.append("insider_trading")
    return list(dict.fromkeys(sections))


def build_feature_family_coverage_row(section_label: str, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    if df.empty or not feature_cols:
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    usable_cols = [col for col in feature_cols if col in df.columns]
    if not usable_cols:
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    mask = df[usable_cols].notna().any(axis=1)
    if not mask.any():
        return {"section_label": section_label, "min_date": None, "max_date": None, "count": 0}
    valid_index = df.index[mask]
    if isinstance(valid_index, pd.MultiIndex):
        dates = pd.to_datetime(valid_index.get_level_values("date"))
    else:
        dates = pd.to_datetime(valid_index)
    return {
        "section_label": section_label,
        "min_date": dates.min().date().isoformat() if len(dates) else None,
        "max_date": dates.max().date().isoformat() if len(dates) else None,
        "count": int(mask.sum()),
    }


def append_representation_embedding_columns(
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
    *,
    config: dict[str, Any] | RepresentationEmbeddingSpec,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    embedding_spec = config if isinstance(config, RepresentationEmbeddingSpec) else RepresentationEmbeddingSpec(**dict(config or {}))
    if symbol_df.empty or not embedding_spec.enabled:
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(embedding_spec.model_name),
            "model_version": str(embedding_spec.model_version),
            "store_dir": str(embedding_spec.store_dir),
            "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
        }

    build_dataset_embeddings, encoder = _resolve_representation_embedding_backend(embedding_spec)
    dataset_rows = representation_embedding_dataset_rows(symbol_df, grouped_feature_columns)
    if not dataset_rows:
        return symbol_df, [], {
            "enabled": False,
            "columns": [],
            "dimension": 0,
            "model_name": str(getattr(encoder, "model_name", embedding_spec.model_name)),
            "model_version": str(getattr(encoder, "model_version", embedding_spec.model_version)),
            "store_dir": str(embedding_spec.store_dir),
            "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
        }

    embedding_rows = build_dataset_embeddings(
        dataset_rows,
        encoder=encoder,
        store_dir=str(embedding_spec.store_dir),
    )
    first_vector_value = embedding_rows[0].get("embedding_vector")
    first_vector = list(first_vector_value) if first_vector_value is not None else []
    embedding_columns = [f"{embedding_spec.column_prefix}{idx}" for idx in range(len(first_vector))]
    embedding_df = pd.DataFrame(
        [
            {column: float(vector[idx]) for idx, column in enumerate(embedding_columns)}
            for vector in [
                list(item.get("embedding_vector")) if item.get("embedding_vector") is not None else []
                for item in embedding_rows
            ]
        ]
    )
    augmented = pd.concat([symbol_df.reset_index(drop=True), embedding_df], axis=1)
    return augmented, embedding_columns, {
        "enabled": bool(embedding_columns),
        "columns": list(embedding_columns),
        "dimension": len(embedding_columns),
        "model_name": str(getattr(encoder, "model_name", embedding_spec.model_name)),
        "model_version": str(getattr(encoder, "model_version", embedding_spec.model_version)),
        "store_dir": str(embedding_spec.store_dir),
        "family_groups": {key: list(value) for key, value in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items()},
    }


def representation_embedding_dataset_rows(
    symbol_df: pd.DataFrame,
    grouped_feature_columns: dict[str, list[str]],
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    embedding_family_columns = representation_embedding_grouped_feature_columns(grouped_feature_columns)
    usable_families = [
        str(family_name)
        for family_name, columns in embedding_family_columns.items()
        if list(columns or [])
    ]
    for row in symbol_df.to_dict(orient="records"):
        families: dict[str, dict[str, Any]] = {}
        for family_name in usable_families:
            values: dict[str, Any] = {}
            for column in list(embedding_family_columns.get(family_name) or []):
                if column not in row:
                    continue
                value = row.get(column)
                if representation_embedding_missing_value(value):
                    continue
                display_name = representation_embedding_feature_name(column, values)
                values[display_name] = value
            if values:
                families[family_name] = values
        if not families:
            raise ValueError(
                f"Representation embedding requested but no family features were available for "
                f"{row.get('symbol')} on {row.get('date')}."
            )
        dataset_rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "date": str(row.get("date") or ""),
                "families": families,
            }
        )
    return dataset_rows


def representation_embedding_grouped_feature_columns(
    grouped_feature_columns: dict[str, list[str]],
) -> dict[str, list[str]]:
    semantic_groups: dict[str, list[str]] = {}
    for family_name, source_families in REPRESENTATION_EMBEDDING_FAMILY_GROUPS.items():
        merged_columns: list[str] = []
        for source_family in source_families:
            merged_columns.extend(str(column) for column in list(grouped_feature_columns.get(source_family) or []))
        semantic_groups[family_name] = list(dict.fromkeys(merged_columns))
    return semantic_groups


def representation_embedding_feature_name(column: str, existing_values: dict[str, Any]) -> str:
    try:
        from features.naming import feature_display_name
    except Exception:
        feature_display_name = lambda value: value
    display_name = str(feature_display_name(str(column)) or str(column)).strip() or str(column)
    if display_name not in existing_values:
        return display_name
    return f"{display_name} [{column}]"


def representation_embedding_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip().lower() in {"", "nan", "none", "null", "<na>", "n/a", "na"}
    if isinstance(value, (list, tuple, set)):
        return not any(not representation_embedding_missing_value(item) for item in value)
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _resolve_representation_embedding_backend(config: RepresentationEmbeddingSpec):
    from analysis.feature_embeddings.encoder import SentenceTransformerEncoder
    from analysis.feature_embeddings.pipeline import build_dataset_embeddings

    encoder = SentenceTransformerEncoder(
        model_name=str(config.model_name),
        model_version=str(config.model_version),
        local_files_only=bool(config.local_files_only),
        device=config.device,
    )
    return build_dataset_embeddings, encoder
