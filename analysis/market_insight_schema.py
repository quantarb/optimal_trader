from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from pipeline.feature_presentation import FeatureDefinition, format_feature_value, get_feature_definition, group_features_by_family
from .market_state import percentile_rank


PRIMARY_ANALOG_HORIZON_DAYS = 60
DEFAULT_FAMILY_LIMIT = 8
NEUTRAL_PERCENTILE = 0.5

STATE_METADATA_FIELDS = {
    "date",
    "symbol",
    "label",
    "market_position",
    "trade_return",
    "hold_days",
    "side",
    "freq",
    "k",
    "oracle_cluster_key",
}


@dataclass(frozen=True)
class CanonicalFeatureValue:
    internal_name: str
    display_name: str
    family: str
    raw_value: Any
    rendered_value: str
    format: str
    decimals: int
    percentile: float | None = None
    zscore: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalAnalog:
    symbol: str
    date: str
    similarity_score: float
    match_type: str
    returns_by_horizon: dict[str, float | None] = field(default_factory=dict)
    drawdowns_by_horizon: dict[str, float | None] = field(default_factory=dict)
    volatility_by_horizon: dict[str, float | None] = field(default_factory=dict)
    explanation_tags: list[str] = field(default_factory=list)
    cluster_id: str = ""
    cluster_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutcomeHorizonSummary:
    horizon_days: int
    median_return: float | None
    mean_return: float | None
    win_rate: float | None
    worst_case: float | None
    best_case: float | None
    tail_risk: float | None
    avg_drawdown: float | None
    avg_volatility: float | None
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalogOutcomeSummary:
    primary_horizon_days: int
    median_return: float | None
    mean_return: float | None
    win_rate: float | None
    worst_case: float | None
    best_case: float | None
    tail_risk: float | None
    avg_drawdown: float | None
    avg_volatility: float | None
    sample_size: int
    horizon_rows: list[OutcomeHorizonSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["horizon_rows"] = [row.to_dict() for row in self.horizon_rows]
        return payload


@dataclass(frozen=True)
class ModelScore:
    name: str
    display_name: str
    value: float | None
    rendered_value: str
    favorable_high: bool = True
    percentile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamiliaritySignals:
    market_familiarity_score: float | None
    market_familiarity_label: str
    confidence_score: float | None
    confidence_label: str
    ae_familiarity_raw: float | None = None
    analog_density: float | None = None
    mean_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterContext:
    cluster_id: str = ""
    description: str = ""
    similarity_score_pct: float | None = None
    feature_signature: list[str] = field(default_factory=list)
    outcome_statistics: dict[str, Any] = field(default_factory=dict)
    example_historical_dates: list[dict[str, Any]] = field(default_factory=list)
    nearest_clusters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketInsightInput:
    symbol: str
    as_of_date: str
    canonical_features: dict[str, list[CanonicalFeatureValue]]
    feature_family_summaries: dict[str, list[str]]
    same_symbol_analogs: list[HistoricalAnalog]
    cross_symbol_analogs: list[HistoricalAnalog]
    analog_outcome_summary: AnalogOutcomeSummary
    same_symbol_outcome_summary: AnalogOutcomeSummary
    cross_symbol_outcome_summary: AnalogOutcomeSummary
    model_scores: list[ModelScore]
    familiarity_signals: FamiliaritySignals
    cluster_context: ClusterContext
    optional_notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "canonical_features": {
                family: [item.to_dict() for item in items]
                for family, items in self.canonical_features.items()
            },
            "feature_family_summaries": dict(self.feature_family_summaries),
            "same_symbol_analogs": [item.to_dict() for item in self.same_symbol_analogs],
            "cross_symbol_analogs": [item.to_dict() for item in self.cross_symbol_analogs],
            "analog_outcome_summary": self.analog_outcome_summary.to_dict(),
            "same_symbol_outcome_summary": self.same_symbol_outcome_summary.to_dict(),
            "cross_symbol_outcome_summary": self.cross_symbol_outcome_summary.to_dict(),
            "model_scores": [item.to_dict() for item in self.model_scores],
            "familiarity_signals": self.familiarity_signals.to_dict(),
            "cluster_context": self.cluster_context.to_dict(),
            "optional_notes": dict(self.optional_notes),
        }


@dataclass(frozen=True)
class PortfolioHoldingInsightInput:
    symbol: str
    opportunity_score: float | None
    confidence_score: float | None
    familiarity_score: float | None
    risk_indicator: str
    cluster_id: str = ""
    cluster_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioInsightInput:
    symbols: list[str]
    as_of_date: str
    holdings: list[PortfolioHoldingInsightInput]
    cluster_exposure_rows: list[dict[str, Any]]
    portfolio_score: float
    regime_similarity_score: float
    risk_concentration_score: float
    optional_notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "as_of_date": self.as_of_date,
            "holdings": [item.to_dict() for item in self.holdings],
            "cluster_exposure_rows": [dict(item) for item in self.cluster_exposure_rows],
            "portfolio_score": self.portfolio_score,
            "regime_similarity_score": self.regime_similarity_score,
            "risk_concentration_score": self.risk_concentration_score,
            "optional_notes": dict(self.optional_notes),
        }


@dataclass(frozen=True)
class StockInsight:
    headline: str
    summary: str
    key_drivers: list[str]
    historical_context: list[str]
    opportunity_context: list[str]
    risk_flags: list[str]
    familiarity_comment: str
    confidence_comment: str
    supporting_evidence: dict[str, Any]
    llm_prompt: str = ""
    mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioInsight:
    headline: str
    summary: str
    strongest_holdings: list[str]
    weakest_holdings: list[str]
    concentration_summary: list[str]
    risk_flags: list[str]
    supporting_evidence: dict[str, Any]
    llm_prompt: str = ""
    mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketSituationExplanation:
    headline: str
    summary: str
    situation_context: list[str]
    supporting_evidence: dict[str, Any]
    llm_prompt: str = ""
    mode: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return float(parsed)


def _feature_baseline_stats(
    state_frame: pd.DataFrame,
    *,
    feature_name: str,
    numeric_value: float | None,
) -> tuple[float | None, float | None]:
    if numeric_value is None or feature_name not in state_frame.columns:
        return None, None
    baseline = pd.to_numeric(state_frame[feature_name], errors="coerce").dropna()
    if baseline.empty:
        return None, None
    percentile = percentile_rank(baseline, numeric_value)
    std = float(baseline.std(ddof=0) or 0.0)
    if std <= 0.0:
        return percentile, None
    return percentile, (numeric_value - float(baseline.mean())) / std


def _feature_priority_score(percentile: float | None, zscore: float | None) -> float:
    if zscore is not None:
        return abs(float(zscore))
    return abs((float(percentile or NEUTRAL_PERCENTILE) - NEUTRAL_PERCENTILE) * 2.0)


def _canonical_feature_item(
    *,
    family_name: str,
    definition: FeatureDefinition,
    raw_value: Any,
    state_frame: pd.DataFrame,
) -> tuple[float, CanonicalFeatureValue] | None:
    if definition.internal_name in STATE_METADATA_FIELDS:
        return None
    numeric_value = _safe_float(raw_value)
    percentile, zscore = _feature_baseline_stats(
        state_frame,
        feature_name=definition.internal_name,
        numeric_value=numeric_value,
    )
    item = CanonicalFeatureValue(
        internal_name=definition.internal_name,
        display_name=definition.display_name,
        family=family_name,
        raw_value=raw_value,
        rendered_value=format_feature_value(definition.internal_name, raw_value),
        format=definition.format,
        decimals=definition.decimals,
        percentile=percentile,
        zscore=zscore,
    )
    return _feature_priority_score(percentile, zscore), item


def _include_family(include_set: set[str], family_name: str) -> bool:
    return not include_set or family_name in include_set


def _row_canonical_features(
    row: Mapping[str, Any],
    state_frame: pd.DataFrame,
    *,
    feature_family_map: Mapping[str, Sequence[str]] | None = None,
    include_families: Sequence[str] | None = None,
    limit_per_family: int = DEFAULT_FAMILY_LIMIT,
) -> dict[str, list[CanonicalFeatureValue]]:
    family_items = group_features_by_family(dict(row), feature_family_map=feature_family_map)
    include_set = {str(value) for value in list(include_families or []) if str(value).strip()}
    output: dict[str, list[CanonicalFeatureValue]] = {}
    for family_name, rows in family_items.items():
        if not _include_family(include_set, str(family_name)):
            continue
        scoped = [
            ranked_item
            for definition, raw_value in rows
            if (ranked_item := _canonical_feature_item(
                family_name=str(family_name),
                definition=definition,
                raw_value=raw_value,
                state_frame=state_frame,
            )) is not None
        ]
        scoped.sort(key=lambda pair: (-pair[0], pair[1].display_name))
        output[str(family_name)] = [item for _, item in scoped[: max(int(limit_per_family), 1)]]
    return output


def _outcome_summary_from_payload(payload: Mapping[str, Any]) -> AnalogOutcomeSummary:
    horizon_rows = [
        OutcomeHorizonSummary(
            horizon_days=int(row.get("horizon_days") or 0),
            median_return=_safe_float(row.get("median_return")),
            mean_return=_safe_float(row.get("mean_return")),
            win_rate=_safe_float(row.get("win_rate")),
            worst_case=_safe_float(row.get("worst_case")),
            best_case=_safe_float(row.get("best_case")),
            tail_risk=_safe_float(row.get("tail_risk")),
            avg_drawdown=_safe_float(row.get("avg_drawdown")),
            avg_volatility=_safe_float(row.get("avg_volatility")),
            sample_size=int(row.get("sample_size") or 0),
        )
        for row in list(payload.get("horizon_rows") or [])
    ]
    return AnalogOutcomeSummary(
        primary_horizon_days=int(payload.get("primary_horizon_days") or PRIMARY_ANALOG_HORIZON_DAYS),
        median_return=_safe_float(payload.get("median_return")),
        mean_return=_safe_float(payload.get("mean_return")),
        win_rate=_safe_float(payload.get("win_rate")),
        worst_case=_safe_float(payload.get("worst_case")),
        best_case=_safe_float(payload.get("best_case")),
        tail_risk=_safe_float(payload.get("tail_risk")),
        avg_drawdown=_safe_float(payload.get("avg_drawdown")),
        avg_volatility=_safe_float(payload.get("avg_volatility")),
        sample_size=int(payload.get("sample_size") or 0),
        horizon_rows=horizon_rows,
    )


def _analogs_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[HistoricalAnalog]:
    return [
        HistoricalAnalog(
            symbol=str(row.get("symbol") or ""),
            date=str(row.get("date") or ""),
            similarity_score=float(row.get("similarity_score") or 0.0),
            match_type=str(row.get("match_type") or ""),
            returns_by_horizon={
                key: _safe_float(row.get(key))
                for key in ("return_5d", "return_20d", "return_60d", "return_90d", "return_180d")
            },
            drawdowns_by_horizon={
                key: _safe_float(row.get(key))
                for key in ("drawdown_20d", "drawdown_60d")
            },
            volatility_by_horizon={
                key: _safe_float(row.get(key))
                for key in ("volatility_20d", "volatility_60d")
            },
            explanation_tags=[
                str((item or {}).get("explanation") or "")
                for item in list(row.get("explanations") or [])
                if str((item or {}).get("explanation") or "").strip()
            ],
            cluster_id=str(row.get("cluster_id") or ""),
            cluster_description=str(row.get("cluster_description") or ""),
        )
        for row in list(rows or [])
    ]


def _model_scores_from_row(
    row: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    state_frame: pd.DataFrame,
) -> list[ModelScore]:
    candidates: list[tuple[str, str, Any, bool]] = [
        ("opportunity_score", "Opportunity Score", opportunity.get("opportunity_score"), True),
        ("confidence_score", "Confidence Score", opportunity.get("confidence_score"), True),
        ("market_familiarity_score", "Market Familiarity", opportunity.get("market_familiarity_score"), True),
        ("risk_score", "Risk Score", opportunity.get("risk_score"), False),
        ("prob_buy", get_feature_definition("prob_buy").display_name, row.get("prob_buy"), True),
        ("ranking", get_feature_definition("ranking").display_name, row.get("ranking"), True),
        ("combined_score", get_feature_definition("combined_score").display_name, row.get("combined_score"), True),
        ("strategy_score", get_feature_definition("strategy_score").display_name, row.get("strategy_score"), True),
        ("prediction_score", get_feature_definition("prediction_score").display_name, row.get("prediction_score"), True),
        ("ae_familiarity", get_feature_definition("ae_familiarity").display_name, row.get("ae_familiarity"), True),
    ]
    rows: list[ModelScore] = []
    for name, display_name, value, favorable_high in candidates:
        numeric_value = _safe_float(value)
        if numeric_value is None:
            continue
        percentile = None
        if name in state_frame.columns:
            baseline = pd.to_numeric(state_frame[name], errors="coerce").dropna()
            if not baseline.empty:
                percentile = percentile_rank(baseline, numeric_value)
        definition = get_feature_definition(name)
        rows.append(
            ModelScore(
                name=name,
                display_name=display_name,
                value=numeric_value,
                rendered_value=format_feature_value(definition.internal_name, numeric_value),
                favorable_high=bool(favorable_high),
                percentile=percentile,
            )
        )
    return rows


def _mean_similarity(analog_rows: list[Mapping[str, Any]]) -> float:
    if not analog_rows:
        return 0.0
    return sum(float(analog_row.get("similarity_score") or 0.0) for analog_row in analog_rows) / float(len(analog_rows))


def _familiarity_signals(
    *,
    row: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    analog_rows: list[Mapping[str, Any]],
) -> FamiliaritySignals:
    return FamiliaritySignals(
        market_familiarity_score=_safe_float(opportunity.get("market_familiarity_score")),
        market_familiarity_label=str(opportunity.get("market_familiarity_label") or ""),
        confidence_score=_safe_float(opportunity.get("confidence_score")),
        confidence_label=str(opportunity.get("confidence_label") or ""),
        ae_familiarity_raw=_safe_float(row.get("ae_familiarity")),
        analog_density=float(len(analog_rows)),
        mean_similarity=_mean_similarity(analog_rows),
    )


def _cluster_context(
    *,
    current_cluster: Mapping[str, Any] | None,
    nearest_clusters: Sequence[Mapping[str, Any]],
) -> ClusterContext:
    cluster_payload = dict(current_cluster or {})
    return ClusterContext(
        cluster_id=str(cluster_payload.get("cluster_id") or ""),
        description=str(cluster_payload.get("description") or ""),
        similarity_score_pct=_safe_float(cluster_payload.get("similarity_score_pct")),
        feature_signature=[str(value) for value in list(cluster_payload.get("feature_signature") or []) if str(value).strip()],
        outcome_statistics=dict(cluster_payload.get("outcome_statistics") or {}),
        example_historical_dates=[dict(item) for item in list(cluster_payload.get("example_historical_dates") or [])],
        nearest_clusters=[dict(item) for item in list(nearest_clusters or [])],
    )


def _feature_family_summaries(
    canonical_features: dict[str, list[CanonicalFeatureValue]],
) -> dict[str, list[str]]:
    return {
        family: [f"{feature.display_name}: {feature.rendered_value}" for feature in rows]
        for family, rows in canonical_features.items()
    }


def _market_analog_components(
    *,
    same_symbol_analogs: Sequence[Mapping[str, Any]],
    cross_symbol_analogs: Sequence[Mapping[str, Any]],
    analog_outcome_summary: Mapping[str, Any],
    same_symbol_outcome_summary: Mapping[str, Any],
    cross_symbol_outcome_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "same_symbol_analogs": _analogs_from_rows(same_symbol_analogs),
        "cross_symbol_analogs": _analogs_from_rows(cross_symbol_analogs),
        "analog_outcome_summary": _outcome_summary_from_payload(analog_outcome_summary),
        "same_symbol_outcome_summary": _outcome_summary_from_payload(same_symbol_outcome_summary),
        "cross_symbol_outcome_summary": _outcome_summary_from_payload(cross_symbol_outcome_summary),
    }


def _market_scoring_components(
    *,
    row: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    state_frame: pd.DataFrame,
    analog_rows: list[Mapping[str, Any]],
    current_cluster: Mapping[str, Any] | None,
    nearest_clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model_scores": _model_scores_from_row(row, opportunity, state_frame),
        "familiarity_signals": _familiarity_signals(
            row=row,
            opportunity=opportunity,
            analog_rows=analog_rows,
        ),
        "cluster_context": _cluster_context(
            current_cluster=current_cluster,
            nearest_clusters=nearest_clusters,
        ),
    }


def _market_insight_components(
    *,
    symbol: str,
    as_of_date: str,
    row: Mapping[str, Any],
    state_frame: pd.DataFrame,
    feature_family_map: Mapping[str, Sequence[str]] | None,
    same_symbol_analogs: Sequence[Mapping[str, Any]],
    cross_symbol_analogs: Sequence[Mapping[str, Any]],
    analog_outcome_summary: Mapping[str, Any],
    same_symbol_outcome_summary: Mapping[str, Any],
    cross_symbol_outcome_summary: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    current_cluster: Mapping[str, Any] | None = None,
    nearest_clusters: Sequence[Mapping[str, Any]] = (),
    optional_notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_features = _row_canonical_features(
        row=row,
        state_frame=state_frame,
        feature_family_map=feature_family_map,
    )
    analog_rows = list(same_symbol_analogs or []) + list(cross_symbol_analogs or [])
    return {
        "symbol": str(symbol).upper(),
        "as_of_date": str(as_of_date),
        "canonical_features": canonical_features,
        "feature_family_summaries": _feature_family_summaries(canonical_features),
        **_market_analog_components(
            same_symbol_analogs=same_symbol_analogs,
            cross_symbol_analogs=cross_symbol_analogs,
            analog_outcome_summary=analog_outcome_summary,
            same_symbol_outcome_summary=same_symbol_outcome_summary,
            cross_symbol_outcome_summary=cross_symbol_outcome_summary,
        ),
        **_market_scoring_components(
            row=row,
            opportunity=opportunity,
            state_frame=state_frame,
            analog_rows=analog_rows,
            current_cluster=current_cluster,
            nearest_clusters=nearest_clusters,
        ),
        "optional_notes": dict(optional_notes or {}),
    }


def _compose_market_insight(**components: Any) -> MarketInsightInput:
    return MarketInsightInput(**components)


def build_market_insight_input(
    *,
    symbol: str,
    as_of_date: str,
    row: Mapping[str, Any],
    state_frame: pd.DataFrame,
    feature_family_map: Mapping[str, Sequence[str]] | None,
    same_symbol_analogs: Sequence[Mapping[str, Any]],
    cross_symbol_analogs: Sequence[Mapping[str, Any]],
    analog_outcome_summary: Mapping[str, Any],
    same_symbol_outcome_summary: Mapping[str, Any],
    cross_symbol_outcome_summary: Mapping[str, Any],
    opportunity: Mapping[str, Any],
    current_cluster: Mapping[str, Any] | None = None,
    nearest_clusters: Sequence[Mapping[str, Any]] = (),
    optional_notes: Mapping[str, Any] | None = None,
) -> MarketInsightInput:
    return _compose_market_insight(
        **_market_insight_components(
            symbol=symbol,
            as_of_date=as_of_date,
            row=row,
            state_frame=state_frame,
            feature_family_map=feature_family_map,
            same_symbol_analogs=same_symbol_analogs,
            cross_symbol_analogs=cross_symbol_analogs,
            analog_outcome_summary=analog_outcome_summary,
            same_symbol_outcome_summary=same_symbol_outcome_summary,
            cross_symbol_outcome_summary=cross_symbol_outcome_summary,
            opportunity=opportunity,
            current_cluster=current_cluster,
            nearest_clusters=nearest_clusters,
            optional_notes=optional_notes,
        )
    )


def _portfolio_holdings(rows: Sequence[Mapping[str, Any]]) -> list[PortfolioHoldingInsightInput]:
    return [
        PortfolioHoldingInsightInput(
            symbol=str(row.get("symbol") or ""),
            opportunity_score=_safe_float(row.get("opportunity_score")),
            confidence_score=_safe_float(row.get("confidence_score")),
            familiarity_score=_safe_float(row.get("market_familiarity_score")),
            risk_indicator=str(row.get("risk_indicator") or ""),
            cluster_id=str(row.get("cluster_id") or ""),
            cluster_description=str(row.get("cluster_description") or ""),
        )
        for row in list(rows or [])
    ]


def _portfolio_insight_components(
    *,
    symbols: Sequence[str],
    as_of_date: str,
    rows: Sequence[Mapping[str, Any]],
    cluster_exposure_rows: Sequence[Mapping[str, Any]],
    portfolio_score: float,
    regime_similarity_score: float,
    risk_concentration_score: float,
    optional_notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "symbols": [str(value).upper() for value in list(symbols or []) if str(value).strip()],
        "as_of_date": str(as_of_date),
        "holdings": _portfolio_holdings(rows),
        "cluster_exposure_rows": [dict(item) for item in list(cluster_exposure_rows or [])],
        "portfolio_score": float(portfolio_score or 0.0),
        "regime_similarity_score": float(regime_similarity_score or 0.0),
        "risk_concentration_score": float(risk_concentration_score or 0.0),
        "optional_notes": dict(optional_notes or {}),
    }


def _compose_portfolio_insight(**components: Any) -> PortfolioInsightInput:
    return PortfolioInsightInput(**components)


def build_portfolio_insight_input(
    *,
    symbols: Sequence[str],
    as_of_date: str,
    rows: Sequence[Mapping[str, Any]],
    cluster_exposure_rows: Sequence[Mapping[str, Any]],
    portfolio_score: float,
    regime_similarity_score: float,
    risk_concentration_score: float,
    optional_notes: Mapping[str, Any] | None = None,
) -> PortfolioInsightInput:
    return _compose_portfolio_insight(
        **_portfolio_insight_components(
            symbols=symbols,
            as_of_date=as_of_date,
            rows=rows,
            cluster_exposure_rows=cluster_exposure_rows,
            portfolio_score=portfolio_score,
            regime_similarity_score=regime_similarity_score,
            risk_concentration_score=risk_concentration_score,
            optional_notes=optional_notes,
        )
    )
