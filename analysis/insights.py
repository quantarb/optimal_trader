from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .historical_outcomes import DEFAULT_HORIZONS
from .insight_composer import build_market_situation_explanation, build_portfolio_insight, build_stock_insight
from .historical_situation_search import (
    DEFAULT_SEARCH_HORIZONS,
    build_historical_situation_search_bundle,
    search_market_state_neighbors,
    summarize_historical_outcomes,
)
from .market_insight_schema import build_market_insight_input, build_portfolio_insight_input
from .market_state import (
    compute_market_state_embedding,
    latest_rows_by_symbol,
    load_market_state_frame,
    resolve_insight_artifacts,
    resolve_price_column,
)
from .opportunity_scoring import compute_opportunity_summary, explain_key_drivers
from .situation_similarity import (
    find_nearest_clusters,
    load_market_situation_cluster_artifact,
    resolve_market_situation_artifact,
)
from .state_embedding import compute_state_embedding as compute_cluster_embedding
from .state_representations import TextEmbeddingConfig


def build_stock_intelligence(
    *,
    symbol: str,
    date: str | None = None,
    strategy_artifact_id: int = 0,
    feature_artifact_id: int = 0,
    label_artifact_id: int = 0,
    prediction_artifact_ids: Sequence[int] | None = None,
    market_situation_artifact_id: int = 0,
    twin_count: int = 10,
    search_method: str = "hybrid",
    reasoning_mode: str = "deterministic",
) -> dict[str, Any]:
    artifacts = resolve_insight_artifacts(
        strategy_artifact_id=int(strategy_artifact_id or 0),
        feature_artifact_id=int(feature_artifact_id or 0),
        label_artifact_id=int(label_artifact_id or 0),
        prediction_artifact_ids=list(prediction_artifact_ids or []),
    )
    payload = compute_market_state_embedding(
        symbol=symbol,
        date=date,
        strategy_artifact=artifacts.strategy_artifact,
        feature_artifact=artifacts.feature_artifact,
        label_artifact=artifacts.label_artifact,
        prediction_artifacts=artifacts.prediction_artifacts,
    )
    frame = payload["frame"]
    row = dict(payload["row"])
    query_date = str(payload["date"])
    embedding_columns = list(payload["embedding_columns"] or [])
    market_situation_artifact = resolve_market_situation_artifact(artifact_id=int(market_situation_artifact_id or 0))
    current_cluster: dict[str, Any] = {}
    nearest_clusters: list[dict[str, Any]] = []
    if market_situation_artifact is not None:
        try:
            cluster_bundle = load_market_situation_cluster_artifact(market_situation_artifact)
            cluster_embedding = compute_cluster_embedding(row, cluster_bundle.embedding_model)
            nearest_clusters = find_nearest_clusters(
                cluster_embedding,
                cluster_bundle,
                side=str(row.get("side") or "").strip().lower() or None,
                top_n=3,
            )
            if nearest_clusters:
                current_cluster = dict(nearest_clusters[0])
        except Exception:
            current_cluster = {}
            nearest_clusters = []
    search_bundle = build_historical_situation_search_bundle(
        frame,
        feature_columns=embedding_columns,
        feature_family_map=dict((payload.get("meta") or {}).get("feature_family_map") or {}),
        text_embedding_config=TextEmbeddingConfig(backend="auto", embedding_dim=128),
    )
    same_symbol_matches = search_market_state_neighbors(
        row,
        search_bundle,
        method=search_method,
        top_k=max(int(twin_count), 1),
        search_mode="same_symbol",
        query_symbol=str(symbol).strip().upper(),
        query_date=query_date,
    )
    cross_symbol_matches = search_market_state_neighbors(
        row,
        search_bundle,
        method=search_method,
        top_k=max(int(twin_count), 1),
        search_mode="cross_symbol",
        query_symbol=str(symbol).strip().upper(),
        query_date=query_date,
    )
    matches = search_market_state_neighbors(
        row,
        search_bundle,
        method=search_method,
        top_k=max(int(twin_count), 1),
        search_mode="mixed",
        query_symbol=str(symbol).strip().upper(),
        query_date=query_date,
    )
    price_col = resolve_price_column(frame)
    mixed_outcomes = summarize_historical_outcomes(matches, frame, price_col=price_col, horizons=DEFAULT_SEARCH_HORIZONS)
    same_symbol_outcomes = summarize_historical_outcomes(same_symbol_matches, frame, price_col=price_col, horizons=DEFAULT_SEARCH_HORIZONS)
    cross_symbol_outcomes = summarize_historical_outcomes(cross_symbol_matches, frame, price_col=price_col, horizons=DEFAULT_SEARCH_HORIZONS)
    matches = list(mixed_outcomes.get("matches") or [])
    same_symbol_matches = list(same_symbol_outcomes.get("matches") or [])
    cross_symbol_matches = list(cross_symbol_outcomes.get("matches") or [])
    outcome_summary = dict(mixed_outcomes.get("summary") or {})
    scoring = compute_opportunity_summary(
        row=row,
        state_frame=frame,
        outcome_summary=outcome_summary,
        similarity_rows=matches,
    )
    drivers = explain_key_drivers(
        row=row,
        state_frame=frame,
        embedding_columns=embedding_columns,
    )

    def _shape_match_rows(source_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        shaped: list[dict[str, Any]] = []
        for match in list(source_rows or []):
            shaped.append(
                {
                    "symbol": str(match.get("symbol") or ""),
                    "date": str(match.get("date") or ""),
                    "similarity_score": round(float(match.get("similarity_score") or 0.0) * 100.0, 2),
                    "numeric_similarity": round(float(match.get("numeric_similarity") or 0.0) * 100.0, 2),
                    "embedding_similarity": round(float(match.get("embedding_similarity") or 0.0) * 100.0, 2),
                    "match_type": str(match.get("match_type") or "mixed"),
                    "return_5d": match.get("return_5d"),
                    "return_20d": match.get("return_20d"),
                    "return_60d": match.get("return_60d"),
                    "return_90d": match.get("return_90d"),
                    "return_180d": match.get("return_180d"),
                    "drawdown_20d": match.get("drawdown_20d"),
                    "drawdown_60d": match.get("drawdown_60d"),
                    "volatility_20d": match.get("volatility_20d"),
                    "volatility_60d": match.get("volatility_60d"),
                    "oracle_cluster_key": match.get("oracle_cluster_key") or "",
                    "cluster_id": match.get("cluster_id") or "",
                    "cluster_description": match.get("cluster_description") or "",
                    "explanations": list(match.get("explanations") or []),
                }
            )
        return shaped

    top_twins = _shape_match_rows(matches)
    same_symbol_twins = _shape_match_rows(same_symbol_matches)
    cross_symbol_twins = _shape_match_rows(cross_symbol_matches)
    reasoning_input = build_market_insight_input(
        symbol=str(symbol).strip().upper(),
        as_of_date=query_date,
        row=row,
        state_frame=frame,
        feature_family_map=dict((payload.get("meta") or {}).get("feature_family_map") or {}),
        same_symbol_analogs=same_symbol_twins,
        cross_symbol_analogs=cross_symbol_twins,
        analog_outcome_summary=outcome_summary,
        same_symbol_outcome_summary=dict(same_symbol_outcomes.get("summary") or {}),
        cross_symbol_outcome_summary=dict(cross_symbol_outcomes.get("summary") or {}),
        opportunity=scoring,
        current_cluster=current_cluster,
        nearest_clusters=nearest_clusters,
        optional_notes={
            "artifacts": {
                "strategy_artifact_id": int(artifacts.strategy_artifact.id) if artifacts.strategy_artifact is not None else 0,
                "feature_artifact_id": int(artifacts.feature_artifact.id) if artifacts.feature_artifact is not None else 0,
                "label_artifact_id": int(artifacts.label_artifact.id) if artifacts.label_artifact is not None else 0,
                "prediction_artifact_ids": [int(artifact.id) for artifact in artifacts.prediction_artifacts],
                "market_situation_artifact_id": int(market_situation_artifact.id) if market_situation_artifact is not None else 0,
            },
            "search_method": str(search_method or "hybrid"),
            "source": str((payload.get("meta") or {}).get("source") or ""),
        },
    )
    resolved_reasoning_mode = str(reasoning_mode or "deterministic")
    stock_insight = build_stock_insight(reasoning_input, mode=resolved_reasoning_mode)
    market_situation_explanation = build_market_situation_explanation(reasoning_input, mode=resolved_reasoning_mode)
    return {
        "symbol": str(symbol).strip().upper(),
        "date": query_date,
        "artifacts": {
            "strategy_artifact_id": int(artifacts.strategy_artifact.id) if artifacts.strategy_artifact is not None else 0,
            "feature_artifact_id": int(artifacts.feature_artifact.id) if artifacts.feature_artifact is not None else 0,
            "label_artifact_id": int(artifacts.label_artifact.id) if artifacts.label_artifact is not None else 0,
            "prediction_artifact_ids": [int(artifact.id) for artifact in artifacts.prediction_artifacts],
            "market_situation_artifact_id": int(market_situation_artifact.id) if market_situation_artifact is not None else 0,
        },
        "state_row": row,
        "outcome_summary": outcome_summary,
        "same_symbol_outcome_summary": dict(same_symbol_outcomes.get("summary") or {}),
        "cross_symbol_outcome_summary": dict(cross_symbol_outcomes.get("summary") or {}),
        "opportunity": scoring,
        "drivers": drivers,
        "historical_twins": top_twins,
        "same_symbol_twins": same_symbol_twins,
        "cross_symbol_twins": cross_symbol_twins,
        "reasoning_input": reasoning_input.to_dict(),
        "stock_insight": stock_insight.to_dict(),
        "market_situation_explanation": market_situation_explanation.to_dict(),
        "reasoning_mode": resolved_reasoning_mode,
        "current_cluster": {
            **dict(current_cluster),
            "similarity_score_pct": round(float(current_cluster.get("similarity_score") or 0.0) * 100.0, 2) if current_cluster else 0.0,
        },
        "nearest_clusters": [
            {
                **dict(item),
                "similarity_score_pct": round(float(item.get("similarity_score") or 0.0) * 100.0, 2),
            }
            for item in nearest_clusters
        ],
        "frame_meta": payload["meta"],
        "search": {
            "method": str(search_method or "hybrid"),
            "horizons": [int(value) for value in DEFAULT_SEARCH_HORIZONS],
        },
    }


def build_opportunity_dashboard(
    *,
    strategy_artifact_id: int = 0,
    feature_artifact_id: int = 0,
    label_artifact_id: int = 0,
    prediction_artifact_ids: Sequence[int] | None = None,
    market_situation_artifact_id: int = 0,
    limit: int = 20,
    search_method: str = "hybrid",
) -> dict[str, Any]:
    artifacts = resolve_insight_artifacts(
        strategy_artifact_id=int(strategy_artifact_id or 0),
        feature_artifact_id=int(feature_artifact_id or 0),
        label_artifact_id=int(label_artifact_id or 0),
        prediction_artifact_ids=list(prediction_artifact_ids or []),
    )
    frame, meta = load_market_state_frame(
        strategy_artifact=artifacts.strategy_artifact,
        feature_artifact=artifacts.feature_artifact,
        label_artifact=artifacts.label_artifact,
        prediction_artifacts=artifacts.prediction_artifacts,
    )
    current_rows = latest_rows_by_symbol(frame)
    if current_rows.empty:
        return {"rows": [], "as_of_date": "", "artifacts": meta}
    price_col = resolve_price_column(frame)
    market_situation_artifact = resolve_market_situation_artifact(artifact_id=int(market_situation_artifact_id or 0))
    cluster_bundle = None
    if market_situation_artifact is not None:
        try:
            cluster_bundle = load_market_situation_cluster_artifact(market_situation_artifact)
        except Exception:
            cluster_bundle = None
    search_bundle = build_historical_situation_search_bundle(
        frame,
        feature_columns=list(meta.get("embedding_columns") or []),
        feature_family_map=dict(meta.get("feature_family_map") or {}),
        text_embedding_config=TextEmbeddingConfig(backend="auto", embedding_dim=128),
    )
    rows: list[dict[str, Any]] = []
    for item in current_rows.to_dict(orient="records"):
        symbol = str(item.get("symbol") or "").strip().upper()
        current_cluster: dict[str, Any] = {}
        if cluster_bundle is not None:
            cluster_embedding = compute_cluster_embedding(item, cluster_bundle.embedding_model)
            nearest_clusters = find_nearest_clusters(
                cluster_embedding,
                cluster_bundle,
                side=str(item.get("side") or "").strip().lower() or None,
                top_n=1,
            )
            if nearest_clusters:
                current_cluster = dict(nearest_clusters[0])
        matches = search_market_state_neighbors(
            item,
            search_bundle,
            method=search_method,
            top_k=8,
            search_mode="mixed",
            query_symbol=symbol,
            query_date=str(pd.Timestamp(item["date"]).date()),
        )
        outcomes = summarize_historical_outcomes(matches, frame, price_col=price_col, horizons=DEFAULT_SEARCH_HORIZONS)["summary"]
        scoring = compute_opportunity_summary(
            row=item,
            state_frame=frame,
            outcome_summary=outcomes,
            similarity_rows=matches,
        )
        rows.append(
            {
                "symbol": symbol,
                "date": str(pd.Timestamp(item["date"]).date()),
                "opportunity_score": scoring["opportunity_score"],
                "confidence_score": scoring["confidence_score"],
                "confidence_label": scoring["confidence_label"],
                "market_familiarity_score": scoring["market_familiarity_score"],
                "market_familiarity_label": scoring["market_familiarity_label"],
                "risk_indicator": scoring["risk_indicator"],
                "median_return": outcomes.get("median_return"),
                "win_rate": outcomes.get("win_rate"),
                "top_similarity_score": max((float(match.get("similarity_score") or 0.0) for match in matches), default=0.0),
                "cluster_id": current_cluster.get("cluster_id") or "",
                "cluster_description": current_cluster.get("description") or "",
                "cluster_median_return": ((current_cluster.get("outcome_statistics") or {}).get("median_return") if current_cluster else None),
                "cluster_win_rate": ((current_cluster.get("outcome_statistics") or {}).get("win_rate") if current_cluster else None),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("opportunity_score") or 0.0),
            float(row.get("confidence_score") or 0.0),
            float(row.get("market_familiarity_score") or 0.0),
        ),
        reverse=True,
    )
    return {
        "rows": rows[: max(int(limit), 1)],
        "as_of_date": str(pd.Timestamp(current_rows["date"].max()).date()),
        "artifacts": {
            "strategy_artifact_id": int(artifacts.strategy_artifact.id) if artifacts.strategy_artifact is not None else 0,
            "feature_artifact_id": int(artifacts.feature_artifact.id) if artifacts.feature_artifact is not None else 0,
            "label_artifact_id": int(artifacts.label_artifact.id) if artifacts.label_artifact is not None else 0,
            "prediction_artifact_ids": [int(artifact.id) for artifact in artifacts.prediction_artifacts],
            "market_situation_artifact_id": int(market_situation_artifact.id) if market_situation_artifact is not None else 0,
        },
        "frame_meta": meta,
        "search": {"method": str(search_method or "hybrid")},
    }


def build_portfolio_analysis(
    *,
    symbols: Sequence[str],
    strategy_artifact_id: int = 0,
    feature_artifact_id: int = 0,
    label_artifact_id: int = 0,
    prediction_artifact_ids: Sequence[int] | None = None,
    market_situation_artifact_id: int = 0,
    search_method: str = "hybrid",
    reasoning_mode: str = "deterministic",
) -> dict[str, Any]:
    cleaned_symbols = []
    seen: set[str] = set()
    for value in list(symbols or []):
        symbol = str(value).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            cleaned_symbols.append(symbol)
    dashboard = build_opportunity_dashboard(
        strategy_artifact_id=strategy_artifact_id,
        feature_artifact_id=feature_artifact_id,
        label_artifact_id=label_artifact_id,
        prediction_artifact_ids=prediction_artifact_ids,
        market_situation_artifact_id=market_situation_artifact_id,
        search_method=search_method,
        limit=500,
    )
    rows = [row for row in list(dashboard.get("rows") or []) if row.get("symbol") in cleaned_symbols]
    if not rows:
        return {
            "symbols": cleaned_symbols,
            "portfolio_score": 0.0,
            "regime_similarity_score": 0.0,
            "risk_concentration_score": 0.0,
            "strong_rows": [],
            "neutral_rows": [],
            "weak_rows": [],
            "rows": [],
            "cluster_exposure_rows": [],
            "portfolio_insight_input": {},
            "portfolio_insight": {},
        }
    portfolio_score = round(sum(float(row.get("opportunity_score") or 0.0) for row in rows) / float(len(rows)), 2)
    regime_similarity_score = round(sum(float(row.get("confidence_score") or 0.0) for row in rows) / float(len(rows)), 2)
    risk_concentration_score = round(sum(100.0 - min(100.0, float(row.get("market_familiarity_score") or 0.0)) for row in rows) / float(len(rows)), 2)
    strong_rows = [row for row in rows if float(row.get("opportunity_score") or 0.0) >= 70.0]
    neutral_rows = [row for row in rows if 45.0 <= float(row.get("opportunity_score") or 0.0) < 70.0]
    weak_rows = [row for row in rows if float(row.get("opportunity_score") or 0.0) < 45.0]
    cluster_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "").strip()
        cluster_description = str(row.get("cluster_description") or "").strip()
        if not cluster_id:
            continue
        key = (cluster_id, cluster_description)
        cluster_counts[key] = cluster_counts.get(key, 0) + 1
    cluster_exposure_rows = [
        {
            "cluster_id": cluster_id,
            "cluster_description": cluster_description,
            "count": count,
            "exposure_pct": round(float(count / max(len(rows), 1)) * 100.0, 2),
        }
        for (cluster_id, cluster_description), count in sorted(cluster_counts.items(), key=lambda item: (-item[1], item[0][0]))
    ]
    portfolio_reasoning_input = build_portfolio_insight_input(
        symbols=cleaned_symbols,
        as_of_date=str(dashboard.get("as_of_date") or ""),
        rows=rows,
        cluster_exposure_rows=cluster_exposure_rows,
        portfolio_score=portfolio_score,
        regime_similarity_score=regime_similarity_score,
        risk_concentration_score=risk_concentration_score,
        optional_notes={
            "artifacts": dict(dashboard.get("artifacts") or {}),
            "search_method": str(search_method or "hybrid"),
        },
    )
    resolved_reasoning_mode = str(reasoning_mode or "deterministic")
    portfolio_insight = build_portfolio_insight(portfolio_reasoning_input, mode=resolved_reasoning_mode)
    return {
        "symbols": cleaned_symbols,
        "portfolio_score": portfolio_score,
        "regime_similarity_score": regime_similarity_score,
        "risk_concentration_score": risk_concentration_score,
        "strong_rows": strong_rows,
        "neutral_rows": neutral_rows,
        "weak_rows": weak_rows,
        "rows": rows,
        "cluster_exposure_rows": cluster_exposure_rows,
        "artifacts": dashboard.get("artifacts") or {},
        "as_of_date": dashboard.get("as_of_date") or "",
        "portfolio_insight_input": portfolio_reasoning_input.to_dict(),
        "portfolio_insight": portfolio_insight.to_dict(),
        "reasoning_mode": resolved_reasoning_mode,
    }
