from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Sequence

import pandas as pd
from django.db import transaction

from fmp.models import EconomicIndicatorSeries, Symbol, TreasuryRateSeries
from features.feature_builders import (
    build_event_features,
    build_fundamental_change_features,
    build_ownership_features,
    build_price_technical_features,
    build_statement_quality_features,
)
from features.macro import EconomicDataConfig, broadcast_series_to_daily, fetch_economic_data_series
from features.views import _load_adjusted_prices
from modules.models.base import FitSpec
from modules.models.sklearn.classifier import SklearnRFClassifier
from modules.models.sklearn.regressor import SklearnRFRegressor
from modules.workflows.training import train_ae

from .models import ModelArtifact, ModelTrainingJob
from .store import save_model_artifact

JOB_CONTEXT_KEY = "__job_context__"

FUNDAMENTAL_PREFIXES = {
    "key_metrics": ("km__",),
    "ratios": ("ratio__",),
}

STATEMENT_PREFIXES = {
    "income_statement": ("is__",),
    "income_statement_growth": ("isg__",),
    "cash_flow": ("cf__",),
    "cash_flow_growth": ("cfg__",),
    "balance_sheet": ("bs__",),
    "balance_sheet_growth": ("bsg__",),
    "financial_growth": ("fg__",),
}

EVENT_PREFIXES = {
    "earnings": ("evt__earn_",),
    "analyst_estimates": ("evt__ae_",),
    "ratings_historical": ("evt__rating_",),
    "grades_historical": ("evt__grade_",),
}


def build_symbol_choices() -> list[tuple[str, str]]:
    rows = Symbol.objects.order_by("symbol").values_list("symbol", "company_name")[:500]
    choices: list[tuple[str, str]] = []
    for symbol, company_name in rows:
        code = str(symbol).strip().upper()
        if not code:
            continue
        label = code if not company_name else f"{code} - {company_name}"
        choices.append((code, label))
    return choices


def merge_job_params(model_params: dict[str, Any], *, symbol: str) -> dict[str, Any]:
    payload = dict(model_params)
    payload[JOB_CONTEXT_KEY] = {"symbol": str(symbol).strip().upper()}
    return payload


def extract_model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(params or {}).items() if key != JOB_CONTEXT_KEY}


def extract_training_symbol(params: dict[str, Any]) -> str:
    context = dict(params or {}).get(JOB_CONTEXT_KEY)
    if isinstance(context, dict):
        raw = context.get("symbol")
        if raw:
            return str(raw).strip().upper()
    return ""


def run_training_job(job: ModelTrainingJob) -> ModelArtifact:
    with transaction.atomic():
        job.status = "running"
        job.save(update_fields=["status", "updated_at"])

    try:
        artifact = _run_training_job_impl(job)
    except Exception as exc:
        with transaction.atomic():
            job.status = "failed"
            detail = f"Execution failed: {exc}"
            job.notes = f"{job.notes}\n{detail}".strip()
            job.save(update_fields=["status", "notes", "updated_at"])
        raise

    with transaction.atomic():
        job.status = "succeeded"
        job.latest_artifact = artifact
        job.save(update_fields=["status", "latest_artifact", "updated_at"])

    return artifact


def _run_training_job_impl(job: ModelTrainingJob) -> ModelArtifact:
    symbol = extract_training_symbol(job.params)
    if not symbol:
        raise ValueError("Training symbol is required.")

    train_df, feature_cols = _build_training_frame(symbol=symbol, selected_families=job.feature_families)
    if not feature_cols:
        raise ValueError("No feature columns were produced for the selected families.")

    model_params = extract_model_params(job.params)
    task_type = str(job.task_type).strip().lower()
    algorithm = str(job.algorithm).strip().lower()

    if algorithm == "random_forest_classifier":
        model_obj = _train_classifier(job=job, train_df=train_df, feature_cols=feature_cols, model_params=model_params)
    elif algorithm == "random_forest_regressor":
        model_obj = _train_regressor(job=job, train_df=train_df, feature_cols=feature_cols, model_params=model_params)
    elif algorithm == "autoencoder":
        model_obj = _train_autoencoder(train_df=train_df, feature_cols=feature_cols)
    else:
        raise ValueError(f"Algorithm {job.algorithm!r} is not supported by the web runner yet.")

    metadata = {
        "job_id": job.id,
        "symbol": symbol,
        "feature_families": list(job.feature_families),
        "model_summary": _model_summary(model_obj),
    }

    return save_model_artifact(
        name=job.name,
        model_obj=model_obj,
        framework=job.framework,
        task_type=task_type,
        target_col=job.target_col,
        feature_cols=feature_cols,
        metrics=_metrics_for(model_obj),
        params=model_params,
        metadata=metadata,
    )


def _build_training_frame(*, symbol: str, selected_families: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    symbol_obj = Symbol.objects.filter(symbol__iexact=symbol).first()
    if symbol_obj is None:
        raise ValueError(f"Symbol {symbol!r} was not found.")

    df_prices = _load_adjusted_prices(symbol_obj, None, None)
    if df_prices.empty:
        raise ValueError(f"No adjusted price data found for {symbol}.")

    target_index = pd.MultiIndex.from_arrays(
        [df_prices.index, [symbol] * len(df_prices)],
        names=["date", "symbol"],
    )
    merged = pd.DataFrame(index=target_index)
    grouped: dict[str, list[str]] = {key: [] for key in selected_families}
    selected = set(selected_families)

    if "prices_div_adj" in selected:
        built = build_price_technical_features(symbol, df_prices)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            grouped["prices_div_adj"] = list(built.feature_cols)

    if selected & set(FUNDAMENTAL_PREFIXES):
        built = build_fundamental_change_features(symbol_obj, target_index, df_prices=df_prices)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in FUNDAMENTAL_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if selected & set(STATEMENT_PREFIXES):
        built = build_statement_quality_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in STATEMENT_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if selected & set(EVENT_PREFIXES):
        built = build_event_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            for family, prefixes in EVENT_PREFIXES.items():
                if family in selected:
                    grouped[family] = [col for col in built.feature_cols if col.startswith(prefixes)]

    if "insider_trading" in selected:
        built = build_ownership_features(symbol_obj, target_index)
        if not built.df.empty:
            merged = merged.join(built.df[built.feature_cols], how="left")
            grouped["insider_trading"] = [col for col in built.feature_cols if col.startswith("own__insider_")]

    if "economic_indicators" in selected:
        economic_series_codes = tuple(
            str(code) for code in EconomicIndicatorSeries.objects.order_by("code").values_list("code", flat=True)
        )
        economic_df = fetch_economic_data_series(
            api_key="",
            start_date=df_prices.index.min().date().isoformat(),
            end_date=df_prices.index.max().date().isoformat(),
            config=EconomicDataConfig(
                economic_indicator_series=economic_series_codes,
                include_treasury_rates=False,
            ),
        )
        if not economic_df.empty:
            daily = broadcast_series_to_daily(economic_df, target_index)
            cols = list(daily.columns)
            merged = merged.join(daily[cols], how="left")
            grouped["economic_indicators"] = cols

    if "treasury_rates" in selected:
        treasury_series_codes = tuple(
            str(code) for code in TreasuryRateSeries.objects.order_by("code").values_list("code", flat=True)
        )
        treasury_df = fetch_economic_data_series(
            api_key="",
            start_date=df_prices.index.min().date().isoformat(),
            end_date=df_prices.index.max().date().isoformat(),
            config=EconomicDataConfig(
                economic_indicator_series=treasury_series_codes,
                include_treasury_rates=False,
            ),
        )
        if not treasury_df.empty:
            daily = broadcast_series_to_daily(treasury_df, target_index)
            cols = list(daily.columns)
            merged = merged.join(daily[cols], how="left")
            grouped["treasury_rates"] = cols

    feature_cols: list[str] = []
    for family in selected_families:
        feature_cols.extend(grouped.get(family, []))
    feature_cols = list(dict.fromkeys(feature_cols))

    train_df = merged.reset_index()
    train_df["close"] = df_prices.reindex(train_df["date"])["close"].to_numpy()
    train_df["sample_weight"] = 1.0
    train_df = train_df.dropna(subset=["date"])
    return train_df, feature_cols


def _attach_target(train_df: pd.DataFrame, *, target_col: str, task_type: str) -> pd.DataFrame:
    df = train_df.copy()
    next_return = pd.to_numeric(df["close"], errors="coerce").pct_change().shift(-1)
    if target_col in df.columns and df[target_col].notna().any():
        return df
    if task_type == "classification":
        df[target_col] = (next_return > 0).astype(int)
    elif task_type == "regression":
        df[target_col] = next_return.astype(float)
    else:
        df[target_col] = next_return.astype(float)
    return df


def _train_classifier(
    *,
    job: ModelTrainingJob,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
) -> Any:
    df = _attach_target(train_df, target_col=job.target_col, task_type="classification")
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=job.target_col,
        weight_col="sample_weight",
        split_ratio=float(job.split_ratio),
    )
    model = SklearnRFClassifier(random_state=1337, **model_params)
    model.fit(df, spec, verbose=False)
    return model


def _train_regressor(
    *,
    job: ModelTrainingJob,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    model_params: dict[str, Any],
) -> Any:
    df = _attach_target(train_df, target_col=job.target_col, task_type="regression")
    model = SklearnRFRegressor(
        test_size=max(0.0, 1.0 - float(job.split_ratio)),
        random_state=1337,
        **model_params,
    )
    spec = FitSpec(
        feature_cols=list(feature_cols),
        target_col=job.target_col,
        weight_col="sample_weight",
        split_ratio=float(job.split_ratio),
    )
    model.fit(df, spec, verbose=False)
    return model


def _train_autoencoder(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Any:
    ae_model, _numeric_cols = train_ae(train_df, feature_cols, verbose=False)
    return ae_model


def _metrics_for(model_obj: Any) -> dict[str, Any]:
    metrics_fn = getattr(model_obj, "metrics_report", None)
    if callable(metrics_fn):
        try:
            metrics = metrics_fn()
            return dict(metrics or {})
        except Exception:
            return {}
    return {}


def _model_summary(model_obj: Any) -> str:
    summarize_fn = getattr(model_obj, "summarize", None)
    if not callable(summarize_fn):
        return ""
    buf = StringIO()
    try:
        with redirect_stdout(buf):
            summarize_fn()
    except Exception:
        return ""
    return buf.getvalue().strip()
