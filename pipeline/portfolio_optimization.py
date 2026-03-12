from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fmp.models import Symbol


DEFAULT_RETURN_COL_CANDIDATES: tuple[str, ...] = (
    "ret_1",
    "px__ret_1",
    "px__ret_1d",
    "px__ret_1_d",
    "asset_return",
)
DEFAULT_PRICE_COL_CANDIDATES: tuple[str, ...] = (
    "close",
    "px__close",
    "adj_close",
    "px__adj_close",
)
SYMBOL_METADATA_COLUMNS = {"sector", "industry", "exchange", "country"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0, *, minimum: int | None = None) -> int:
    try:
        resolved = int(value) if value not in (None, "") else int(default)
    except Exception:
        resolved = int(default)
    if minimum is not None:
        resolved = max(int(minimum), resolved)
    return resolved


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _unique_str_list(values: Any) -> tuple[str, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, str):
        raw_values = values.split(",")
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    out: list[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _first_available_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    available = {str(column) for column in columns}
    for candidate in list(candidates or []):
        if str(candidate) in available:
            return str(candidate)
    return ""


def _winsorized_zscore(series: pd.Series, *, clip: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    mean_value = float(valid.mean())
    std_value = float(valid.std(ddof=0))
    if std_value <= 1e-12:
        centered = valid - mean_value
        if centered.abs().max() <= 1e-12:
            out = pd.Series(0.0, index=series.index, dtype=float)
            out.loc[valid.index] = 0.0
            return out
        std_value = float(centered.abs().max())
    zscore = (valid - mean_value) / std_value
    if clip > 0.0:
        zscore = zscore.clip(lower=-float(clip), upper=float(clip))
    out = pd.Series(0.0, index=series.index, dtype=float)
    out.loc[zscore.index] = zscore.astype(float)
    return out


def _bucket_labels(scores: pd.Series, *, bucket_count: int) -> tuple[pd.Series, pd.Series]:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    display_rank = pd.Series("", index=scores.index, dtype=object)
    bucket_labels = pd.Series(0, index=scores.index, dtype=int)
    if valid.empty:
        return display_rank, bucket_labels
    rank_positions = valid.rank(method="first", ascending=True)
    display = valid.rank(method="first", ascending=False).astype(int)
    buckets = (((rank_positions - 1.0) * float(max(int(bucket_count), 2))) // float(len(rank_positions))).astype(int) + 1
    display_rank.loc[display.index] = display.astype(str)
    bucket_labels.loc[buckets.index] = buckets.astype(int)
    return display_rank, bucket_labels


def _regularize_covariance(
    covariance: np.ndarray,
    *,
    shrinkage: float,
    diagonal_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    diag = np.diag(matrix).astype(float)
    positive_diag = diag[np.isfinite(diag) & (diag > 0.0)]
    median_variance = float(np.median(positive_diag)) if positive_diag.size else 1e-4
    variance_floor = max(float(diagonal_floor), median_variance * 0.05, 1e-8)
    diag = np.where(np.isfinite(diag) & (diag > variance_floor), diag, variance_floor)
    diagonal = np.diag(diag)
    resolved_shrinkage = min(max(float(shrinkage), 0.0), 1.0)
    regularized = ((1.0 - resolved_shrinkage) * matrix) + (resolved_shrinkage * diagonal)
    regularized = 0.5 * (regularized + regularized.T)
    eigenvalues = np.linalg.eigvalsh(regularized)
    min_eigenvalue = float(eigenvalues.min()) if len(eigenvalues) else variance_floor
    if min_eigenvalue < variance_floor:
        regularized = regularized + np.eye(regularized.shape[0]) * (variance_floor - min_eigenvalue + 1e-10)
        eigenvalues = np.linalg.eigvalsh(regularized)
        min_eigenvalue = float(eigenvalues.min()) if len(eigenvalues) else variance_floor
    max_eigenvalue = float(eigenvalues.max()) if len(eigenvalues) else variance_floor
    condition_number = max_eigenvalue / max(min_eigenvalue, 1e-12)
    return regularized, {
        "median_variance": float(median_variance),
        "variance_floor": float(variance_floor),
        "min_eigenvalue": float(min_eigenvalue),
        "max_eigenvalue": float(max_eigenvalue),
        "condition_number": float(condition_number),
    }


def _bounded_allocation(scores: np.ndarray, *, budget: float, cap: float) -> np.ndarray:
    weights = np.zeros(len(scores), dtype=float)
    resolved_budget = max(float(budget), 0.0)
    resolved_cap = max(float(cap), 0.0)
    if resolved_budget <= 1e-12 or resolved_cap <= 1e-12 or len(scores) == 0:
        return weights
    positive_scores = np.clip(np.asarray(scores, dtype=float), a_min=0.0, a_max=None)
    if float(positive_scores.sum()) <= 1e-12:
        positive_scores = np.ones(len(scores), dtype=float)
    active = positive_scores > 1e-12
    remaining_budget = min(resolved_budget, float(active.sum()) * resolved_cap)
    working_scores = positive_scores.copy()
    while remaining_budget > 1e-12 and active.any():
        scale = remaining_budget / float(working_scores[active].sum())
        proposal = working_scores * scale
        capped = active & (proposal >= (resolved_cap - 1e-12))
        if not capped.any():
            weights[active] = proposal[active]
            remaining_budget = 0.0
            break
        weights[capped] = resolved_cap
        remaining_budget -= float(capped.sum()) * resolved_cap
        working_scores[capped] = 0.0
        active = working_scores > 1e-12
    return weights


def _heuristic_long_short_weights(
    expected_returns: np.ndarray,
    *,
    gross_exposure_limit: float,
    net_exposure_target: float,
    max_name_weight: float,
    allow_short: bool,
) -> np.ndarray:
    signal = np.asarray(expected_returns, dtype=float)
    gross_limit = max(float(gross_exposure_limit), 0.0)
    name_cap = max(float(max_name_weight), 0.0)
    net_target = float(net_exposure_target)
    if len(signal) == 0 or gross_limit <= 0.0 or name_cap <= 0.0:
        return np.zeros(len(signal), dtype=float)

    positive_capacity = float(len(signal) * name_cap)
    if not allow_short:
        long_budget = min(max(net_target, 0.0), gross_limit, positive_capacity)
        return _bounded_allocation(np.clip(signal, a_min=0.0, a_max=None), budget=long_budget, cap=name_cap)

    negative_capacity = float(len(signal) * name_cap)
    desired_long = max((gross_limit + net_target) / 2.0, 0.0)
    desired_short = max((gross_limit - net_target) / 2.0, 0.0)
    long_budget = min(desired_long, positive_capacity)
    short_budget = min(desired_short, negative_capacity)
    if net_target >= 0.0:
        long_budget = min(long_budget, short_budget + net_target)
        short_budget = max(0.0, min(short_budget, long_budget - net_target))
    else:
        short_budget = min(short_budget, long_budget - net_target)
        long_budget = max(0.0, min(long_budget, short_budget + net_target))
    if abs(net_target) <= 1e-9:
        matched_budget = min(long_budget, short_budget)
        long_budget = matched_budget
        short_budget = matched_budget
    long_weights = _bounded_allocation(np.clip(signal, a_min=0.0, a_max=None), budget=long_budget, cap=name_cap)
    short_weights = _bounded_allocation(np.clip(-signal, a_min=0.0, a_max=None), budget=short_budget, cap=name_cap)
    return long_weights - short_weights


@dataclass(frozen=True)
class PortfolioRiskModelConfig:
    model_type: str = "sample_covariance"
    lookback_days: int = 63
    min_observations: int = 20
    shrinkage: float = 0.15
    diagonal_floor: float = 1e-6
    factor_count: int = 3


@dataclass(frozen=True)
class PortfolioConstraintConfig:
    gross_exposure_limit: float = 1.0
    net_exposure_target: float = 0.0
    max_name_weight: float = 0.05
    neutrality_columns: tuple[str, ...] = ()
    allow_short: bool = True


@dataclass(frozen=True)
class PortfolioOptimizationConfig:
    expected_return_input: str = "ranking_score"
    alpha_scale: float = 0.05
    alpha_quantile: float = 0.2
    normalize_expected_returns: bool = True
    demean_expected_returns: bool = True
    winsorize_limit: float = 3.0
    solver_maxiter: int = 200
    solver_ftol: float = 1e-9
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.0
    turnover_cap: float | None = None
    bucket_count: int = 10
    return_col_candidates: tuple[str, ...] = DEFAULT_RETURN_COL_CANDIDATES
    price_col_candidates: tuple[str, ...] = DEFAULT_PRICE_COL_CANDIDATES
    risk_model: PortfolioRiskModelConfig = field(default_factory=PortfolioRiskModelConfig)
    constraints: PortfolioConstraintConfig = field(default_factory=PortfolioConstraintConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskModelEstimate:
    symbols: tuple[str, ...]
    covariance: np.ndarray
    idiosyncratic_variance: np.ndarray
    factor_loadings: np.ndarray | None
    factor_names: tuple[str, ...]
    model_type: str
    observations: int
    shrinkage: float
    variance_floor: float
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    weights: pd.Series
    expected_returns: pd.Series
    display_rank: pd.Series
    bucket_labels: pd.Series
    risk_model: RiskModelEstimate
    success: bool
    status: str
    objective_value: float
    expected_portfolio_return: float
    portfolio_variance: float
    turnover: float
    gross_exposure: float
    net_exposure: float
    max_abs_weight: float
    iterations: int
    constraint_violation: float
    neutrality_exposures: dict[str, float]


def build_portfolio_optimization_config(
    config: Mapping[str, Any],
    *,
    gross_exposure: float,
    selection_side: str,
) -> PortfolioOptimizationConfig:
    base = dict(config or {})
    opt = dict(base.get("portfolio_optimization") or {})
    risk = dict(opt.get("risk_model") or {})
    constraint_values = dict(opt.get("constraints") or {})

    neutrality_columns = list(_unique_str_list(constraint_values.get("neutrality_columns") or opt.get("neutrality_columns")))
    if _safe_bool(constraint_values.get("sector_neutral", opt.get("sector_neutral")), False) and "sector" not in neutrality_columns:
        neutrality_columns.append("sector")
    for column in _unique_str_list(constraint_values.get("factor_neutral_columns") or opt.get("factor_neutral_columns")):
        if column not in neutrality_columns:
            neutrality_columns.append(column)

    allow_short = str(selection_side or "").strip().lower() != "long_only"
    resolved_net_target = _safe_float(
        constraint_values.get("net_exposure_target", opt.get("net_exposure_target")),
        0.0 if allow_short else float(gross_exposure),
    )
    resolved_gross_limit = max(
        0.0,
        _safe_float(
            constraint_values.get("gross_exposure_limit", opt.get("gross_exposure_limit")),
            float(gross_exposure),
        ),
    )
    resolved_name_cap = max(
        0.0,
        _safe_float(
            constraint_values.get("max_name_weight", opt.get("max_name_weight")),
            min(float(gross_exposure), 0.05 if allow_short else float(gross_exposure)),
        ),
    )

    turnover_cap_value = opt.get("turnover_cap")
    turnover_cap = None if turnover_cap_value in (None, "", "null") else max(0.0, _safe_float(turnover_cap_value))
    return PortfolioOptimizationConfig(
        expected_return_input=str(opt.get("expected_return_input") or "ranking_score").strip().lower() or "ranking_score",
        alpha_scale=max(0.0, _safe_float(opt.get("alpha_scale"), 0.05)),
        alpha_quantile=min(max(_safe_float(opt.get("alpha_quantile"), 0.2), 0.0), 0.5),
        normalize_expected_returns=_safe_bool(opt.get("normalize_expected_returns"), True),
        demean_expected_returns=_safe_bool(opt.get("demean_expected_returns"), True),
        winsorize_limit=max(0.0, _safe_float(opt.get("winsorize_limit"), 3.0)),
        solver_maxiter=_safe_int(opt.get("solver_maxiter"), 200, minimum=1),
        solver_ftol=max(1e-12, _safe_float(opt.get("solver_ftol"), 1e-9)),
        risk_aversion=max(0.0, _safe_float(opt.get("risk_aversion"), 5.0)),
        turnover_penalty=max(0.0, _safe_float(opt.get("turnover_penalty"), 0.0)),
        turnover_cap=turnover_cap,
        bucket_count=_safe_int(opt.get("bucket_count"), 10, minimum=2),
        return_col_candidates=_unique_str_list(opt.get("return_col_candidates")) or DEFAULT_RETURN_COL_CANDIDATES,
        price_col_candidates=_unique_str_list(opt.get("price_col_candidates")) or DEFAULT_PRICE_COL_CANDIDATES,
        risk_model=PortfolioRiskModelConfig(
            model_type=str(risk.get("model_type") or opt.get("risk_model_type") or "sample_covariance").strip().lower(),
            lookback_days=_safe_int(risk.get("lookback_days", opt.get("risk_lookback_days")), 63, minimum=5),
            min_observations=_safe_int(risk.get("min_observations", opt.get("risk_min_observations")), 20, minimum=2),
            shrinkage=min(max(_safe_float(risk.get("shrinkage", opt.get("risk_shrinkage")), 0.15), 0.0), 1.0),
            diagonal_floor=max(1e-10, _safe_float(risk.get("diagonal_floor", opt.get("risk_diagonal_floor")), 1e-6)),
            factor_count=_safe_int(risk.get("factor_count", opt.get("risk_factor_count")), 3, minimum=1),
        ),
        constraints=PortfolioConstraintConfig(
            gross_exposure_limit=float(resolved_gross_limit),
            net_exposure_target=float(resolved_net_target),
            max_name_weight=float(resolved_name_cap),
            neutrality_columns=tuple(neutrality_columns),
            allow_short=bool(allow_short),
        ),
    )


def ensure_constraint_columns(
    feature_df: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    requested = [str(column).strip() for column in list(columns or []) if str(column).strip()]
    missing = [column for column in requested if column not in feature_df.columns and column in SYMBOL_METADATA_COLUMNS]
    if not missing or feature_df.empty or "symbol" not in feature_df.columns:
        return feature_df
    symbols = [
        str(symbol).strip().upper()
        for symbol in feature_df["symbol"].astype(str).tolist()
        if str(symbol).strip()
    ]
    lookup = {
        str(row.symbol).strip().upper(): row
        for row in Symbol.objects.filter(symbol__in=sorted(set(symbols))).only("symbol", *missing)
    }
    out = feature_df.copy()
    symbol_series = out["symbol"].astype(str).str.strip().str.upper()
    for column in missing:
        out[column] = symbol_series.map(
            lambda symbol: str(getattr(lookup.get(str(symbol)), column, "") or "").strip() or "Unknown"
        )
    return out


def build_return_panel(
    feature_df: pd.DataFrame,
    *,
    return_col_candidates: Sequence[str],
    price_col_candidates: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    if feature_df.empty:
        return pd.DataFrame(), ""
    out = feature_df[["date", "symbol"]].copy()
    return_col = _first_available_column(feature_df.columns, return_col_candidates)
    if return_col:
        out["asset_return"] = pd.to_numeric(feature_df.get(return_col), errors="coerce")
    else:
        price_col = _first_available_column(feature_df.columns, price_col_candidates)
        if not price_col:
            return pd.DataFrame(), ""
        out["_price"] = pd.to_numeric(feature_df.get(price_col), errors="coerce")
        out["asset_return"] = out.groupby("symbol", sort=False)["_price"].pct_change(fill_method=None)
        out = out.drop(columns=["_price"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out = out.dropna(subset=["date", "symbol"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(), return_col
    pivot = out.pivot_table(index="date", columns="symbol", values="asset_return", aggfunc="last").sort_index()
    return pivot, str(return_col or "derived_price_return")


def build_expected_return_series(
    group: pd.DataFrame,
    *,
    score_field: str,
    config: PortfolioOptimizationConfig,
    higher_score_is_better: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw_scores = pd.to_numeric(group.get(score_field), errors="coerce")
    if not higher_score_is_better:
        raw_scores = -1.0 * raw_scores
    valid_scores = raw_scores.dropna().astype(float)
    expected_returns = pd.Series(0.0, index=group.index, dtype=float)
    display_rank, bucket_labels = _bucket_labels(valid_scores, bucket_count=config.bucket_count)
    if valid_scores.empty:
        return expected_returns, display_rank, bucket_labels

    input_style = str(config.expected_return_input or "ranking_score").strip().lower()
    if input_style in {"predicted_return", "predicted_returns"}:
        transformed = valid_scores.copy()
        if config.normalize_expected_returns:
            transformed = _winsorized_zscore(transformed, clip=config.winsorize_limit).loc[valid_scores.index]
        elif config.demean_expected_returns:
            transformed = transformed - float(transformed.mean())
        transformed = transformed * float(config.alpha_scale or 1.0)
    elif input_style in {"quantile_membership", "quantile_alpha", "top_bottom_quantile"}:
        transformed = pd.Series(0.0, index=valid_scores.index, dtype=float)
        if len(valid_scores) == 1:
            transformed.iloc[0] = float(config.alpha_scale)
        else:
            quantile = min(max(float(config.alpha_quantile), 0.0), 0.5)
            high_threshold = float(valid_scores.quantile(1.0 - quantile)) if quantile > 0.0 else float(valid_scores.max())
            low_threshold = float(valid_scores.quantile(quantile)) if quantile > 0.0 else float(valid_scores.min())
            transformed.loc[valid_scores >= high_threshold] = float(config.alpha_scale)
            if config.constraints.allow_short:
                transformed.loc[valid_scores <= low_threshold] = -float(config.alpha_scale)
    else:
        rank_alpha = (valid_scores.rank(method="first", pct=True) - 0.5) * 2.0
        if config.normalize_expected_returns:
            rank_alpha = _winsorized_zscore(rank_alpha, clip=config.winsorize_limit).loc[valid_scores.index]
        transformed = rank_alpha * float(config.alpha_scale or 1.0)

    if config.constraints.allow_short and config.demean_expected_returns and input_style not in {"quantile_membership", "quantile_alpha", "top_bottom_quantile"}:
        transformed = transformed - float(transformed.mean())
    if not config.constraints.allow_short:
        transformed = transformed.clip(lower=0.0)
    expected_returns.loc[transformed.index] = pd.to_numeric(transformed, errors="coerce").fillna(0.0)
    return expected_returns.astype(float), display_rank, bucket_labels


def build_neutrality_matrix(
    group: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    exposures: list[np.ndarray] = []
    labels: list[str] = []
    for column in list(columns or []):
        if column not in group.columns:
            continue
        series = group[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= 2 and numeric.abs().sum() > 1e-12:
            filled = numeric.fillna(float(numeric.median()) if numeric.notna().any() else 0.0)
            exposures.append(filled.to_numpy(dtype=float))
            labels.append(str(column))
            continue
        categories = series.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        unique_values = sorted(categories.unique().tolist())
        if len(unique_values) <= 1:
            continue
        for value in unique_values[:-1]:
            vector = categories.eq(value).astype(float).to_numpy(dtype=float)
            if float(vector.sum()) <= 0.0:
                continue
            exposures.append(vector)
            labels.append(f"{column}={value}")
    if not exposures:
        return np.zeros((0, len(group.index)), dtype=float), []
    return np.vstack(exposures).astype(float), labels


def estimate_risk_model(
    return_panel: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    symbols: Sequence[str],
    config: PortfolioRiskModelConfig,
) -> RiskModelEstimate:
    ordered_symbols = tuple(str(symbol).strip().upper() for symbol in list(symbols or []))
    if not ordered_symbols:
        raise ValueError("estimate_risk_model requires at least one symbol.")
    history = return_panel.reindex(columns=list(ordered_symbols)).loc[return_panel.index <= pd.Timestamp(as_of_date)].tail(int(config.lookback_days))
    history = history.astype(float)
    sample_count = int(len(history))
    if history.empty:
        identity = np.eye(len(ordered_symbols), dtype=float) * max(float(config.diagonal_floor), 1e-4)
        return RiskModelEstimate(
            symbols=ordered_symbols,
            covariance=identity,
            idiosyncratic_variance=np.diag(identity).astype(float),
            factor_loadings=None,
            factor_names=(),
            model_type=str(config.model_type or "sample_covariance"),
            observations=0,
            shrinkage=float(config.shrinkage),
            variance_floor=float(config.diagonal_floor),
            condition_number=1.0,
            min_eigenvalue=float(np.diag(identity).min()),
            max_eigenvalue=float(np.diag(identity).max()),
        )

    model_type = str(config.model_type or "sample_covariance").strip().lower()
    if model_type in {"factor_covariance", "factor"} and len(ordered_symbols) >= 2 and sample_count >= 2:
        filled = history.fillna(0.0)
        centered = filled - filled.mean(axis=0)
        matrix = centered.to_numpy(dtype=float)
        _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
        factor_count = min(int(config.factor_count), int(vt.shape[0]), len(ordered_symbols))
        loadings = vt[:factor_count].T if factor_count > 0 else np.zeros((len(ordered_symbols), 0), dtype=float)
        factor_scores = matrix @ loadings if factor_count > 0 else np.zeros((matrix.shape[0], 0), dtype=float)
        factor_covariance = np.cov(factor_scores, rowvar=False, ddof=0) if factor_count > 0 else np.zeros((0, 0), dtype=float)
        factor_covariance = np.atleast_2d(factor_covariance) if factor_count > 1 else np.array([[float(np.var(factor_scores[:, 0], ddof=0))]]) if factor_count == 1 else np.zeros((0, 0), dtype=float)
        common_covariance = loadings @ factor_covariance @ loadings.T if factor_count > 0 else np.zeros((len(ordered_symbols), len(ordered_symbols)), dtype=float)
        residual = matrix - (factor_scores @ loadings.T if factor_count > 0 else 0.0)
        idiosyncratic = np.var(residual, axis=0, ddof=0).astype(float)
        covariance = common_covariance + np.diag(idiosyncratic)
        factor_names = tuple(f"factor_{index}" for index in range(1, factor_count + 1))
        factor_loadings = loadings.astype(float) if factor_count > 0 else None
    else:
        covariance_frame = history.cov(min_periods=max(2, int(config.min_observations // 2))).reindex(index=ordered_symbols, columns=ordered_symbols)
        covariance = covariance_frame.to_numpy(dtype=float)
        filled = history.fillna(0.0)
        idiosyncratic = np.nan_to_num(np.diag(covariance), nan=0.0, posinf=0.0, neginf=0.0)
        factor_names = ()
        factor_loadings = None

    regularized, diagnostics = _regularize_covariance(
        covariance,
        shrinkage=float(config.shrinkage),
        diagonal_floor=float(config.diagonal_floor),
    )
    idiosyncratic = np.where(np.isfinite(idiosyncratic) & (idiosyncratic > 0.0), idiosyncratic, np.diag(regularized))
    return RiskModelEstimate(
        symbols=ordered_symbols,
        covariance=regularized.astype(float),
        idiosyncratic_variance=np.asarray(idiosyncratic, dtype=float),
        factor_loadings=factor_loadings,
        factor_names=tuple(str(name) for name in factor_names),
        model_type=model_type,
        observations=sample_count,
        shrinkage=float(config.shrinkage),
        variance_floor=float(diagnostics["variance_floor"]),
        condition_number=float(diagnostics["condition_number"]),
        min_eigenvalue=float(diagnostics["min_eigenvalue"]),
        max_eigenvalue=float(diagnostics["max_eigenvalue"]),
    )


def optimize_mean_variance_portfolio(
    expected_returns: pd.Series,
    *,
    risk_model: RiskModelEstimate,
    previous_weights: Mapping[str, Any] | pd.Series | None,
    config: PortfolioOptimizationConfig,
    neutrality_matrix: np.ndarray | None = None,
    neutrality_labels: Sequence[str] | None = None,
) -> PortfolioOptimizationResult:
    symbols = [str(symbol) for symbol in list(expected_returns.index)]
    mu = pd.to_numeric(expected_returns, errors="coerce").fillna(0.0).reindex(symbols).to_numpy(dtype=float)
    if previous_weights is None:
        previous_series = pd.Series(dtype=float)
    elif isinstance(previous_weights, pd.Series):
        previous_series = pd.to_numeric(previous_weights, errors="coerce")
    else:
        previous_series = pd.Series(dict(previous_weights or {}), dtype=float)
    previous = previous_series.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    covariance = np.asarray(risk_model.covariance, dtype=float)
    asset_count = len(symbols)
    gross_limit = max(float(config.constraints.gross_exposure_limit), 0.0)
    max_name_weight = max(float(config.constraints.max_name_weight), 0.0)
    if asset_count == 0 or gross_limit <= 0.0 or max_name_weight <= 0.0:
        zero = pd.Series(0.0, index=symbols, dtype=float)
        return PortfolioOptimizationResult(
            weights=zero,
            expected_returns=expected_returns.reindex(symbols).fillna(0.0).astype(float),
            display_rank=pd.Series("", index=symbols, dtype=object),
            bucket_labels=pd.Series(0, index=symbols, dtype=int),
            risk_model=risk_model,
            success=True,
            status="empty_universe",
            objective_value=0.0,
            expected_portfolio_return=0.0,
            portfolio_variance=0.0,
            turnover=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            max_abs_weight=0.0,
            iterations=0,
            constraint_violation=0.0,
            neutrality_exposures={},
        )

    feasible_net_target = float(config.constraints.net_exposure_target)
    if not config.constraints.allow_short:
        feasible_net_target = min(max(feasible_net_target, 0.0), min(gross_limit, asset_count * max_name_weight))
    else:
        feasible_net_target = min(max(feasible_net_target, -gross_limit), gross_limit)

    initial_weights = _heuristic_long_short_weights(
        mu,
        gross_exposure_limit=gross_limit,
        net_exposure_target=feasible_net_target,
        max_name_weight=max_name_weight,
        allow_short=bool(config.constraints.allow_short),
    )
    x0 = np.concatenate([np.clip(initial_weights, 0.0, None), np.clip(-initial_weights, 0.0, None)])
    short_cap = max_name_weight if config.constraints.allow_short else 0.0
    bounds = [(0.0, max_name_weight)] * asset_count + [(0.0, short_cap)] * asset_count
    net_vector = np.concatenate([np.ones(asset_count, dtype=float), -np.ones(asset_count, dtype=float)])
    gross_vector = np.ones(asset_count * 2, dtype=float)
    exposure_matrix = np.asarray(neutrality_matrix, dtype=float) if neutrality_matrix is not None else np.zeros((0, asset_count), dtype=float)
    exposure_labels = [str(label) for label in list(neutrality_labels or [])]

    def objective(x: np.ndarray) -> float:
        w = x[:asset_count] - x[asset_count:]
        diff = w - previous
        return float(
            (-1.0 * (mu @ w))
            + (float(config.risk_aversion) * (w @ covariance @ w))
            + (float(config.turnover_penalty) * (diff @ diff))
        )

    def objective_jac(x: np.ndarray) -> np.ndarray:
        w = x[:asset_count] - x[asset_count:]
        diff = w - previous
        grad_w = (
            (-1.0 * mu)
            + (2.0 * float(config.risk_aversion) * (covariance @ w))
            + (2.0 * float(config.turnover_penalty) * diff)
        )
        return np.concatenate([grad_w, -grad_w]).astype(float)

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda x, vector=net_vector, target=feasible_net_target: float(vector @ x) - float(target),
            "jac": lambda _x, vector=net_vector: vector,
        },
        {
            "type": "ineq",
            "fun": lambda x, vector=gross_vector, limit=gross_limit: float(limit) - float(vector @ x),
            "jac": lambda _x, vector=gross_vector: -vector,
        },
    ]
    for row_index, exposure_row in enumerate(exposure_matrix):
        vector = np.concatenate([exposure_row, -exposure_row]).astype(float)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, vector=vector: float(vector @ x),
                "jac": lambda _x, vector=vector: vector,
            }
        )

    try:
        solver_result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "ftol": float(config.solver_ftol), "maxiter": int(config.solver_maxiter)},
        )
        solved_weights = (
            np.asarray(solver_result.x[:asset_count], dtype=float)
            - np.asarray(solver_result.x[asset_count:], dtype=float)
        )
        success = bool(solver_result.success) and np.all(np.isfinite(solved_weights))
        status = str(solver_result.message or "ok")
        objective_value = float(solver_result.fun) if np.isfinite(solver_result.fun) else float(objective(x0))
        iterations = int(getattr(solver_result, "nit", 0) or 0)
    except Exception as exc:
        solved_weights = initial_weights
        success = False
        status = f"solver_error:{exc}"
        objective_value = float(objective(np.concatenate([np.clip(initial_weights, 0.0, None), np.clip(-initial_weights, 0.0, None)])))
        iterations = 0

    if not success:
        solved_weights = initial_weights

    solved_weights = np.asarray(solved_weights, dtype=float)
    solved_weights[np.abs(solved_weights) <= 1e-12] = 0.0
    raw_turnover = 0.5 * float(np.abs(solved_weights - previous).sum())
    if config.turnover_cap is not None and raw_turnover > float(config.turnover_cap) + 1e-12:
        prev_net = float(previous.sum())
        prev_gross = float(np.abs(previous).sum())
        if abs(prev_net - feasible_net_target) <= 1e-6 and prev_gross <= gross_limit + 1e-6:
            blend = float(config.turnover_cap) / max(raw_turnover, 1e-12)
            solved_weights = previous + (blend * (solved_weights - previous))
            raw_turnover = 0.5 * float(np.abs(solved_weights - previous).sum())

    gross_exposure = float(np.abs(solved_weights).sum())
    net_exposure = float(solved_weights.sum())
    max_abs_weight = float(np.abs(solved_weights).max()) if len(solved_weights) else 0.0
    neutrality_exposures = {
        exposure_labels[index]: float(exposure_matrix[index] @ solved_weights)
        for index in range(len(exposure_labels))
    }
    constraint_violation = max(
        max(0.0, gross_exposure - gross_limit),
        abs(net_exposure - feasible_net_target),
        max(0.0, max_abs_weight - max_name_weight),
        max((abs(value) for value in neutrality_exposures.values()), default=0.0),
    )
    expected_portfolio_return = float(mu @ solved_weights)
    portfolio_variance = float(solved_weights @ covariance @ solved_weights)
    weight_series = pd.Series(solved_weights, index=symbols, dtype=float)
    display_rank, bucket_labels = _bucket_labels(pd.Series(mu, index=symbols, dtype=float), bucket_count=config.bucket_count)
    return PortfolioOptimizationResult(
        weights=weight_series,
        expected_returns=expected_returns.reindex(symbols).fillna(0.0).astype(float),
        display_rank=display_rank.reindex(symbols, fill_value=""),
        bucket_labels=bucket_labels.reindex(symbols, fill_value=0).astype(int),
        risk_model=risk_model,
        success=success and constraint_violation <= 1e-4,
        status=str(status),
        objective_value=float(objective_value),
        expected_portfolio_return=float(expected_portfolio_return),
        portfolio_variance=float(portfolio_variance),
        turnover=float(raw_turnover),
        gross_exposure=float(gross_exposure),
        net_exposure=float(net_exposure),
        max_abs_weight=float(max_abs_weight),
        iterations=int(iterations),
        constraint_violation=float(constraint_violation),
        neutrality_exposures=neutrality_exposures,
    )


__all__ = [
    "DEFAULT_PRICE_COL_CANDIDATES",
    "DEFAULT_RETURN_COL_CANDIDATES",
    "PortfolioConstraintConfig",
    "PortfolioOptimizationConfig",
    "PortfolioOptimizationResult",
    "PortfolioRiskModelConfig",
    "RiskModelEstimate",
    "build_expected_return_series",
    "build_neutrality_matrix",
    "build_portfolio_optimization_config",
    "build_return_panel",
    "ensure_constraint_columns",
    "estimate_risk_model",
    "optimize_mean_variance_portfolio",
]
