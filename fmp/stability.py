from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from django.db.models import Count, Max, Min, Q
from django.utils import timezone

from fmp.models import Symbol, SymbolSectionHistorical, SymbolSectionState
from fmp.symbol_dates import effective_symbol_history_start


@dataclass(frozen=True)
class HistoricalSectionStability:
    stable: bool
    reason: str
    mode: str
    score: float
    count: int
    distinct_dates: int
    min_date: date | None
    max_date: date | None
    coverage_ratio: float
    density_ratio: float | None
    fetched_recently: bool


def should_defer_historical_refetch(assessment: HistoricalSectionStability) -> bool:
    """Honor the endpoint cooldown even when stored history is incomplete."""

    return bool(assessment.fetched_recently)


def _mode_for_endpoint(endpoint) -> str:
    configured = str(getattr(endpoint, "stability_mode", "auto") or "auto").lower()
    if configured != "auto":
        return configured
    if bool(getattr(endpoint, "dedupe_by_date", False)):
        return "daily"
    if tuple(getattr(endpoint, "supported_periods", ()) or ()):
        return "periodic"
    return "event"


def _effective_target_start(endpoint, target_start: date, target_end: date) -> date:
    years = getattr(endpoint, "min_history_years", None)
    if years is None or int(years) <= 0:
        return target_start
    endpoint_start = target_end - timedelta(days=365 * int(years))
    return max(target_start, endpoint_start)


def _business_days(start: date, end: date) -> int:
    if end < start:
        return 0
    total_days = (end - start).days + 1
    full_weeks, remaining = divmod(total_days, 7)
    weekdays = full_weeks * 5
    for offset in range(remaining):
        if (start + timedelta(days=full_weeks * 7 + offset)).weekday() < 5:
            weekdays += 1
    return weekdays


def assess_historical_section_stability(
    symbol: Symbol,
    endpoint,
    *,
    target_start: date,
    target_end: date,
    now=None,
) -> HistoricalSectionStability:
    now = now or timezone.now()
    section_key = str(endpoint.key)
    mode = _mode_for_endpoint(endpoint)
    effective_start = _effective_target_start(endpoint, target_start, target_end)
    effective_start = effective_symbol_history_start(symbol, effective_start)
    queryset = SymbolSectionHistorical.objects.filter(symbol=symbol, section_key=section_key)
    aggregate = queryset.aggregate(
        count=Count("id"),
        distinct_dates=Count("record_date", distinct=True),
        dated_count=Count("id", filter=Q(record_date__isnull=False)),
        min_date=Min("record_date"),
        max_date=Max("record_date"),
    )
    count = int(aggregate["count"] or 0)
    distinct_dates = int(aggregate["distinct_dates"] or 0)
    dated_count = int(aggregate["dated_count"] or 0)
    min_date = aggregate["min_date"]
    max_date = aggregate["max_date"]

    state = SymbolSectionState.objects.filter(symbol=symbol, section_key=section_key).first()
    threshold_days = max(1, int(endpoint.threshold_days))
    fetched_recently = bool(
        state
        and state.last_fetched_at
        and state.last_fetched_at >= now - timedelta(days=threshold_days)
    )

    # Periodic fundamentals normally begin at the first reporting date after
    # listing, not on the listing day itself. Do not classify that initial
    # quarter as missing history.
    if (
        mode == "periodic"
        and min_date is not None
        and effective_start < min_date <= effective_start + timedelta(days=120)
    ):
        effective_start = min_date

    window_days = max(1, (target_end - effective_start).days + 1)
    covered_start = max(min_date, effective_start) if min_date else None
    if mode == "periodic" and max_date is not None:
        covered_end = min(max_date + timedelta(days=120), target_end)
    else:
        covered_end = min(max_date, target_end) if max_date else None
    covered_days = (
        (covered_end - covered_start).days + 1
        if covered_start is not None and covered_end is not None and covered_end >= covered_start
        else 0
    )
    coverage_ratio = min(1.0, max(0.0, covered_days / window_days))
    minimum_observations = max(0, int(getattr(endpoint, "minimum_observations", 1) or 0))

    if mode == "event":
        valid_dates = count == 0 or dated_count / count >= 0.95
        stable = fetched_recently and valid_dates and (count == 0 or count >= minimum_observations)
        reason = "stable_event_section" if stable else "event_section_not_recently_confirmed"
        if fetched_recently and count > 0 and not valid_dates:
            reason = "too_many_missing_record_dates"
        score = (0.7 if fetched_recently else 0.0) + (0.3 if valid_dates else 0.0)
        return HistoricalSectionStability(
            stable, reason, mode, round(score, 3), count, distinct_dates,
            min_date, max_date, coverage_ratio, None, fetched_recently,
        )

    if count < minimum_observations or min_date is None or max_date is None:
        return HistoricalSectionStability(
            False, "insufficient_observations", mode, 0.0, count, distinct_dates,
            min_date, max_date, coverage_ratio, 0.0, fetched_recently,
        )

    in_window_dates = queryset.filter(
        record_date__gte=effective_start,
        record_date__lte=target_end,
    ).values("record_date").distinct().count()
    if mode == "daily":
        expected = max(1, _business_days(effective_start, min(target_end, max_date)))
    else:
        candidate_params = dict(endpoint.candidates[0][1]) if endpoint.candidates else {}
        period = str(candidate_params.get("period") or "quarter").lower()
        observations_per_year = 1 if period == "annual" else 4
        expected = max(1, int(window_days / 365 * observations_per_year))
    density_ratio = min(1.0, in_window_dates / expected)

    if mode == "daily":
        expected_latest = target_end
        while expected_latest.weekday() >= 5:
            expected_latest -= timedelta(days=1)
        recent_cutoff = expected_latest - timedelta(days=threshold_days)
    else:
        recent_cutoff = target_end - timedelta(days=threshold_days)
    if mode == "periodic":
        recent_cutoff = target_end - timedelta(days=max(120, threshold_days))
    recent_enough = max_date >= recent_cutoff
    valid_dates = dated_count / count >= 0.95
    coverage_ok = coverage_ratio >= 0.90
    density_floor = 0.90 if mode == "daily" else 0.75
    density_ok = density_ratio >= density_floor
    stable = fetched_recently and recent_enough and coverage_ok and density_ok and valid_dates

    if not valid_dates:
        reason = "too_many_missing_record_dates"
    elif not coverage_ok:
        reason = "insufficient_date_coverage"
    elif not density_ok:
        reason = "sparse_observation_density"
    elif not recent_enough:
        reason = "latest_observation_is_stale"
    elif not fetched_recently:
        reason = "section_not_fetched_recently"
    else:
        reason = "stable_historical_section"
    score = (
        0.25 * coverage_ratio
        + 0.25 * density_ratio
        + 0.25 * float(recent_enough)
        + 0.15 * float(fetched_recently)
        + 0.10 * float(valid_dates)
    )
    return HistoricalSectionStability(
        stable, reason, mode, round(score, 3), count, distinct_dates,
        min_date, max_date, coverage_ratio, density_ratio, fetched_recently,
    )


__all__ = [
    "HistoricalSectionStability",
    "assess_historical_section_stability",
    "should_defer_historical_refetch",
]
