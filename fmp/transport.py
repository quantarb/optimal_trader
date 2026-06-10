from __future__ import annotations

import time
import json
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

from fmp.endpoints.base import Candidate, EndpointDefinition
from fmp.endpoints.helpers import DEFAULT_LIMIT, DEFAULT_MAX_PAGES


def to_records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [item if isinstance(item, dict) else {"value": item} for item in data]
    if isinstance(data, dict):
        list_values = [value for value in data.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return to_records(list_values[0])
        return [data]
    return [{"value": data}]


def fetch_first_success(client, candidates: Iterable[Candidate]) -> Any:
    last_error: Exception | None = None
    attempted = False
    for path, params in candidates:
        attempted = True
        try:
            return client.get_json(path, params=dict(params or {}))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    if not attempted:
        raise RuntimeError("No endpoint candidates provided.")
    raise RuntimeError("No endpoint candidate succeeded.")


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def candidates_support_date_window(
    candidates: Iterable[Candidate],
    *,
    endpoint: EndpointDefinition | None = None,
) -> bool:
    if endpoint is not None and endpoint.supports_date_window:
        return True
    return any("from" in dict(params or {}) and "to" in dict(params or {}) for _, params in candidates)


def with_date_window(
    candidates: Iterable[Candidate],
    *,
    from_date: date,
    to_date: date,
    endpoint: EndpointDefinition | None = None,
) -> list[Candidate]:
    supports_window = candidates_support_date_window(candidates, endpoint=endpoint)
    out: list[Candidate] = []
    for path, params in candidates:
        next_params = dict(params or {})
        if supports_window:
            next_params["from"] = from_date.isoformat()
            next_params["to"] = to_date.isoformat()
        out.append((path, next_params))
    return out


def _fetch_paginated(client, path: str, params: dict[str, Any], endpoint: EndpointDefinition) -> list[dict[str, Any]]:
    page = int(params.get("page", 0) or 0)
    limit = max(1, int(endpoint.page_size or params.get("limit") or DEFAULT_LIMIT))
    max_pages = max(1, int(endpoint.max_pages or DEFAULT_MAX_PAGES))
    records: list[dict[str, Any]] = []
    previous_signature: str | None = None
    for _ in range(max_pages):
        page_params = dict(params)
        page_params.update({"page": page, "limit": limit})
        rows = to_records(client.get_json(path, params=page_params))
        if not rows:
            break
        signature = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
        if signature == previous_signature:
            break
        records.extend(rows)
        previous_signature = signature
        page += 1
    return records


def _fetch_chunked(client, path: str, params: dict[str, Any], chunk_years: int) -> list[dict[str, Any]]:
    start = parse_date(params.get("from"))
    end = parse_date(params.get("to"))
    if start is None or end is None or start > end:
        return to_records(client.get_json(path, params=params))
    records: list[dict[str, Any]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=365 * max(1, chunk_years)))
        chunk_params = dict(params)
        chunk_params.update({"from": current.isoformat(), "to": chunk_end.isoformat()})
        records.extend(to_records(client.get_json(path, params=chunk_params)))
        current = chunk_end + timedelta(days=1)
    return records


def fetch_historical_records(client, endpoint: EndpointDefinition) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for path, raw_params in endpoint.candidates:
        try:
            params = {key: value for key, value in dict(raw_params or {}).items() if not str(key).startswith("__")}
            if endpoint.pagination == "page":
                return _fetch_paginated(client, path, params, endpoint)
            chunk_years = endpoint.chunk_years or int(dict(raw_params or {}).get("__chunk_years", 0) or 0)
            if chunk_years and candidates_support_date_window(endpoint.candidates, endpoint=endpoint):
                return _fetch_chunked(client, path, params, chunk_years)
            return to_records(client.get_json(path, params=params))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No endpoint candidates provided.")


def run_with_retries(
    fetch_fn: Callable[[], Any],
    *,
    max_attempts: int,
    base_delay_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Any, int]:
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            return fetch_fn(), attempt - 1
        except Exception:
            if attempt >= attempts:
                raise
            sleep_fn(float(base_delay_s) * (2 ** (attempt - 1)))
    raise RuntimeError("Retry loop exited unexpectedly.")
