from __future__ import annotations

from typing import Any

from ..models import PageSnapshot, TableMetric


def table_risk_score(table: TableMetric) -> float:
    density = max(1, table.row_count) * max(1, table.column_count)
    risk = density / 500.0
    if table.row_count > 300:
        risk += 3.0
    elif table.row_count > 100:
        risk += 1.5
    if not table.has_pagination and table.row_count > 100:
        risk += 3.0
    return round(risk, 2)


def page_has_visible_issue_markers(snapshot: PageSnapshot) -> bool:
    return bool(snapshot.empty_markers or snapshot.error_markers or snapshot.response_error)


def snapshot_metric(snapshot: PageSnapshot, key: str, default: float = 0.0) -> float:
    value = getattr(snapshot, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def coerce_table_metrics(raw_tables: list[dict[str, Any]]) -> list[TableMetric]:
    tables: list[TableMetric] = []
    for index, item in enumerate(list(raw_tables or [])):
        metric = TableMetric(
            index=int(item.get("index") or index),
            identifier=str(item.get("identifier") or f"table-{index}"),
            row_count=int(item.get("row_count") or 0),
            column_count=int(item.get("column_count") or 0),
            visible_row_count=int(item.get("visible_row_count") or 0),
            visible_column_count=int(item.get("visible_column_count") or 0),
            has_pagination=bool(item.get("has_pagination")),
            page_size=int(item.get("page_size")) if item.get("page_size") not in (None, "", 0) else None,
            has_sort_controls=bool(item.get("has_sort_controls")),
            text_density=float(item.get("text_density") or 0.0),
        )
        metric.readability_risk_score = table_risk_score(metric)
        tables.append(metric)
    return tables
