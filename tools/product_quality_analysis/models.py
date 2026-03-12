from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RouteTarget(BaseModel):
    name: str
    path: str
    group: str = "core"
    description: str = ""
    tier: str | None = None
    symbol: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def full_url(self, base_url: str) -> str:
        if self.path.startswith("http://") or self.path.startswith("https://"):
            return self.path
        return f"{str(base_url).rstrip('/')}{self.path}"


class TableMetric(BaseModel):
    index: int
    identifier: str
    row_count: int
    column_count: int
    visible_row_count: int = 0
    visible_column_count: int = 0
    has_pagination: bool = False
    page_size: int | None = None
    has_sort_controls: bool = False
    text_density: float = 0.0
    readability_risk_score: float = 0.0


class PageSnapshot(BaseModel):
    name: str
    url: str
    group: str = "core"
    tier: str | None = None
    ok: bool = False
    status_code: int | None = None
    load_time_ms: float | None = None
    dom_node_count: int = 0
    interactive_count: int = 0
    table_metrics: list[TableMetric] = Field(default_factory=list)
    card_count: int = 0
    chart_like_count: int = 0
    heading_count: int = 0
    unique_colors_used: int = 0
    unique_font_sizes: int = 0
    unique_spacing_values: int = 0
    layout_signature: list[str] = Field(default_factory=list)
    component_signatures: list[str] = Field(default_factory=list)
    text_sample: str = ""
    headings: list[str] = Field(default_factory=list)
    empty_markers: list[str] = Field(default_factory=list)
    error_markers: list[str] = Field(default_factory=list)
    console_errors: list[str] = Field(default_factory=list)
    html_path: str = ""
    screenshot_path: str = ""
    response_error: str = ""
    warning_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    artifact_id: int
    artifact_type: str
    tier: str | None = None
    uri: str
    created_at: str = ""


class ArtifactInventory(BaseModel):
    tier: str
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    symbol_count: int = 0
    date_count: int = 0


class DataCoverageFinding(BaseModel):
    field_name: str
    matched_column: str | None = None
    dataset_label: str = ""
    coverage_rate: float = 0.0
    symbol_count: int = 0
    present_symbols: list[str] = Field(default_factory=list)
    missing_symbols: list[str] = Field(default_factory=list)
    sample_values: dict[str, Any] = Field(default_factory=dict)
    severity: Severity = Severity.LOW
    page_label_present: bool | None = None


class RankedIssue(BaseModel):
    issue_id: str
    title: str
    severity: Severity
    score: float
    page: str = ""
    category: str = ""
    confidence: float = 1.0
    recommendation: str = ""
    evidence: list[str] = Field(default_factory=list)
    metric_name: str = ""
    metric_value: float | int | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationFinding(BaseModel):
    issue_id: str
    title: str
    metric_name: str
    before_value: float | int | str | None = None
    after_value: float | int | str | None = None
    status: str
    details: str = ""


class AnalysisSnapshot(BaseModel):
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))
    label: str
    base_url: str
    routes: list[RouteTarget] = Field(default_factory=list)
    page_snapshots: list[PageSnapshot] = Field(default_factory=list)
    artifact_inventory: list[ArtifactInventory] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    pagination: dict[str, Any] = Field(default_factory=dict)
    readability: dict[str, Any] = Field(default_factory=dict)
    dom_complexity: dict[str, Any] = Field(default_factory=dict)
    ui_consistency: dict[str, Any] = Field(default_factory=dict)
    design_tokens: dict[str, Any] = Field(default_factory=dict)
    layout_similarity: dict[str, Any] = Field(default_factory=dict)
    empty_states: dict[str, Any] = Field(default_factory=dict)
    scalability: dict[str, Any] = Field(default_factory=dict)
    lighthouse: dict[str, Any] = Field(default_factory=dict)
    axe: dict[str, Any] = Field(default_factory=dict)
    stylelint: dict[str, Any] = Field(default_factory=dict)
    visual_regression: dict[str, Any] = Field(default_factory=dict)
    issues: list[RankedIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    base_url: str
    output_dir: Path
    snapshot_dir: Path
    crawl_dir: Path
    label: str
    browser_timeout_ms: int
    dom_warning_threshold: int
    dom_critical_threshold: int
    table_warning_threshold: int
    table_critical_threshold: int
    critical_symbols: list[str] = Field(default_factory=list)
    default_symbol_fallbacks: list[str] = Field(default_factory=list)
    symbol_tiers: list[str] = Field(default_factory=list)
    field_aliases: dict[str, list[str]] = Field(default_factory=dict)
    display_labels: dict[str, list[str]] = Field(default_factory=dict)
    routes: list[RouteTarget] = Field(default_factory=list)
