from __future__ import annotations

from ..integrations.data_quality_runner import latest_rows, load_artifact_frame
from ..models import AnalysisConfig, ArtifactInventory, DataCoverageFinding, PageSnapshot, RankedIssue, Severity


def _match_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        candidate = lowered.get(str(alias).lower())
        if candidate:
            return candidate
    return None


def analyze_data_completeness(
    config: AnalysisConfig,
    inventories: list[ArtifactInventory],
    page_snapshots: list[PageSnapshot],
) -> dict:
    findings: list[DataCoverageFinding] = []
    issues: list[RankedIssue] = []
    stock_pages = [snapshot for snapshot in page_snapshots if "stock" in snapshot.name]
    stock_page_text = " ".join(snapshot.text_sample for snapshot in stock_pages).lower()
    all_symbols_present: set[str] = set()

    for inventory in inventories:
        strategy = inventory.artifacts.get("STRATEGY_DATASET") or inventory.artifacts.get("FEATURES")
        if strategy is None:
            continue
        frame = latest_rows(load_artifact_frame(strategy.uri))
        if frame.empty:
            continue
        columns = [str(column) for column in frame.columns]
        if "symbol" in frame.columns:
            all_symbols_present.update(str(value).strip().upper() for value in frame["symbol"].tolist())
        for field_name, aliases in config.field_aliases.items():
            matched_column = _match_column(columns, aliases)
            coverage_rate = 0.0
            present_symbols: list[str] = []
            missing_symbols: list[str] = []
            sample_values: dict[str, str | float | int | None] = {}
            if matched_column is not None:
                series = frame[matched_column]
                valid = series.notna() & (series.astype(str).str.strip() != "")
                coverage_rate = float(valid.mean()) if len(frame) else 0.0
                if "symbol" in frame.columns:
                    present_symbols = frame.loc[valid, "symbol"].astype(str).head(8).tolist()
                    missing_symbols = frame.loc[~valid, "symbol"].astype(str).head(8).tolist()
                    sample_values = {
                        str(row["symbol"]): row[matched_column]
                        for row in frame.loc[valid, ["symbol", matched_column]].head(5).to_dict(orient="records")
                    }
            page_label_present = None
            if field_name in config.display_labels and stock_pages:
                page_label_present = any(label.lower() in stock_page_text for label in config.display_labels[field_name])
            severity = Severity.LOW
            if field_name in {"current_price", "return_60d", "return_120d"} and coverage_rate < 0.8:
                severity = Severity.CRITICAL
            elif coverage_rate < 0.9:
                severity = Severity.HIGH
            findings.append(
                DataCoverageFinding(
                    field_name=field_name,
                    matched_column=matched_column,
                    dataset_label=inventory.tier,
                    coverage_rate=round(coverage_rate, 4),
                    symbol_count=int(len(frame)),
                    present_symbols=present_symbols,
                    missing_symbols=missing_symbols,
                    sample_values=sample_values,
                    severity=severity,
                    page_label_present=page_label_present,
                )
            )
            if matched_column is None:
                issues.append(
                    RankedIssue(
                        issue_id=f"data-missing-column:{inventory.tier}:{field_name}",
                        title=f"{inventory.tier} dataset does not expose {field_name}",
                        severity=Severity.CRITICAL if field_name in {"return_60d", "return_120d", "current_price"} else Severity.HIGH,
                        score=0.0,
                        page=inventory.tier,
                        category="data_quality",
                        recommendation="Backfill the field into the artifact frame or map the existing alias carrying the same business meaning.",
                        evidence=[f"No alias matched for {field_name} in the latest {inventory.tier} artifact stack."],
                        metric_name="field_coverage_rate",
                        metric_value=0.0,
                        metadata={"trust_impact": 5, "frequency": 4, "scalability_risk": 2, "usability_impact": 4, "implementation_feasibility": 4},
                    )
                )
            elif field_name in {"return_60d", "return_120d", "current_price"} and coverage_rate < 0.85:
                issues.append(
                    RankedIssue(
                        issue_id=f"data-low-coverage:{inventory.tier}:{field_name}",
                        title=f"{field_name} coverage is thin in the latest {inventory.tier} artifact stack",
                        severity=Severity.CRITICAL,
                        score=0.0,
                        page=inventory.tier,
                        category="data_quality",
                        recommendation="Increase retained history or compute the field from the underlying cached price series before rendering the page.",
                        evidence=[
                            f"Matched column: {matched_column}",
                            f"Coverage rate: {coverage_rate:.0%}",
                            f"Missing examples: {', '.join(missing_symbols[:5]) or 'n/a'}",
                        ],
                        metric_name="field_coverage_rate",
                        metric_value=round(coverage_rate, 4),
                        metadata={"trust_impact": 5, "frequency": 4, "scalability_risk": 3, "usability_impact": 5, "implementation_feasibility": 3},
                    )
                )
            if page_label_present is False and matched_column is not None:
                issues.append(
                    RankedIssue(
                        issue_id=f"ui-missing-display:{inventory.tier}:{field_name}",
                        title=f"Stock intelligence does not surface {field_name} even though the data exists",
                        severity=Severity.HIGH,
                        score=0.0,
                        page="pipeline_stock",
                        category="data_quality",
                        recommendation="Add the field to the stock-intelligence summary panel so users can validate the signal against current price action.",
                        evidence=[
                            f"Artifact alias {matched_column} exists in {inventory.tier}.",
                            f"Expected labels missing from stock page: {', '.join(config.display_labels.get(field_name, []))}",
                        ],
                        metric_name="display_field_presence",
                        metric_value=0.0,
                        metadata={"trust_impact": 4, "frequency": 4, "scalability_risk": 1, "usability_impact": 4, "implementation_feasibility": 5},
                    )
                )

    critical_symbol_rate = 0.0
    if config.critical_symbols:
        present = [symbol for symbol in config.critical_symbols if symbol in all_symbols_present]
        critical_symbol_rate = float(len(present) / max(1, len(config.critical_symbols)))
        if critical_symbol_rate < 0.5:
            issues.append(
                RankedIssue(
                    issue_id="data-critical-symbol-coverage",
                    title="Latest artifact stacks omit most critical symbols",
                    severity=Severity.CRITICAL,
                    score=0.0,
                    page="artifacts",
                    category="data_quality",
                    recommendation="Refresh the latest pipeline artifacts against a universe that includes the tier-1 symbols developers use to sanity check the UI.",
                    evidence=[
                        f"Critical symbol validation rate: {critical_symbol_rate:.0%}",
                        f"Present: {', '.join(symbol for symbol in config.critical_symbols if symbol in all_symbols_present) or 'none'}",
                    ],
                    metric_name="tier1_symbol_validation_rate",
                    metric_value=round(critical_symbol_rate, 4),
                    metadata={"trust_impact": 5, "frequency": 5, "scalability_risk": 2, "usability_impact": 5, "implementation_feasibility": 2},
                )
            )

    return {
        "field_coverage": [finding.model_dump(mode="json") for finding in findings],
        "tier1_symbol_validation_rate": round(critical_symbol_rate, 4),
        "issues": [issue.model_dump(mode="json") for issue in issues],
    }
