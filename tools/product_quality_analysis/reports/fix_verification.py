from __future__ import annotations

from ..models import AnalysisSnapshot, VerificationFinding


def compare_snapshots(before: AnalysisSnapshot, after: AnalysisSnapshot) -> list[VerificationFinding]:
    after_index = {issue.issue_id: issue for issue in after.issues}
    findings: list[VerificationFinding] = []
    for issue in before.issues:
        current = after_index.get(issue.issue_id)
        if current is None:
            findings.append(
                VerificationFinding(
                    issue_id=issue.issue_id,
                    title=issue.title,
                    metric_name=issue.metric_name or "score",
                    before_value=issue.metric_value if issue.metric_value is not None else issue.score,
                    after_value=0,
                    status="resolved",
                    details="Issue no longer appears in the rerun snapshot.",
                )
            )
            continue
        before_value = issue.metric_value if issue.metric_value is not None else issue.score
        after_value = current.metric_value if current.metric_value is not None else current.score
        status = "unchanged"
        try:
            if float(after_value) < float(before_value):
                status = "improved"
            elif float(after_value) > float(before_value):
                status = "regressed"
        except Exception:
            if str(after_value) != str(before_value):
                status = "changed"
        findings.append(
            VerificationFinding(
                issue_id=issue.issue_id,
                title=issue.title,
                metric_name=issue.metric_name or "score",
                before_value=before_value,
                after_value=after_value,
                status=status,
                details=current.recommendation or issue.recommendation,
            )
        )
    return findings


def fix_verification_markdown(findings: list[VerificationFinding]) -> str:
    lines = ["# Fix Verification", ""]
    if not findings:
        lines.extend(["No before/after findings were available.", ""])
        return "\n".join(lines)
    lines.append("| Issue | Metric | Before | After | Status |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for item in findings:
        lines.append(f"| {item.title} | {item.metric_name} | {item.before_value} | {item.after_value} | {item.status} |")
    lines.append("")
    return "\n".join(lines)
