#!/usr/bin/env python3
"""
Batch journal review system for auditing multiple medical ML projects.

Runs parallel audits across N projects using audit_external_project.run_audit(),
then produces a comparison matrix, cross-cutting analysis, and aggregated
remediation priorities.

Usage:
    python3 scripts/batch_journal_review.py --manifest batch_manifest.json --output batch_report.json
    python3 scripts/batch_journal_review.py --manifest batch_manifest.json --target-journal nature_medicine --workers 4
    python3 scripts/batch_journal_review.py --manifest batch_manifest.json --format markdown --output batch_report.md
    python3 scripts/batch_journal_review.py --manifest batch_manifest.json --summary-csv batch_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Contract version for batch review reports.
BATCH_REPORT_CONTRACT_VERSION = "batch_review_report.v1"
BATCH_MANIFEST_CONTRACT_VERSION = "batch_manifest.v1"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProjectEntry:
    """A single project in the batch manifest."""

    id: str
    path: str
    label: str = ""
    notes: str = ""


@dataclass
class AuditResult:
    """Result of auditing a single project."""

    project_id: str
    project_label: str
    project_path: str
    success: bool
    audit_report: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class BatchResult:
    """Aggregated result of a batch audit."""

    target_journal: Optional[str]
    total_projects: int
    results: List[AuditResult]
    comparison_matrix: List[Dict[str, Any]] = field(default_factory=list)
    cross_cutting_analysis: Dict[str, Any] = field(default_factory=dict)
    aggregated_remediation: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> Tuple[List[ProjectEntry], Optional[List[str]]]:
    """Load a batch manifest JSON file.

    Returns:
        Tuple of (project entries, target_journals list or None).
    """
    with open(path) as f:
        data = json.load(f)

    contract = data.get("contract_version", "")
    if contract and contract != BATCH_MANIFEST_CONTRACT_VERSION:
        print(
            f"[WARN] Manifest contract_version={contract!r}, expected {BATCH_MANIFEST_CONTRACT_VERSION!r}",
            file=sys.stderr,
        )

    entries: List[ProjectEntry] = []
    for proj in data.get("projects", []):
        entries.append(
            ProjectEntry(
                id=str(proj.get("id", "")),
                path=str(proj.get("path", "")),
                label=str(proj.get("label", proj.get("id", ""))),
                notes=str(proj.get("notes", "")),
            )
        )

    target_journals = data.get("target_journals")
    return entries, target_journals


# ---------------------------------------------------------------------------
# Single-project audit (runs in subprocess pool)
# ---------------------------------------------------------------------------


def _audit_single_project_worker(
    project_path: str,
    project_id: str,
    project_label: str,
    target_journal: Optional[str],
) -> AuditResult:
    """Worker function for ProcessPoolExecutor.

    Imports audit_external_project inside the worker to avoid pickling issues.
    """
    t0 = time.monotonic()
    try:
        # Import inside worker for process-pool compatibility.
        import audit_external_project  # type: ignore[import-untyped]

        pdir = Path(project_path).expanduser().resolve()
        if not pdir.exists():
            return AuditResult(
                project_id=project_id,
                project_label=project_label,
                project_path=project_path,
                success=False,
                error=f"Project directory not found: {pdir}",
                elapsed_seconds=time.monotonic() - t0,
            )

        report = audit_external_project.run_audit(
            project_dir=pdir,
            target_journal=target_journal,
            output_path=None,
            as_json=True,
        )
        return AuditResult(
            project_id=project_id,
            project_label=project_label,
            project_path=project_path,
            success=True,
            audit_report=report,
            elapsed_seconds=time.monotonic() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return AuditResult(
            project_id=project_id,
            project_label=project_label,
            project_path=project_path,
            success=False,
            error=str(exc),
            elapsed_seconds=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Batch audit orchestration
# ---------------------------------------------------------------------------


def run_batch_audit(
    entries: List[ProjectEntry],
    target_journal: Optional[str] = None,
    max_workers: int = 2,
) -> BatchResult:
    """Run audits on all projects in parallel."""
    results: List[AuditResult] = []

    if max_workers <= 1 or len(entries) <= 1:
        # Sequential for single project or single worker.
        for entry in entries:
            result = _audit_single_project_worker(
                project_path=entry.path,
                project_id=entry.id,
                project_label=entry.label,
                target_journal=target_journal,
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_entry = {}
            for entry in entries:
                fut = pool.submit(
                    _audit_single_project_worker,
                    project_path=entry.path,
                    project_id=entry.id,
                    project_label=entry.label,
                    target_journal=target_journal,
                )
                future_to_entry[fut] = entry

            for fut in as_completed(future_to_entry):
                results.append(fut.result())

    # Sort results by original manifest order.
    id_order = {e.id: idx for idx, e in enumerate(entries)}
    results.sort(key=lambda r: id_order.get(r.project_id, 999))

    batch = BatchResult(
        target_journal=target_journal,
        total_projects=len(entries),
        results=results,
    )
    batch.comparison_matrix = build_comparison_matrix(results)
    batch.cross_cutting_analysis = build_cross_cutting_analysis(results)
    batch.aggregated_remediation = build_aggregated_remediation(results)
    batch.summary = build_summary(results)
    return batch


# ---------------------------------------------------------------------------
# Analysis builders
# ---------------------------------------------------------------------------


def _get_dimension_scores(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract per-dimension scores from an audit report."""
    dims = report.get("dimensions", {})
    out: Dict[str, float] = {}
    for key, dim_data in dims.items():
        if isinstance(dim_data, dict):
            out[key] = float(dim_data.get("score", 0.0))
    return out


def _get_total_score(report: Dict[str, Any]) -> float:
    return float(report.get("total_score", 0.0))


def _get_grade(report: Dict[str, Any]) -> str:
    return str(report.get("grade", {}).get("label_en", "Unknown"))


def _get_top_gaps(report: Dict[str, Any]) -> List[str]:
    """Extract top remediation actions from audit report."""
    rems = report.get("remediation_priorities", [])
    return [str(r.get("action", "")) for r in rems[:5] if isinstance(r, dict)]


def _get_journal_compliance(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract journal compliance info."""
    journal_sec = report.get("journal_gap_analysis", {})
    if not journal_sec:
        return {"mandatory_met": 0, "mandatory_total": 0, "unmet": []}
    met = int(journal_sec.get("mandatory_met", 0))
    total = int(journal_sec.get("mandatory_total", 0))
    unmet = [str(u) for u in journal_sec.get("unmet_mandatory", [])]
    return {"mandatory_met": met, "mandatory_total": total, "unmet": unmet}


def build_comparison_matrix(results: List[AuditResult]) -> List[Dict[str, Any]]:
    """Build per-project comparison matrix."""
    matrix: List[Dict[str, Any]] = []
    for r in results:
        if not r.success:
            matrix.append(
                {
                    "project": r.project_id,
                    "label": r.project_label,
                    "status": "error",
                    "error": r.error,
                    "total_score": 0.0,
                    "grade": "Error",
                    "dimensions": {},
                    "journal_compliance": {},
                    "top_gaps": [],
                }
            )
            continue

        rpt = r.audit_report
        matrix.append(
            {
                "project": r.project_id,
                "label": r.project_label,
                "status": "audited",
                "total_score": _get_total_score(rpt),
                "grade": _get_grade(rpt),
                "dimensions": _get_dimension_scores(rpt),
                "journal_compliance": _get_journal_compliance(rpt),
                "top_gaps": _get_top_gaps(rpt),
            }
        )
    return matrix


def build_cross_cutting_analysis(results: List[AuditResult]) -> Dict[str, Any]:
    """Identify cross-cutting patterns across projects."""
    successful = [r for r in results if r.success]
    if not successful:
        return {"most_failed_dimensions": [], "most_common_gaps": []}

    # Per-dimension aggregation.
    dim_scores: Dict[str, List[float]] = {}
    dim_weights: Dict[str, float] = {}
    for r in successful:
        dims = r.audit_report.get("dimensions", {})
        for key, dim_data in dims.items():
            if not isinstance(dim_data, dict):
                continue
            score = float(dim_data.get("score", 0.0))
            weight = float(dim_data.get("weight", 1.0))
            dim_scores.setdefault(key, []).append(score)
            dim_weights[key] = weight

    most_failed_dims: List[Dict[str, Any]] = []
    for key, scores in dim_scores.items():
        weight = dim_weights.get(key, 1.0)
        if weight <= 0:
            continue
        avg_pct = (sum(scores) / len(scores)) / weight * 100 if weight > 0 else 0
        below_80 = sum(1 for s in scores if (s / weight * 100) < 80)
        most_failed_dims.append(
            {
                "dimension": key,
                "avg_score_pct": round(avg_pct, 1),
                "projects_below_80pct": below_80,
            }
        )
    most_failed_dims.sort(key=lambda d: d["avg_score_pct"])

    # Most common gaps.
    gap_counter: Dict[str, int] = {}
    for r in successful:
        rems = r.audit_report.get("remediation_priorities", [])
        for rem in rems:
            action = str(rem.get("action", "")) if isinstance(rem, dict) else ""
            if action:
                gap_counter[action] = gap_counter.get(action, 0) + 1
    most_common_gaps = [
        {"requirement": action, "missing_in_n_projects": count}
        for action, count in sorted(gap_counter.items(), key=lambda x: -x[1])
    ]

    return {
        "most_failed_dimensions": most_failed_dims[:10],
        "most_common_gaps": most_common_gaps[:10],
    }


def build_aggregated_remediation(results: List[AuditResult]) -> List[Dict[str, Any]]:
    """Build deduplicated remediation priorities across all projects."""
    successful = [r for r in results if r.success]
    action_data: Dict[str, Dict[str, Any]] = {}

    for r in successful:
        rems = r.audit_report.get("remediation_priorities", [])
        for rem in rems:
            if not isinstance(rem, dict):
                continue
            action = str(rem.get("action", ""))
            if not action:
                continue
            if action not in action_data:
                action_data[action] = {
                    "action": action,
                    "affects_n_projects": 0,
                    "severity": str(rem.get("severity", "MEDIUM")),
                    "dimension": str(rem.get("dimension", "")),
                    "score_impact_est": float(rem.get("score_impact", 0.0)),
                }
            action_data[action]["affects_n_projects"] += 1
            # Keep highest severity.
            sev_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            existing_sev = sev_order.get(action_data[action]["severity"], 0)
            new_sev = sev_order.get(str(rem.get("severity", "")), 0)
            if new_sev > existing_sev:
                action_data[action]["severity"] = str(rem.get("severity", "MEDIUM"))

    items = sorted(
        action_data.values(),
        key=lambda x: (
            -{"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["severity"], 0),
            -x["affects_n_projects"],
        ),
    )

    for idx, item in enumerate(items, 1):
        item["priority"] = idx

    return items[:20]


def build_summary(results: List[AuditResult]) -> Dict[str, Any]:
    """Build high-level summary statistics."""
    successful = [r for r in results if r.success]
    scores = [_get_total_score(r.audit_report) for r in successful]

    if not scores:
        return {
            "audited": 0,
            "errors": len(results),
            "publication_ready": 0,
            "needs_work": 0,
            "major_issues": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "score_range": [0.0, 0.0],
        }

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    grades = [_get_grade(r.audit_report) for r in successful]
    pub_ready = sum(1 for g in grades if g == "Publication-grade")
    major = sum(1 for g in grades if g in ("Major issues", "Not publishable"))
    needs_work = len(successful) - pub_ready - major

    return {
        "audited": len(successful),
        "errors": len(results) - len(successful),
        "publication_ready": pub_ready,
        "needs_work": needs_work,
        "major_issues": major,
        "mean_score": round(sum(scores) / len(scores), 1),
        "median_score": round(median, 1),
        "score_range": [round(min(scores), 1), round(max(scores), 1)],
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_json_report(batch: BatchResult) -> str:
    """Format batch result as JSON."""
    payload = {
        "contract_version": BATCH_REPORT_CONTRACT_VERSION,
        "target_journal": batch.target_journal,
        "total_projects": batch.total_projects,
        "summary": batch.summary,
        "comparison_matrix": batch.comparison_matrix,
        "cross_cutting_analysis": batch.cross_cutting_analysis,
        "aggregated_remediation": batch.aggregated_remediation,
        "per_project_details": [],
    }

    for r in batch.results:
        detail: Dict[str, Any] = {
            "project_id": r.project_id,
            "project_label": r.project_label,
            "project_path": r.project_path,
            "success": r.success,
            "elapsed_seconds": round(r.elapsed_seconds, 2),
        }
        if r.success:
            detail["total_score"] = _get_total_score(r.audit_report)
            detail["grade"] = _get_grade(r.audit_report)
        else:
            detail["error"] = r.error
        payload["per_project_details"].append(detail)

    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_markdown_report(batch: BatchResult) -> str:
    """Format batch result as Markdown."""
    lines: List[str] = []
    lines.append("# Batch Journal Review Report")
    lines.append("")
    if batch.target_journal:
        lines.append(f"**Target Journal**: {batch.target_journal}")
    lines.append(f"**Total Projects**: {batch.total_projects}")
    lines.append("")

    # Summary.
    s = batch.summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Audited | {s.get('audited', 0)} |")
    lines.append(f"| Errors | {s.get('errors', 0)} |")
    lines.append(f"| Publication-ready | {s.get('publication_ready', 0)} |")
    lines.append(f"| Needs work | {s.get('needs_work', 0)} |")
    lines.append(f"| Major issues | {s.get('major_issues', 0)} |")
    lines.append(f"| Mean score | {s.get('mean_score', 0.0)} |")
    lines.append(f"| Median score | {s.get('median_score', 0.0)} |")
    sr = s.get("score_range", [0, 0])
    lines.append(f"| Score range | {sr[0]} – {sr[1]} |")
    lines.append("")

    # Comparison matrix.
    lines.append("## Comparison Matrix")
    lines.append("")
    lines.append("| Project | Score | Grade | Top Gap |")
    lines.append("|---------|-------|-------|---------|")
    for entry in batch.comparison_matrix:
        project = entry.get("project", "")
        score = entry.get("total_score", 0.0)
        grade = entry.get("grade", "")
        gaps = entry.get("top_gaps", [])
        top_gap = gaps[0] if gaps else "—"
        lines.append(f"| {project} | {score} | {grade} | {top_gap} |")
    lines.append("")

    # Cross-cutting analysis.
    lines.append("## Cross-Cutting Analysis")
    lines.append("")
    lines.append("### Most Failed Dimensions")
    lines.append("")
    cca = batch.cross_cutting_analysis
    for dim in cca.get("most_failed_dimensions", [])[:5]:
        lines.append(
            f"- **{dim['dimension']}**: avg {dim['avg_score_pct']}%, "
            f"{dim['projects_below_80pct']} projects below 80%"
        )
    lines.append("")

    lines.append("### Most Common Gaps")
    lines.append("")
    for gap in cca.get("most_common_gaps", [])[:5]:
        lines.append(f"- {gap['requirement']} (missing in {gap['missing_in_n_projects']} projects)")
    lines.append("")

    # Aggregated remediation.
    lines.append("## Aggregated Remediation Priorities")
    lines.append("")
    lines.append("| # | Action | Severity | Affects | Dimension |")
    lines.append("|---|--------|----------|---------|-----------|")
    for rem in batch.aggregated_remediation[:10]:
        lines.append(
            f"| {rem.get('priority', '')} | {rem['action']} | {rem['severity']} | "
            f"{rem['affects_n_projects']} projects | {rem.get('dimension', '')} |"
        )
    lines.append("")

    return "\n".join(lines)


def format_text_report(batch: BatchResult) -> str:
    """Format batch result as plain text."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("BATCH JOURNAL REVIEW REPORT")
    lines.append("=" * 70)
    if batch.target_journal:
        lines.append(f"Target Journal: {batch.target_journal}")
    lines.append(f"Total Projects: {batch.total_projects}")
    lines.append("")

    s = batch.summary
    lines.append(f"Audited: {s.get('audited', 0)}  |  Errors: {s.get('errors', 0)}")
    lines.append(
        f"Publication-ready: {s.get('publication_ready', 0)}  |  "
        f"Needs work: {s.get('needs_work', 0)}  |  "
        f"Major issues: {s.get('major_issues', 0)}"
    )
    lines.append(
        f"Mean score: {s.get('mean_score', 0.0)}  |  "
        f"Median: {s.get('median_score', 0.0)}  |  "
        f"Range: {s.get('score_range', [0, 0])}"
    )
    lines.append("")
    lines.append("-" * 70)
    lines.append("PROJECT SCORES")
    lines.append("-" * 70)

    for entry in batch.comparison_matrix:
        status = entry.get("status", "")
        if status == "error":
            lines.append(f"  [{entry['project']}] ERROR: {entry.get('error', '')}")
        else:
            lines.append(
                f"  [{entry['project']}] Score: {entry['total_score']}  "
                f"Grade: {entry['grade']}"
            )
            gaps = entry.get("top_gaps", [])
            if gaps:
                lines.append(f"    Top gaps: {'; '.join(gaps[:3])}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("TOP REMEDIATION PRIORITIES")
    lines.append("-" * 70)
    for rem in batch.aggregated_remediation[:10]:
        lines.append(
            f"  #{rem.get('priority', '')} [{rem['severity']}] {rem['action']} "
            f"(affects {rem['affects_n_projects']} projects)"
        )
    lines.append("")

    return "\n".join(lines)


def format_summary_csv(batch: BatchResult) -> str:
    """Format comparison matrix as CSV."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["project_id", "label", "total_score", "grade", "status", "top_gap"])
    for entry in batch.comparison_matrix:
        gaps = entry.get("top_gaps", [])
        writer.writerow(
            [
                entry.get("project", ""),
                entry.get("label", ""),
                entry.get("total_score", 0.0),
                entry.get("grade", ""),
                entry.get("status", ""),
                gaps[0] if gaps else "",
            ]
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch journal review: audit N medical ML projects in parallel.\n\n"
            "Produces comparison matrix, cross-cutting analysis, and aggregated\n"
            "remediation priorities.\n\n"
            "Examples:\n"
            "  python3 scripts/batch_journal_review.py --manifest batch_manifest.json --output report.json\n"
            "  python3 scripts/batch_journal_review.py --manifest batch_manifest.json --target-journal nature_medicine\n"
            "  python3 scripts/batch_journal_review.py --manifest batch_manifest.json --format markdown\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to batch manifest JSON file.",
    )
    parser.add_argument(
        "--target-journal",
        type=str,
        default=None,
        choices=[
            "nature_medicine",
            "lancet_digital_health",
            "jama",
            "bmj",
            "npj_digital_medicine",
        ],
        help="Target journal for gap analysis (overrides manifest).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout).",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["json", "markdown", "text"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional: also emit a summary CSV file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Max parallel workers (default: 2).",
    )

    args = parser.parse_args()

    # Load manifest.
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"[FAIL] Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    entries, manifest_journals = load_manifest(manifest_path)
    if not entries:
        print("[FAIL] No projects found in manifest.", file=sys.stderr)
        return 2

    # Resolve target journal.
    target_journal = args.target_journal
    if not target_journal and manifest_journals:
        target_journal = manifest_journals[0]

    print(f"[INFO] Batch audit: {len(entries)} projects, workers={args.workers}", file=sys.stderr)
    if target_journal:
        print(f"[INFO] Target journal: {target_journal}", file=sys.stderr)

    # Run batch audit.
    batch = run_batch_audit(
        entries=entries,
        target_journal=target_journal,
        max_workers=args.workers,
    )

    # Format output.
    if args.output_format == "markdown":
        output_text = format_markdown_report(batch)
    elif args.output_format == "text":
        output_text = format_text_report(batch)
    else:
        output_text = format_json_report(batch)

    # Write output.
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"[OK] Report written to {out_path}", file=sys.stderr)
    else:
        print(output_text)

    # Optional CSV.
    if args.summary_csv:
        csv_path = Path(args.summary_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(format_summary_csv(batch), encoding="utf-8")
        print(f"[OK] Summary CSV written to {csv_path}", file=sys.stderr)

    # Exit code based on results.
    errors = sum(1 for r in batch.results if not r.success)
    if errors == len(batch.results):
        return 2
    if errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
