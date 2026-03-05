#!/usr/bin/env python3
"""
Evidence Health Check — scan all gate reports and produce a completeness dashboard.

Usage:
    python3 scripts/report_health_check.py --evidence-dir evidence/
    python3 scripts/report_health_check.py --evidence-dir evidence/ --json
    python3 scripts/report_health_check.py --evidence-dir evidence/ --output health.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# All known gate report filenames in execution order.
EXPECTED_REPORTS: List[str] = [
    "request_contract_report.json",
    "manifest.json",
    "split_protocol_report.json",
    "leakage_report.json",
    "definition_guard_report.json",
    "lineage_report.json",
    "covariate_shift_report.json",
    "imbalance_policy_report.json",
    "missingness_policy_report.json",
    "tuning_leakage_report.json",
    "model_selection_audit_report.json",
    "feature_engineering_audit_report.json",
    "clinical_metrics_report.json",
    "prediction_replay_report.json",
    "distribution_generalization_report.json",
    "generalization_gap_report.json",
    "robustness_gate_report.json",
    "seed_stability_report.json",
    "external_validation_gate_report.json",
    "calibration_dca_report.json",
    "ci_matrix_gate_report.json",
    "metric_consistency_report.json",
    "evaluation_quality_report.json",
    "permutation_report.json",
    "reporting_bias_report.json",
    "execution_attestation_report.json",
    "self_critique_report.json",
    "publication_gate_report.json",
]


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON report, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _gate_name(filename: str) -> str:
    """Derive gate name from filename."""
    return filename.replace("_report.json", "").replace(".json", "")


def check_health(evidence_dir: Path) -> Dict[str, Any]:
    """Scan evidence directory and produce health summary."""
    gate_results: List[Dict[str, Any]] = []
    failure_codes: Counter[str] = Counter()
    statuses: Counter[str] = Counter()

    for filename in EXPECTED_REPORTS:
        name = _gate_name(filename)
        report = _load_report(evidence_dir / filename)

        if report is None:
            gate_results.append({
                "gate": name,
                "file": filename,
                "status": "missing",
                "failure_count": 0,
                "warning_count": 0,
            })
            statuses["missing"] += 1
            continue

        status = str(report.get("status", "unknown"))
        failures = report.get("failures", [])
        warnings = report.get("warnings", [])
        f_count = len(failures) if isinstance(failures, list) else 0
        w_count = len(warnings) if isinstance(warnings, list) else 0

        gate_results.append({
            "gate": name,
            "file": filename,
            "status": status,
            "failure_count": f_count,
            "warning_count": w_count,
            "execution_time": report.get("execution_time_seconds"),
        })
        statuses[status] += 1

        if isinstance(failures, list):
            for f in failures:
                if isinstance(f, dict) and "code" in f:
                    failure_codes[str(f["code"])] += 1

    total = len(EXPECTED_REPORTS)
    present = total - statuses.get("missing", 0)
    passed = statuses.get("pass", 0)
    failed = statuses.get("fail", 0)
    missing = statuses.get("missing", 0)

    completeness = round(present / total * 100, 1) if total > 0 else 0.0
    pass_rate = round(passed / present * 100, 1) if present > 0 else 0.0

    # Top failure codes
    top_failures = [
        {"code": code, "count": count}
        for code, count in failure_codes.most_common(10)
    ]

    # Recommendations
    recommendations: List[str] = []
    if missing > 0:
        recommendations.append(
            f"Run {missing} missing gate(s) to complete the pipeline."
        )
    if failed > 0:
        recommendations.append(
            f"Address {failed} failing gate(s) before publication."
        )
    if completeness == 100.0 and pass_rate == 100.0:
        recommendations.append(
            "All gates pass — pipeline is ready for publication."
        )

    return {
        "schema_version": "health_check.v1",
        "evidence_dir": str(evidence_dir),
        "total_gates": total,
        "present": present,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "completeness_pct": completeness,
        "pass_rate_pct": pass_rate,
        "gates": gate_results,
        "top_failure_codes": top_failures,
        "recommendations": recommendations,
    }


def format_text(result: Dict[str, Any]) -> str:
    """Format health check result as human-readable text."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  ML Leakage Guard — Evidence Health Check")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Evidence:     {result['evidence_dir']}")
    lines.append(f"  Completeness: {result['completeness_pct']}% ({result['present']}/{result['total_gates']} reports)")
    lines.append(f"  Pass Rate:    {result['pass_rate_pct']}% ({result['passed']}/{result['present']} present)")
    lines.append(f"  Failed:       {result['failed']}")
    lines.append(f"  Missing:      {result['missing']}")
    lines.append("")

    # Gate status table
    lines.append("-" * 60)
    lines.append(f"  {'Gate':<42} {'Status':<10}")
    lines.append("-" * 60)

    status_icons = {"pass": "✓", "fail": "✗", "missing": "—", "unknown": "?"}
    for g in result["gates"]:
        icon = status_icons.get(g["status"], "?")
        name = g["gate"]
        status = g["status"].upper()
        suffix = ""
        if g["failure_count"] > 0:
            suffix = f" ({g['failure_count']} failures)"
        lines.append(f"  {icon} {name:<40} {status}{suffix}")

    lines.append("")

    # Top failure codes
    if result["top_failure_codes"]:
        lines.append("-" * 60)
        lines.append("  Top Failure Codes:")
        for entry in result["top_failure_codes"]:
            lines.append(f"    [{entry['count']}x] {entry['code']}")
        lines.append("")

    # Recommendations
    if result["recommendations"]:
        lines.append("-" * 60)
        lines.append("  Recommendations:")
        for rec in result["recommendations"]:
            lines.append(f"    • {rec}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Scan gate reports and produce an evidence health dashboard."
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to evidence directory containing gate report JSONs.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--output", help="Write result to file instead of stdout.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence = Path(args.evidence_dir).expanduser().resolve()

    if not evidence.is_dir():
        print(f"Evidence directory not found: {evidence}", file=sys.stderr)
        return 1

    result = check_health(evidence)

    if args.json_output:
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = format_text(result)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Report written to: {out_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
