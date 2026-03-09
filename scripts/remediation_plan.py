#!/usr/bin/env python3
"""
Remediation Plan Generator — prioritized action plan from evidence directory.

Scans all gate reports, collects failures and warnings, groups by root cause,
orders by pipeline dependency, and produces a step-by-step remediation plan
with concrete CLI commands.

Usage:
    python3 scripts/remediation_plan.py --evidence-dir evidence/
    python3 scripts/remediation_plan.py --evidence-dir evidence/ --json
    python3 scripts/remediation_plan.py --evidence-dir evidence/ --output plan.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Gate execution order — earlier gates block later ones.
GATE_ORDER: List[Tuple[str, str]] = [
    ("request_contract", "request_contract_report.json"),
    ("manifest", "manifest.json"),
    ("split_protocol", "split_protocol_report.json"),
    ("leakage", "leakage_report.json"),
    ("definition_guard", "definition_guard_report.json"),
    ("lineage", "lineage_report.json"),
    ("covariate_shift", "covariate_shift_report.json"),
    ("imbalance_policy", "imbalance_policy_report.json"),
    ("missingness_policy", "missingness_policy_report.json"),
    ("tuning_leakage", "tuning_leakage_report.json"),
    ("schema_preflight", "schema_preflight_report.json"),
    ("model_selection_audit", "model_selection_audit_report.json"),
    ("feature_engineering_audit", "feature_engineering_audit_report.json"),
    ("clinical_metrics", "clinical_metrics_report.json"),
    ("prediction_replay", "prediction_replay_report.json"),
    ("distribution_generalization", "distribution_generalization_report.json"),
    ("generalization_gap", "generalization_gap_report.json"),
    ("robustness", "robustness_gate_report.json"),
    ("seed_stability", "seed_stability_report.json"),
    ("external_validation", "external_validation_gate_report.json"),
    ("calibration_dca", "calibration_dca_report.json"),
    ("ci_matrix", "ci_matrix_gate_report.json"),
    ("metric_consistency", "metric_consistency_report.json"),
    ("evaluation_quality", "evaluation_quality_report.json"),
    ("permutation", "permutation_report.json"),
    ("reporting_bias", "reporting_bias_report.json"),
    ("execution_attestation", "execution_attestation_report.json"),
    ("self_critique", "self_critique_report.json"),
    ("security_audit", "security_audit_gate_report.json"),
    ("publication", "publication_gate_report.json"),
]

# Code prefix → (priority 1-5, category, remediation command hint)
_REMEDIATION_MAP: List[Tuple[str, int, str, str]] = [
    # Data integrity — fix first
    ("io_error", 1, "data", "Check file paths and permissions."),
    ("input_csv_read_failed", 1, "data", "Verify CSV files exist and are readable."),
    ("missing_split", 1, "data", "Run: python3 scripts/split_data.py ..."),
    ("row_overlap", 1, "data", "Re-split data with patient-level separation: python3 scripts/split_data.py --patient-id-col patient_id"),
    ("patient_overlap", 1, "data", "Re-split ensuring patient-level exclusivity."),
    ("id_overlap", 1, "data", "Re-split data ensuring ID-level exclusivity."),
    ("temporal_overlap", 1, "data", "Enforce temporal ordering: python3 scripts/split_data.py --time-col event_time"),
    ("temporal_boundary", 1, "data", "Set strict chronological boundaries in split_protocol_spec."),
    ("entity_overlap", 1, "data", "Re-split data ensuring entity-level exclusivity."),

    # Leakage — critical
    ("leakage", 1, "leakage", "Review feature pipeline for information leakage."),
    ("test_data_usage", 1, "leakage", "Isolate test split completely from training pipeline."),
    ("test_split_used", 1, "leakage", "Use validation split for selection/calibration, never test."),
    ("resampling_scope", 1, "leakage", "Ensure resampling is confined within training split."),
    ("tuning_", 2, "leakage", "Review tuning_protocol_spec for leakage-safe configuration."),
    ("suspicious_feature", 2, "leakage", "Review flagged feature names for potential target leakage."),

    # Split/protocol
    ("split_seed_not_locked", 2, "protocol", "Lock random seed in split_protocol_spec.json."),
    ("split_not_frozen", 2, "protocol", "Set frozen: true in split_protocol_spec.json."),
    ("split_", 2, "protocol", "Review and fix split_protocol_spec.json."),

    # Definition/schema
    ("definition_", 2, "schema", "Review phenotype_definition_spec for target variable."),
    ("target_", 2, "schema", "Ensure target column is binary with no missing values."),
    ("feature_", 2, "schema", "Review feature group spec and lineage."),

    # Model quality
    ("covariate_shift", 3, "model", "Investigate distributional shift; consider stratified splitting."),
    ("prevalence_shift", 3, "model", "Use stratified splitting to maintain prevalence balance."),
    ("imbalance", 3, "model", "Review imbalance_policy_spec for appropriate strategy."),
    ("missingness_", 3, "model", "Review missingness_policy for imputation strategy."),
    ("model_selection", 3, "model", "Review model_selection_report.json for candidate pool issues."),
    ("selection_", 3, "model", "Review selection criteria and one-SE rule application."),
    ("clinical_", 3, "model", "Adjust model or thresholds per performance_policy.json."),
    ("threshold_", 3, "model", "Review threshold_policy in performance_policy.json."),
    ("calibration", 3, "model", "Try different calibration methods (sigmoid, isotonic)."),

    # Robustness/generalization
    ("robustness_", 3, "robustness", "Investigate metric stability across data subgroups."),
    ("seed_stability", 3, "robustness", "Results vary too much across seeds; add regularization."),
    ("seed_", 3, "robustness", "Ensure consistent seed usage across pipeline."),
    ("distribution", 3, "robustness", "Review feature distributions between internal/external cohorts."),
    ("generalization", 3, "robustness", "Model may be overfitting; try simpler models."),
    ("overfit_", 3, "robustness", "Reduce model complexity or increase regularization."),
    ("external_", 3, "robustness", "Check external cohort data quality and compatibility."),
    ("permutation_", 3, "robustness", "Model may not outperform random; review features."),
    ("ci_", 3, "robustness", "Increase bootstrap resamples or review CI width thresholds."),
    ("metric_", 3, "robustness", "Verify reported metrics match computed values."),

    # Attestation/signing — fix after model issues
    ("signature_", 4, "attestation", "Regenerate signatures with valid keys."),
    ("signing_", 4, "attestation", "Check key validity, expiration, and revocation status."),
    ("witness_", 4, "attestation", "Ensure sufficient independent witnesses for attestation."),
    ("timestamp_", 4, "attestation", "Verify timestamp authority and ordering."),
    ("transparency_", 4, "attestation", "Check transparency log entries and signatures."),
    ("execution_attestation_", 4, "attestation", "Run: python3 scripts/generate_execution_attestation.py ..."),

    # Reporting/publication — fix last
    ("reporting_", 5, "publication", "Complete reporting_bias_checklist.json per guidelines."),
    ("publication_", 5, "publication", "Address all upstream gate failures before publication claim."),
    ("component_not_passed", 5, "publication", "Fix the failing component gate first."),
    ("component_not_strict", 5, "publication", "Re-run component gate with --strict flag."),
    ("component_has_failures", 5, "publication", "Fix failures in the component gate."),
    ("manifest_", 5, "publication", "Re-run manifest_lock.py to update manifest."),
    ("missing_component", 5, "publication", "Run the missing component gate."),
    ("missing_or_invalid", 5, "publication", "Regenerate the missing/invalid artifact."),
    ("quality_score", 5, "publication", "Improve upstream gate results to raise quality score."),

    # Generic
    ("gate_timeout", 2, "infra", "Increase --timeout or optimize data/model complexity."),
    ("input_error", 2, "infra", "Check input file paths and formats."),
    ("path_not_found", 2, "infra", "Verify the file exists at the specified path."),
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


def _lookup_remediation(code: str) -> Tuple[int, str, str]:
    """Look up priority, category, and remediation for a failure code."""
    for prefix, priority, category, remedy in _REMEDIATION_MAP:
        if code.startswith(prefix):
            return priority, category, remedy
    return 3, "other", "Review the gate script source for details."


def collect_issues(evidence_dir: Path) -> List[Dict[str, Any]]:
    """Scan all gate reports and collect issues with metadata."""
    all_issues: List[Dict[str, Any]] = []

    for order_idx, (gate_name, filename) in enumerate(GATE_ORDER):
        report = _load_report(evidence_dir / filename)

        if report is None:
            all_issues.append({
                "gate": gate_name,
                "gate_order": order_idx,
                "code": "gate_report_missing",
                "severity": "error",
                "message": f"Gate report not found: {filename}",
                "priority": 2,
                "category": "missing",
                "remediation": f"Run the {gate_name} gate to generate {filename}.",
            })
            continue

        status = str(report.get("status", "unknown"))
        if status == "pass":
            continue

        # Collect failures
        failures = report.get("failures", report.get("issues", []))
        if isinstance(failures, list):
            for issue in failures:
                if not isinstance(issue, dict):
                    continue
                code = str(issue.get("code", "unknown"))
                message = str(issue.get("message", ""))
                priority, category, remedy = _lookup_remediation(code)
                all_issues.append({
                    "gate": gate_name,
                    "gate_order": order_idx,
                    "code": code,
                    "severity": "error",
                    "message": message,
                    "priority": priority,
                    "category": category,
                    "remediation": remedy,
                })

        # Collect warnings
        warnings = report.get("warnings", [])
        if isinstance(warnings, list):
            for issue in warnings:
                if not isinstance(issue, dict):
                    continue
                code = str(issue.get("code", "unknown"))
                message = str(issue.get("message", ""))
                priority, category, remedy = _lookup_remediation(code)
                all_issues.append({
                    "gate": gate_name,
                    "gate_order": order_idx,
                    "code": code,
                    "severity": "warning",
                    "message": message,
                    "priority": max(priority, 4),  # warnings are lower priority
                    "category": category,
                    "remediation": remedy,
                })

    return all_issues


def build_plan(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a prioritized remediation plan from collected issues."""
    # Sort by priority then gate order
    sorted_issues = sorted(issues, key=lambda x: (x["priority"], x["gate_order"]))

    # Group by category
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for issue in sorted_issues:
        cat = issue["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(issue)

    # Deduplicate remediation steps (same code across gates)
    seen_codes: Dict[str, int] = {}
    steps: List[Dict[str, Any]] = []
    for issue in sorted_issues:
        code = issue["code"]
        if code in seen_codes:
            seen_codes[code] += 1
            continue
        seen_codes[code] = 1
        steps.append({
            "step": len(steps) + 1,
            "priority": issue["priority"],
            "category": issue["category"],
            "gate": issue["gate"],
            "code": code,
            "severity": issue["severity"],
            "action": issue["remediation"],
        })

    # Count occurrences
    for step in steps:
        step["occurrences"] = seen_codes[step["code"]]

    error_count = sum(1 for i in issues if i["severity"] == "error")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    gates_affected = len(set(i["gate"] for i in issues))

    return {
        "schema_version": "remediation_plan.v1",
        "total_issues": len(issues),
        "errors": error_count,
        "warnings": warning_count,
        "gates_affected": gates_affected,
        "unique_codes": len(seen_codes),
        "steps": steps,
        "categories": {
            cat: len(items)
            for cat, items in sorted(categories.items(), key=lambda x: min(i["priority"] for i in x[1]))
        },
    }


PRIORITY_LABELS = {
    1: "CRITICAL",
    2: "HIGH",
    3: "MEDIUM",
    4: "LOW",
    5: "INFO",
}

PRIORITY_ICONS = {
    1: "🔴",
    2: "🟠",
    3: "🟡",
    4: "🔵",
    5: "⚪",
}


def to_markdown(plan: Dict[str, Any]) -> str:
    """Render remediation plan as Markdown."""
    lines: List[str] = []
    lines.append("# Remediation Plan")
    lines.append("")
    lines.append(f"**Total Issues**: {plan['total_issues']} "
                 f"({plan['errors']} errors, {plan['warnings']} warnings)")
    lines.append(f"**Gates Affected**: {plan['gates_affected']}")
    lines.append(f"**Unique Issue Codes**: {plan['unique_codes']}")
    lines.append("")

    if not plan["steps"]:
        lines.append("No issues found — all gates are passing or missing reports.")
        return "\n".join(lines)

    # Category overview
    lines.append("## Issue Categories")
    for cat, count in plan["categories"].items():
        lines.append(f"- **{cat}**: {count} issue(s)")
    lines.append("")

    # Steps
    lines.append("## Action Steps")
    lines.append("")

    current_priority = 0
    for step in plan["steps"]:
        p = step["priority"]
        if p != current_priority:
            current_priority = p
            label = PRIORITY_LABELS.get(p, f"P{p}")
            icon = PRIORITY_ICONS.get(p, "•")
            lines.append(f"### {icon} {label}")
            lines.append("")

        occ = f" (×{step['occurrences']})" if step["occurrences"] > 1 else ""
        sev = "⛔" if step["severity"] == "error" else "⚠️"
        lines.append(f"**{step['step']}.** {sev} `{step['code']}`{occ} — _{step['gate']}_")
        lines.append(f"   {step['action']}")
        lines.append("")

    return "\n".join(lines)


def to_text(plan: Dict[str, Any]) -> str:
    """Render remediation plan as plain text."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  ML Leakage Guard — Remediation Plan")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Issues:  {plan['total_issues']} ({plan['errors']} errors, {plan['warnings']} warnings)")
    lines.append(f"  Gates:   {plan['gates_affected']} affected")
    lines.append(f"  Codes:   {plan['unique_codes']} unique")
    lines.append("")

    if not plan["steps"]:
        lines.append("  No issues found.")
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append("-" * 60)
    current_priority = 0
    for step in plan["steps"]:
        p = step["priority"]
        if p != current_priority:
            current_priority = p
            label = PRIORITY_LABELS.get(p, f"P{p}")
            lines.append(f"\n  [{label}]")
            lines.append("-" * 60)

        sev = "FAIL" if step["severity"] == "error" else "WARN"
        occ = f" (x{step['occurrences']})" if step["occurrences"] > 1 else ""
        lines.append(f"  {step['step']:>3}. [{sev}] {step['code']}{occ}")
        lines.append(f"       Gate: {step['gate']}")
        lines.append(f"       Fix:  {step['action']}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a prioritized remediation plan from evidence reports."
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to evidence directory containing gate report JSONs.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON.",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Output as Markdown (default is plain text).",
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

    issues = collect_issues(evidence)
    plan = build_plan(issues)

    if args.json_output:
        output = json.dumps(plan, indent=2, ensure_ascii=False)
    elif args.markdown:
        output = to_markdown(plan)
    else:
        output = to_text(plan)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Plan written to: {out_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
