#!/usr/bin/env python3
"""
Run Comparison Tool for ML Leakage Guard.

Compares two evidence directories and produces a JSON diff report
and Markdown summary highlighting gate status changes, metric deltas,
and new/resolved failure codes.

Usage:
    python3 scripts/compare_runs.py \
        --baseline evidence_v1 \
        --candidate evidence_v2 \
        --output comparison_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare two pipeline runs.")
    parser.add_argument("--baseline", required=True, help="Path to baseline evidence directory.")
    parser.add_argument("--candidate", required=True, help="Path to candidate evidence directory.")
    parser.add_argument("--output", help="Output JSON report path.")
    parser.add_argument("--markdown", help="Output Markdown summary path.")
    return parser.parse_args()


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


def _extract_status(report: Optional[Dict[str, Any]]) -> str:
    """Extract status from a gate report."""
    if report is None:
        return "missing"
    return str(report.get("status", "unknown"))


def _extract_codes(report: Optional[Dict[str, Any]]) -> Set[str]:
    """Extract failure codes from a gate report."""
    if report is None:
        return set()
    issues = report.get("issues", [])
    codes: Set[str] = set()
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict) and "code" in issue:
                codes.add(str(issue["code"]))
    return codes


def _extract_metric(report: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    """Extract a numeric metric from a report by key path."""
    if report is None:
        return None
    val = report.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


REPORT_FILES = [
    "request_contract_report.json",
    "manifest.json",
    "execution_attestation_report.json",
    "reporting_bias_report.json",
    "leakage_report.json",
    "split_protocol_report.json",
    "covariate_shift_report.json",
    "definition_guard_report.json",
    "lineage_report.json",
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
    "publication_gate_report.json",
    "self_critique_report.json",
]

METRIC_KEYS = [
    "actual_metric",
    "primary_metric_value",
    "pr_auc",
    "roc_auc",
    "sensitivity",
    "specificity",
    "npv",
    "ppv",
    "brier_score",
]


def compare(baseline_dir: Path, candidate_dir: Path) -> Dict[str, Any]:
    """Compare two evidence directories."""
    gate_diffs: List[Dict[str, Any]] = []
    metric_diffs: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, str]] = []
    resolved_failures: List[Dict[str, str]] = []

    for filename in REPORT_FILES:
        gate_name = filename.replace("_report.json", "").replace(".json", "")
        base_report = _load_report(baseline_dir / filename)
        cand_report = _load_report(candidate_dir / filename)

        base_status = _extract_status(base_report)
        cand_status = _extract_status(cand_report)

        if base_status != cand_status:
            gate_diffs.append({
                "gate": gate_name,
                "file": filename,
                "baseline_status": base_status,
                "candidate_status": cand_status,
            })

        base_codes = _extract_codes(base_report)
        cand_codes = _extract_codes(cand_report)

        for code in cand_codes - base_codes:
            new_failures.append({"gate": gate_name, "code": code})
        for code in base_codes - cand_codes:
            resolved_failures.append({"gate": gate_name, "code": code})

    # Compare key metrics from evaluation reports
    for metric_key in METRIC_KEYS:
        for filename in ["evaluation_report.json", "model_selection_report.json",
                         "metric_consistency_report.json"]:
            base_r = _load_report(baseline_dir / filename)
            cand_r = _load_report(candidate_dir / filename)
            base_val = _extract_metric(base_r, metric_key)
            cand_val = _extract_metric(cand_r, metric_key)
            if base_val is not None and cand_val is not None and base_val != cand_val:
                metric_diffs.append({
                    "metric": metric_key,
                    "source": filename,
                    "baseline": round(base_val, 6),
                    "candidate": round(cand_val, 6),
                    "delta": round(cand_val - base_val, 6),
                })

    # Pipeline-level summary
    base_pipeline = (
        _load_report(baseline_dir / "dag_pipeline_report.json")
        or _load_report(baseline_dir / "strict_pipeline_report.json")
    )
    cand_pipeline = (
        _load_report(candidate_dir / "dag_pipeline_report.json")
        or _load_report(candidate_dir / "strict_pipeline_report.json")
    )

    return {
        "schema_version": "v1.0",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "pipeline_status": {
            "baseline": _extract_status(base_pipeline),
            "candidate": _extract_status(cand_pipeline),
        },
        "gate_status_changes": gate_diffs,
        "new_failure_codes": new_failures,
        "resolved_failure_codes": resolved_failures,
        "metric_deltas": metric_diffs,
        "summary": {
            "gates_changed": len(gate_diffs),
            "new_failures": len(new_failures),
            "resolved_failures": len(resolved_failures),
            "metric_changes": len(metric_diffs),
        },
    }


def to_markdown(result: Dict[str, Any]) -> str:
    """Convert comparison result to Markdown summary."""
    lines = ["# Run Comparison Report", ""]
    lines.append(f"- **Baseline**: `{result['baseline_dir']}`")
    lines.append(f"- **Candidate**: `{result['candidate_dir']}`")
    lines.append(f"- **Pipeline**: {result['pipeline_status']['baseline']} → {result['pipeline_status']['candidate']}")
    lines.append("")

    s = result["summary"]
    lines.append("## Summary")
    lines.append(f"- Gate status changes: **{s['gates_changed']}**")
    lines.append(f"- New failures: **{s['new_failures']}**")
    lines.append(f"- Resolved failures: **{s['resolved_failures']}**")
    lines.append(f"- Metric changes: **{s['metric_changes']}**")
    lines.append("")

    if result["gate_status_changes"]:
        lines.append("## Gate Status Changes")
        lines.append("| Gate | Baseline | Candidate |")
        lines.append("|------|----------|-----------|")
        for d in result["gate_status_changes"]:
            lines.append(f"| {d['gate']} | {d['baseline_status']} | {d['candidate_status']} |")
        lines.append("")

    if result["metric_deltas"]:
        lines.append("## Metric Deltas")
        lines.append("| Metric | Source | Baseline | Candidate | Delta |")
        lines.append("|--------|--------|----------|-----------|-------|")
        for d in result["metric_deltas"]:
            sign = "+" if d["delta"] > 0 else ""
            lines.append(f"| {d['metric']} | {d['source']} | {d['baseline']:.4f} | {d['candidate']:.4f} | {sign}{d['delta']:.4f} |")
        lines.append("")

    if result["new_failure_codes"]:
        lines.append("## New Failure Codes")
        for d in result["new_failure_codes"]:
            lines.append(f"- `{d['gate']}`: `{d['code']}`")
        lines.append("")

    if result["resolved_failure_codes"]:
        lines.append("## Resolved Failure Codes")
        for d in result["resolved_failure_codes"]:
            lines.append(f"- `{d['gate']}`: `{d['code']}`")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    args = parse_args()
    baseline = Path(args.baseline).expanduser().resolve()
    candidate = Path(args.candidate).expanduser().resolve()

    if not baseline.is_dir():
        print(f"Baseline directory not found: {baseline}", file=sys.stderr)
        return 1
    if not candidate.is_dir():
        print(f"Candidate directory not found: {candidate}", file=sys.stderr)
        return 1

    result = compare(baseline, candidate)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"JSON report: {out}")

    md = to_markdown(result)

    if args.markdown:
        md_path = Path(args.markdown).expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        print(f"Markdown report: {md_path}")
    else:
        print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
