#!/usr/bin/env python3
"""
Aggregate fail-closed publication gate for medical prediction evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate publication-grade evidence gate.")
    parser.add_argument("--request-report", required=True, help="Path to request contract report JSON.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument("--leakage-report", required=True, help="Path to leakage report JSON.")
    parser.add_argument("--split-protocol-report", required=True, help="Path to split protocol gate report JSON.")
    parser.add_argument("--definition-report", required=True, help="Path to definition-variable guard report JSON.")
    parser.add_argument("--lineage-report", required=True, help="Path to lineage gate report JSON.")
    parser.add_argument("--imbalance-report", required=True, help="Path to imbalance policy gate report JSON.")
    parser.add_argument("--missingness-report", required=True, help="Path to missingness policy gate report JSON.")
    parser.add_argument("--tuning-report", required=True, help="Path to tuning leakage gate report JSON.")
    parser.add_argument("--metric-report", required=True, help="Path to metric consistency report JSON.")
    parser.add_argument("--permutation-report", required=True, help="Path to permutation gate report JSON.")
    parser.add_argument("--report", help="Optional output publication gate report path.")
    parser.add_argument("--strict", action="store_true", help="Require strict-mode component reports.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")
    return data


def validate_component_status(
    name: str,
    report: Optional[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    strict: bool,
) -> None:
    if report is None:
        add_issue(failures, "missing_component_report", "Missing required component report.", {"component": name})
        return

    status = str(report.get("status", "")).lower()
    if status != "pass":
        add_issue(
            failures,
            "component_not_passed",
            "Required component report did not pass.",
            {"component": name, "status": report.get("status")},
        )

    if strict and report.get("strict_mode") is not True:
        add_issue(
            failures,
            "component_not_strict",
            "Required component report was not generated in strict mode.",
            {"component": name},
        )
    elif report.get("strict_mode") is not True:
        add_issue(
            warnings,
            "component_not_strict",
            "Component report was not generated in strict mode.",
            {"component": name},
        )

    failure_count = report.get("failure_count")
    if isinstance(failure_count, int) and failure_count > 0:
        add_issue(
            failures,
            "component_has_failures",
            "Component report contains failures.",
            {"component": name, "failure_count": failure_count},
        )


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    loaded: Dict[str, Dict[str, Any]] = {}

    files = {
        "request_report": args.request_report,
        "manifest": args.manifest,
        "leakage_report": args.leakage_report,
        "split_protocol_report": args.split_protocol_report,
        "definition_report": args.definition_report,
        "lineage_report": args.lineage_report,
        "imbalance_report": args.imbalance_report,
        "missingness_report": args.missingness_report,
        "tuning_report": args.tuning_report,
        "metric_report": args.metric_report,
        "permutation_report": args.permutation_report,
    }

    for name, path in files.items():
        try:
            loaded[name] = load_json(path)
        except Exception as exc:
            add_issue(
                failures,
                "invalid_or_missing_json",
                "Failed to load required JSON artifact.",
                {"artifact": name, "path": str(Path(path).expanduser()), "error": str(exc)},
            )

    manifest = loaded.get("manifest")
    if manifest is not None:
        if str(manifest.get("status", "")).lower() != "pass":
            add_issue(
                failures,
                "manifest_not_passed",
                "Manifest status is not pass.",
                {"status": manifest.get("status")},
            )

        files_meta = manifest.get("files")
        if not isinstance(files_meta, list) or not files_meta:
            add_issue(
                failures,
                "manifest_missing_files",
                "Manifest does not contain locked file entries.",
                {},
            )

        errors = manifest.get("errors", [])
        if isinstance(errors, list) and errors:
            add_issue(
                failures,
                "manifest_has_errors",
                "Manifest contains errors.",
                {"error_count": len(errors)},
            )

        comparison = manifest.get("comparison")
        if isinstance(comparison, dict) and comparison.get("matched") is False:
            add_issue(
                failures,
                "manifest_comparison_mismatch",
                "Manifest comparison against baseline did not match.",
                {
                    "missing_in_current": comparison.get("missing_in_current", []),
                    "missing_in_baseline": comparison.get("missing_in_baseline", []),
                    "hash_mismatches": comparison.get("hash_mismatches", []),
                },
            )

    for component in (
        "request_report",
        "leakage_report",
        "split_protocol_report",
        "definition_report",
        "lineage_report",
        "imbalance_report",
        "missingness_report",
        "tuning_report",
        "metric_report",
        "permutation_report",
    ):
        validate_component_status(component, loaded.get(component), failures, warnings, args.strict)

    metric_report = loaded.get("metric_report")
    if isinstance(metric_report, dict):
        actual_metric = metric_report.get("actual_metric")
        if not (
            isinstance(actual_metric, (int, float))
            and not isinstance(actual_metric, bool)
            and math.isfinite(float(actual_metric))
        ):
            add_issue(
                failures,
                "metric_report_missing_actual",
                "Metric consistency report must contain finite numeric actual_metric.",
                {"actual_metric_type": type(actual_metric).__name__ if actual_metric is not None else None},
            )

    should_fail = bool(failures) or (args.strict and bool(warnings))
    quality_score = max(0.0, min(100.0, 100.0 - 20.0 * len(failures) - 2.5 * len(warnings)))

    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "quality_score": round(quality_score, 2),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "artifacts": {
                name: {
                    "path": str(Path(path).expanduser().resolve()),
                    "loaded": name in loaded,
                    "status": loaded.get(name, {}).get("status"),
                }
                for name, path in files.items()
            }
        },
    }

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
