#!/usr/bin/env python3
"""
Fail-closed robustness gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _gate_utils import add_issue, load_json_from_str as load_json_object


DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "time_slices": {
        "pr_auc_drop_warn": 0.10,
        "pr_auc_drop_fail": 0.14,
        "pr_auc_range_warn": 0.15,
        "pr_auc_range_fail": 0.20,
        "min_slice_size": 8,
        "min_positive": 2,
    },
    "patient_hash_groups": {
        "pr_auc_drop_warn": 0.10,
        "pr_auc_drop_fail": 0.14,
        "pr_auc_range_warn": 0.15,
        "pr_auc_range_fail": 0.20,
        "min_slice_size": 8,
        "min_positive": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate robustness report for time-slice and group-holdout stability.")
    parser.add_argument("--robustness-report", required=True, help="Path to robustness_report.json.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON for threshold overrides.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_bucket_thresholds(policy: Optional[Dict[str, Any]], bucket: str) -> Dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS[bucket])
    if not isinstance(policy, dict):
        return out
    raw = policy.get("robustness_thresholds")
    if not isinstance(raw, dict):
        return out
    block = raw.get(bucket)
    if not isinstance(block, dict):
        return out
    for key in out:
        value = block.get(key)
        if is_finite_number(value) and float(value) >= 0.0:
            out[key] = float(value)
    return out


def metric_in_unit_interval(value: float) -> bool:
    return 0.0 <= value <= 1.0 and math.isfinite(value)


def validate_bucket(
    bucket_name: str,
    block: Dict[str, Any],
    row_key: str,
    thresholds: Dict[str, float],
    overall_pr_auc: float,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, float]:
    rows = block.get(row_key)
    if not isinstance(rows, list) or not rows:
        add_issue(
            failures,
            "robustness_missing_bucket_rows",
            "Robustness bucket must include non-empty rows.",
            {"bucket": bucket_name, "row_key": row_key},
        )
        return {}

    pr_auc_values: List[float] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            add_issue(
                failures,
                "robustness_invalid_bucket_row",
                "Robustness row must be an object.",
                {"bucket": bucket_name, "index": idx, "actual_type": type(row).__name__},
            )
            continue
        n = row.get("n")
        pos = row.get("positive_count")
        metrics = row.get("metrics")
        if not is_finite_number(n) or int(float(n)) <= 0:
            add_issue(
                failures,
                "robustness_invalid_bucket_row",
                "Row n must be positive integer-like.",
                {"bucket": bucket_name, "index": idx, "n": n},
            )
            continue
        if not is_finite_number(pos) or int(float(pos)) < 0:
            add_issue(
                failures,
                "robustness_invalid_bucket_row",
                "Row positive_count must be non-negative integer-like.",
                {"bucket": bucket_name, "index": idx, "positive_count": pos},
            )
            continue
        if not isinstance(metrics, dict):
            add_issue(
                failures,
                "robustness_invalid_bucket_row",
                "Row must include metrics object.",
                {"bucket": bucket_name, "index": idx},
            )
            continue
        pr_auc = metrics.get("pr_auc")
        if not is_finite_number(pr_auc):
            add_issue(
                failures,
                "robustness_non_finite_metric",
                "Row metrics.pr_auc must be finite numeric.",
                {"bucket": bucket_name, "index": idx, "value": pr_auc},
            )
            continue
        pr_auc_f = float(pr_auc)
        if not metric_in_unit_interval(pr_auc_f):
            add_issue(
                failures,
                "robustness_metric_out_of_range",
                "Row metrics.pr_auc must be in [0,1].",
                {"bucket": bucket_name, "index": idx, "value": pr_auc_f},
            )
            continue
        pr_auc_values.append(pr_auc_f)

        if int(float(n)) < int(thresholds["min_slice_size"]):
            add_issue(
                warnings,
                "robustness_low_slice_size",
                "Slice/group sample size is below robustness minimum.",
                {
                    "bucket": bucket_name,
                    "index": idx,
                    "n": int(float(n)),
                    "min_slice_size": int(thresholds["min_slice_size"]),
                },
            )
        if int(float(pos)) < int(thresholds["min_positive"]):
            add_issue(
                warnings,
                "robustness_low_positive_count",
                "Slice/group positive count is below robustness minimum.",
                {
                    "bucket": bucket_name,
                    "index": idx,
                    "positive_count": int(float(pos)),
                    "min_positive": int(thresholds["min_positive"]),
                },
            )

    if not pr_auc_values:
        add_issue(
            failures,
            "robustness_missing_metric_values",
            "No valid pr_auc values found in robustness bucket.",
            {"bucket": bucket_name},
        )
        return {}

    min_pr_auc = min(pr_auc_values)
    max_pr_auc = max(pr_auc_values)
    pr_auc_range = max_pr_auc - min_pr_auc
    worst_drop = overall_pr_auc - min_pr_auc

    if worst_drop > float(thresholds["pr_auc_drop_fail"]):
        add_issue(
            failures,
            "robustness_pr_auc_drop_exceeds_threshold",
            "Worst-slice/group PR-AUC drop exceeds fail threshold.",
            {
                "bucket": bucket_name,
                "overall_pr_auc": overall_pr_auc,
                "worst_pr_auc": min_pr_auc,
                "worst_drop": worst_drop,
                "fail_threshold": float(thresholds["pr_auc_drop_fail"]),
            },
        )
    elif worst_drop > float(thresholds["pr_auc_drop_warn"]):
        add_issue(
            warnings,
            "robustness_pr_auc_drop_near_threshold",
            "Worst-slice/group PR-AUC drop exceeds warning threshold.",
            {
                "bucket": bucket_name,
                "overall_pr_auc": overall_pr_auc,
                "worst_pr_auc": min_pr_auc,
                "worst_drop": worst_drop,
                "warn_threshold": float(thresholds["pr_auc_drop_warn"]),
            },
        )

    if pr_auc_range > float(thresholds["pr_auc_range_fail"]):
        add_issue(
            failures,
            "robustness_pr_auc_range_exceeds_threshold",
            "Slice/group PR-AUC range exceeds fail threshold.",
            {
                "bucket": bucket_name,
                "range": pr_auc_range,
                "fail_threshold": float(thresholds["pr_auc_range_fail"]),
            },
        )
    elif pr_auc_range > float(thresholds["pr_auc_range_warn"]):
        add_issue(
            warnings,
            "robustness_pr_auc_range_near_threshold",
            "Slice/group PR-AUC range exceeds warning threshold.",
            {
                "bucket": bucket_name,
                "range": pr_auc_range,
                "warn_threshold": float(thresholds["pr_auc_range_warn"]),
            },
        )

    return {
        "pr_auc_min": float(min_pr_auc),
        "pr_auc_max": float(max_pr_auc),
        "pr_auc_range": float(pr_auc_range),
        "pr_auc_worst_drop_from_overall": float(worst_drop),
        "n_rows": float(len(pr_auc_values)),
    }


def compare_summary_fields(
    declared: Dict[str, Any],
    computed: Dict[str, float],
    bucket: str,
    failures: List[Dict[str, Any]],
) -> None:
    for key, computed_value in computed.items():
        declared_value = declared.get(key)
        if not is_finite_number(declared_value):
            add_issue(
                failures,
                "robustness_summary_missing",
                "Robustness summary missing required numeric field.",
                {"bucket": bucket, "field": key, "declared": declared_value},
            )
            continue
        if abs(float(declared_value) - float(computed_value)) > 1e-9:
            add_issue(
                failures,
                "robustness_summary_mismatch",
                "Declared robustness summary does not match computed value.",
                {
                    "bucket": bucket,
                    "field": key,
                    "declared": float(declared_value),
                    "computed": float(computed_value),
                },
            )


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    robustness_path = Path(args.robustness_report).expanduser().resolve()
    if not robustness_path.exists():
        add_issue(
            failures,
            "missing_robustness_report",
            "robustness_report file not found.",
            {"path": str(robustness_path)},
        )
        return finish(args, failures, warnings, {}, {}, {})

    try:
        robustness = load_json_object(str(robustness_path))
    except Exception as exc:
        add_issue(
            failures,
            "invalid_robustness_report",
            "Unable to parse robustness_report JSON.",
            {"path": str(robustness_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {}, {}, {})

    policy_payload: Optional[Dict[str, Any]] = None
    if args.performance_policy:
        policy_path = Path(args.performance_policy).expanduser().resolve()
        try:
            policy_payload = load_json_object(str(policy_path))
        except Exception as exc:
            add_issue(
                failures,
                "invalid_performance_policy",
                "Unable to parse performance policy JSON.",
                {"path": str(policy_path), "error": str(exc)},
            )

    thresholds = {
        "time_slices": parse_bucket_thresholds(policy_payload, "time_slices"),
        "patient_hash_groups": parse_bucket_thresholds(policy_payload, "patient_hash_groups"),
    }

    primary_metric = str(robustness.get("primary_metric", "")).strip().lower()
    if primary_metric != "pr_auc":
        add_issue(
            failures,
            "robustness_primary_metric_mismatch",
            "robustness_report.primary_metric must be pr_auc in strict mode.",
            {"primary_metric": robustness.get("primary_metric")},
        )

    overall_metrics = robustness.get("overall_test_metrics")
    overall_pr_auc = None
    if isinstance(overall_metrics, dict) and is_finite_number(overall_metrics.get("pr_auc")):
        overall_pr_auc = float(overall_metrics.get("pr_auc"))
        if not metric_in_unit_interval(overall_pr_auc):
            add_issue(
                failures,
                "robustness_metric_out_of_range",
                "overall_test_metrics.pr_auc must be in [0,1].",
                {"value": overall_pr_auc},
            )
    else:
        add_issue(
            failures,
            "robustness_missing_overall_metric",
            "robustness_report.overall_test_metrics.pr_auc must be finite numeric.",
            {"overall_test_metrics": overall_metrics},
        )

    computed: Dict[str, Dict[str, float]] = {}
    if overall_pr_auc is not None and metric_in_unit_interval(overall_pr_auc):
        time_block = robustness.get("time_slices")
        if not isinstance(time_block, dict):
            add_issue(
                failures,
                "robustness_missing_bucket",
                "robustness_report must include time_slices object.",
                {},
            )
        else:
            computed["time_slices"] = validate_bucket(
                bucket_name="time_slices",
                block=time_block,
                row_key="slices",
                thresholds=thresholds["time_slices"],
                overall_pr_auc=overall_pr_auc,
                failures=failures,
                warnings=warnings,
            )

        group_block = robustness.get("patient_hash_groups")
        if not isinstance(group_block, dict):
            add_issue(
                failures,
                "robustness_missing_bucket",
                "robustness_report must include patient_hash_groups object.",
                {},
            )
        else:
            computed["patient_hash_groups"] = validate_bucket(
                bucket_name="patient_hash_groups",
                block=group_block,
                row_key="groups",
                thresholds=thresholds["patient_hash_groups"],
                overall_pr_auc=overall_pr_auc,
                failures=failures,
                warnings=warnings,
            )

    declared_summary = robustness.get("summary")
    if not isinstance(declared_summary, dict):
        add_issue(
            failures,
            "robustness_summary_missing",
            "robustness_report.summary must be an object.",
            {},
        )
    else:
        for bucket in ("time_slices", "patient_hash_groups"):
            declared_bucket = declared_summary.get(bucket)
            computed_bucket = computed.get(bucket)
            if not isinstance(declared_bucket, dict) or not isinstance(computed_bucket, dict):
                continue
            compare_summary_fields(declared_bucket, computed_bucket, bucket, failures)

    return finish(args, failures, warnings, robustness, computed, thresholds)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    source_payload: Dict[str, Any],
    computed_summary: Dict[str, Dict[str, float]],
    thresholds: Dict[str, Dict[str, float]],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "robustness_report": str(Path(args.robustness_report).expanduser().resolve()),
            "primary_metric": source_payload.get("primary_metric"),
            "overall_test_metrics": source_payload.get("overall_test_metrics"),
            "computed": computed_summary,
            "thresholds": thresholds,
        },
    }

    if args.report:
        out = Path(args.report).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
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
