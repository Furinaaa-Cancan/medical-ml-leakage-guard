#!/usr/bin/env python3
"""
Fail-closed generalization gap gate for train/valid/test overfitting detection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_THRESHOLDS: Dict[Tuple[str, str, str], Tuple[float, float]] = {
    ("train", "valid", "pr_auc"): (0.05, 0.08),
    ("valid", "test", "pr_auc"): (0.04, 0.06),
    ("train", "test", "f2_beta"): (0.07, 0.10),
    ("valid", "test", "brier"): (0.02, 0.03),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate train/valid/test metric gaps for overfitting risk.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation report JSON.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be object.")
    return payload


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            parsed = float(token)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def get_nested_threshold(
    policy: Optional[Dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    default_warn: float,
    default_fail: float,
) -> Tuple[float, float]:
    warn = default_warn
    fail = default_fail
    if isinstance(policy, dict):
        gap = policy.get("gap_thresholds")
        if isinstance(gap, dict):
            pair_key = f"{left}_{right}"
            pair_block = gap.get(pair_key)
            if pair_block is None:
                pair_block = gap.get(f"{left}-{right}")
            if pair_block is None:
                pair_block = gap.get(left + "_" + right)
            if pair_block is None:
                pair_block = gap.get(left + right)
            if isinstance(pair_block, dict):
                metric_block = pair_block.get(metric)
                if isinstance(metric_block, dict):
                    raw_warn = to_float(metric_block.get("warn"))
                    raw_fail = to_float(metric_block.get("fail"))
                    if raw_warn is not None:
                        warn = raw_warn
                    if raw_fail is not None:
                        fail = raw_fail
            # Flat-key fallback support.
            flat_warn = to_float(gap.get(f"{left}_{right}_{metric}_warn"))
            flat_fail = to_float(gap.get(f"{left}_{right}_{metric}_fail"))
            if flat_warn is not None:
                warn = flat_warn
            if flat_fail is not None:
                fail = flat_fail
    return warn, fail


def read_metric(split_metrics: Dict[str, Any], split: str, metric: str) -> Optional[float]:
    block = split_metrics.get(split)
    if not isinstance(block, dict):
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return to_float(metrics.get(metric))


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    try:
        evaluation = load_json(args.evaluation_report)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            "Unable to parse evaluation report JSON.",
            {"path": str(Path(args.evaluation_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    policy: Optional[Dict[str, Any]] = None
    if args.performance_policy:
        try:
            policy = load_json(args.performance_policy)
        except Exception as exc:
            add_issue(
                failures,
                "invalid_performance_policy",
                "Unable to parse performance policy JSON.",
                {"path": str(Path(args.performance_policy).expanduser()), "error": str(exc)},
            )
            return finish(args, failures, warnings, {})

    split_metrics = evaluation.get("split_metrics")
    if not isinstance(split_metrics, dict):
        add_issue(
            failures,
            "missing_split_metrics",
            "evaluation_report must provide split_metrics.train/valid/test for gap analysis.",
            {"migration_hint": "Upgrade evaluation report schema with split_metrics blocks."},
        )
        return finish(args, failures, warnings, {})

    for required in ("train", "valid", "test"):
        if not isinstance(split_metrics.get(required), dict):
            add_issue(
                failures,
                "missing_split_metrics",
                "Missing split block required for gap analysis.",
                {"split": required},
            )

    gap_rows: List[Dict[str, Any]] = []
    for (left, right, metric), (default_warn, default_fail) in DEFAULT_THRESHOLDS.items():
        left_value = read_metric(split_metrics, left, metric)
        right_value = read_metric(split_metrics, right, metric)
        if left_value is None or right_value is None:
            add_issue(
                failures,
                "missing_required_metric",
                "Gap analysis requires metric in both compared splits.",
                {"left_split": left, "right_split": right, "metric": metric},
            )
            continue

        warn_th, fail_th = get_nested_threshold(policy, left, right, metric, default_warn, default_fail)
        if warn_th < 0.0 or fail_th < 0.0 or warn_th > fail_th:
            add_issue(
                failures,
                "invalid_gap_threshold",
                "Gap thresholds must satisfy 0 <= warn <= fail.",
                {
                    "left_split": left,
                    "right_split": right,
                    "metric": metric,
                    "warn": warn_th,
                    "fail": fail_th,
                },
            )
            continue

        # Directional bad gap:
        # - For performance metrics (pr_auc, f2_beta): earlier split minus later split.
        # - For Brier (lower is better): later split minus earlier split.
        if metric == "brier":
            gap_value = right_value - left_value
        else:
            gap_value = left_value - right_value

        row = {
            "left_split": left,
            "right_split": right,
            "metric": metric,
            "left_value": left_value,
            "right_value": right_value,
            "directional_gap": gap_value,
            "warn_threshold": warn_th,
            "fail_threshold": fail_th,
        }
        gap_rows.append(row)

        if gap_value > fail_th:
            add_issue(
                failures,
                "overfit_gap_exceeds_threshold",
                "Generalization gap exceeds fail threshold.",
                row,
            )
        elif gap_value > warn_th:
            add_issue(
                warnings,
                "overfit_gap_warning",
                "Generalization gap exceeds warning threshold.",
                row,
            )

    summary = {
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
        "performance_policy": str(Path(args.performance_policy).expanduser().resolve()) if args.performance_policy else None,
        "gaps": gap_rows,
    }
    return finish(args, failures, warnings, summary)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
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
