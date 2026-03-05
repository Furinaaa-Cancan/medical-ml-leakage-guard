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

from _gate_utils import add_issue, load_json_from_str as load_json, to_float
from _gate_framework import (
    GateBase,
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)


DEFAULT_THRESHOLDS: Dict[Tuple[str, str, str], Tuple[float, float]] = {
    ("train", "valid", "pr_auc"): (0.05, 0.08),
    ("valid", "test", "pr_auc"): (0.04, 0.06),
    ("train", "test", "f2_beta"): (0.07, 0.10),
    ("valid", "test", "brier"): (0.02, 0.03),
}


# ---------------------------------------------------------------------------
# Gate-specific remediation hints
# ---------------------------------------------------------------------------

register_remediations({
    "invalid_evaluation_report": "Check that --evaluation-report points to a valid JSON file with the expected schema.",
    "invalid_performance_policy": "Check that --performance-policy points to a valid JSON file. This is optional; omit to use defaults.",
    "missing_split_metrics": "Ensure evaluation_report contains split_metrics.{train,valid,test} blocks with per-split metric dicts.",
    "missing_required_metric": "Both compared splits must contain the metric. Re-run evaluation to populate missing metrics.",
    "invalid_gap_threshold": "Gap thresholds must satisfy 0 <= warn <= fail. Fix the values in performance_policy.gap_thresholds.",
    "overfit_gap_exceeds_threshold": "The model shows excessive overfitting. Consider regularization, data augmentation, or simpler model architectures.",
    "overfit_gap_warning": "Moderate overfitting detected. Monitor this gap; it may worsen with distribution shift.",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate train/valid/test metric gaps for overfitting risk.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation report JSON.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()




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
    from _gate_utils import get_gate_elapsed, write_json as _write_report

    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"

    fi = [GateIssue.from_legacy(f, Severity.ERROR) for f in failures]
    wi = [GateIssue.from_legacy(w, Severity.WARNING) for w in warnings]
    for issue in fi + wi:
        if not issue.remediation:
            issue.remediation = get_remediation(issue.code)

    report = build_report_envelope(
        gate_name="generalization_gap_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary=summary,
        input_files={
            "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
            "performance_policy": str(Path(args.performance_policy).expanduser().resolve()) if args.performance_policy else None,
        },
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="generalization_gap_gate",
        status=status,
        failures=fi,
        warnings=wi,
        strict=bool(args.strict),
        elapsed=get_gate_elapsed(),
    )

    return 2 if should_fail else 0


if __name__ == "__main__":
    from _gate_utils import start_gate_timer
    start_gate_timer()
    raise SystemExit(main())
