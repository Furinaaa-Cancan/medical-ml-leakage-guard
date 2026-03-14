#!/usr/bin/env python3
"""
Fail-closed sample size adequacy gate for publication-grade medical prediction.

Validates that the dataset has sufficient sample size relative to the number
of predictors, following Riley et al. (2019, 2025) EPV criteria and
FDA/MHRA/Health Canada Good ML Practice principles.

References:
- Riley et al., BMJ 2019: Minimum sample size for binary prediction models
- Riley et al., Lancet Digital Health 2025: Sample size for AI prediction
- Tsegaye et al., J Clin Epidemiol 2025: ML models in oncology need larger samples
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_utils import add_issue, load_json_from_str as load_json_object
from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)


register_remediations({
    "missing_evaluation_report": (
        "Provide --evaluation-report pointing to evaluation_report.json."
    ),
    "invalid_evaluation_report": (
        "Fix JSON syntax in evaluation_report.json."
    ),
    "missing_sample_size_info": (
        "evaluation_report.json must contain 'sample_size_adequacy' or "
        "sufficient metadata (n_train, n_features) for EPV computation."
    ),
    "epv_below_minimum": (
        "Events per variable (EPV) is below the minimum threshold. "
        "Options: 1) Collect more data, 2) Reduce predictors via feature selection, "
        "3) Use penalized regression to reduce effective parameters. "
        "Reference: Riley et al. 2019 recommends EPV >= 10 for binary outcomes."
    ),
    "epv_below_recommended": (
        "EPV is below the recommended level for robust ML models. "
        "Consider increasing sample size or reducing feature count. "
        "Reference: Tsegaye et al. 2025 showed ML models need even larger samples."
    ),
    "total_sample_too_small": (
        "Total sample size is below the minimum for reliable prediction modeling. "
        "At minimum, need 100+ events and 100+ non-events across all splits."
    ),
    "events_too_few": (
        "Too few positive events for reliable model development. "
        "Consider: 1) Broader case definition, 2) Longer observation window, "
        "3) Multi-center data collection."
    ),
    "test_set_events_too_few": (
        "Test set has insufficient events for reliable performance estimation. "
        "Bootstrap CI widths will be unreliable with <50 events in test."
    ),
    "shrinkage_factor_low": (
        "Estimated shrinkage factor is below 0.9, indicating >10% expected "
        "overfitting. Consider: penalized regression, reduced predictor set, "
        "or larger sample. Reference: Riley et al. 2019."
    ),
})


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "epv_minimum": 10.0,
    "epv_recommended": 20.0,
    "min_total_events": 100,
    "min_total_non_events": 100,
    "min_test_events": 50,
    "min_train_samples": 200,
    "shrinkage_factor_target": 0.90,
}


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _estimate_shrinkage(n_events: int, n_features: int) -> Optional[float]:
    """Approximate shrinkage factor using Van Houwelingen formula.

    S ≈ (E - p) / E where E = events, p = parameters.
    This is a rough approximation; Riley et al. provides more precise criteria.
    """
    if n_events <= 0 or n_features <= 0:
        return None
    if n_events <= n_features:
        return 0.0
    return (n_events - n_features) / n_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate sample size adequacy for medical prediction model."
    )
    parser.add_argument(
        "--evaluation-report",
        required=True,
        help="Path to evaluation_report.json.",
    )
    parser.add_argument(
        "--report",
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings.",
    )
    parser.add_argument(
        "--epv-minimum",
        type=float,
        default=None,
        help="Override minimum EPV threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.epv_minimum is not None:
        thresholds["epv_minimum"] = args.epv_minimum

    # Load evaluation report
    eval_path = Path(args.evaluation_report)
    if not eval_path.is_file():
        add_issue(
            failures,
            "missing_evaluation_report",
            f"Evaluation report not found: {eval_path}",
            {"path": str(eval_path)},
        )
        return _finish(args, failures, warnings, info, thresholds, None)

    try:
        eval_report = load_json_object(eval_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            f"Cannot parse evaluation report: {exc}",
            {"path": str(eval_path)},
        )
        return _finish(args, failures, warnings, info, thresholds, None)

    # Extract sample size info
    ssa = eval_report.get("sample_size_adequacy", {})
    metadata = eval_report.get("metadata", {})
    split_summary = eval_report.get("split_summary", {})

    # Try to get key numbers
    n_events = _to_float(ssa.get("n_events"))
    n_non_events = _to_float(ssa.get("n_non_events"))
    n_features = _to_float(ssa.get("n_features"))
    n_total = _to_float(ssa.get("n_total"))
    epv = _to_float(ssa.get("events_per_variable"))

    # Fallback extraction from metadata
    if n_features is None:
        n_features = _to_float(metadata.get("n_features"))
    if n_total is None:
        n_total = _to_float(metadata.get("n_train"))

    # Extract test set info
    test_info = split_summary.get("test", {})
    test_events = _to_float(test_info.get("n_positive", test_info.get("positive_count")))

    if n_events is None or n_features is None:
        # Try to compute from split summaries
        train_info = split_summary.get("train", {})
        if train_info:
            n_events = _to_float(train_info.get("n_positive", train_info.get("positive_count")))
            n_non_events = _to_float(train_info.get("n_negative", train_info.get("negative_count")))
            if n_total is None:
                n_total = _to_float(train_info.get("n", train_info.get("total")))

    if n_events is None or n_features is None:
        add_issue(
            failures,
            "missing_sample_size_info",
            "Cannot determine n_events or n_features from evaluation report.",
            {"found_keys": list(ssa.keys()) + list(metadata.keys())},
        )
        return _finish(args, failures, warnings, info, thresholds, eval_report)

    n_events_int = int(n_events)
    n_features_int = int(n_features)

    # Compute EPV if not provided
    if epv is None and n_features_int > 0:
        epv = n_events / n_features
    elif n_features_int == 0:
        epv = float("inf")

    # Check EPV
    if epv is not None and epv < thresholds["epv_minimum"]:
        add_issue(
            failures,
            "epv_below_minimum",
            f"EPV = {epv:.1f} is below minimum {thresholds['epv_minimum']:.0f}. "
            f"({n_events_int} events / {n_features_int} features)",
            {
                "epv": epv,
                "n_events": n_events_int,
                "n_features": n_features_int,
                "threshold": thresholds["epv_minimum"],
            },
        )
    elif epv is not None and epv < thresholds["epv_recommended"]:
        add_issue(
            warnings,
            "epv_below_recommended",
            f"EPV = {epv:.1f} is below recommended {thresholds['epv_recommended']:.0f}. "
            f"ML models may need even higher EPV (Tsegaye et al. 2025).",
            {
                "epv": epv,
                "n_events": n_events_int,
                "n_features": n_features_int,
                "threshold": thresholds["epv_recommended"],
            },
        )

    # Check total events
    if n_events < thresholds["min_total_events"]:
        add_issue(
            warnings if n_events >= 50 else failures,
            "events_too_few",
            f"Only {n_events_int} events total (minimum: {int(thresholds['min_total_events'])}).",
            {"n_events": n_events_int, "threshold": thresholds["min_total_events"]},
        )

    # Check non-events
    if n_non_events is not None and n_non_events < thresholds["min_total_non_events"]:
        add_issue(
            warnings,
            "events_too_few",
            f"Only {int(n_non_events)} non-events total.",
            {"n_non_events": int(n_non_events)},
        )

    # Check test events
    if test_events is not None and test_events < thresholds["min_test_events"]:
        add_issue(
            warnings,
            "test_set_events_too_few",
            f"Test set has {int(test_events)} events (recommend >= {int(thresholds['min_test_events'])}).",
            {"test_events": int(test_events), "threshold": thresholds["min_test_events"]},
        )

    # Shrinkage factor estimate
    shrinkage = _estimate_shrinkage(n_events_int, n_features_int)
    if shrinkage is not None and shrinkage < thresholds["shrinkage_factor_target"]:
        add_issue(
            warnings,
            "shrinkage_factor_low",
            f"Estimated shrinkage factor = {shrinkage:.3f} "
            f"(target >= {thresholds['shrinkage_factor_target']:.2f}).",
            {
                "shrinkage": shrinkage,
                "target": thresholds["shrinkage_factor_target"],
                "n_events": n_events_int,
                "n_features": n_features_int,
            },
        )

    # Summary
    summary = {
        "n_events": n_events_int,
        "n_non_events": int(n_non_events) if n_non_events else None,
        "n_features": n_features_int,
        "n_total": int(n_total) if n_total else None,
        "events_per_variable": round(epv, 2) if epv is not None else None,
        "estimated_shrinkage_factor": (
            round(shrinkage, 4) if shrinkage is not None else None
        ),
        "test_events": int(test_events) if test_events else None,
        "adequacy_verdict": (
            "adequate"
            if (epv is not None and epv >= thresholds["epv_recommended"])
            else "marginal"
            if (epv is not None and epv >= thresholds["epv_minimum"])
            else "insufficient"
        ),
    }

    add_issue(
        info,
        "sample_size_summary",
        f"EPV={epv:.1f}, events={n_events_int}, features={n_features_int}, "
        f"shrinkage={shrinkage:.3f}" if shrinkage else
        f"EPV={epv:.1f}, events={n_events_int}, features={n_features_int}",
        summary,
    )

    return _finish(args, failures, warnings, info, thresholds, eval_report, summary)


def _finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    info: List[Dict[str, Any]],
    thresholds: Dict[str, float],
    eval_report: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"

    report = build_report_envelope(
        gate_name="sample_size_gate",
        status=status,
        failures=failures,
        warnings=warnings,
        info=info,
        extra={
            "thresholds": thresholds,
            "summary": summary or {},
        },
    )

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json
        out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print_gate_summary("Sample Size Adequacy Gate", status, failures, warnings)
    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
