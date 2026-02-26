#!/usr/bin/env python3
"""
Fail-closed seed stability gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "pr_auc_std_max": 0.03,
    "pr_auc_range_max": 0.08,
    "f2_beta_std_max": 0.05,
    "f2_beta_range_max": 0.12,
    "brier_std_max": 0.02,
    "brier_range_max": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multi-seed stability evidence for strict publication-grade workflow.")
    parser.add_argument("--seed-sensitivity-report", required=True, help="Path to seed_sensitivity_report.json.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON for threshold overrides.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def load_json_object(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object.")
    return payload


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_int_like(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def parse_thresholds(policy: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    if not isinstance(policy, dict):
        return out
    raw = policy.get("seed_stability_thresholds")
    if not isinstance(raw, dict):
        return out
    for key in DEFAULT_THRESHOLDS:
        value = raw.get(key)
        if is_finite_number(value) and float(value) > 0.0:
            out[key] = float(value)
    return out


def metric_bounds_ok(metric: str, value: float) -> bool:
    if metric in {"pr_auc", "f2_beta", "brier"}:
        return 0.0 <= value <= 1.0
    return math.isfinite(value)


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    report_path = Path(args.seed_sensitivity_report).expanduser().resolve()
    if not report_path.exists():
        add_issue(
            failures,
            "missing_seed_sensitivity_report",
            "seed_sensitivity_report file not found.",
            {"path": str(report_path)},
        )
        return finish(args, failures, warnings, {}, None, dict(DEFAULT_THRESHOLDS))

    try:
        payload = load_json_object(str(report_path))
    except Exception as exc:
        add_issue(
            failures,
            "invalid_seed_sensitivity_report",
            "Unable to parse seed_sensitivity_report JSON.",
            {"path": str(report_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {}, None, dict(DEFAULT_THRESHOLDS))

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

    thresholds = parse_thresholds(policy_payload)

    primary_metric = str(payload.get("primary_metric", "")).strip().lower()
    if primary_metric != "pr_auc":
        add_issue(
            failures,
            "seed_stability_primary_metric_mismatch",
            "seed_sensitivity_report.primary_metric must be pr_auc in strict publication-grade mode.",
            {"primary_metric": payload.get("primary_metric")},
        )

    selection_data = str(payload.get("selection_data", "")).strip().lower()
    if "test" in selection_data:
        add_issue(
            failures,
            "seed_stability_selection_data_invalid",
            "selection_data must not reference test scope.",
            {"selection_data": selection_data},
        )

    threshold_selection_split = str(payload.get("threshold_selection_split", "")).strip().lower()
    if threshold_selection_split not in {"valid", "cv_inner", "nested_cv"}:
        add_issue(
            failures,
            "seed_stability_threshold_split_invalid",
            "threshold_selection_split must be valid/cv_inner/nested_cv.",
            {"threshold_selection_split": threshold_selection_split},
        )

    per_seed = payload.get("per_seed_results")
    if not isinstance(per_seed, list):
        add_issue(
            failures,
            "seed_stability_missing_per_seed_results",
            "per_seed_results must be a non-empty list.",
            {},
        )
        per_seed = []

    min_seed_count = 5 if args.strict else 3
    if len(per_seed) < min_seed_count:
        add_issue(
            failures,
            "insufficient_seed_runs",
            "Seed stability requires enough independent seed runs.",
            {"observed_seed_runs": len(per_seed), "required_min_seed_runs": min_seed_count},
        )

    seen_seeds: set[int] = set()
    per_metric_values: Dict[str, List[float]] = {"pr_auc": [], "f2_beta": [], "brier": []}
    for idx, row in enumerate(per_seed):
        if not isinstance(row, dict):
            add_issue(
                failures,
                "invalid_seed_result_entry",
                "Each per_seed_results entry must be an object.",
                {"index": idx, "actual_type": type(row).__name__},
            )
            continue
        seed = parse_int_like(row.get("seed"))
        if seed is None:
            add_issue(
                failures,
                "invalid_seed_result_entry",
                "per_seed_results entry must include integer seed.",
                {"index": idx, "seed": row.get("seed")},
            )
        else:
            if seed in seen_seeds:
                add_issue(
                    failures,
                    "duplicate_seed_result",
                    "seed values in per_seed_results must be unique.",
                    {"seed": seed},
                )
            seen_seeds.add(seed)

        metrics = row.get("test_metrics")
        if not isinstance(metrics, dict):
            add_issue(
                failures,
                "invalid_seed_result_entry",
                "per_seed_results entry must include test_metrics object.",
                {"index": idx},
            )
            continue
        for metric_name in ("pr_auc", "f2_beta", "brier"):
            value = metrics.get(metric_name)
            if not is_finite_number(value):
                add_issue(
                    failures,
                    "seed_metric_non_finite",
                    "Seed metric must be finite numeric value.",
                    {"index": idx, "metric": metric_name, "value": value},
                )
                continue
            value_f = float(value)
            if not metric_bounds_ok(metric_name, value_f):
                add_issue(
                    failures,
                    "seed_metric_out_of_range",
                    "Seed metric out of legal range.",
                    {"index": idx, "metric": metric_name, "value": value_f},
                )
                continue
            per_metric_values[metric_name].append(value_f)

    computed_summary: Dict[str, Dict[str, float]] = {}
    for metric_name, values in per_metric_values.items():
        if not values:
            add_issue(
                failures,
                "seed_metric_missing",
                "No valid values found for required seed stability metric.",
                {"metric": metric_name},
            )
            continue
        mean = float(statistics.mean(values))
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        minimum = float(min(values))
        maximum = float(max(values))
        computed_summary[metric_name] = {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "range": maximum - minimum,
            "n": float(len(values)),
        }

    declared_summary = payload.get("summary")
    if not isinstance(declared_summary, dict):
        add_issue(
            failures,
            "seed_summary_missing",
            "seed_sensitivity_report.summary must be an object.",
            {},
        )
        declared_summary = {}

    tol = 1e-9
    for metric_name, summary in computed_summary.items():
        declared_metric = declared_summary.get(metric_name)
        if not isinstance(declared_metric, dict):
            add_issue(
                failures,
                "seed_summary_missing_metric",
                "seed_sensitivity_report.summary missing metric block.",
                {"metric": metric_name},
            )
            continue
        for key in ("mean", "std", "min", "max", "range"):
            declared_value = declared_metric.get(key)
            if not is_finite_number(declared_value):
                add_issue(
                    failures,
                    "seed_summary_invalid_value",
                    "Declared summary value must be finite numeric.",
                    {"metric": metric_name, "field": key, "value": declared_value},
                )
                continue
            if abs(float(declared_value) - float(summary[key])) > tol:
                add_issue(
                    failures,
                    "seed_summary_mismatch",
                    "Declared summary does not match per-seed metric values.",
                    {
                        "metric": metric_name,
                        "field": key,
                        "declared": float(declared_value),
                        "computed": float(summary[key]),
                    },
                )

    def apply_threshold(metric: str, field: str, threshold_key: str) -> None:
        summary = computed_summary.get(metric)
        if not isinstance(summary, dict):
            return
        observed = float(summary[field])
        limit = float(thresholds[threshold_key])
        if observed > limit:
            add_issue(
                failures,
                "seed_stability_exceeds_threshold",
                "Seed stability metric exceeds fail threshold.",
                {
                    "metric": metric,
                    "field": field,
                    "observed": observed,
                    "threshold": limit,
                    "threshold_key": threshold_key,
                },
            )
        elif observed > (0.8 * limit):
            add_issue(
                warnings,
                "seed_stability_near_threshold",
                "Seed stability metric is close to fail threshold.",
                {
                    "metric": metric,
                    "field": field,
                    "observed": observed,
                    "threshold": limit,
                    "threshold_key": threshold_key,
                },
            )

    apply_threshold("pr_auc", "std", "pr_auc_std_max")
    apply_threshold("pr_auc", "range", "pr_auc_range_max")
    apply_threshold("f2_beta", "std", "f2_beta_std_max")
    apply_threshold("f2_beta", "range", "f2_beta_range_max")
    apply_threshold("brier", "std", "brier_std_max")
    apply_threshold("brier", "range", "brier_range_max")

    return finish(args, failures, warnings, payload, computed_summary, thresholds)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    source_payload: Dict[str, Any],
    computed_summary: Optional[Dict[str, Dict[str, float]]],
    thresholds: Dict[str, float],
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
            "seed_sensitivity_report": str(Path(args.seed_sensitivity_report).expanduser().resolve()),
            "primary_metric": source_payload.get("primary_metric"),
            "model_id": source_payload.get("model_id"),
            "n_seed_results": len(source_payload.get("per_seed_results", []))
            if isinstance(source_payload.get("per_seed_results"), list)
            else 0,
            "computed_metrics": computed_summary or {},
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
