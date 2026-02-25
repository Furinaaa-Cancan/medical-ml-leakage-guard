#!/usr/bin/env python3
"""
Fail-closed clinical metrics gate for medical binary prediction reports.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "ppv",
    "npv",
    "sensitivity",
    "specificity",
    "f1",
    "f2_beta",
    "roc_auc",
    "pr_auc",
    "brier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clinical binary-classification metrics for all splits.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation report JSON.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance for metric consistency checks.")
    parser.add_argument("--report", help="Optional output report JSON path.")
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


def canonical_metric_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


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


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def safe_ratio(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def compute_confusion_metrics(tp: int, fp: int, tn: int, fn: int, beta: float) -> Dict[str, Optional[float]]:
    total = tp + fp + tn + fn
    precision = safe_ratio(float(tp), float(tp + fp))
    recall = safe_ratio(float(tp), float(tp + fn))
    specificity = safe_ratio(float(tn), float(tn + fp))
    npv = safe_ratio(float(tn), float(tn + fn))
    accuracy = safe_ratio(float(tp + tn), float(total))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    f2 = None
    beta_sq = beta * beta
    if precision is not None and recall is not None and ((beta_sq * precision) + recall) > 0:
        f2 = ((1.0 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "f2_beta": f2,
    }


def metric_in_unit_range(metric_name: str) -> bool:
    # All required metrics here are bounded to [0,1] for binary classification.
    return canonical_metric_token(metric_name) in {
        canonical_metric_token(name)
        for name in DEFAULT_REQUIRED_METRICS
    }


def get_required_metrics(policy: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(policy, dict):
        return list(DEFAULT_REQUIRED_METRICS)
    raw = policy.get("required_metrics")
    if not isinstance(raw, list):
        return list(DEFAULT_REQUIRED_METRICS)
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out if out else list(DEFAULT_REQUIRED_METRICS)


def parse_beta(policy: Optional[Dict[str, Any]]) -> float:
    if not isinstance(policy, dict):
        return 2.0
    value = to_float(policy.get("beta"))
    if value is None or value <= 0.0:
        return 2.0
    return float(value)


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not isinstance(args.tolerance, (int, float)) or not math.isfinite(float(args.tolerance)) or float(args.tolerance) < 0:
        add_issue(
            failures,
            "invalid_tolerance",
            "tolerance must be finite and >= 0.",
            {"tolerance": args.tolerance},
        )
        return finish(args, failures, warnings, {})

    try:
        payload = load_json(args.evaluation_report)
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

    required_metrics = get_required_metrics(policy)
    beta = parse_beta(policy)
    allowed_threshold_splits = {"valid", "cv_inner", "nested_cv"}

    split_metrics = payload.get("split_metrics")
    if not isinstance(split_metrics, dict):
        add_issue(
            failures,
            "missing_split_metrics",
            "evaluation_report must include split_metrics.train/valid/test.",
            {"migration_hint": "Upgrade evaluation_report schema with split_metrics and confusion_matrix blocks."},
        )
        return finish(args, failures, warnings, {"required_metrics": required_metrics, "beta": beta})

    threshold_selection = payload.get("threshold_selection")
    if not isinstance(threshold_selection, dict):
        add_issue(
            failures,
            "missing_threshold_selection",
            "evaluation_report must include threshold_selection with selection_split.",
            {"migration_hint": "Add threshold_selection.selection_split and selected_threshold."},
        )
    else:
        selection_split = str(threshold_selection.get("selection_split", "")).strip().lower()
        if not selection_split:
            add_issue(
                failures,
                "missing_threshold_selection_split",
                "threshold_selection.selection_split is required.",
                {},
            )
        elif selection_split not in allowed_threshold_splits:
            add_issue(
                failures,
                "test_split_used_for_threshold_selection",
                "threshold_selection.selection_split must be valid/cv_inner/nested_cv (never train/test).",
                {"selection_split": selection_split, "allowed": sorted(allowed_threshold_splits)},
            )

    split_summary: Dict[str, Any] = {}
    for split_name in ("train", "valid", "test"):
        split_block = split_metrics.get(split_name)
        if not isinstance(split_block, dict):
            add_issue(
                failures,
                "missing_split_metrics",
                "Missing split_metrics entry for required split.",
                {"split": split_name},
            )
            continue

        metrics = split_block.get("metrics")
        confusion = split_block.get("confusion_matrix")
        if not isinstance(metrics, dict):
            add_issue(
                failures,
                "missing_split_metric_block",
                "split_metrics.<split>.metrics must be an object.",
                {"split": split_name},
            )
            continue
        if not isinstance(confusion, dict):
            add_issue(
                failures,
                "missing_confusion_matrix",
                "split_metrics.<split>.confusion_matrix must be an object.",
                {"split": split_name},
            )
            continue

        cm: Dict[str, int] = {}
        for key in ("tp", "fp", "tn", "fn"):
            value = to_int(confusion.get(key))
            if value is None or value < 0:
                add_issue(
                    failures,
                    "invalid_confusion_matrix",
                    "confusion_matrix values must be non-negative integers.",
                    {"split": split_name, "field": key, "value": confusion.get(key)},
                )
            else:
                cm[key] = value
        if len(cm) != 4:
            continue

        total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
        if total <= 0:
            add_issue(
                failures,
                "invalid_confusion_matrix",
                "confusion_matrix total must be > 0.",
                {"split": split_name, "confusion_matrix": cm},
            )
            continue

        derived = compute_confusion_metrics(cm["tp"], cm["fp"], cm["tn"], cm["fn"], beta=beta)

        normalized_required = {canonical_metric_token(m): m for m in required_metrics}
        for metric_token, metric_name in normalized_required.items():
            raw_value = metrics.get(metric_name)
            if raw_value is None:
                # Allow canonical aliases in report, e.g., "recall" for sensitivity.
                alias_keys = []
                if metric_token == canonical_metric_token("sensitivity"):
                    alias_keys = ["recall"]
                if metric_token == canonical_metric_token("f2_beta"):
                    alias_keys = ["f2", "fbeta"]
                if metric_token == canonical_metric_token("precision"):
                    alias_keys = ["ppv"]
                for alias in alias_keys:
                    if alias in metrics:
                        raw_value = metrics.get(alias)
                        break
            numeric = to_float(raw_value)
            if numeric is None:
                add_issue(
                    failures,
                    "missing_required_metric",
                    "Required metric missing or non-numeric.",
                    {"split": split_name, "metric": metric_name},
                )
                continue
            if metric_in_unit_range(metric_name) and not (0.0 <= numeric <= 1.0):
                add_issue(
                    failures,
                    "metric_out_of_range",
                    "Metric must be within [0,1].",
                    {"split": split_name, "metric": metric_name, "value": numeric},
                )

            # Derived-formula checks for threshold-dependent metrics.
            if metric_token in {
                canonical_metric_token("accuracy"),
                canonical_metric_token("precision"),
                canonical_metric_token("ppv"),
                canonical_metric_token("npv"),
                canonical_metric_token("sensitivity"),
                canonical_metric_token("specificity"),
                canonical_metric_token("f1"),
                canonical_metric_token("f2_beta"),
            }:
                derived_key = {
                    canonical_metric_token("accuracy"): "accuracy",
                    canonical_metric_token("precision"): "precision",
                    canonical_metric_token("ppv"): "ppv",
                    canonical_metric_token("npv"): "npv",
                    canonical_metric_token("sensitivity"): "sensitivity",
                    canonical_metric_token("specificity"): "specificity",
                    canonical_metric_token("f1"): "f1",
                    canonical_metric_token("f2_beta"): "f2_beta",
                }[metric_token]
                derived_value = derived.get(derived_key)
                if derived_value is None:
                    add_issue(
                        failures,
                        "metric_formula_uncomputable",
                        "Unable to derive metric from confusion matrix due to zero denominator.",
                        {"split": split_name, "metric": metric_name, "confusion_matrix": cm},
                    )
                elif abs(float(derived_value) - float(numeric)) > float(args.tolerance):
                    add_issue(
                        failures,
                        "metric_formula_mismatch",
                        "Metric value does not match confusion-matrix-derived value.",
                        {
                            "split": split_name,
                            "metric": metric_name,
                            "reported": numeric,
                            "derived": derived_value,
                            "tolerance": float(args.tolerance),
                        },
                    )

        precision = to_float(metrics.get("precision"))
        ppv = to_float(metrics.get("ppv"))
        if precision is None or ppv is None:
            add_issue(
                failures,
                "missing_required_metric",
                "Both precision and ppv must be present.",
                {"split": split_name},
            )
        elif abs(precision - ppv) > float(args.tolerance):
            add_issue(
                failures,
                "metric_formula_mismatch",
                "precision must equal ppv for binary classification.",
                {"split": split_name, "precision": precision, "ppv": ppv, "tolerance": float(args.tolerance)},
            )

        split_summary[split_name] = {
            "confusion_matrix": cm,
            "metrics_present": sorted([k for k, v in metrics.items() if to_float(v) is not None]),
        }

    summary = {
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
        "performance_policy": str(Path(args.performance_policy).expanduser().resolve()) if args.performance_policy else None,
        "required_metrics": required_metrics,
        "beta": beta,
        "splits": split_summary,
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
