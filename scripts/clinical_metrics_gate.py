#!/usr/bin/env python3
"""
Fail-closed clinical metrics gate for medical binary prediction reports.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)
from _gate_utils import add_issue, canonical_metric_token as _shared_canonical_metric_token, load_json_from_str as load_json, to_float, to_int as _shared_to_int


register_remediations({
    "clinical_floor_violation": "Clinical metric is below the required minimum floor. Improve model or adjust operating point.",
    "clinical_metric_missing": "Required clinical metric is missing from evaluation report. Re-run evaluation.",
    "operating_point_missing": "Evaluation report must specify an operating point (threshold) for clinical metrics.",
})


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
    parser.add_argument("--external-validation-report", help="Optional external_validation_report JSON path.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numeric tolerance for metric consistency checks.")
    parser.add_argument("--report", help="Optional output report JSON path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()




def canonical_metric_token(value: str) -> str:
    return _shared_canonical_metric_token(value)




def to_int(value: Any) -> Optional[int]:
    return _shared_to_int(value)


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
    # Publication-grade base: always enforce the full default clinical panel.
    required = list(DEFAULT_REQUIRED_METRICS)
    seen_tokens = {canonical_metric_token(x) for x in required}
    if not isinstance(policy, dict):
        return required
    raw = policy.get("required_metrics")
    if not isinstance(raw, list):
        return required
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            continue
        token = canonical_metric_token(item.strip())
        if token and token not in seen_tokens:
            required.append(item.strip())
            seen_tokens.add(token)
    return required


def parse_beta(policy: Optional[Dict[str, Any]]) -> float:
    if not isinstance(policy, dict):
        return 2.0
    value = to_float(policy.get("beta"))
    if value is None or value <= 0.0:
        return 2.0
    return float(value)


def parse_clinical_floors(policy: Optional[Dict[str, Any]]) -> Dict[str, float]:
    floors = {
        "sensitivity_min": 0.85,
        "npv_min": 0.90,
        "specificity_min": 0.40,
        "ppv_min": 0.55,
    }
    if not isinstance(policy, dict):
        return floors
    clinical = policy.get("clinical_floors")
    if isinstance(clinical, dict):
        for key in floors:
            value = to_float(clinical.get(key))
            if value is not None and 0.0 <= value <= 1.0:
                floors[key] = float(value)
    threshold_policy = policy.get("threshold_policy")
    if isinstance(threshold_policy, dict):
        policy_clinical = threshold_policy.get("clinical_floors")
        if isinstance(policy_clinical, dict):
            for key in floors:
                value = to_float(policy_clinical.get(key))
                if value is not None and 0.0 <= value <= 1.0:
                    floors[key] = float(value)
    return floors


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
    clinical_floors = parse_clinical_floors(policy)
    allowed_threshold_splits = {"valid", "cv_inner", "nested_cv"}
    selection_split: Optional[str] = None
    selected_threshold: Optional[float] = None
    guard_constraints_satisfied: Optional[bool] = None

    if isinstance(policy, dict):
        policy_required = policy.get("required_metrics")
        if isinstance(policy_required, list):
            policy_tokens = {
                canonical_metric_token(x.strip())
                for x in policy_required
                if isinstance(x, str) and x.strip()
            }
            missing_mandatory = [
                metric for metric in DEFAULT_REQUIRED_METRICS if canonical_metric_token(metric) not in policy_tokens
            ]
            if missing_mandatory:
                add_issue(
                    failures,
                    "performance_policy_missing_required_metric",
                    "performance_policy.required_metrics must include the full mandatory clinical panel.",
                    {"missing_metrics": missing_mandatory, "mandatory_metrics": DEFAULT_REQUIRED_METRICS},
                )
        threshold_policy = policy.get("threshold_policy")
        if isinstance(threshold_policy, dict):
            policy_split = str(threshold_policy.get("selection_split", "")).strip().lower()
            if policy_split and policy_split not in allowed_threshold_splits:
                add_issue(
                    failures,
                    "invalid_performance_policy",
                    "performance_policy.threshold_policy.selection_split must be valid/cv_inner/nested_cv.",
                    {"selection_split": policy_split, "allowed": sorted(allowed_threshold_splits)},
                )
            elif policy_split:
                selection_split = policy_split

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
        else:
            reported_split = selection_split
            selection_split = selection_split
            if isinstance(policy, dict):
                threshold_policy = policy.get("threshold_policy")
                if isinstance(threshold_policy, dict):
                    policy_split = str(threshold_policy.get("selection_split", "")).strip().lower()
                    if policy_split and policy_split != reported_split:
                        add_issue(
                            failures,
                            "threshold_selection_policy_mismatch",
                            "evaluation_report threshold_selection.selection_split must match performance_policy.threshold_policy.selection_split.",
                            {"evaluation_selection_split": reported_split, "policy_selection_split": policy_split},
                        )

        threshold_value = to_float(threshold_selection.get("selected_threshold"))
        if threshold_value is None or not (0.0 <= threshold_value <= 1.0):
            add_issue(
                failures,
                "invalid_threshold_selection_value",
                "threshold_selection.selected_threshold must be numeric in [0,1].",
                {"selected_threshold": threshold_selection.get("selected_threshold")},
            )
        else:
            selected_threshold = threshold_value

        if selection_split == "cv_inner":
            guard_flag = threshold_selection.get("constraints_satisfied_guard_split")
            if not isinstance(guard_flag, bool):
                add_issue(
                    failures,
                    "threshold_guard_constraints_not_met",
                    "selection_split=cv_inner requires threshold_selection.constraints_satisfied_guard_split=true.",
                    {
                        "constraints_satisfied_guard_split": guard_flag,
                        "migration_hint": (
                            "Populate threshold_selection.constraints_satisfied_guard_split "
                            "and ensure guard split meets all clinical floors."
                        ),
                    },
                )
            elif not guard_flag:
                add_issue(
                    failures,
                    "threshold_guard_constraints_not_met",
                    "Guard split constraints are not satisfied for selection_split=cv_inner.",
                    {"constraints_satisfied_guard_split": bool(guard_flag)},
                )
            else:
                guard_constraints_satisfied = True

    split_summary: Dict[str, Any] = {}
    metadata = payload.get("metadata")
    metadata_fingerprints = metadata.get("data_fingerprints") if isinstance(metadata, dict) else None
    top_level_metrics = payload.get("metrics")
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

        if isinstance(split_block.get("n_samples"), (int, float)) and not isinstance(split_block.get("n_samples"), bool):
            n_samples = to_int(split_block.get("n_samples"))
            if n_samples is None or n_samples <= 0:
                add_issue(
                    failures,
                    "invalid_split_sample_count",
                    "split_metrics.<split>.n_samples must be a positive integer when provided.",
                    {"split": split_name, "n_samples": split_block.get("n_samples")},
                )
            elif n_samples != total:
                add_issue(
                    failures,
                    "confusion_matrix_row_count_mismatch",
                    "confusion_matrix total must equal split_metrics.<split>.n_samples.",
                    {"split": split_name, "confusion_total": total, "n_samples": n_samples},
                )

        if isinstance(metadata_fingerprints, dict):
            split_fp = metadata_fingerprints.get(split_name)
            if isinstance(split_fp, dict):
                row_count = to_int(split_fp.get("row_count"))
                if row_count is not None and row_count != total:
                    add_issue(
                        failures,
                        "confusion_matrix_row_count_mismatch",
                        "confusion_matrix total must match metadata.data_fingerprints split row_count.",
                        {"split": split_name, "confusion_total": total, "fingerprint_row_count": row_count},
                    )

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

    if not isinstance(top_level_metrics, dict):
        add_issue(
            failures,
            "missing_top_level_metrics",
            "evaluation_report.metrics must be an object and match split_metrics.test.metrics.",
            {},
        )
    else:
        test_block = split_metrics.get("test")
        test_metrics = test_block.get("metrics") if isinstance(test_block, dict) else None
        if isinstance(test_metrics, dict):
            for metric_name in DEFAULT_REQUIRED_METRICS:
                test_value = to_float(test_metrics.get(metric_name))
                top_value = to_float(top_level_metrics.get(metric_name))
                if test_value is None or top_value is None:
                    add_issue(
                        failures,
                        "missing_top_level_metrics",
                        "Both top-level and test-split metrics must contain required numeric metric.",
                        {
                            "metric": metric_name,
                            "test_value": test_metrics.get(metric_name),
                            "top_level_value": top_level_metrics.get(metric_name),
                        },
                    )
                    continue
                if abs(float(test_value) - float(top_value)) > float(args.tolerance):
                    add_issue(
                        failures,
                        "top_level_test_metric_mismatch",
                        "evaluation_report.metrics must equal split_metrics.test.metrics for required metrics.",
                        {
                            "metric": metric_name,
                            "top_level_value": float(top_value),
                            "test_split_value": float(test_value),
                            "tolerance": float(args.tolerance),
                        },
                    )

    if isinstance(threshold_selection, dict) and selection_split == "valid":
        valid_block = split_metrics.get("valid")
        valid_metrics = valid_block.get("metrics") if isinstance(valid_block, dict) else None
        valid_confusion = valid_block.get("confusion_matrix") if isinstance(valid_block, dict) else None
        selected_metrics_valid = threshold_selection.get("selected_metrics_on_valid")
        if not isinstance(selected_metrics_valid, dict):
            selected_metrics_valid = threshold_selection.get("selected_metrics_on_selection_split")
        selected_confusion_valid = threshold_selection.get("selected_confusion_on_valid")
        if not isinstance(selected_confusion_valid, dict):
            selected_confusion_valid = threshold_selection.get("selected_confusion_on_selection_split")

        if not isinstance(selected_metrics_valid, dict):
            add_issue(
                failures,
                "missing_threshold_selection_snapshot",
                "threshold_selection must include selected_metrics_on_valid for selection_split=valid.",
                {},
            )
        if not isinstance(selected_confusion_valid, dict):
            add_issue(
                failures,
                "missing_threshold_selection_snapshot",
                "threshold_selection must include selected_confusion_on_valid for selection_split=valid.",
                {},
            )

        if isinstance(valid_metrics, dict) and isinstance(selected_metrics_valid, dict):
            for metric_name in DEFAULT_REQUIRED_METRICS:
                valid_value = to_float(valid_metrics.get(metric_name))
                selected_value = to_float(selected_metrics_valid.get(metric_name))
                if valid_value is None or selected_value is None:
                    continue
                if abs(float(valid_value) - float(selected_value)) > float(args.tolerance):
                    add_issue(
                        failures,
                        "threshold_selection_metric_mismatch",
                        "threshold_selection selected_metrics_on_valid must match split_metrics.valid.metrics.",
                        {
                            "metric": metric_name,
                            "threshold_selection_value": selected_value,
                            "valid_split_value": valid_value,
                            "tolerance": float(args.tolerance),
                        },
                    )

        if isinstance(valid_confusion, dict) and isinstance(selected_confusion_valid, dict):
            for key in ("tp", "fp", "tn", "fn"):
                expected = to_int(valid_confusion.get(key))
                observed = to_int(selected_confusion_valid.get(key))
                if expected is None or observed is None:
                    continue
                if expected != observed:
                    add_issue(
                        failures,
                        "threshold_selection_confusion_mismatch",
                        "threshold_selection selected_confusion_on_valid must match split_metrics.valid.confusion_matrix.",
                        {"field": key, "threshold_selection_value": observed, "valid_split_value": expected},
                    )

    test_metrics_for_floor = (
        split_metrics.get("test", {}).get("metrics")
        if isinstance(split_metrics.get("test"), dict)
        else None
    )
    if isinstance(test_metrics_for_floor, dict):
        floor_codes = {
            "sensitivity_min": "clinical_floor_sensitivity_not_met",
            "npv_min": "clinical_floor_npv_not_met",
            "specificity_min": "clinical_floor_specificity_not_met",
            "ppv_min": "clinical_floor_ppv_not_met",
        }
        metric_names = {
            "sensitivity_min": "sensitivity",
            "npv_min": "npv",
            "specificity_min": "specificity",
            "ppv_min": "ppv",
        }
        for floor_key, floor_value in clinical_floors.items():
            metric_name = metric_names[floor_key]
            metric_value = to_float(test_metrics_for_floor.get(metric_name))
            if metric_value is None:
                continue
            if metric_value < float(floor_value):
                add_issue(
                    failures,
                    floor_codes[floor_key],
                    "Clinical floor not met on internal test split.",
                    {
                        "split": "test",
                        "metric": metric_name,
                        "value": float(metric_value),
                        "required_min": float(floor_value),
                    },
                )

    if args.external_validation_report:
        try:
            external_payload = load_json(args.external_validation_report)
        except Exception as exc:
            add_issue(
                failures,
                "missing_required_metric",
                "Unable to parse external_validation_report for external clinical floor checks.",
                {"path": str(Path(args.external_validation_report).expanduser()), "error": str(exc)},
            )
            external_payload = {}
        cohorts = external_payload.get("cohorts") if isinstance(external_payload, dict) else None
        if isinstance(cohorts, list):
            floor_codes = {
                "sensitivity_min": "clinical_floor_sensitivity_not_met",
                "npv_min": "clinical_floor_npv_not_met",
                "specificity_min": "clinical_floor_specificity_not_met",
                "ppv_min": "clinical_floor_ppv_not_met",
            }
            metric_names = {
                "sensitivity_min": "sensitivity",
                "npv_min": "npv",
                "specificity_min": "specificity",
                "ppv_min": "ppv",
            }
            for cohort in cohorts:
                if not isinstance(cohort, dict):
                    continue
                cohort_id = str(cohort.get("cohort_id", "")).strip()
                metrics_ext = cohort.get("metrics")
                if not isinstance(metrics_ext, dict):
                    continue
                for floor_key, floor_value in clinical_floors.items():
                    metric_name = metric_names[floor_key]
                    metric_value = to_float(metrics_ext.get(metric_name))
                    if metric_value is None:
                        continue
                    if metric_value < float(floor_value):
                        add_issue(
                            failures,
                            floor_codes[floor_key],
                            "Clinical floor not met on external cohort.",
                            {
                                "split": f"external:{cohort_id}" if cohort_id else "external",
                                "metric": metric_name,
                                "value": float(metric_value),
                                "required_min": float(floor_value),
                            },
                        )

    summary = {
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
        "external_validation_report": str(Path(args.external_validation_report).expanduser().resolve())
        if args.external_validation_report
        else None,
        "performance_policy": str(Path(args.performance_policy).expanduser().resolve()) if args.performance_policy else None,
        "required_metrics": required_metrics,
        "beta": beta,
        "clinical_floors": clinical_floors,
        "threshold_selection_split": selection_split,
        "selected_threshold": selected_threshold,
        "threshold_guard_constraints_satisfied": guard_constraints_satisfied,
        "splits": split_summary,
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

    input_files = {
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
    }
    if getattr(args, "external_validation_report", None):
        input_files["external_validation_report"] = str(Path(args.external_validation_report).expanduser().resolve())
    if getattr(args, "performance_policy", None):
        input_files["performance_policy"] = str(Path(args.performance_policy).expanduser().resolve())

    report = build_report_envelope(
        gate_name="clinical_metrics_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary=summary,
        input_files=input_files,
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="clinical_metrics_gate",
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
