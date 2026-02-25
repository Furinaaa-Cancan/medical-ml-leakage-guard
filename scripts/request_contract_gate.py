#!/usr/bin/env python3
"""
Validate structured request contract for medical prediction leakage-safe workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REQUIRED_STRING_FIELDS = [
    "study_id",
    "run_id",
    "target_name",
    "prediction_unit",
    "index_time_col",
    "label_col",
    "patient_id_col",
    "primary_metric",
    "phenotype_definition_spec",
    "claim_tier_target",
]

ALLOWED_CLAIM_TIERS = {"leakage-audited", "publication-grade"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate structured request JSON for medical prediction workflow.")
    parser.add_argument("--request", required=True, help="Path to request JSON.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Enable strict contract requirements.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def must_be_non_empty_str(request: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[str]:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        add_issue(
            failures,
            "invalid_field",
            "Required field must be a non-empty string.",
            {"field": key},
        )
        return None
    return value.strip()


def validate_thresholds(
    request: Dict[str, Any], failures: List[Dict[str, Any]], warnings: List[Dict[str, Any]], strict: bool
) -> Dict[str, float]:
    thresholds = request.get("thresholds", {})
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, dict):
        add_issue(
            failures,
            "invalid_thresholds",
            "thresholds must be an object.",
            {"actual_type": type(thresholds).__name__},
        )
        return {}

    parsed: Dict[str, float] = {}
    for key in ("alpha", "min_delta", "min_baseline_delta", "ci_max_width"):
        if key in thresholds:
            value = thresholds[key]
            if is_finite_number(value):
                parsed[key] = float(value)
            else:
                add_issue(
                    failures,
                    "invalid_threshold_value",
                    "Threshold value must be a finite number.",
                    {"field": key, "actual_type": type(value).__name__},
                )

    if "ci_min_resamples" in thresholds:
        value = thresholds["ci_min_resamples"]
        if isinstance(value, bool):
            value = None
        if isinstance(value, int):
            parsed["ci_min_resamples"] = float(value)
        elif isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
            parsed["ci_min_resamples"] = float(int(value))
        else:
            add_issue(
                failures,
                "invalid_threshold_value",
                "Threshold value must be an integer.",
                {"field": "ci_min_resamples", "actual_type": type(thresholds["ci_min_resamples"]).__name__},
            )

    if "alpha" in parsed and not (0.0 < parsed["alpha"] <= 1.0):
        add_issue(
            failures,
            "invalid_threshold_alpha_range",
            "thresholds.alpha must be within (0, 1].",
            {"alpha": parsed["alpha"]},
        )
    if "min_delta" in parsed and parsed["min_delta"] < 0.0:
        add_issue(
            failures,
            "invalid_threshold_min_delta_range",
            "thresholds.min_delta must be >= 0.",
            {"min_delta": parsed["min_delta"]},
        )
    if "min_baseline_delta" in parsed and parsed["min_baseline_delta"] < 0.0:
        add_issue(
            failures,
            "invalid_threshold_min_baseline_delta_range",
            "thresholds.min_baseline_delta must be >= 0.",
            {"min_baseline_delta": parsed["min_baseline_delta"]},
        )
    if "ci_max_width" in parsed and parsed["ci_max_width"] <= 0.0:
        add_issue(
            failures,
            "invalid_threshold_ci_max_width_range",
            "thresholds.ci_max_width must be > 0.",
            {"ci_max_width": parsed["ci_max_width"]},
        )
    if "ci_min_resamples" in parsed and parsed["ci_min_resamples"] < 1.0:
        add_issue(
            failures,
            "invalid_threshold_ci_min_resamples_range",
            "thresholds.ci_min_resamples must be >= 1.",
            {"ci_min_resamples": parsed["ci_min_resamples"]},
        )

    if strict and "alpha" not in parsed:
        add_issue(
            warnings,
            "missing_threshold_alpha",
            "thresholds.alpha not provided; workflow will use default.",
            {"default": 0.01},
        )
    if strict and "min_delta" not in parsed:
        add_issue(
            warnings,
            "missing_threshold_min_delta",
            "thresholds.min_delta not provided; workflow will use default.",
            {"default": 0.03},
        )
    return parsed


def require_numeric(request: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[float]:
    value = request.get(key)
    if is_finite_number(value):
        return float(value)
    add_issue(
        failures,
        "invalid_numeric_field",
        "Required field must be a finite number.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def is_valid_dot_path(path: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", path))


def canonical_metric_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def validate_evaluation_report_shape(
    evaluation_report_path: str,
    failures: List[Dict[str, Any]],
) -> None:
    path = Path(evaluation_report_path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            "Unable to parse evaluation_report_file JSON.",
            {"path": str(path), "error": str(exc)},
        )
        return

    split_metrics = payload.get("split_metrics")
    if not isinstance(split_metrics, dict):
        add_issue(
            failures,
            "evaluation_report_missing_split_metrics",
            "evaluation_report_file must include split_metrics with train/valid/test entries.",
            {
                "path": str(path),
                "migration_hint": "Add split_metrics.{train,valid,test}.metrics and confusion_matrix blocks.",
            },
        )
    else:
        for split_name in ("train", "valid", "test"):
            block = split_metrics.get(split_name)
            if not isinstance(block, dict):
                add_issue(
                    failures,
                    "evaluation_report_missing_split_metrics",
                    "Required split missing from evaluation_report split_metrics.",
                    {"path": str(path), "split": split_name},
                )
                continue
            if not isinstance(block.get("metrics"), dict):
                add_issue(
                    failures,
                    "evaluation_report_missing_split_metrics",
                    "split_metrics.<split>.metrics must be an object.",
                    {"path": str(path), "split": split_name},
                )
            if not isinstance(block.get("confusion_matrix"), dict):
                add_issue(
                    failures,
                    "evaluation_report_missing_split_metrics",
                    "split_metrics.<split>.confusion_matrix must be an object.",
                    {"path": str(path), "split": split_name},
                )

    threshold_selection = payload.get("threshold_selection")
    if not isinstance(threshold_selection, dict):
        add_issue(
            failures,
            "evaluation_report_missing_threshold_selection",
            "evaluation_report_file must include threshold_selection metadata.",
            {
                "path": str(path),
                "migration_hint": "Add threshold_selection.selection_split and selected_threshold.",
            },
        )
    else:
        selection_split = threshold_selection.get("selection_split")
        if not isinstance(selection_split, str) or not selection_split.strip():
            add_issue(
                failures,
                "evaluation_report_missing_threshold_selection",
                "threshold_selection.selection_split must be a non-empty string.",
                {"path": str(path)},
            )
        else:
            token = selection_split.strip().lower()
            allowed = {"valid", "cv_inner", "nested_cv"}
            if token not in allowed:
                add_issue(
                    failures,
                    "threshold_selection_split_invalid",
                    "threshold_selection.selection_split must be valid/cv_inner/nested_cv (never train/test).",
                    {"path": str(path), "selection_split": selection_split, "allowed": sorted(allowed)},
                )


def validate_performance_policy_spec(
    policy_path: str,
    failures: List[Dict[str, Any]],
    expected_primary_metric: str,
) -> None:
    path = Path(policy_path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_performance_policy_spec",
            "Unable to parse performance_policy_spec JSON.",
            {"path": str(path), "error": str(exc)},
        )
        return

    required_keys = [
        "required_metrics",
        "primary_metric",
        "threshold_policy",
        "clinical_floors",
        "gap_thresholds",
        "beta",
    ]
    for key in required_keys:
        if key not in payload:
            add_issue(
                failures,
                "missing_performance_policy_field",
                "performance_policy_spec is missing required field.",
                {"path": str(path), "field": key},
            )

    primary_metric = payload.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.primary_metric must be a non-empty string.",
            {"path": str(path)},
        )
    else:
        if canonical_metric_token(primary_metric) != canonical_metric_token(expected_primary_metric):
            add_issue(
                failures,
                "performance_policy_metric_mismatch",
                "performance_policy_spec.primary_metric must match request primary_metric.",
                {
                    "path": str(path),
                    "policy_primary_metric": primary_metric,
                    "request_primary_metric": expected_primary_metric,
                },
            )

    required_metrics = payload.get("required_metrics")
    if not isinstance(required_metrics, list) or not required_metrics:
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.required_metrics must be a non-empty list.",
            {"path": str(path)},
        )

    beta = payload.get("beta")
    if not is_finite_number(beta) or float(beta) <= 0.0:
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.beta must be finite and > 0.",
            {"path": str(path), "beta": beta},
        )

    threshold_policy = payload.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.threshold_policy must be an object.",
            {"path": str(path)},
        )
    else:
        selection_split = threshold_policy.get("selection_split")
        allowed = {"valid", "cv_inner", "nested_cv"}
        if not isinstance(selection_split, str) or selection_split.strip().lower() not in allowed:
            add_issue(
                failures,
                "invalid_performance_policy_field",
                "threshold_policy.selection_split must be valid/cv_inner/nested_cv.",
                {"path": str(path), "selection_split": selection_split, "allowed": sorted(allowed)},
            )

    clinical_floors = payload.get("clinical_floors")
    if not isinstance(clinical_floors, dict):
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.clinical_floors must be an object.",
            {"path": str(path)},
        )
    else:
        for key in ("sensitivity_min", "npv_min"):
            value = clinical_floors.get(key)
            if not is_finite_number(value) or not (0.0 <= float(value) <= 1.0):
                add_issue(
                    failures,
                    "invalid_performance_policy_field",
                    "clinical floor must be finite within [0,1].",
                    {"path": str(path), "field": key, "value": value},
                )

    gap_thresholds = payload.get("gap_thresholds")
    if not isinstance(gap_thresholds, dict):
        add_issue(
            failures,
            "invalid_performance_policy_field",
            "performance_policy_spec.gap_thresholds must be an object.",
            {"path": str(path)},
        )


def validate_optional_path(
    request: Dict[str, Any],
    key: str,
    base: Path,
    failures: List[Dict[str, Any]],
    required: bool,
    normalized: Dict[str, Any],
) -> None:
    value = request.get(key)
    if value is None:
        if required:
            add_issue(
                failures,
                "missing_required_path",
                "Required path field is missing.",
                {"field": key},
            )
        return

    if not isinstance(value, str) or not value.strip():
        add_issue(
            failures,
            "invalid_path_field",
            "Path field must be a non-empty string.",
            {"field": key},
        )
        return

    resolved = resolve_path(base, value.strip())
    normalized[key] = str(resolved)
    if not resolved.exists():
        add_issue(
            failures,
            "path_not_found",
            "Path field points to a missing file.",
            {"field": key, "path": str(resolved)},
        )
    elif not resolved.is_file():
        add_issue(
            failures,
            "path_not_file",
            "Path field must point to a file.",
            {"field": key, "path": str(resolved)},
        )


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    normalized: Dict[str, Any] = {}

    request_path = Path(args.request).expanduser().resolve()
    if not request_path.exists():
        add_issue(
            failures,
            "missing_request_file",
            "Request JSON file not found.",
            {"path": str(request_path)},
        )
        return finish(args, failures, warnings, normalized)

    try:
        with request_path.open("r", encoding="utf-8") as fh:
            request = json.load(fh)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_request_json",
            "Unable to parse request JSON.",
            {"path": str(request_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, normalized)

    if not isinstance(request, dict):
        add_issue(
            failures,
            "invalid_request_json",
            "Request JSON root must be an object.",
            {"path": str(request_path)},
        )
        return finish(args, failures, warnings, normalized)

    request_base = request_path.parent
    normalized["path_resolution_base"] = str(request_base)

    for key in REQUIRED_STRING_FIELDS:
        value = must_be_non_empty_str(request, key, failures)
        if value is not None:
            normalized[key] = value

    claim_tier = normalized.get("claim_tier_target")
    if claim_tier and claim_tier not in ALLOWED_CLAIM_TIERS:
        add_issue(
            failures,
            "invalid_claim_tier_target",
            "claim_tier_target must be one of the allowed values.",
            {"allowed": sorted(ALLOWED_CLAIM_TIERS), "actual": claim_tier},
        )

    split_paths = request.get("split_paths")
    if not isinstance(split_paths, dict):
        add_issue(
            failures,
            "invalid_split_paths",
            "split_paths must be an object.",
            {},
        )
    else:
        normalized_splits: Dict[str, str] = {}
        for key in ("train", "test"):
            val = split_paths.get(key)
            if not isinstance(val, str) or not val.strip():
                add_issue(
                    failures,
                    "missing_split_path",
                    "Required split path missing.",
                    {"split": key},
                )
                continue
            resolved = resolve_path(request_base, val.strip())
            normalized_splits[key] = str(resolved)
            if not resolved.exists():
                add_issue(
                    failures,
                    "split_path_not_found",
                    "Split file path does not exist.",
                    {"split": key, "path": str(resolved)},
                )
            elif not resolved.is_file():
                add_issue(
                    failures,
                    "split_path_not_file",
                    "Split path must point to a file.",
                    {"split": key, "path": str(resolved)},
                )

        valid_val = split_paths.get("valid")
        if valid_val is not None:
            if not isinstance(valid_val, str) or not valid_val.strip():
                add_issue(
                    failures,
                    "invalid_split_path",
                    "valid split path must be a non-empty string when provided.",
                    {},
                )
            else:
                resolved = resolve_path(request_base, valid_val.strip())
                normalized_splits["valid"] = str(resolved)
                if not resolved.exists():
                    add_issue(
                        failures,
                        "split_path_not_found",
                        "Split file path does not exist.",
                        {"split": "valid", "path": str(resolved)},
                    )
                elif not resolved.is_file():
                    add_issue(
                        failures,
                        "split_path_not_file",
                        "Split path must point to a file.",
                        {"split": "valid", "path": str(resolved)},
                    )
        elif args.strict:
            add_issue(
                warnings,
                "missing_valid_split",
                "valid split is absent; strict workflows usually require train/valid/test.",
                {},
            )

        seen_paths: Dict[str, str] = {}
        for split_name, split_path in normalized_splits.items():
            prev_split = seen_paths.get(split_path)
            if prev_split is not None:
                add_issue(
                    failures,
                    "duplicate_split_path",
                    "Different splits must not point to the same file path.",
                    {"split_a": prev_split, "split_b": split_name, "path": split_path},
                )
            else:
                seen_paths[split_path] = split_name
        normalized["split_paths"] = normalized_splits

    phenotype_path = normalized.get("phenotype_definition_spec")
    if phenotype_path:
        resolved = resolve_path(request_base, phenotype_path)
        normalized["phenotype_definition_spec"] = str(resolved)
        if not resolved.exists():
            add_issue(
                failures,
                "phenotype_definition_spec_not_found",
                "phenotype_definition_spec path does not exist.",
                {"path": str(resolved)},
            )

    # Publication-grade requests must include lineage, split/imbalance/tuning protocol specs, and evaluated metric.
    require_lineage = normalized.get("claim_tier_target") == "publication-grade"

    if require_lineage:
        primary_metric = str(normalized.get("primary_metric", "")).strip()
        if canonical_metric_token(primary_metric) != canonical_metric_token("pr_auc"):
            add_issue(
                failures,
                "unsupported_primary_metric",
                "Publication-grade strict workflow requires primary_metric=pr_auc.",
                {"primary_metric": primary_metric, "expected": "pr_auc"},
            )

    validate_optional_path(
        request=request,
        key="feature_lineage_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="split_protocol_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="imbalance_policy_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="tuning_protocol_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="missingness_policy_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="reporting_bias_checklist_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="execution_attestation_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="performance_policy_spec",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="model_selection_report_file",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )

    validate_optional_path(
        request=request,
        key="evaluation_report_file",
        base=request_base,
        failures=failures,
        required=require_lineage,
        normalized=normalized,
    )
    evaluation_report_file = normalized.get("evaluation_report_file")
    if isinstance(evaluation_report_file, str) and evaluation_report_file:
        validate_evaluation_report_shape(evaluation_report_file, failures)
    performance_policy_spec = normalized.get("performance_policy_spec")
    if isinstance(performance_policy_spec, str) and performance_policy_spec:
        validate_performance_policy_spec(
            performance_policy_spec,
            failures=failures,
            expected_primary_metric=str(normalized.get("primary_metric", "")).strip(),
        )

    metric_path = request.get("evaluation_metric_path")
    if metric_path is None:
        if require_lineage:
            add_issue(
                failures,
                "missing_required_field",
                "Publication-grade request requires evaluation_metric_path to pin canonical metric source.",
                {"field": "evaluation_metric_path"},
            )
    elif isinstance(metric_path, str) and metric_path.strip():
        metric_path_clean = metric_path.strip()
        if not is_valid_dot_path(metric_path_clean):
            add_issue(
                failures,
                "invalid_field",
                "evaluation_metric_path must be a dot path using alphanumeric/underscore segments.",
                {"field": "evaluation_metric_path", "value": metric_path_clean},
            )
        else:
            metric_leaf = metric_path_clean.split(".")[-1]
            primary_metric = str(normalized.get("primary_metric", "")).strip()
            if primary_metric and canonical_metric_token(metric_leaf) != canonical_metric_token(primary_metric):
                add_issue(
                    failures,
                    "metric_path_metric_mismatch",
                    "evaluation_metric_path leaf must match primary_metric.",
                    {
                        "primary_metric": primary_metric,
                        "evaluation_metric_path": metric_path_clean,
                        "metric_leaf": metric_leaf,
                    },
                )
            normalized["evaluation_metric_path"] = metric_path_clean
    else:
        add_issue(
            failures,
            "invalid_field",
            "evaluation_metric_path must be a non-empty string when provided.",
            {"field": "evaluation_metric_path"},
        )

    if request.get("actual_primary_metric") is not None:
        actual_metric = request.get("actual_primary_metric")
        if is_finite_number(actual_metric):
            normalized["actual_primary_metric"] = float(actual_metric)
        else:
            add_issue(
                failures,
                "invalid_numeric_field",
                "actual_primary_metric must be a finite number when provided.",
                {"actual_type": type(actual_metric).__name__},
            )
    elif require_lineage:
        require_numeric(request, "actual_primary_metric", failures)

    null_required = normalized.get("claim_tier_target") == "publication-grade"
    validate_optional_path(
        request=request,
        key="permutation_null_metrics_file",
        base=request_base,
        failures=failures,
        required=null_required,
        normalized=normalized,
    )

    normalized["thresholds"] = validate_thresholds(request, failures, warnings, args.strict)

    context = request.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, dict):
        add_issue(
            failures,
            "invalid_context",
            "context must be an object when provided.",
            {"actual_type": type(context).__name__},
        )
    else:
        normalized["context"] = context

    return finish(args, failures, warnings, normalized, request_path=request_path)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    normalized_request: Dict[str, Any],
    request_path: Optional[Path] = None,
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "request_path": str(request_path) if request_path else None,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "normalized_request": normalized_request,
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
