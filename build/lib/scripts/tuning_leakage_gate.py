#!/usr/bin/env python3
"""
Fail-closed hyperparameter tuning leakage gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)
from _gate_utils import add_issue


register_remediations({
    "tuning_spec_missing": "Provide a valid tuning_protocol_spec JSON.",
    "search_strategy_missing": "tuning_protocol_spec must declare a search_strategy (grid, random, bayesian, etc.).",
    "cv_not_enabled": "Cross-validation must be enabled for publication-grade tuning.",
    "cv_folds_too_low": "CV fold count is too low. Use at least 5 folds for robust tuning.",
    "nested_cv_without_cv": "nested_cv model selection requires cv.enabled=true.",
    "cv_inner_without_cv": "cv_inner model selection requires cv.enabled=true.",
    "leakage_risk_refit_on_all": "Refitting on all data risks leakage. Use train+valid only.",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tuning protocol against leakage-safe requirements.")
    parser.add_argument("--tuning-spec", required=True, help="Path to tuning protocol JSON.")
    parser.add_argument("--id-col", help="Runtime ID column used for grouped CV validation.")
    parser.add_argument(
        "--has-valid-split",
        action="store_true",
        help="Indicate that a dedicated validation split exists in this run.",
    )
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def require_str(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[str]:
    value = spec.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    add_issue(
        failures,
        "invalid_tuning_field",
        "Tuning field must be a non-empty string.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def require_bool(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[bool]:
    value = spec.get(key)
    if isinstance(value, bool):
        return value
    add_issue(
        failures,
        "invalid_tuning_field",
        "Tuning field must be boolean.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def require_int(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[int]:
    value = spec.get(key)
    if isinstance(value, bool):
        value = None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    add_issue(
        failures,
        "invalid_tuning_field",
        "Tuning field must be integer.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def contains_test_token(value: Optional[str]) -> bool:
    if not value:
        return False
    token = value.strip().lower()
    if "no_test" in token or "without_test" in token or "exclude_test" in token or "notest" in token:
        return False
    parts = [p for p in re.split(r"[^a-z0-9]+", token) if p]
    if "test" in parts:
        return True
    for part in parts:
        if part.startswith("test") or part.endswith("test"):
            if part not in {"latest", "attest", "tested", "testing"}:
                return True
    return False


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    has_valid_split = bool(args.has_valid_split)

    spec_path = Path(args.tuning_spec).expanduser().resolve()
    if not spec_path.exists():
        add_issue(
            failures,
            "missing_tuning_spec",
            "Tuning protocol spec file not found.",
            {"path": str(spec_path)},
        )
        return finish(args, failures, warnings, {})

    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
        if not isinstance(spec, dict):
            raise ValueError("Tuning spec root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_tuning_spec",
            "Failed to parse tuning protocol spec JSON.",
            {"path": str(spec_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    search_method = require_str(spec, "search_method", failures)
    model_selection_data = require_str(spec, "model_selection_data", failures)
    early_stopping_data = require_str(spec, "early_stopping_data", failures)
    preprocessing_fit_scope = require_str(spec, "preprocessing_fit_scope", failures)
    feature_selection_scope = require_str(spec, "feature_selection_scope", failures)
    resampling_scope = require_str(spec, "resampling_scope", failures)
    final_model_refit_scope = require_str(spec, "final_model_refit_scope", failures)
    objective_metric = require_str(spec, "objective_metric", failures)
    hyperparameter_trials = require_int(spec, "hyperparameter_trials", failures)

    test_used_for_model_selection = require_bool(spec, "test_used_for_model_selection", failures)
    test_used_for_early_stopping = require_bool(spec, "test_used_for_early_stopping", failures)
    test_used_for_threshold_selection = require_bool(spec, "test_used_for_threshold_selection", failures)
    test_used_for_calibration = require_bool(spec, "test_used_for_calibration", failures)
    outer_evaluation_split_locked = require_bool(spec, "outer_evaluation_split_locked", failures)
    random_seed_controlled = require_bool(spec, "random_seed_controlled", failures)

    allowed_search_methods = {
        "grid_search",
        "random_search",
        "bayesian_optimization",
        "optuna",
        "hyperband",
        "manual_pre_registered",
    }
    if search_method and search_method not in allowed_search_methods:
        add_issue(
            failures,
            "unsupported_search_method",
            "search_method must be a pre-registered and approved strategy.",
            {"search_method": search_method, "allowed": sorted(allowed_search_methods)},
        )

    allowed_train_only_scopes = {"train_only", "fold_train_only", "cv_inner_train_only"}
    for field_name, field_value in (
        ("preprocessing_fit_scope", preprocessing_fit_scope),
        ("feature_selection_scope", feature_selection_scope),
        ("resampling_scope", resampling_scope),
    ):
        if field_value and field_value not in allowed_train_only_scopes:
            add_issue(
                failures,
                "invalid_scope",
                "Scope must be train-only to avoid leakage.",
                {"field": field_name, "value": field_value, "allowed": sorted(allowed_train_only_scopes)},
            )

    allowed_early_stopping_data = {"none", "valid", "cv_inner", "nested_cv"}
    if early_stopping_data and early_stopping_data not in allowed_early_stopping_data:
        add_issue(
            failures,
            "invalid_early_stopping_data",
            "early_stopping_data must use approved non-test sources.",
            {"early_stopping_data": early_stopping_data, "allowed": sorted(allowed_early_stopping_data)},
        )

    allowed_final_model_refit_scope = {
        "train_only",
        "train_plus_valid_no_test",
        "outer_train_only",
    }
    if final_model_refit_scope and final_model_refit_scope not in allowed_final_model_refit_scope:
        add_issue(
            failures,
            "invalid_final_model_refit_scope",
            "final_model_refit_scope must not include test data and must be explicitly approved.",
            {
                "final_model_refit_scope": final_model_refit_scope,
                "allowed": sorted(allowed_final_model_refit_scope),
            },
        )

    if model_selection_data == "valid" and not has_valid_split:
        add_issue(
            failures,
            "valid_model_selection_without_valid_split",
            "model_selection_data=valid requires an actual validation split.",
            {"has_valid_split": has_valid_split},
        )
    if early_stopping_data == "valid" and not has_valid_split:
        add_issue(
            failures,
            "valid_early_stopping_without_valid_split",
            "early_stopping_data=valid requires an actual validation split.",
            {"has_valid_split": has_valid_split},
        )
    if final_model_refit_scope == "train_plus_valid_no_test" and not has_valid_split:
        add_issue(
            failures,
            "train_plus_valid_refit_without_valid_split",
            "final_model_refit_scope=train_plus_valid_no_test requires a validation split.",
            {"has_valid_split": has_valid_split},
        )

    for field_name, field_value in (
        ("model_selection_data", model_selection_data),
        ("early_stopping_data", early_stopping_data),
        ("final_model_refit_scope", final_model_refit_scope),
    ):
        if contains_test_token(field_value):
            add_issue(
                failures,
                "test_data_usage_detected",
                "Test data token detected in tuning protocol field.",
                {"field": field_name, "value": field_value},
            )

    for field_name, field_value in (
        ("test_used_for_model_selection", test_used_for_model_selection),
        ("test_used_for_early_stopping", test_used_for_early_stopping),
        ("test_used_for_threshold_selection", test_used_for_threshold_selection),
        ("test_used_for_calibration", test_used_for_calibration),
    ):
        if field_value is True:
            add_issue(
                failures,
                "explicit_test_usage",
                "Tuning protocol indicates explicit test usage, which is forbidden.",
                {"field": field_name},
            )

    if outer_evaluation_split_locked is not None and outer_evaluation_split_locked is not True:
        add_issue(
            failures,
            "outer_evaluation_not_locked",
            "outer_evaluation_split_locked must be true.",
            {},
        )
    if random_seed_controlled is not None and random_seed_controlled is not True:
        add_issue(
            failures,
            "seed_not_controlled",
            "random_seed_controlled must be true.",
            {},
        )

    if hyperparameter_trials is not None and hyperparameter_trials <= 0:
        add_issue(
            failures,
            "invalid_hyperparameter_trials",
            "hyperparameter_trials must be >= 1.",
            {"hyperparameter_trials": hyperparameter_trials},
        )

    if objective_metric and contains_test_token(objective_metric):
        add_issue(
            failures,
            "invalid_objective_metric",
            "objective_metric value appears to reference test data.",
            {"objective_metric": objective_metric},
        )

    if search_method and search_method.lower() in {"manual_test_peeking", "ad_hoc_test_selection"}:
        add_issue(
            failures,
            "unsafe_search_method",
            "Search method is leakage-prone.",
            {"search_method": search_method},
        )

    allowed_model_selection_data = {"valid", "cv_inner", "nested_cv"}
    if model_selection_data and model_selection_data not in allowed_model_selection_data:
        add_issue(
            failures,
            "invalid_model_selection_data",
            "model_selection_data must be valid/cv_inner/nested_cv.",
            {"model_selection_data": model_selection_data, "allowed": sorted(allowed_model_selection_data)},
        )

    cv = spec.get("cv")
    if not isinstance(cv, dict):
        add_issue(
            failures,
            "missing_cv_config",
            "cv section must be provided as object.",
            {},
        )
    else:
        cv_enabled = cv.get("enabled")
        if not isinstance(cv_enabled, bool):
            add_issue(
                failures,
                "invalid_cv_field",
                "cv.enabled must be boolean.",
                {"actual_type": type(cv_enabled).__name__ if cv_enabled is not None else None},
            )
            cv_enabled = None

        cv_type = cv.get("type")
        if cv_enabled:
            if not isinstance(cv_type, str) or not cv_type.strip():
                add_issue(
                    failures,
                    "invalid_cv_field",
                    "cv.type must be non-empty string when cv is enabled.",
                    {},
                )
            else:
                allowed_cv_types = {
                    "group_k_fold",
                    "stratified_group_k_fold",
                    "time_series_split",
                    "group_time_series_split",
                }
                if cv_type not in allowed_cv_types:
                    add_issue(
                        failures,
                        "unsupported_cv_type",
                        "Unsupported CV type for strict medical evaluation.",
                        {"cv_type": cv_type, "allowed": sorted(allowed_cv_types)},
                    )
                if "group" in cv_type:
                    group_col = cv.get("group_col")
                    if not isinstance(group_col, str) or not group_col.strip():
                        add_issue(
                            failures,
                            "missing_cv_group_col",
                            "Grouped CV type requires cv.group_col.",
                            {"cv_type": cv_type},
                        )
                    elif args.id_col and group_col.strip() != args.id_col:
                        add_issue(
                            failures,
                            "cv_group_col_mismatch",
                            "cv.group_col must match runtime id-col.",
                            {"cv_group_col": group_col.strip(), "runtime_id_col": args.id_col},
                        )

            n_splits = cv.get("n_splits")
            if isinstance(n_splits, bool):
                n_splits = None
            if not isinstance(n_splits, int):
                add_issue(
                    failures,
                    "invalid_cv_field",
                    "cv.n_splits must be integer when cv is enabled.",
                    {"actual_type": type(n_splits).__name__ if n_splits is not None else None},
                )
            elif n_splits < 3:
                add_issue(
                    failures,
                    "insufficient_cv_splits",
                    "cv.n_splits must be >= 3 for stable tuning.",
                    {"n_splits": n_splits},
                )

            nested = cv.get("nested")
            if not isinstance(nested, bool):
                add_issue(
                    failures,
                    "invalid_cv_field",
                    "cv.nested must be boolean when cv is enabled.",
                    {"actual_type": type(nested).__name__ if nested is not None else None},
                )
            elif model_selection_data == "nested_cv" and nested is not True:
                add_issue(
                    failures,
                    "nested_cv_required",
                    "model_selection_data=nested_cv requires cv.nested=true.",
                    {},
                )
            elif model_selection_data == "cv_inner" and nested is True:
                add_issue(
                    warnings,
                    "nested_cv_overconfigured",
                    "cv.nested=true with model_selection_data=cv_inner; verify protocol intent.",
                    {},
                )

        if model_selection_data == "cv_inner" and cv_enabled is not True:
            add_issue(
                failures,
                "cv_inner_without_cv",
                "model_selection_data=cv_inner requires cv.enabled=true.",
                {},
            )
        if model_selection_data == "nested_cv" and cv_enabled is not True:
            add_issue(
                failures,
                "nested_cv_without_cv",
                "model_selection_data=nested_cv requires cv.enabled=true.",
                {},
            )

    return finish(args, failures, warnings, spec)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    spec: Dict[str, Any],
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
        "tuning_spec": str(Path(args.tuning_spec).expanduser().resolve()),
    }

    report = build_report_envelope(
        gate_name="tuning_leakage_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary={
            "fields_present": sorted(spec.keys()) if isinstance(spec, dict) else [],
            "has_valid_split": bool(args.has_valid_split),
        },
        input_files=input_files,
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="tuning_leakage_gate",
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
