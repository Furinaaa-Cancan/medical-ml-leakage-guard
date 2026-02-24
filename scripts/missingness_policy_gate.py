#!/usr/bin/env python3
"""
Fail-closed missingness policy gate for large-scale medical prediction workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a", "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate missingness/imputation policy and split missingness profiles.")
    parser.add_argument("--policy-spec", required=True, help="Path to missingness policy JSON.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--target-col", default="y", help="Target/label column name.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def require_str(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[str]:
    value = spec.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    add_issue(
        failures,
        "invalid_policy_field",
        "Policy field must be a non-empty string.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def require_bool(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]], default: Optional[bool] = None) -> Optional[bool]:
    if key not in spec:
        return default
    value = spec.get(key)
    if isinstance(value, bool):
        return value
    add_issue(
        failures,
        "invalid_policy_field",
        "Policy field must be boolean.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return default


def require_number(
    spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]], default: Optional[float] = None
) -> Optional[float]:
    if key not in spec:
        return default
    value = spec.get(key)
    if isinstance(value, bool):
        value = None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        add_issue(
            failures,
            "invalid_policy_field",
            "Policy numeric field must be finite number.",
            {"field": key, "actual_type": type(value).__name__ if value is not None else None},
        )
        return default
    if not math.isfinite(parsed):
        add_issue(
            failures,
            "invalid_policy_field",
            "Policy numeric field must be finite number.",
            {"field": key, "value": value},
        )
        return default
    return parsed


def require_str_list(
    spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]], default: Optional[List[str]] = None
) -> List[str]:
    if key not in spec:
        return list(default or [])
    value = spec.get(key)
    if not isinstance(value, list):
        add_issue(
            failures,
            "invalid_policy_field",
            "Policy field must be list of strings.",
            {"field": key, "actual_type": type(value).__name__ if value is not None else None},
        )
        return list(default or [])
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        else:
            add_issue(
                failures,
                "invalid_policy_field",
                "Policy list items must be non-empty strings.",
                {"field": key},
            )
            return list(default or [])
    return out


def validate_ratio_field(
    key: str, value: Optional[float], failures: List[Dict[str, Any]], low: float = 0.0, high: float = 1.0
) -> None:
    if value is None:
        return
    if value < low or value > high:
        add_issue(
            failures,
            "invalid_policy_range",
            "Policy ratio must be within allowed range.",
            {"field": key, "value": value, "range": [low, high]},
        )


def read_missing_stats(path: str, split_name: str, target_col: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name}: file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{split_name}: missing CSV header.")
        headers = [(h or "").strip() for h in reader.fieldnames]
        if target_col not in headers:
            raise ValueError(f"{split_name}: missing target_col '{target_col}'.")

        missing_by_col: Dict[str, int] = {h: 0 for h in headers}
        row_count = 0
        for row in reader:
            row_count += 1
            for col in headers:
                raw = row.get(col)
                value = "" if raw is None else str(raw)
                if is_missing(value):
                    missing_by_col[col] += 1

    col_count = len(headers)
    total_cells = row_count * col_count
    total_missing = sum(missing_by_col.values())
    total_missing_ratio = None if total_cells <= 0 else (total_missing / float(total_cells))
    missing_ratio_by_col: Dict[str, Optional[float]] = {}
    for col in headers:
        missing_ratio_by_col[col] = None if row_count <= 0 else (missing_by_col[col] / float(row_count))

    return {
        "path": str(Path(path).expanduser().resolve()),
        "headers": headers,
        "row_count": row_count,
        "col_count": col_count,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "total_missing_ratio": total_missing_ratio,
        "missing_by_col": missing_by_col,
        "missing_ratio_by_col": missing_ratio_by_col,
        "target_missing_count": missing_by_col.get(target_col, 0),
    }


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    policy_path = Path(args.policy_spec).expanduser().resolve()
    if not policy_path.exists():
        add_issue(
            failures,
            "missing_policy_spec",
            "Missingness policy file not found.",
            {"path": str(policy_path)},
        )
        return finish(args, failures, warnings, {}, {})

    try:
        with policy_path.open("r", encoding="utf-8") as fh:
            policy = json.load(fh)
        if not isinstance(policy, dict):
            raise ValueError("Missingness policy root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_policy_spec",
            "Unable to parse missingness policy JSON.",
            {"path": str(policy_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {}, {})

    strategy = require_str(policy, "strategy", failures)
    imputer_fit_scope = require_str(policy, "imputer_fit_scope", failures)

    add_missing_indicators = require_bool(policy, "add_missing_indicators", failures, default=False)
    complete_case_analysis = require_bool(policy, "complete_case_analysis", failures, default=False)
    forbid_test_usage = require_bool(policy, "forbid_test_usage", failures, default=True)
    test_used_for_fit = require_bool(policy, "test_used_for_fit", failures, default=False)
    valid_used_for_fit = require_bool(policy, "valid_used_for_fit", failures, default=False)
    use_target_in_imputation = require_bool(policy, "use_target_in_imputation", failures, default=False)

    max_feature_missing_ratio = require_number(policy, "max_feature_missing_ratio", failures, default=0.6)
    min_non_missing_per_feature = require_number(policy, "min_non_missing_per_feature", failures, default=50.0)
    indicator_required_above_ratio = require_number(policy, "indicator_required_above_ratio", failures, default=0.1)
    missingness_drift_tolerance = require_number(policy, "missingness_drift_tolerance", failures, default=0.2)

    large_data_row_threshold = require_number(policy, "large_data_row_threshold", failures, default=1_000_000.0)
    large_data_col_threshold = require_number(policy, "large_data_col_threshold", failures, default=300.0)
    mice_max_rows = require_number(policy, "mice_max_rows", failures, default=200_000.0)
    mice_max_cols = require_number(policy, "mice_max_cols", failures, default=200.0)

    allowed_high_missing_features = set(
        x.lower() for x in require_str_list(policy, "allowed_high_missing_features", failures, default=[])
    )
    large_data_restricted_methods = set(
        x.lower()
        for x in require_str_list(
            policy,
            "large_data_restricted_methods",
            failures,
            default=["mice", "missforest", "knn"],
        )
    )

    validate_ratio_field("max_feature_missing_ratio", max_feature_missing_ratio, failures)
    validate_ratio_field("indicator_required_above_ratio", indicator_required_above_ratio, failures)
    validate_ratio_field("missingness_drift_tolerance", missingness_drift_tolerance, failures)

    if min_non_missing_per_feature is not None and min_non_missing_per_feature < 1:
        add_issue(
            failures,
            "invalid_policy_range",
            "min_non_missing_per_feature must be >= 1.",
            {"value": min_non_missing_per_feature},
        )

    if strategy:
        allowed_strategies = {
            "none",
            "simple",
            "simple_with_indicator",
            "mice",
            "missforest",
            "knn",
            "drop_rows",
        }
        if strategy not in allowed_strategies:
            add_issue(
                failures,
                "unsupported_missingness_strategy",
                "Unsupported missingness strategy.",
                {"strategy": strategy, "allowed": sorted(allowed_strategies)},
            )

    if imputer_fit_scope:
        allowed_fit_scopes = {"train_only", "cv_inner_train_only", "fold_train_only"}
        if imputer_fit_scope not in allowed_fit_scopes:
            add_issue(
                failures,
                "invalid_imputer_fit_scope",
                "Imputer fit scope must be train-only.",
                {"imputer_fit_scope": imputer_fit_scope, "allowed": sorted(allowed_fit_scopes)},
            )

    if forbid_test_usage is not None and forbid_test_usage is not True:
        add_issue(
            failures,
            "test_usage_not_forbidden",
            "forbid_test_usage must be true.",
            {},
        )
    if test_used_for_fit is True:
        add_issue(
            failures,
            "test_used_for_imputer_fit",
            "Imputer fit must not use test data.",
            {},
        )
    if valid_used_for_fit is True:
        add_issue(
            warnings,
            "valid_used_for_imputer_fit",
            "Imputer fit uses valid split; strict protocols prefer train-only fitting.",
            {},
        )
    if use_target_in_imputation is True:
        add_issue(
            failures,
            "target_used_in_imputation",
            "Target/outcome signal must not be used in imputation model.",
            {},
        )

    splits: Dict[str, Dict[str, Any]] = {}
    split_paths = {"train": args.train, "test": args.test}
    if args.valid:
        split_paths["valid"] = args.valid

    try:
        for split_name, path in split_paths.items():
            splits[split_name] = read_missing_stats(path, split_name, args.target_col)
    except Exception as exc:
        add_issue(
            failures,
            "split_missingness_read_error",
            "Failed to profile split missingness.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, policy, splits)

    for split_name, stats in splits.items():
        if stats["row_count"] <= 0:
            add_issue(
                failures,
                "empty_split",
                "Split must not be empty.",
                {"split": split_name},
            )
        if stats["target_missing_count"] > 0:
            add_issue(
                failures,
                "target_missing_values",
                "Target column contains missing values.",
                {"split": split_name, "target_missing_count": stats["target_missing_count"]},
            )

    train_stats = splits.get("train")
    if train_stats:
        train_rows = int(train_stats["row_count"])
        train_cols = int(train_stats["col_count"])
        feature_count = max(train_cols - 1, 0)

        if strategy and strategy.lower() in large_data_restricted_methods:
            if (
                large_data_row_threshold is not None
                and large_data_col_threshold is not None
                and train_rows >= int(large_data_row_threshold)
                and feature_count >= int(large_data_col_threshold)
            ):
                add_issue(
                    failures,
                    "strategy_not_scalable_for_large_data",
                    "Chosen imputation strategy is not approved for this data scale.",
                    {
                        "strategy": strategy,
                        "train_rows": train_rows,
                        "feature_count": feature_count,
                        "large_data_row_threshold": int(large_data_row_threshold),
                        "large_data_col_threshold": int(large_data_col_threshold),
                    },
                )

        if strategy == "mice":
            if mice_max_rows is not None and train_rows > int(mice_max_rows):
                add_issue(
                    failures,
                    "mice_row_scale_exceeded",
                    "MICE is not approved beyond configured row limit.",
                    {"train_rows": train_rows, "mice_max_rows": int(mice_max_rows)},
                )
            if mice_max_cols is not None and feature_count > int(mice_max_cols):
                add_issue(
                    failures,
                    "mice_feature_scale_exceeded",
                    "MICE is not approved beyond configured feature limit.",
                    {"feature_count": feature_count, "mice_max_cols": int(mice_max_cols)},
                )

        if strategy == "none":
            total_missing = int(train_stats["total_missing"])
            if total_missing > 0 and not complete_case_analysis:
                add_issue(
                    failures,
                    "missingness_unhandled",
                    "Missing values exist but strategy is 'none' without complete-case analysis.",
                    {"train_total_missing": total_missing},
                )

        missing_by_col = train_stats["missing_by_col"]
        ratio_by_col = train_stats["missing_ratio_by_col"]
        for col in train_stats["headers"]:
            if col == args.target_col:
                continue
            ratio = ratio_by_col.get(col)
            if ratio is None:
                continue
            non_missing = train_rows - int(missing_by_col.get(col, 0))
            col_lower = col.lower()

            if (
                max_feature_missing_ratio is not None
                and ratio > max_feature_missing_ratio
                and col_lower not in allowed_high_missing_features
            ):
                add_issue(
                    failures,
                    "feature_missingness_too_high",
                    "Feature missingness exceeds policy threshold.",
                    {
                        "feature": col,
                        "missing_ratio": ratio,
                        "max_feature_missing_ratio": max_feature_missing_ratio,
                    },
                )

            if min_non_missing_per_feature is not None and non_missing < int(min_non_missing_per_feature):
                add_issue(
                    failures,
                    "insufficient_non_missing_samples",
                    "Feature has too few non-missing samples.",
                    {
                        "feature": col,
                        "non_missing_count": non_missing,
                        "minimum_required": int(min_non_missing_per_feature),
                    },
                )

            if (
                indicator_required_above_ratio is not None
                and ratio >= indicator_required_above_ratio
                and strategy not in {"drop_rows", "none"}
                and add_missing_indicators is not True
            ):
                add_issue(
                    warnings,
                    "missing_indicator_recommended",
                    "High-missingness feature without missing-indicator flag.",
                    {
                        "feature": col,
                        "missing_ratio": ratio,
                        "indicator_required_above_ratio": indicator_required_above_ratio,
                    },
                )

    # Missingness drift audit (train vs valid/test) for sufficiently large splits.
    if "train" in splits and missingness_drift_tolerance is not None:
        train = splits["train"]
        train_rows = int(train["row_count"])
        for split_name in ("valid", "test"):
            if split_name not in splits:
                continue
            other = splits[split_name]
            other_rows = int(other["row_count"])
            if min(train_rows, other_rows) < 50:
                continue
            shared_cols = set(train["headers"]) & set(other["headers"])
            for col in shared_cols:
                if col == args.target_col:
                    continue
                train_ratio = train["missing_ratio_by_col"].get(col)
                other_ratio = other["missing_ratio_by_col"].get(col)
                if train_ratio is None or other_ratio is None:
                    continue
                diff = abs(train_ratio - other_ratio)
                if diff > missingness_drift_tolerance:
                    add_issue(
                        warnings,
                        "missingness_shift",
                        "Missingness ratio shift exceeds tolerance across splits.",
                        {
                            "feature": col,
                            "train_missing_ratio": train_ratio,
                            f"{split_name}_missing_ratio": other_ratio,
                            "abs_diff": diff,
                            "tolerance": missingness_drift_tolerance,
                        },
                    )

    return finish(args, failures, warnings, policy, splits)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    policy: Dict[str, Any],
    splits: Dict[str, Dict[str, Any]],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))

    split_summary: Dict[str, Any] = {}
    for split_name, stats in splits.items():
        split_summary[split_name] = {
            "path": stats.get("path"),
            "row_count": stats.get("row_count"),
            "col_count": stats.get("col_count"),
            "total_missing": stats.get("total_missing"),
            "total_missing_ratio": stats.get("total_missing_ratio"),
            "target_missing_count": stats.get("target_missing_count"),
            "top_missing_features": sorted(
                (
                    {
                        "feature": col,
                        "missing_count": int(stats["missing_by_col"].get(col, 0)),
                        "missing_ratio": stats["missing_ratio_by_col"].get(col),
                    }
                    for col in stats.get("headers", [])
                    if col != args.target_col
                ),
                key=lambda item: (item.get("missing_ratio") or 0.0),
                reverse=True,
            )[:10],
        }

    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "policy_spec": str(Path(args.policy_spec).expanduser().resolve()),
        "strategy": policy.get("strategy"),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "policy_fields_present": sorted(policy.keys()) if isinstance(policy, dict) else [],
            "splits": split_summary,
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
