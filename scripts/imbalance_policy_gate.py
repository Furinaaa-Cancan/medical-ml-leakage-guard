#!/usr/bin/env python3
"""
Fail-closed class-imbalance policy gate for medical prediction workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate imbalance handling policy and split label distributions.")
    parser.add_argument("--policy-spec", required=True, help="Path to imbalance policy JSON.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--target-col", default="y", help="Label column name.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def parse_label(raw: str) -> Optional[int]:
    s = raw.strip()
    if not s:
        return None
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    try:
        parsed = float(s)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if parsed == 0.0:
        return 0
    if parsed == 1.0:
        return 1
    return None


def read_label_stats(path: str, split_name: str, target_col: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name}: file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{split_name}: missing CSV header.")
        headers = [(h or "").strip() for h in reader.fieldnames]
        if target_col not in headers:
            raise ValueError(f"{split_name}: missing target_col '{target_col}'.")

        pos = 0
        neg = 0
        invalid = 0
        row_count = 0
        for row in reader:
            row_count += 1
            label = parse_label((row.get(target_col) or ""))
            if label is None:
                invalid += 1
            elif label == 1:
                pos += 1
            else:
                neg += 1

    ratio = None
    minority = min(pos, neg)
    majority = max(pos, neg)
    if minority > 0:
        ratio = majority / float(minority)
    elif majority > 0:
        ratio = math.inf

    prevalence = None
    denom = pos + neg
    if denom > 0:
        prevalence = pos / float(denom)

    return {
        "path": str(Path(path).expanduser().resolve()),
        "row_count": row_count,
        "positive_count": pos,
        "negative_count": neg,
        "invalid_label_rows": invalid,
        "prevalence": prevalence,
        "imbalance_ratio_majority_to_minority": ratio,
        "minority_count": minority,
        "majority_count": majority,
    }


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


def require_bool(spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]]) -> Optional[bool]:
    value = spec.get(key)
    if isinstance(value, bool):
        return value
    add_issue(
        failures,
        "invalid_policy_field",
        "Policy field must be boolean.",
        {"field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def require_number(
    spec: Dict[str, Any], key: str, failures: List[Dict[str, Any]], default: Optional[float] = None
) -> Optional[float]:
    if key not in spec:
        return default
    value = spec.get(key)
    if isinstance(value, bool):
        add_issue(
            failures,
            "invalid_policy_field",
            "Policy numeric field must be finite number.",
            {"field": key, "actual_type": "bool"},
        )
        return default
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


def normalize_scope_list(raw: Any, field: str, failures: List[Dict[str, Any]]) -> Optional[List[str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        add_issue(
            failures,
            "invalid_policy_field",
            "Policy field must be list of non-empty strings.",
            {"field": field, "actual_type": type(raw).__name__},
        )
        return None

    out: List[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            add_issue(
                failures,
                "invalid_policy_field",
                "Policy list items must be non-empty strings.",
                {"field": field},
            )
            return None
        out.append(item.strip().lower())
    return sorted(set(out))


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    spec_path = Path(args.policy_spec).expanduser().resolve()
    if not spec_path.exists():
        add_issue(
            failures,
            "missing_policy_spec",
            "Imbalance policy spec file not found.",
            {"path": str(spec_path)},
        )
        return finish(args, failures, warnings, {}, {})

    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
        if not isinstance(spec, dict):
            raise ValueError("Imbalance policy spec root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_policy_spec",
            "Failed to parse imbalance policy spec JSON.",
            {"path": str(spec_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {}, {})

    strategy = require_str(spec, "strategy", failures)
    fit_scope = require_str(spec, "fit_scope", failures)
    threshold_selection_split = require_str(spec, "threshold_selection_split", failures)
    calibration_split = require_str(spec, "calibration_split", failures)
    forbid_test_usage = require_bool(spec, "forbid_test_usage", failures)

    ratio_alert = require_number(spec, "imbalance_alert_ratio", failures, default=5.0)
    min_minority_cases = require_number(spec, "minimum_minority_cases", failures, default=20.0)

    allowed_strategies = {
        "class_weight",
        "focal_loss",
        "oversample_train_only",
        "undersample_train_only",
        "smote_train_only",
        "none",
    }
    allowed_fit_scopes = {"train_only", "cv_inner_train_only", "fold_train_only"}

    if strategy and strategy not in allowed_strategies:
        add_issue(
            failures,
            "unsupported_imbalance_strategy",
            "Unsupported imbalance strategy.",
            {"strategy": strategy, "allowed": sorted(allowed_strategies)},
        )
    if fit_scope and fit_scope not in allowed_fit_scopes:
        add_issue(
            failures,
            "invalid_fit_scope",
            "fit_scope must ensure train-only fitting.",
            {"fit_scope": fit_scope, "allowed": sorted(allowed_fit_scopes)},
        )
    if forbid_test_usage is not None and forbid_test_usage is not True:
        add_issue(
            failures,
            "test_usage_not_forbidden",
            "forbid_test_usage must be true.",
            {},
        )

    allowed_postprocessing_splits = {"valid", "cv_inner", "nested_cv", "none", "not_applicable", "na"}
    has_valid_split = bool(args.valid)
    for field_name, split_name in (
        ("threshold_selection_split", threshold_selection_split),
        ("calibration_split", calibration_split),
    ):
        if not split_name:
            continue
        split_token = split_name.lower()
        if split_token not in allowed_postprocessing_splits:
            add_issue(
                failures,
                "invalid_postprocessing_split",
                "Post-processing split must be one of the approved non-test scopes.",
                {"field": field_name, "split": split_name, "allowed": sorted(allowed_postprocessing_splits)},
            )
        if split_token == "test":
            add_issue(
                failures,
                "test_split_used_for_postprocessing",
                "Test split must not be used for threshold/calibration selection.",
                {"field": field_name, "split": split_name},
            )
        if split_token == "train":
            add_issue(
                failures,
                "train_split_used_for_postprocessing",
                "Train split must not be reused for threshold/calibration selection.",
                {"field": field_name, "split": split_name},
            )
        if split_token == "valid" and not has_valid_split:
            add_issue(
                failures,
                "valid_split_required_but_missing",
                "Policy requires valid split for post-processing, but valid split path is not provided.",
                {"field": field_name, "split": split_name, "has_valid_split": has_valid_split},
            )

    resampling_required = strategy in {"oversample_train_only", "undersample_train_only", "smote_train_only"}
    raw_resampling = spec.get("resampling_applied_to")
    normalized_scope = normalize_scope_list(raw_resampling, "resampling_applied_to", failures)
    if resampling_required:
        if normalized_scope is None or not normalized_scope:
            add_issue(
                failures,
                "missing_resampling_scope",
                "Resampling strategy requires resampling_applied_to list.",
                {"strategy": strategy},
            )
        elif normalized_scope != ["train"]:
            add_issue(
                failures,
                "resampling_scope_leakage",
                "Resampling must be applied only to train split.",
                {"resampling_applied_to": normalized_scope},
            )
    elif normalized_scope not in (None, [], ["none"]):
        add_issue(
            failures,
            "unexpected_resampling_scope",
            "Non-resampling strategy must not declare active resampling scopes.",
            {"strategy": strategy, "resampling_applied_to": normalized_scope},
        )

    splits: Dict[str, Dict[str, Any]] = {}
    split_paths = {"train": args.train, "test": args.test}
    if args.valid:
        split_paths["valid"] = args.valid

    try:
        for split_name, path in split_paths.items():
            splits[split_name] = read_label_stats(path, split_name, args.target_col)
    except Exception as exc:
        add_issue(
            failures,
            "split_label_read_error",
            "Failed to read split labels for imbalance audit.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, spec, splits, strategy=strategy)

    for split_name, stats in splits.items():
        if stats["row_count"] <= 0:
            add_issue(
                failures,
                "empty_split",
                "Split must not be empty.",
                {"split": split_name},
            )
        if stats["invalid_label_rows"] > 0:
            add_issue(
                failures,
                "invalid_labels",
                "Split contains non-binary or invalid labels.",
                {"split": split_name, "invalid_label_rows": stats["invalid_label_rows"]},
            )
        if stats["positive_count"] == 0 or stats["negative_count"] == 0:
            add_issue(
                failures,
                "single_class_split",
                "Split must contain both classes to support robust model validation.",
                {
                    "split": split_name,
                    "positive_count": stats["positive_count"],
                    "negative_count": stats["negative_count"],
                },
            )

    train_stats = splits.get("train", {})
    train_ratio = train_stats.get("imbalance_ratio_majority_to_minority")
    train_minority = train_stats.get("minority_count")

    if isinstance(train_minority, int) and min_minority_cases is not None and train_minority < int(min_minority_cases):
        add_issue(
            failures,
            "insufficient_minority_samples",
            "Train split minority class count is below required minimum.",
            {"minority_count": train_minority, "minimum_required": int(min_minority_cases)},
        )

    if strategy == "none" and train_ratio is not None and ratio_alert is not None:
        if math.isinf(train_ratio) or train_ratio > ratio_alert:
            add_issue(
                failures,
                "imbalance_unmitigated",
                "Severe class imbalance detected but strategy is 'none'.",
                {"train_ratio": train_ratio, "ratio_alert": ratio_alert},
            )

    return finish(args, failures, warnings, spec, splits, strategy=strategy)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    spec: Dict[str, Any],
    splits: Dict[str, Dict[str, Any]],
    strategy: Optional[str] = None,
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    split_summary: Dict[str, Any] = {}
    for split_name, stats in splits.items():
        split_summary[split_name] = {
            "path": stats.get("path"),
            "row_count": stats.get("row_count"),
            "positive_count": stats.get("positive_count"),
            "negative_count": stats.get("negative_count"),
            "minority_count": stats.get("minority_count"),
            "majority_count": stats.get("majority_count"),
            "prevalence": stats.get("prevalence"),
            "imbalance_ratio_majority_to_minority": stats.get("imbalance_ratio_majority_to_minority"),
            "invalid_label_rows": stats.get("invalid_label_rows"),
        }

    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "strategy": strategy,
        "policy_spec": str(Path(args.policy_spec).expanduser().resolve()),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "policy_fields_present": sorted(spec.keys()) if isinstance(spec, dict) else [],
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
