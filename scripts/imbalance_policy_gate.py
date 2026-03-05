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
from typing import Any, Dict, List, Optional, Tuple

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
    "imbalance_policy_missing": "Provide a valid imbalance_policy_spec JSON with strategy and thresholds.",
    "extreme_imbalance": "Class imbalance ratio exceeds safe threshold. Apply resampling or cost-sensitive learning.",
    "strategy_not_declared": "Imbalance handling strategy not declared in policy spec.",
    "prevalence_mismatch": "Observed prevalence differs significantly between splits. Check stratification.",
    "reconciliation_mismatch": "Declared imbalance strategy doesn't match execution evidence.",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate imbalance handling policy and split label distributions.")
    parser.add_argument("--policy-spec", required=True, help="Path to imbalance policy JSON.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--evaluation-report", help="Optional evaluation_report.json for execution-vs-policy reconciliation.")
    parser.add_argument("--target-col", default="y", help="Label column name.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


STRATEGY_ALIAS_MAP = {
    "auto": "auto",
    "none": "none",
    "class_weight": "class_weight",
    "balanced": "class_weight",
    "class_weight_balanced": "class_weight",
    "oversample_train_only": "random_oversample",
    "undersample_train_only": "random_undersample",
    "smote_train_only": "smote",
    "random_oversample": "random_oversample",
    "random_undersample": "random_undersample",
    "smote": "smote",
    "adasyn": "adasyn",
}

SUPPORTED_STRATEGIES = {"none", "class_weight", "random_oversample", "random_undersample", "smote", "adasyn"}
ALLOWED_POLICY_STRATEGIES = set(SUPPORTED_STRATEGIES) | {"auto"}


def normalize_strategy_token(token: str) -> Optional[str]:
    norm = str(token or "").strip().lower()
    if not norm:
        return None
    return STRATEGY_ALIAS_MAP.get(norm)


def resolve_auto_strategy(canonical_strategy: Optional[str], train_ratio: Optional[float]) -> Optional[str]:
    if canonical_strategy != "auto":
        return canonical_strategy
    if train_ratio is None:
        return "none"
    if math.isfinite(float(train_ratio)) and float(train_ratio) >= 1.5:
        return "class_weight"
    return "none"


def load_evaluation_report(path: str, failures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    if not report_path.exists():
        add_issue(
            failures,
            "missing_evaluation_report",
            "evaluation_report file not found for imbalance reconciliation.",
            {"path": str(report_path)},
        )
        return None
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            "Failed to parse evaluation_report JSON for imbalance reconciliation.",
            {"path": str(report_path), "error": str(exc)},
        )
        return None
    if not isinstance(payload, dict):
        add_issue(
            failures,
            "invalid_evaluation_report",
            "evaluation_report JSON root must be object.",
            {"path": str(report_path)},
        )
        return None
    return payload


def extract_evaluation_selected_strategy(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None, None
    imbalance_meta = metadata.get("imbalance")
    if not isinstance(imbalance_meta, dict):
        return None, None
    selected_raw = imbalance_meta.get("selected_strategy")
    if not isinstance(selected_raw, str) or not selected_raw.strip():
        return None, None
    selected_canonical = normalize_strategy_token(selected_raw)
    return selected_raw.strip(), selected_canonical


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

    strategy_raw = require_str(spec, "strategy", failures)
    fit_scope = require_str(spec, "fit_scope", failures)
    threshold_selection_split = require_str(spec, "threshold_selection_split", failures)
    calibration_split = require_str(spec, "calibration_split", failures)
    forbid_test_usage = require_bool(spec, "forbid_test_usage", failures)

    ratio_alert = require_number(spec, "imbalance_alert_ratio", failures, default=5.0)
    min_minority_cases = require_number(spec, "minimum_minority_cases", failures, default=20.0)

    allowed_fit_scopes = {"train_only", "cv_inner_train_only", "fold_train_only"}

    strategy = normalize_strategy_token(strategy_raw) if strategy_raw else None
    if strategy_raw and strategy is None:
        add_issue(
            failures,
            "unsupported_imbalance_strategy",
            "Unsupported imbalance strategy.",
            {"strategy": strategy_raw, "allowed": sorted(ALLOWED_POLICY_STRATEGIES)},
        )
    if strategy and strategy not in ALLOWED_POLICY_STRATEGIES:
        add_issue(
            failures,
            "unsupported_imbalance_strategy",
            "Unsupported imbalance strategy.",
            {"strategy": strategy, "allowed": sorted(ALLOWED_POLICY_STRATEGIES)},
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

    strategy_for_scope = strategy
    resampling_required = strategy_for_scope in {"random_oversample", "random_undersample", "smote", "adasyn"}
    raw_resampling = spec.get("resampling_applied_to")
    normalized_scope = normalize_scope_list(raw_resampling, "resampling_applied_to", failures)
    if resampling_required:
        if normalized_scope is None or not normalized_scope:
            add_issue(
                failures,
                "missing_resampling_scope",
                "Resampling strategy requires resampling_applied_to list.",
                {"strategy": strategy_for_scope},
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
            {"strategy": strategy_for_scope, "resampling_applied_to": normalized_scope},
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
            f"Failed to read '{split_name}' split labels for imbalance audit.",
            {"error": str(exc), "path": str(path)},
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
    strategy_resolved = resolve_auto_strategy(strategy, train_ratio if isinstance(train_ratio, (int, float)) else None)

    if isinstance(train_minority, int) and min_minority_cases is not None and train_minority < int(min_minority_cases):
        add_issue(
            failures,
            "insufficient_minority_samples",
            "Train split minority class count is below required minimum.",
            {"minority_count": train_minority, "minimum_required": int(min_minority_cases)},
        )

    if strategy_resolved == "none" and train_ratio is not None and ratio_alert is not None:
        if math.isinf(train_ratio) or train_ratio > ratio_alert:
            add_issue(
                failures,
                "imbalance_unmitigated",
                "Severe class imbalance detected but strategy is 'none'.",
                {"train_ratio": train_ratio, "ratio_alert": ratio_alert},
            )

    reconciliation: Dict[str, Any] = {
        "policy_strategy_raw": strategy_raw,
        "policy_strategy_canonical": strategy,
        "policy_strategy_resolved": strategy_resolved,
        "evaluation_report": None,
        "evaluation_selected_strategy_raw": None,
        "evaluation_selected_strategy_canonical": None,
        "match": None,
    }
    if args.evaluation_report:
        eval_payload = load_evaluation_report(args.evaluation_report, failures)
        reconciliation["evaluation_report"] = str(Path(args.evaluation_report).expanduser().resolve())
        if isinstance(eval_payload, dict):
            selected_raw, selected_canonical = extract_evaluation_selected_strategy(eval_payload)
            reconciliation["evaluation_selected_strategy_raw"] = selected_raw
            reconciliation["evaluation_selected_strategy_canonical"] = selected_canonical
            if selected_raw is None:
                add_issue(
                    failures,
                    "evaluation_report_imbalance_metadata_missing",
                    "evaluation_report.metadata.imbalance.selected_strategy is missing.",
                    {},
                )
            elif selected_canonical is None:
                add_issue(
                    failures,
                    "evaluation_report_imbalance_strategy_invalid",
                    "evaluation_report selected imbalance strategy is unsupported.",
                    {"selected_strategy": selected_raw, "allowed": sorted(SUPPORTED_STRATEGIES)},
                )
            elif strategy_resolved and selected_canonical != strategy_resolved:
                add_issue(
                    failures,
                    "imbalance_execution_policy_mismatch",
                    "Executed imbalance strategy does not match policy strategy.",
                    {
                        "policy_strategy_resolved": strategy_resolved,
                        "evaluation_selected_strategy": selected_canonical,
                    },
                )
                reconciliation["match"] = False
            elif strategy_resolved:
                reconciliation["match"] = True

    return finish(
        args,
        failures,
        warnings,
        spec,
        splits,
        strategy=strategy_resolved,
        reconciliation=reconciliation,
    )


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    spec: Dict[str, Any],
    splits: Dict[str, Dict[str, Any]],
    strategy: Optional[str] = None,
    reconciliation: Optional[Dict[str, Any]] = None,
) -> int:
    from _gate_utils import get_gate_elapsed, write_json as _write_report

    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"

    fi = [GateIssue.from_legacy(f, Severity.ERROR) for f in failures]
    wi = [GateIssue.from_legacy(w, Severity.WARNING) for w in warnings]
    for issue in fi + wi:
        if not issue.remediation:
            issue.remediation = get_remediation(issue.code)

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

    input_files = {
        "policy_spec": str(Path(args.policy_spec).expanduser().resolve()),
        "train": str(Path(args.train).expanduser().resolve()),
        "test": str(Path(args.test).expanduser().resolve()),
    }
    if getattr(args, "valid", None):
        input_files["valid"] = str(Path(args.valid).expanduser().resolve())
    if getattr(args, "evaluation_report", None):
        input_files["evaluation_report"] = str(Path(args.evaluation_report).expanduser().resolve())

    report = build_report_envelope(
        gate_name="imbalance_policy_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary={
            "strategy": strategy,
            "policy_fields_present": sorted(spec.keys()) if isinstance(spec, dict) else [],
            "splits": split_summary,
            "execution_reconciliation": reconciliation if isinstance(reconciliation, dict) else None,
        },
        input_files=input_files,
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="imbalance_policy_gate",
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
