#!/usr/bin/env python3
"""
Fail-closed guard against disease-definition-variable leakage in medical prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block predictors that are used to define the same disease endpoint."
    )
    parser.add_argument("--target", required=True, help="Target name in definition spec, e.g. sepsis.")
    parser.add_argument("--definition-spec", required=True, help="Path to phenotype definition JSON.")
    parser.add_argument("--train", required=True, help="Training CSV.")
    parser.add_argument("--valid", help="Validation CSV.")
    parser.add_argument("--test", help="Test CSV.")
    parser.add_argument("--target-col", default="y", help="Target column to ignore from predictor checks.")
    parser.add_argument(
        "--ignore-cols",
        default="",
        help="Comma-separated non-predictor columns to ignore (ids/timestamps/etc).",
    )
    parser.add_argument("--report", help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Allow missing target in spec and use only global forbidden rules.",
    )
    return parser.parse_args()


def read_csv_header(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Missing header row: {path}")
        return [h.strip() for h in header]


def parse_comma_set(raw: str) -> Set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def resolve_target_block(spec: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    targets = spec.get("targets")
    if not isinstance(targets, dict):
        return None
    if target in targets and isinstance(targets[target], dict):
        return targets[target]
    lowered = target.lower()
    for key, value in targets.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
            return value
    return None


def list_from(spec: Dict[str, Any], key: str) -> List[str]:
    raw = spec.get(key, [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Field '{key}' must be a list.")
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def compile_patterns(patterns: Iterable[str]) -> Tuple[List[re.Pattern[str]], List[str]]:
    compiled: List[re.Pattern[str]] = []
    errors: List[str] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, flags=re.IGNORECASE))
        except re.error as exc:
            errors.append(f"Invalid regex '{pattern}': {exc}")
    return compiled, errors


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    split_paths = {"train": args.train}
    if args.valid:
        split_paths["valid"] = args.valid
    if args.test:
        split_paths["test"] = args.test

    try:
        headers_by_split = {name: read_csv_header(path) for name, path in split_paths.items()}
    except Exception as exc:
        add_issue(failures, "input_error", "Failed to read split headers.", {"error": str(exc)})
        return finish(args, failures, warnings, {}, [], [], [])

    header_sets = {name: set(cols) for name, cols in headers_by_split.items()}
    union_headers = set().union(*header_sets.values())
    intersection_headers = set.intersection(*header_sets.values()) if header_sets else set()
    if union_headers != intersection_headers:
        add_issue(
            warnings,
            "column_mismatch",
            "Split files have non-identical headers.",
            {"union_count": len(union_headers), "intersection_count": len(intersection_headers)},
        )

    spec_path = Path(args.definition_spec).expanduser().resolve()
    if not spec_path.exists():
        add_issue(
            failures,
            "missing_definition_spec",
            "Definition spec not found.",
            {"path": str(spec_path)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], [])

    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_definition_spec",
            "Unable to parse definition spec JSON.",
            {"error": str(exc), "path": str(spec_path)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], [])

    if not isinstance(spec, dict):
        add_issue(
            failures,
            "invalid_definition_spec",
            "Definition spec must be a JSON object.",
            {"path": str(spec_path)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], [])

    target_block = resolve_target_block(spec, args.target)
    if target_block is None and not args.allow_missing_target:
        add_issue(
            failures,
            "target_not_found",
            "Target not found in definition spec.",
            {"target": args.target, "path": str(spec_path)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], [])
    target_block = target_block or {}

    try:
        global_forbidden_vars = list_from(spec, "global_forbidden_variables")
        global_patterns = list_from(spec, "global_forbidden_patterns")
        target_defining_vars = list_from(target_block, "defining_variables")
        target_forbidden_vars = list_from(target_block, "forbidden_variables")
        target_patterns = list_from(target_block, "forbidden_patterns")
    except ValueError as exc:
        add_issue(
            failures,
            "invalid_definition_spec",
            "Definition spec fields have invalid type.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], [])

    forbidden_exact = (
        global_forbidden_vars + target_defining_vars + target_forbidden_vars
    )
    forbidden_patterns = global_patterns + target_patterns

    if args.strict and not forbidden_exact and not forbidden_patterns:
        add_issue(
            failures,
            "empty_forbidden_rules",
            "No forbidden variables or patterns defined for strict mode.",
            {"target": args.target},
        )

    compiled_patterns, regex_errors = compile_patterns(forbidden_patterns)
    for err in regex_errors:
        add_issue(failures, "invalid_forbidden_pattern", "Invalid forbidden regex.", {"error": err})

    ignore_cols = parse_comma_set(args.ignore_cols)
    ignore_cols.add(args.target_col)

    forbidden_exact_norm = {norm(x): x for x in forbidden_exact}
    checked_features = sorted([h for h in union_headers if h not in ignore_cols])

    if not checked_features:
        add_issue(
            warnings,
            "no_features_checked",
            "No predictor columns were checked after applying ignore columns.",
            {"ignored_columns": sorted(ignore_cols)},
        )

    exact_hits: List[Dict[str, str]] = []
    pattern_hits: List[Dict[str, str]] = []

    for feature in checked_features:
        feature_norm = norm(feature)
        if feature_norm in forbidden_exact_norm:
            exact_hits.append({"feature": feature, "matched_rule": forbidden_exact_norm[feature_norm]})
        for pattern in compiled_patterns:
            if pattern.search(feature):
                pattern_hits.append({"feature": feature, "matched_pattern": pattern.pattern})

    if exact_hits:
        add_issue(
            failures,
            "definition_variable_leakage",
            "Detected predictor columns that are explicitly forbidden by disease definition rules.",
            {"hits": exact_hits},
        )
    if pattern_hits:
        add_issue(
            failures,
            "definition_proxy_leakage",
            "Detected predictor columns matching forbidden proxy patterns.",
            {"hits": pattern_hits},
        )

    return finish(
        args,
        failures,
        warnings,
        headers_by_split,
        sorted(forbidden_exact),
        forbidden_patterns,
        checked_features,
    )


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    headers_by_split: Dict[str, List[str]],
    forbidden_exact: List[str],
    forbidden_patterns: List[str],
    checked_features: List[str],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "target": args.target,
        "definition_spec": str(Path(args.definition_spec).expanduser().resolve()),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "splits": {k: {"column_count": len(v), "columns": v} for k, v in headers_by_split.items()},
            "forbidden_exact_count": len(forbidden_exact),
            "forbidden_pattern_count": len(forbidden_patterns),
            "checked_feature_count": len(checked_features),
            "checked_features": checked_features,
        },
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
