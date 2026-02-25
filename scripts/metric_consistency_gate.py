#!/usr/bin/env python3
"""
Metric consistency gate to prevent manual metric injection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and validate primary metric from evaluation report JSON.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation report JSON.")
    parser.add_argument("--metric-name", required=True, help="Primary metric name, e.g. roc_auc.")
    parser.add_argument(
        "--metric-path",
        help="Optional dot path to metric in evaluation report, e.g. metrics.roc_auc.",
    )
    parser.add_argument(
        "--expected",
        type=float,
        help="Optional expected metric value from request contract.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance for expected metric comparison.",
    )
    parser.add_argument(
        "--required-evaluation-split",
        help="Required split declaration in evaluation report (for example: test).",
    )
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def canonical_metric_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_by_dot_path(payload: Dict[str, Any], path: str) -> Tuple[Optional[Any], str]:
    cur: Any = payload
    traversed = []
    for part in path.split("."):
        key = part.strip()
        if not key:
            return None, ".".join(traversed)
        traversed.append(key)
        if not isinstance(cur, dict) or key not in cur:
            return None, ".".join(traversed)
        cur = cur[key]
    return cur, ".".join(traversed)


def to_float(value: Any) -> Optional[float]:
    if is_finite_number(value):
        return float(value)
    if isinstance(value, (int, float)):  # catches +/-inf and nan
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = float(s)
            if math.isfinite(parsed):
                return parsed
            return None
        except ValueError:
            return None
    return None


def collect_candidate_metrics(payload: Dict[str, Any], metric_name: str) -> List[Tuple[str, float, Any]]:
    hits: List[Tuple[str, float, Any]] = []
    candidate_paths = [
        metric_name,
        f"metrics.{metric_name}",
        f"test.{metric_name}",
        f"results.{metric_name}",
        f"evaluation.{metric_name}",
        f"summary.{metric_name}",
        f"final.{metric_name}",
    ]
    for path in candidate_paths:
        value, resolved = get_by_dot_path(payload, path)
        numeric = to_float(value)
        if numeric is not None:
            hits.append((resolved, numeric, value))
    return hits


def collect_metric_leaf_hits(payload: Any, metric_name: str) -> List[Dict[str, Any]]:
    target_token = canonical_metric_token(metric_name)
    hits: List[Dict[str, Any]] = []

    def walk(node: Any, path: str = "", leaf_key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                next_path = f"{path}.{key}" if path else key
                walk(value, next_path, key)
            return
        if isinstance(node, list):
            for idx, value in enumerate(node):
                next_path = f"{path}[{idx}]"
                walk(value, next_path, leaf_key)
            return

        if leaf_key is None:
            return
        if canonical_metric_token(leaf_key) != target_token:
            return
        numeric = to_float(node)
        if numeric is None:
            return
        hits.append({"path": path, "value": numeric, "raw": node})

    walk(payload)
    dedup: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for hit in hits:
        dedup[(str(hit["path"]), float(hit["value"]))] = hit
    return sorted(dedup.values(), key=lambda item: str(item["path"]))


def path_tokens(path: str) -> List[str]:
    normalized = re.sub(r"\[\d+\]", "", path)
    return [token for token in normalized.split(".") if token]


def is_auxiliary_metric_path(path: str) -> bool:
    tokens = [token.lower() for token in path_tokens(path)]
    if not tokens:
        return False

    auxiliary_tokens = {
        "baseline",
        "baselines",
        "confidence_intervals",
        "metrics_ci",
        "null_distribution",
        "null_metrics",
        "uncertainty",
        "threshold_selection",
        "metadata",
    }
    return any(token in auxiliary_tokens for token in tokens)


def is_allowed_primary_metric_path(path: str, required_split: Optional[str]) -> bool:
    if is_auxiliary_metric_path(path):
        return False

    tokens = [normalize_split_token(token) for token in path_tokens(path)]
    if "splitmetrics" in tokens:
        if not required_split:
            return False
        return normalize_split_token(required_split) in tokens
    return True


def has_conflicting_values(hits: List[Dict[str, Any]], tolerance: float = 1e-12) -> bool:
    observed: List[float] = []
    for hit in hits:
        value = hit.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        v = float(value)
        if not observed:
            observed.append(v)
            continue
        if all(abs(v - ref) > tolerance for ref in observed):
            return True
    return False


def normalize_split_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def extract_declared_split(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for key in ("split", "evaluation_split", "dataset_split"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return key, raw.strip()
        if raw is not None and not isinstance(raw, str):
            return key, None

    meta = payload.get("meta")
    if isinstance(meta, dict):
        for key in ("split", "evaluation_split", "dataset_split"):
            raw = meta.get(key)
            if isinstance(raw, str) and raw.strip():
                return f"meta.{key}", raw.strip()
            if raw is not None and not isinstance(raw, str):
                return f"meta.{key}", None
    return None, None


def extract_metric(
    payload: Dict[str, Any], metric_name: str, metric_path: Optional[str]
) -> Tuple[Optional[float], Optional[str], Optional[Any], List[Dict[str, Any]], bool]:
    if metric_path:
        value, resolved = get_by_dot_path(payload, metric_path)
        numeric = to_float(value)
        return (
            numeric,
            resolved,
            value,
            [{"path": resolved, "value": numeric}] if numeric is not None else [],
            False,
        )

    candidates = collect_candidate_metrics(payload, metric_name)
    if not candidates:
        return None, None, None, [], False

    first_path, first_numeric, first_raw = candidates[0]
    ambiguity = any(abs(v - first_numeric) > 1e-12 for _, v, _ in candidates[1:])
    serialized_candidates = [{"path": path, "value": value} for path, value, _ in candidates]
    return first_numeric, first_path, first_raw, serialized_candidates, ambiguity


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if args.expected is not None and not is_finite_number(args.expected):
        add_issue(
            failures,
            "invalid_expected_metric",
            "Expected metric must be a finite number.",
            {"expected": args.expected},
        )
        return finish(args, failures, warnings, actual_metric=None, metric_source_path=None)

    if not is_finite_number(args.tolerance) or float(args.tolerance) < 0.0:
        add_issue(
            failures,
            "invalid_tolerance",
            "Tolerance must be a finite non-negative number.",
            {"tolerance": args.tolerance},
        )
        return finish(args, failures, warnings, actual_metric=None, metric_source_path=None)

    if args.metric_path:
        leaf = args.metric_path.split(".")[-1].strip()
        if canonical_metric_token(leaf) != canonical_metric_token(args.metric_name):
            add_issue(
                failures,
                "metric_path_metric_mismatch",
                "Metric path leaf must match metric-name.",
                {
                    "metric_name": args.metric_name,
                    "metric_path": args.metric_path,
                    "metric_leaf": leaf,
                },
            )
            return finish(args, failures, warnings, actual_metric=None, metric_source_path=None)

    report_path = Path(args.evaluation_report).expanduser().resolve()
    if not report_path.exists():
        add_issue(
            failures,
            "missing_evaluation_report",
            "Evaluation report file not found.",
            {"path": str(report_path)},
        )
        return finish(args, failures, warnings, actual_metric=None, metric_source_path=None)

    try:
        with report_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            "Unable to parse evaluation report JSON.",
            {"path": str(report_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, actual_metric=None, metric_source_path=None)

    actual, source_path, raw_value, candidates, ambiguous = extract_metric(payload, args.metric_name, args.metric_path)
    leaf_metric_hits = collect_metric_leaf_hits(payload, args.metric_name)
    primary_leaf_hits = [
        hit
        for hit in leaf_metric_hits
        if is_allowed_primary_metric_path(str(hit.get("path", "")), args.required_evaluation_split)
    ]

    if args.required_evaluation_split:
        split_key, declared_split = extract_declared_split(payload)
        if split_key is None:
            add_issue(
                failures,
                "missing_evaluation_split",
                "Evaluation report must declare which split the metric comes from.",
                {"required_split": args.required_evaluation_split},
            )
        elif declared_split is None:
            add_issue(
                failures,
                "invalid_evaluation_split",
                "Declared evaluation split must be a non-empty string.",
                {"split_field": split_key},
            )
        elif normalize_split_token(declared_split) != normalize_split_token(args.required_evaluation_split):
            add_issue(
                failures,
                "evaluation_split_mismatch",
                "Evaluation split does not match required split.",
                {
                    "required_split": args.required_evaluation_split,
                    "declared_split": declared_split,
                    "split_field": split_key,
                },
            )

    if actual is None:
        add_issue(
            failures,
            "metric_not_found",
            "Primary metric not found or non-numeric in evaluation report.",
            {
                "metric_name": args.metric_name,
                "metric_path": args.metric_path,
                "evaluation_report": str(report_path),
            },
        )
        return finish(args, failures, warnings, actual_metric=None, metric_source_path=source_path)

    if ambiguous:
        add_issue(
            failures,
            "ambiguous_metric_sources",
            "Multiple metric sources found with inconsistent values; provide --metric-path.",
            {"metric_name": args.metric_name, "candidates": candidates},
        )
    elif has_conflicting_values(primary_leaf_hits):
        add_issue(
            failures,
            "ambiguous_metric_sources",
            "Primary metric appears in multiple non-auxiliary locations with inconsistent numeric values.",
            {"metric_name": args.metric_name, "leaf_metric_hits": primary_leaf_hits},
        )

    if args.expected is not None:
        diff = abs(actual - args.expected)
        if diff > args.tolerance:
            add_issue(
                failures,
                "metric_mismatch",
                "Expected metric does not match evaluation report metric.",
                {
                    "expected": args.expected,
                    "actual": actual,
                    "abs_diff": diff,
                    "tolerance": args.tolerance,
                    "source_path": source_path,
                },
            )
    else:
        add_issue(
            warnings,
            "missing_expected_metric",
            "No expected metric provided; only extraction was validated.",
            {"actual": actual, "source_path": source_path},
        )

    return finish(
        args,
        failures,
        warnings,
        actual_metric=actual,
        metric_source_path=source_path,
        raw_value=raw_value,
        candidate_metrics=candidates,
        leaf_metric_hits=leaf_metric_hits,
        primary_leaf_metric_hits=primary_leaf_hits,
    )


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    actual_metric: Optional[float],
    metric_source_path: Optional[str],
    raw_value: Any = None,
    candidate_metrics: Optional[List[Dict[str, Any]]] = None,
    leaf_metric_hits: Optional[List[Dict[str, Any]]] = None,
    primary_leaf_metric_hits: Optional[List[Dict[str, Any]]] = None,
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "metric_name": args.metric_name,
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
        "metric_source_path": metric_source_path,
        "candidate_metrics": candidate_metrics or [],
        "leaf_metric_hits": leaf_metric_hits or [],
        "primary_leaf_metric_hits": primary_leaf_metric_hits or [],
        "actual_metric": actual_metric,
        "raw_metric_value": raw_value,
        "expected_metric": args.expected,
        "tolerance": args.tolerance,
        "required_evaluation_split": args.required_evaluation_split,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }

    if args.report:
        out = Path(args.report).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    if actual_metric is not None:
        print(f"Metric {args.metric_name} = {actual_metric} (source={metric_source_path})")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
