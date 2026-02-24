#!/usr/bin/env python3
"""
Falsification gate using permutation-null metric distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether model metric beats permutation null.")
    parser.add_argument("--metric-name", required=True, help="Metric name, e.g. roc_auc.")
    parser.add_argument("--actual", type=float, required=True, help="Observed metric on real labels.")
    parser.add_argument(
        "--null-metrics-file",
        required=True,
        help="Path to null metrics file (.json list/dict or text with one numeric value per line).",
    )
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--higher-is-better", action="store_true", default=True)
    direction.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.01, help="One-sided significance threshold.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum gap between actual and null mean.")
    parser.add_argument(
        "--min-permutations",
        type=int,
        default=100,
        help="Minimum recommended null sample size.",
    )
    parser.add_argument("--report", help="Optional JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def parse_text_values(payload: str) -> List[float]:
    values: List[float] = []
    for line in payload.splitlines():
        s = line.strip()
        if not s:
            continue
        values.append(float(s))
    return values


def load_null_metrics(path: Path) -> List[float]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        if isinstance(parsed, dict):
            for key in ("metrics", "null_metrics", "values"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return [float(x) for x in value]
    except json.JSONDecodeError:
        pass

    return parse_text_values(raw)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "min": min(values), "max": max(values)}


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    null_path = Path(args.null_metrics_file).expanduser().resolve()
    if not null_path.exists():
        add_issue(
            failures,
            "missing_null_metrics_file",
            "Null metrics file not found.",
            {"path": str(null_path)},
        )
        return finish(args, failures, warnings, [])

    try:
        null_metrics = load_null_metrics(null_path)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_null_metrics_file",
            "Failed to parse null metrics file.",
            {"path": str(null_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, [])

    if not null_metrics:
        add_issue(
            failures,
            "empty_null_distribution",
            "Null metrics distribution is empty.",
            {"path": str(null_path)},
        )
        return finish(args, failures, warnings, [])

    if len(null_metrics) < args.min_permutations:
        add_issue(
            warnings,
            "low_permutation_count",
            "Permutation count below recommended minimum.",
            {"count": len(null_metrics), "minimum": args.min_permutations},
        )

    higher_is_better = not args.lower_is_better
    if higher_is_better:
        extreme = sum(1 for x in null_metrics if x >= args.actual)
        delta = args.actual - statistics.fmean(null_metrics)
    else:
        extreme = sum(1 for x in null_metrics if x <= args.actual)
        delta = statistics.fmean(null_metrics) - args.actual
    p_value = (extreme + 1.0) / (len(null_metrics) + 1.0)

    if p_value > args.alpha:
        add_issue(
            failures,
            "permutation_not_significant",
            "Observed metric is not significant against permutation null.",
            {"p_value": p_value, "alpha": args.alpha},
        )

    if delta < args.min_delta:
        add_issue(
            failures,
            "insufficient_effect_delta",
            "Observed metric does not exceed null mean by required delta.",
            {"delta": delta, "min_delta": args.min_delta},
        )

    return finish(args, failures, warnings, null_metrics, p_value=p_value, delta=delta)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    null_metrics: List[float],
    p_value: float = math.nan,
    delta: float = math.nan,
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    stats = summarize(null_metrics)
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "metric_name": args.metric_name,
        "higher_is_better": not args.lower_is_better,
        "actual": args.actual,
        "null_count": len(null_metrics),
        "null_summary": stats,
        "p_value_one_sided": p_value,
        "effect_delta_vs_null_mean": delta,
        "alpha": args.alpha,
        "min_delta": args.min_delta,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    print(
        f"Metric={args.metric_name} actual={args.actual:.6f} null_mean={stats['mean']:.6f} "
        f"delta={delta:.6f} p={p_value:.6g}"
    )
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
