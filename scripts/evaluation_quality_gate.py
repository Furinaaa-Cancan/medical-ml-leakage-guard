#!/usr/bin/env python3
"""
Fail-closed evaluation quality gate for publication-grade medical prediction.

Checks:
1. Primary metric is finite and consistent with optional expected value.
2. Confidence interval exists for primary metric and is valid.
3. Baseline comparison exists and primary metric improves over baseline.
4. No non-finite numeric values are present in evaluation artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _gate_utils import add_issue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate confidence intervals and baseline comparison in evaluation report.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation report JSON.")
    parser.add_argument("--ci-matrix-report", help="Optional CI matrix report JSON path.")
    parser.add_argument("--metric-name", required=True, help="Primary metric name (for example: roc_auc).")
    parser.add_argument("--metric-path", help="Optional dot path to primary metric value in evaluation report.")
    parser.add_argument("--primary-metric", type=float, help="Optional expected primary metric value.")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Absolute tolerance for primary metric comparison.")
    parser.add_argument("--min-resamples", type=int, default=200, help="Minimum required bootstrap/resampling count.")
    parser.add_argument("--min-baseline-delta", type=float, default=0.01, help="Minimum required margin over baseline.")
    parser.add_argument("--max-ci-width", type=float, default=0.20, help="Warning threshold for CI width.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def canonical_metric_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


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
    }
    return any(token in auxiliary_tokens for token in tokens)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def to_float(value: Any) -> Optional[float]:
    if is_finite_number(value):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        try:
            parsed = float(s)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def normalize_split_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_by_dot_path(payload: Dict[str, Any], path: str) -> Tuple[Optional[Any], str]:
    cur: Any = payload
    traversed: List[str] = []
    for part in path.split("."):
        key = part.strip()
        if not key:
            return None, ".".join(traversed)
        traversed.append(key)
        if not isinstance(cur, dict) or key not in cur:
            return None, ".".join(traversed)
        cur = cur[key]
    return cur, ".".join(traversed)


def extract_primary_metric(payload: Dict[str, Any], metric_name: str, metric_path: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if isinstance(metric_path, str) and metric_path.strip():
        value, resolved = get_by_dot_path(payload, metric_path.strip())
        return to_float(value), resolved

    candidates = [
        metric_name,
        f"metrics.{metric_name}",
        f"test.{metric_name}",
        f"results.{metric_name}",
        f"evaluation.{metric_name}",
        f"summary.{metric_name}",
    ]
    for path in candidates:
        value, resolved = get_by_dot_path(payload, path)
        numeric = to_float(value)
        if numeric is not None:
            return numeric, resolved
    return None, None


def find_non_finite_values(payload: Any) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                next_path = f"{path}.{key}" if path else key
                walk(value, next_path)
            return
        if isinstance(node, list):
            for idx, value in enumerate(node):
                next_path = f"{path}[{idx}]"
                walk(value, next_path)
            return

        if isinstance(node, float) and not math.isfinite(node):
            hits.append({"path": path, "value": node, "kind": "non_finite_float"})
            return
        if isinstance(node, str):
            token = node.strip().lower()
            if token in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                hits.append({"path": path, "value": node, "kind": "non_finite_string"})

    walk(payload, "")
    return hits


def parse_ci_block(raw: Any) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[int]]:
    lower: Optional[float] = None
    upper: Optional[float] = None
    method: Optional[str] = None
    n_resamples: Optional[int] = None

    if isinstance(raw, list) and len(raw) == 2:
        lower = to_float(raw[0])
        upper = to_float(raw[1])
        return lower, upper, None, None

    if not isinstance(raw, dict):
        return None, None, None, None

    method_raw = raw.get("method") or raw.get("ci_method")
    if isinstance(method_raw, str) and method_raw.strip():
        method = method_raw.strip()

    n_raw = raw.get("n_resamples")
    if n_raw is None:
        n_raw = raw.get("n_bootstrap")
    if n_raw is None:
        n_raw = raw.get("bootstrap_iterations")
    if isinstance(n_raw, bool):
        n_raw = None
    if isinstance(n_raw, int):
        n_resamples = int(n_raw)
    elif isinstance(n_raw, float) and math.isfinite(n_raw) and float(n_raw).is_integer():
        n_resamples = int(n_raw)

    if "ci_95" in raw:
        ci95 = raw.get("ci_95")
        if isinstance(ci95, list) and len(ci95) == 2:
            lower = to_float(ci95[0])
            upper = to_float(ci95[1])
            return lower, upper, method, n_resamples

    for lo_key, hi_key in (
        ("lower", "upper"),
        ("lo", "hi"),
        ("ci_lower", "ci_upper"),
        ("lower_95", "upper_95"),
    ):
        if lo_key in raw and hi_key in raw:
            lower = to_float(raw.get(lo_key))
            upper = to_float(raw.get(hi_key))
            return lower, upper, method, n_resamples

    return lower, upper, method, n_resamples


def extract_primary_metric_ci(payload: Dict[str, Any], metric_name: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[int], Optional[str]]:
    metric_token = canonical_metric_token(metric_name)
    candidates: List[Tuple[str, Any]] = []

    for path in (
        f"metrics_ci.{metric_name}",
        f"confidence_intervals.{metric_name}",
        f"uncertainty.metrics.{metric_name}",
    ):
        value, resolved = get_by_dot_path(payload, path)
        if value is not None:
            candidates.append((resolved, value))

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get(f"{metric_name}_ci")
        if value is not None:
            candidates.append((f"metrics.{metric_name}_ci", value))

    # Generic fallback scan for matching metric token.
    def walk(node: Any, path: str = "", parent_key: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                next_path = f"{path}.{key}" if path else key
                if is_auxiliary_metric_path(next_path):
                    continue
                if canonical_metric_token(key) == metric_token:
                    candidates.append((next_path, value))
                walk(value, next_path, key)
            return
        if isinstance(node, list):
            for idx, value in enumerate(node):
                walk(value, f"{path}[{idx}]")

    walk(payload)

    seen_paths = set()
    for path, raw in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        lower, upper, method, n_resamples = parse_ci_block(raw)
        if lower is not None and upper is not None:
            return lower, upper, method, n_resamples, path

    return None, None, None, None, None


def extract_primary_metric_ci_from_ci_matrix(
    ci_payload: Dict[str, Any],
    metric_name: str,
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[str]]:
    split_metrics_ci = ci_payload.get("split_metrics_ci")
    if not isinstance(split_metrics_ci, dict):
        return None, None, None, None
    test_block = split_metrics_ci.get("test")
    if not isinstance(test_block, dict):
        return None, None, None, None
    metrics = test_block.get("metrics")
    if not isinstance(metrics, dict):
        return None, None, None, None
    metric_block = metrics.get(metric_name)
    if not isinstance(metric_block, dict):
        return None, None, None, None
    ci_95 = metric_block.get("ci_95")
    if not isinstance(ci_95, list) or len(ci_95) != 2:
        return None, None, None, None
    lo = to_float(ci_95[0])
    hi = to_float(ci_95[1])
    n_resamples_raw = metric_block.get("n_resamples")
    n_resamples: Optional[int] = None
    if isinstance(n_resamples_raw, int):
        n_resamples = int(n_resamples_raw)
    elif isinstance(n_resamples_raw, float) and math.isfinite(n_resamples_raw) and float(n_resamples_raw).is_integer():
        n_resamples = int(n_resamples_raw)
    return lo, hi, n_resamples, f"split_metrics_ci.test.metrics.{metric_name}"


def infer_higher_is_better(metric_name: str) -> bool:
    token = canonical_metric_token(metric_name)
    lower_better_tokens = ("loss", "error", "brier", "rmse", "mae", "mse", "nll", "logloss")
    return not any(t in token for t in lower_better_tokens)


def extract_baseline_metrics(payload: Dict[str, Any], metric_name: str) -> Dict[str, float]:
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        return {}

    target_token = canonical_metric_token(metric_name)
    out: Dict[str, float] = {}

    def extract_metric_from_node(node: Any) -> Optional[float]:
        if isinstance(node, dict):
            metrics = node.get("metrics")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(key, str) and canonical_metric_token(key) == target_token:
                        numeric = to_float(value)
                        if numeric is not None:
                            return numeric
            for key, value in node.items():
                if isinstance(key, str) and canonical_metric_token(key) == target_token:
                    numeric = to_float(value)
                    if numeric is not None:
                        return numeric
        return None

    for baseline_name, baseline_payload in baselines.items():
        if not isinstance(baseline_name, str) or not baseline_name.strip():
            continue
        value = extract_metric_from_node(baseline_payload)
        if value is not None:
            out[baseline_name.strip()] = value
    return out


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if args.min_resamples < 1:
        add_issue(
            failures,
            "invalid_min_resamples",
            "min-resamples must be >= 1.",
            {"min_resamples": args.min_resamples},
        )
        return finish(args, failures, warnings, summary={})

    if not is_finite_number(args.tolerance) or float(args.tolerance) < 0.0:
        add_issue(
            failures,
            "invalid_tolerance",
            "tolerance must be finite and >= 0.",
            {"tolerance": args.tolerance},
        )
        return finish(args, failures, warnings, summary={})

    if not is_finite_number(args.min_baseline_delta) or float(args.min_baseline_delta) < 0.0:
        add_issue(
            failures,
            "invalid_min_baseline_delta",
            "min-baseline-delta must be finite and >= 0.",
            {"min_baseline_delta": args.min_baseline_delta},
        )
        return finish(args, failures, warnings, summary={})

    if not is_finite_number(args.max_ci_width) or float(args.max_ci_width) <= 0.0:
        add_issue(
            failures,
            "invalid_max_ci_width",
            "max-ci-width must be finite and > 0.",
            {"max_ci_width": args.max_ci_width},
        )
        return finish(args, failures, warnings, summary={})

    report_path = Path(args.evaluation_report).expanduser().resolve()
    if not report_path.exists():
        add_issue(
            failures,
            "missing_evaluation_report",
            "Evaluation report file not found.",
            {"path": str(report_path)},
        )
        return finish(args, failures, warnings, summary={"evaluation_report": str(report_path)})

    try:
        with report_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_evaluation_report",
            "Failed to parse evaluation report JSON.",
            {"path": str(report_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, summary={"evaluation_report": str(report_path)})

    ci_matrix_payload: Optional[Dict[str, Any]] = None
    ci_matrix_path: Optional[Path] = None
    if args.ci_matrix_report:
        ci_matrix_path = Path(args.ci_matrix_report).expanduser().resolve()
        if not ci_matrix_path.exists():
            add_issue(
                failures,
                "missing_ci_matrix_report",
                "CI matrix report file not found.",
                {"path": str(ci_matrix_path)},
            )
        else:
            try:
                with ci_matrix_path.open("r", encoding="utf-8") as fh:
                    ci_matrix_payload = json.load(fh)
                if not isinstance(ci_matrix_payload, dict):
                    raise ValueError("JSON root must be object.")
            except Exception as exc:
                add_issue(
                    failures,
                    "missing_ci_matrix_report",
                    "Failed to parse CI matrix report JSON.",
                    {"path": str(ci_matrix_path), "error": str(exc)},
                )
                ci_matrix_payload = None
            if isinstance(ci_matrix_payload, dict):
                status = str(ci_matrix_payload.get("status", "")).strip().lower()
                if status and status != "pass":
                    add_issue(
                        failures,
                        "ci_matrix_not_passed",
                        "CI matrix report status must be pass.",
                        {"status": ci_matrix_payload.get("status"), "path": str(ci_matrix_path)},
                    )

    non_finite_hits = find_non_finite_values(payload)
    if non_finite_hits:
        add_issue(
            failures,
            "non_finite_values_detected",
            "Evaluation report contains non-finite numeric values.",
            {"hits": non_finite_hits[:50], "hit_count": len(non_finite_hits)},
        )

    primary_metric, metric_source_path = extract_primary_metric(payload, args.metric_name, args.metric_path)
    if primary_metric is None:
        add_issue(
            failures,
            "primary_metric_not_found",
            "Primary metric is missing or non-numeric in evaluation report.",
            {
                "metric_name": args.metric_name,
                "metric_path": args.metric_path,
                "evaluation_report": str(report_path),
            },
        )
        return finish(
            args,
            failures,
            warnings,
            summary={"evaluation_report": str(report_path), "metric_source_path": metric_source_path},
        )

    if args.primary_metric is not None:
        if not is_finite_number(args.primary_metric):
            add_issue(
                failures,
                "invalid_primary_metric_expected",
                "primary-metric must be finite.",
                {"primary_metric": args.primary_metric},
            )
        elif abs(primary_metric - float(args.primary_metric)) > float(args.tolerance):
            add_issue(
                failures,
                "primary_metric_mismatch",
                "Primary metric does not match expected metric value.",
                {
                    "metric_name": args.metric_name,
                    "expected": float(args.primary_metric),
                    "actual": primary_metric,
                    "tolerance": float(args.tolerance),
                },
            )

    ci_lower, ci_upper, ci_method, ci_n_resamples, ci_source_path = extract_primary_metric_ci(payload, args.metric_name)
    if (ci_lower is None or ci_upper is None) and isinstance(ci_matrix_payload, dict):
        ci_m_lo, ci_m_hi, ci_m_n, ci_m_src = extract_primary_metric_ci_from_ci_matrix(ci_matrix_payload, args.metric_name)
        if ci_m_lo is not None and ci_m_hi is not None:
            ci_lower = ci_m_lo
            ci_upper = ci_m_hi
            ci_n_resamples = ci_m_n
            ci_method = "bootstrap_ci_matrix"
            ci_source_path = ci_m_src
    if ci_lower is None or ci_upper is None:
        add_issue(
            failures,
            "missing_primary_metric_ci",
            "Primary metric confidence interval is missing.",
            {"metric_name": args.metric_name},
        )
    else:
        if ci_lower > ci_upper:
            add_issue(
                failures,
                "invalid_primary_metric_ci_bounds",
                "Confidence interval lower bound must be <= upper bound.",
                {"ci_lower": ci_lower, "ci_upper": ci_upper},
            )
        if not (ci_lower <= primary_metric <= ci_upper):
            add_issue(
                failures,
                "primary_metric_outside_ci",
                "Primary metric must lie within its reported confidence interval.",
                {"primary_metric": primary_metric, "ci_lower": ci_lower, "ci_upper": ci_upper},
            )

        ci_width = ci_upper - ci_lower
        if ci_width > float(args.max_ci_width):
            add_issue(
                failures,
                "ci_width_exceeds_threshold",
                "Primary metric confidence interval width exceeds threshold.",
                {"ci_width": ci_width, "max_ci_width": float(args.max_ci_width)},
            )

        if ci_method is None or not isinstance(ci_method, str) or not ci_method.strip():
            add_issue(
                failures,
                "missing_ci_method",
                "Confidence interval method must be declared.",
                {"metric_name": args.metric_name, "ci_source_path": ci_source_path},
            )
        if ci_n_resamples is None or ci_n_resamples < int(args.min_resamples):
            add_issue(
                failures,
                "insufficient_ci_resamples",
                "Confidence interval resample count is below required minimum.",
                {
                    "metric_name": args.metric_name,
                    "ci_n_resamples": ci_n_resamples,
                    "min_resamples": int(args.min_resamples),
                },
            )

    baseline_metrics = extract_baseline_metrics(payload, args.metric_name)
    baseline_name: Optional[str] = None
    baseline_value: Optional[float] = None
    baseline_delta: Optional[float] = None
    higher_is_better: Optional[bool] = None
    baseline_selection_strategy = "best_available_baseline"
    if not baseline_metrics:
        add_issue(
            failures,
            "missing_baseline_metrics",
            "Evaluation report must include baseline metrics for the primary metric.",
            {"metric_name": args.metric_name},
        )
    else:
        higher_is_better = infer_higher_is_better(args.metric_name)
        selected_model_id = str(payload.get("model_id", "")).strip().lower()
        if selected_model_id.startswith("logistic") and "prevalence_model" in baseline_metrics:
            baseline_name = "prevalence_model"
            baseline_value = float(baseline_metrics["prevalence_model"])
            baseline_selection_strategy = "family_aware_prevalence_for_logistic_model"
        else:
            baseline_items = sorted(baseline_metrics.items(), key=lambda x: x[1], reverse=higher_is_better)
            baseline_name, baseline_value = baseline_items[0]

        if higher_is_better:
            baseline_delta = primary_metric - float(baseline_value)
        else:
            baseline_delta = float(baseline_value) - primary_metric

        if baseline_delta < float(args.min_baseline_delta):
            add_issue(
                failures,
                "baseline_improvement_insufficient",
                "Primary metric improvement over baseline is below required margin.",
                {
                    "metric_name": args.metric_name,
                    "higher_is_better": higher_is_better,
                    "primary_metric": primary_metric,
                    "baseline_name": baseline_name,
                    "baseline_metric": baseline_value,
                    "delta": baseline_delta,
                    "min_baseline_delta": float(args.min_baseline_delta),
                    "baseline_selection_strategy": baseline_selection_strategy,
                },
            )

    summary = {
        "evaluation_report": str(report_path),
        "ci_matrix_report": str(ci_matrix_path) if ci_matrix_path else None,
        "metric_name": args.metric_name,
        "metric_source_path": metric_source_path,
        "primary_metric": primary_metric,
        "ci_source_path": ci_source_path,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": ci_method,
        "ci_n_resamples": ci_n_resamples,
        "baseline_metrics": baseline_metrics,
        "higher_is_better": higher_is_better,
        "reference_baseline_name": baseline_name,
        "reference_baseline_metric": baseline_value,
        "baseline_delta": baseline_delta,
        "baseline_selection_strategy": baseline_selection_strategy,
        "min_baseline_delta": float(args.min_baseline_delta),
        "non_finite_hit_count": len(non_finite_hits),
    }
    return finish(args, failures, warnings, summary=summary)


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
