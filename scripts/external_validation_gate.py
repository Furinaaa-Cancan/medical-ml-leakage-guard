#!/usr/bin/env python3
"""
Fail-closed external validation gate.

Validates external cohort transport performance and replays cohort metrics from
row-level prediction trace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from _gate_utils import add_issue, load_json_from_str as load_json_obj, to_float


REQUIRED_METRICS = [
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

SUPPORTED_EXTERNAL_TYPES = {"cross_period", "cross_institution"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external cohort report with replayed trace metrics.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external_validation_report.json.")
    parser.add_argument("--prediction-trace", required=True, help="Path to prediction_trace CSV/CSV.GZ.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--report", help="Optional output report JSON path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metric_panel(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, beta: float) -> Tuple[Dict[str, float], Dict[str, int]]:
    cm = confusion_counts(y_true, y_pred)
    tp = float(cm["tp"])
    fp = float(cm["fp"])
    tn = float(cm["tn"])
    fn = float(cm["fn"])
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    npv = safe_ratio(tn, tn + fn)
    accuracy = safe_ratio(tp + tn, tp + fp + tn + fn)
    f1 = 0.0 if (precision + sensitivity) <= 0 else (2.0 * precision * sensitivity) / (precision + sensitivity)
    beta_sq = beta * beta
    f2 = 0.0 if ((beta_sq * precision) + sensitivity) <= 0 else ((1.0 + beta_sq) * precision * sensitivity) / (
        (beta_sq * precision) + sensitivity
    )
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    return (
        {
            "accuracy": accuracy,
            "precision": precision,
            "ppv": precision,
            "npv": npv,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "f2_beta": f2,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier,
        },
        cm,
    )


def normalize_binary(values: pd.Series) -> Optional[np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(arr)):
        return None
    if not np.all(np.isin(arr, [0.0, 1.0])):
        return None
    return arr.astype(int)


def parse_thresholds(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "metric_tolerance": 1e-6,
        "beta": 2.0,
        "min_cohort_count": 1.0,
        "min_rows_per_cohort": 20.0,
        "min_positive_per_cohort": 3.0,
        "max_pr_auc_drop": 0.08,
        "max_f2_beta_drop": 0.10,
        "max_brier_increase": 0.05,
        "require_cross_period": True,
        "require_cross_institution": True,
    }
    if not isinstance(policy, dict):
        return out
    block = policy.get("external_validation_thresholds")
    if not isinstance(block, dict):
        return out
    for key in out:
        if key in {"require_cross_period", "require_cross_institution"}:
            raw_bool = block.get(key)
            if isinstance(raw_bool, bool):
                out[key] = bool(raw_bool)
            continue
        raw = to_float(block.get(key))
        if raw is None:
            continue
        if key in {"min_cohort_count", "min_rows_per_cohort", "min_positive_per_cohort"}:
            if raw >= 1:
                out[key] = float(raw)
        elif key == "beta":
            if raw > 0.0:
                out[key] = float(raw)
        elif raw >= 0.0:
            out[key] = float(raw)
    return out


def compare_metric(
    cohort_id: str,
    metric_name: str,
    observed: float,
    expected: Any,
    tolerance: float,
    failures: List[Dict[str, Any]],
) -> None:
    expected_f = to_float(expected)
    if expected_f is None:
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "External report metric is missing or non-numeric.",
            {"cohort_id": cohort_id, "metric": metric_name, "expected": expected},
        )
        return
    if abs(float(observed) - float(expected_f)) > float(tolerance):
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "Replayed external metric does not match external_validation_report metric.",
            {
                "cohort_id": cohort_id,
                "metric": metric_name,
                "observed": float(observed),
                "expected": float(expected_f),
                "tolerance": float(tolerance),
            },
        )


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    ext_path = Path(args.external_validation_report).expanduser().resolve()
    if not ext_path.exists():
        add_issue(
            failures,
            "external_validation_missing",
            "external_validation_report file does not exist.",
            {"path": str(ext_path)},
        )
        return finish(args, failures, warnings, {})

    trace_path = Path(args.prediction_trace).expanduser().resolve()
    if not trace_path.exists():
        add_issue(
            failures,
            "external_validation_missing",
            "prediction_trace file does not exist.",
            {"path": str(trace_path)},
        )
        return finish(args, failures, warnings, {})

    eval_path = Path(args.evaluation_report).expanduser().resolve()
    if not eval_path.exists():
        add_issue(
            failures,
            "external_validation_missing",
            "evaluation_report file does not exist.",
            {"path": str(eval_path)},
        )
        return finish(args, failures, warnings, {})

    try:
        ext_report = load_json_obj(str(ext_path))
    except Exception as exc:
        add_issue(
            failures,
            "external_validation_missing",
            "Unable to parse external_validation_report JSON.",
            {"path": str(ext_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        evaluation = load_json_obj(str(eval_path))
    except Exception as exc:
        add_issue(
            failures,
            "external_validation_missing",
            "Unable to parse evaluation_report JSON.",
            {"path": str(eval_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    policy: Optional[Dict[str, Any]] = None
    if args.performance_policy:
        try:
            policy = load_json_obj(args.performance_policy)
        except Exception as exc:
            add_issue(
                failures,
                "external_validation_missing",
                "Unable to parse performance_policy JSON.",
                {"path": str(Path(args.performance_policy).expanduser()), "error": str(exc)},
            )
            return finish(args, failures, warnings, {})
    thresholds = parse_thresholds(policy)
    metric_tol = float(thresholds["metric_tolerance"])
    beta = float(thresholds["beta"])

    cohorts = ext_report.get("cohorts")
    if not isinstance(cohorts, list):
        cohorts = []
    if len(cohorts) < int(thresholds["min_cohort_count"]):
        add_issue(
            failures,
            "external_validation_min_cohort_not_met",
            "External validation requires at least one cohort for publication-grade claims.",
            {
                "cohort_count": len(cohorts),
                "min_cohort_count": int(thresholds["min_cohort_count"]),
            },
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    try:
        trace_df = pd.read_csv(trace_path)
    except Exception as exc:
        add_issue(
            failures,
            "external_validation_missing",
            "Unable to read prediction_trace file.",
            {"path": str(trace_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    required_trace_cols = {"scope", "cohort_id", "cohort_type", "y_true", "y_score", "y_pred", "selected_threshold"}
    if not required_trace_cols.issubset(set(trace_df.columns)):
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "prediction_trace missing columns required for external replay validation.",
            {"missing_columns": sorted(required_trace_cols - set(trace_df.columns))},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    trace_df = trace_df.copy()
    trace_df["scope"] = trace_df["scope"].astype(str).str.strip().str.lower()
    trace_df["cohort_id"] = trace_df["cohort_id"].astype(str).str.strip()
    trace_df["cohort_type"] = trace_df["cohort_type"].astype(str).str.strip().str.lower()
    trace_df["y_score"] = pd.to_numeric(trace_df["y_score"], errors="coerce")
    trace_df["selected_threshold"] = pd.to_numeric(trace_df["selected_threshold"], errors="coerce")
    y_true_all = normalize_binary(trace_df["y_true"])
    y_pred_all = normalize_binary(trace_df["y_pred"])
    if y_true_all is None or y_pred_all is None:
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "prediction_trace y_true/y_pred must be binary 0/1.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})
    trace_df["y_true"] = y_true_all
    trace_df["y_pred"] = y_pred_all

    if np.any(~np.isfinite(trace_df["y_score"].to_numpy(dtype=float))):
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "prediction_trace y_score contains non-finite values.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})
    if np.any(trace_df["y_score"].to_numpy(dtype=float) < 0.0) or np.any(trace_df["y_score"].to_numpy(dtype=float) > 1.0):
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "prediction_trace y_score must be within [0,1].",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    eval_metrics = evaluation.get("metrics")
    if not isinstance(eval_metrics, dict):
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "evaluation_report.metrics is required for transport-gap checks.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    internal_pr_auc = to_float(eval_metrics.get("pr_auc"))
    internal_f2 = to_float(eval_metrics.get("f2_beta"))
    internal_brier = to_float(eval_metrics.get("brier"))
    if internal_pr_auc is None or internal_f2 is None or internal_brier is None:
        add_issue(
            failures,
            "external_validation_metric_replay_mismatch",
            "evaluation_report.metrics must include pr_auc/f2_beta/brier.",
            {
                "pr_auc": eval_metrics.get("pr_auc"),
                "f2_beta": eval_metrics.get("f2_beta"),
                "brier": eval_metrics.get("brier"),
            },
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    observed_types: set[str] = set()
    passed_types: set[str] = set()
    replayed: List[Dict[str, Any]] = []
    for idx, cohort in enumerate(cohorts):
        if not isinstance(cohort, dict):
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "Each external cohort entry must be an object.",
                {"index": idx},
            )
            continue

        cohort_id = str(cohort.get("cohort_id", "")).strip()
        cohort_type = str(cohort.get("cohort_type", "")).strip().lower()
        failure_count_before = len(failures)
        if not cohort_id:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "external_validation_report cohort entry is missing cohort_id.",
                {"index": idx},
            )
            continue
        if cohort_type:
            observed_types.add(cohort_type)

        cohort_rows = trace_df[(trace_df["scope"] == "external") & (trace_df["cohort_id"] == cohort_id)]
        if cohort_rows.empty:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "prediction_trace does not contain rows for external cohort.",
                {"cohort_id": cohort_id},
            )
            continue

        threshold_values = np.unique(cohort_rows["selected_threshold"].to_numpy(dtype=float))
        if threshold_values.size == 0:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "External cohort rows must include selected_threshold.",
                {"cohort_id": cohort_id},
            )
            continue
        if float(np.max(threshold_values)) - float(np.min(threshold_values)) > 1e-9:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "selected_threshold must be stable within an external cohort.",
                {
                    "cohort_id": cohort_id,
                    "min_selected_threshold": float(np.min(threshold_values)),
                    "max_selected_threshold": float(np.max(threshold_values)),
                },
            )
            continue
        threshold_value = float(threshold_values[0])

        y_true = cohort_rows["y_true"].to_numpy(dtype=int)
        y_score = cohort_rows["y_score"].to_numpy(dtype=float)
        y_pred_reported = cohort_rows["y_pred"].to_numpy(dtype=int)
        y_pred_replayed = (y_score >= threshold_value).astype(int)
        if np.any(y_pred_reported != y_pred_replayed):
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "External cohort y_pred must equal (y_score >= selected_threshold).",
                {"cohort_id": cohort_id},
            )
            continue

        n_rows = int(y_true.shape[0])
        n_pos = int(np.sum(y_true == 1))
        if n_rows < int(thresholds["min_rows_per_cohort"]) or n_pos < int(thresholds["min_positive_per_cohort"]):
            add_issue(
                failures,
                "external_validation_insufficient_events",
                "External cohort does not satisfy minimum sample/event requirements.",
                {
                    "cohort_id": cohort_id,
                    "row_count": n_rows,
                    "positive_count": n_pos,
                    "min_rows_per_cohort": int(thresholds["min_rows_per_cohort"]),
                    "min_positive_per_cohort": int(thresholds["min_positive_per_cohort"]),
                },
            )
            continue
        if len(np.unique(y_true)) < 2:
            add_issue(
                failures,
                "external_validation_insufficient_events",
                "External cohort must contain both classes for replay metrics.",
                {"cohort_id": cohort_id},
            )
            continue

        replay_metrics, replay_cm = metric_panel(y_true, y_score, y_pred_replayed, beta=beta)

        report_metrics = cohort.get("metrics")
        if not isinstance(report_metrics, dict):
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "External cohort metrics block is missing.",
                {"cohort_id": cohort_id},
            )
            continue
        for metric_name in REQUIRED_METRICS:
            compare_metric(
                cohort_id=cohort_id,
                metric_name=metric_name,
                observed=float(replay_metrics[metric_name]),
                expected=report_metrics.get(metric_name),
                tolerance=metric_tol,
                failures=failures,
            )

        report_cm = cohort.get("confusion_matrix")
        if isinstance(report_cm, dict):
            for key in ("tp", "fp", "tn", "fn"):
                expected_i = to_int(report_cm.get(key))
                observed_i = int(replay_cm[key])
                if expected_i is not None and expected_i != observed_i:
                    add_issue(
                        failures,
                        "external_validation_metric_replay_mismatch",
                        "External cohort confusion matrix does not match replayed values.",
                        {"cohort_id": cohort_id, "field": key, "observed": observed_i, "expected": expected_i},
                    )

        report_rows = to_int(cohort.get("row_count"))
        report_pos = to_int(cohort.get("positive_count"))
        if report_rows is not None and report_rows != n_rows:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "External cohort row_count does not match replayed row count.",
                {"cohort_id": cohort_id, "observed": n_rows, "expected": report_rows},
            )
        if report_pos is not None and report_pos != n_pos:
            add_issue(
                failures,
                "external_validation_metric_replay_mismatch",
                "External cohort positive_count does not match replayed value.",
                {"cohort_id": cohort_id, "observed": n_pos, "expected": report_pos},
            )

        pr_auc_drop = float(internal_pr_auc - float(replay_metrics["pr_auc"]))
        f2_drop = float(internal_f2 - float(replay_metrics["f2_beta"]))
        brier_increase = float(float(replay_metrics["brier"]) - internal_brier)
        if (
            pr_auc_drop > float(thresholds["max_pr_auc_drop"])
            or f2_drop > float(thresholds["max_f2_beta_drop"])
            or brier_increase > float(thresholds["max_brier_increase"])
        ):
            add_issue(
                failures,
                "external_validation_transport_drop_exceeds_threshold",
                "External cohort transport degradation exceeds policy threshold.",
                {
                    "cohort_id": cohort_id,
                    "pr_auc_drop": pr_auc_drop,
                    "f2_beta_drop": f2_drop,
                    "brier_increase": brier_increase,
                    "max_pr_auc_drop": float(thresholds["max_pr_auc_drop"]),
                    "max_f2_beta_drop": float(thresholds["max_f2_beta_drop"]),
                    "max_brier_increase": float(thresholds["max_brier_increase"]),
                },
            )

        if len(failures) == failure_count_before and cohort_type in SUPPORTED_EXTERNAL_TYPES:
            passed_types.add(cohort_type)

        replayed.append(
            {
                "cohort_id": cohort_id,
                "cohort_type": cohort_type,
                "row_count": n_rows,
                "positive_count": n_pos,
                "selected_threshold": threshold_value,
                "metrics": replay_metrics,
                "confusion_matrix": replay_cm,
                "transport_gap": {
                    "pr_auc_drop_from_internal_test": pr_auc_drop,
                    "f2_beta_drop_from_internal_test": f2_drop,
                    "brier_increase_from_internal_test": brier_increase,
                },
            }
        )

    if not (observed_types & SUPPORTED_EXTERNAL_TYPES):
        add_issue(
            failures,
            "external_validation_type_coverage_not_met",
            "At least one supported external cohort type must be present (cross_period or cross_institution).",
            {"observed_types": sorted(observed_types), "required_any_of": sorted(SUPPORTED_EXTERNAL_TYPES)},
        )

    if bool(thresholds.get("require_cross_period", True)) and "cross_period" not in passed_types:
        add_issue(
            failures,
            "external_validation_cross_period_not_met",
            "Publication-grade external validation requires at least one passing cross_period cohort.",
            {"passed_types": sorted(passed_types), "observed_types": sorted(observed_types)},
        )
    if bool(thresholds.get("require_cross_institution", True)) and "cross_institution" not in passed_types:
        add_issue(
            failures,
            "external_validation_cross_institution_not_met",
            "Publication-grade external validation requires at least one passing cross_institution cohort.",
            {"passed_types": sorted(passed_types), "observed_types": sorted(observed_types)},
        )

    summary = {
        "external_validation_report": str(ext_path),
        "prediction_trace": str(trace_path),
        "evaluation_report": str(eval_path),
        "thresholds": thresholds,
        "cohort_count_reported": len(cohorts),
        "cohort_count_replayed": len(replayed),
        "observed_external_types": sorted(observed_types),
        "passed_external_types": sorted(passed_types),
        "replayed_cohorts": replayed,
    }
    return finish(args, failures, warnings, summary)


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
        from _gate_utils import write_json as _write_report
        _write_report(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
