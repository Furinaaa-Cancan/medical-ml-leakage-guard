#!/usr/bin/env python3
"""
Fail-closed prediction replay gate.

Recomputes metrics from row-level prediction traces and verifies alignment with
evaluation_report metrics and split fingerprints.
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


REQUIRED_TRACE_COLUMNS = [
    "scope",
    "cohort_id",
    "cohort_type",
    "hashed_patient_id",
    "y_true",
    "y_score",
    "y_pred",
    "selected_threshold",
    "model_id",
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay prediction metrics from row-level trace.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json.")
    parser.add_argument("--prediction-trace", required=True, help="Path to prediction_trace CSV/CSV.GZ.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--report", help="Optional output report JSON.")
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
    metrics = {
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
    }
    return metrics, cm


def parse_thresholds(policy: Optional[Dict[str, Any]]) -> Dict[str, float]:
    defaults = {
        "metric_tolerance": 1e-6,
        "threshold_tolerance": 1e-9,
        "beta": 2.0,
    }
    if not isinstance(policy, dict):
        return defaults
    block = policy.get("prediction_replay_thresholds")
    if not isinstance(block, dict):
        return defaults
    for key in ("metric_tolerance", "threshold_tolerance", "beta"):
        raw = to_float(block.get(key))
        if raw is not None and raw > 0.0:
            defaults[key] = float(raw)
    return defaults


def load_trace(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def normalize_binary(series: pd.Series) -> Optional[np.ndarray]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(arr)):
        return None
    if not np.all(np.isin(arr, [0.0, 1.0])):
        return None
    return arr.astype(int)


def require_metric_block(
    split_metrics: Dict[str, Any],
    split_name: str,
    failures: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    block = split_metrics.get(split_name)
    if not isinstance(block, dict):
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "evaluation_report.split_metrics is missing required split block.",
            {"split": split_name},
        )
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "evaluation_report.split_metrics.<split>.metrics must be an object.",
            {"split": split_name},
        )
        return None
    return block


def compare_metric(
    split_name: str,
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
            "prediction_metric_replay_mismatch",
            "Expected metric in evaluation report is missing or non-numeric.",
            {"split": split_name, "metric": metric_name, "expected": expected},
        )
        return
    if abs(float(observed) - float(expected_f)) > float(tolerance):
        add_issue(
            failures,
            "prediction_metric_replay_mismatch",
            "Replayed metric does not match evaluation report metric.",
            {
                "split": split_name,
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

    eval_path = Path(args.evaluation_report).expanduser().resolve()
    if not eval_path.exists():
        add_issue(
            failures,
            "prediction_trace_missing",
            "evaluation_report file does not exist.",
            {"path": str(eval_path)},
        )
        return finish(args, failures, warnings, {})

    trace_path = Path(args.prediction_trace).expanduser().resolve()
    if not trace_path.exists():
        add_issue(
            failures,
            "prediction_trace_missing",
            "prediction_trace file does not exist.",
            {"path": str(trace_path)},
        )
        return finish(args, failures, warnings, {})

    try:
        evaluation = load_json_obj(str(eval_path))
    except Exception as exc:
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
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
                "prediction_trace_schema_invalid",
                "Unable to parse performance_policy JSON.",
                {"path": str(Path(args.performance_policy).expanduser()), "error": str(exc)},
            )
            return finish(args, failures, warnings, {})
    thresholds = parse_thresholds(policy)
    metric_tol = float(thresholds["metric_tolerance"])
    threshold_tol = float(thresholds["threshold_tolerance"])
    beta = float(thresholds["beta"])

    try:
        trace_df = load_trace(str(trace_path))
    except Exception as exc:
        add_issue(
            failures,
            "prediction_trace_missing",
            "Unable to read prediction_trace file.",
            {"path": str(trace_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    missing_cols = [c for c in REQUIRED_TRACE_COLUMNS if c not in trace_df.columns]
    if missing_cols:
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "prediction_trace is missing required columns.",
            {"missing_columns": missing_cols, "required_columns": REQUIRED_TRACE_COLUMNS},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    # Normalize and validate trace columns.
    scope = trace_df["scope"].astype(str).str.strip().str.lower()
    y_true = normalize_binary(trace_df["y_true"])
    y_pred = normalize_binary(trace_df["y_pred"])
    y_score = pd.to_numeric(trace_df["y_score"], errors="coerce").to_numpy(dtype=float)
    selected_threshold = pd.to_numeric(trace_df["selected_threshold"], errors="coerce").to_numpy(dtype=float)

    if y_true is None or y_pred is None:
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "y_true and y_pred must be binary 0/1 values.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    if np.any(~np.isfinite(y_score)) or np.any(~np.isfinite(selected_threshold)):
        add_issue(
            failures,
            "prediction_trace_non_finite",
            "prediction_trace contains non-finite y_score or selected_threshold values.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    if np.any(y_score < 0.0) or np.any(y_score > 1.0):
        add_issue(
            failures,
            "prediction_score_out_of_range",
            "prediction_trace y_score must be in [0,1].",
            {
                "min_y_score": float(np.min(y_score)) if y_score.size else None,
                "max_y_score": float(np.max(y_score)) if y_score.size else None,
            },
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    if np.any(selected_threshold < 0.0) or np.any(selected_threshold > 1.0):
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "prediction_trace selected_threshold must be in [0,1].",
            {
                "min_selected_threshold": float(np.min(selected_threshold)) if selected_threshold.size else None,
                "max_selected_threshold": float(np.max(selected_threshold)) if selected_threshold.size else None,
            },
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    trace_df = trace_df.copy()
    trace_df["scope"] = scope
    trace_df["y_true"] = y_true
    trace_df["y_pred"] = y_pred
    trace_df["y_score"] = y_score
    trace_df["selected_threshold"] = selected_threshold
    trace_df["model_id"] = trace_df["model_id"].astype(str).str.strip()

    internal_scopes = ("train", "valid", "test")
    split_metrics = evaluation.get("split_metrics")
    if not isinstance(split_metrics, dict):
        add_issue(
            failures,
            "prediction_trace_schema_invalid",
            "evaluation_report must include split_metrics.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    eval_metadata = evaluation.get("metadata")
    eval_fingerprints = eval_metadata.get("data_fingerprints") if isinstance(eval_metadata, dict) else None
    top_metrics = evaluation.get("metrics")

    summary_splits: Dict[str, Any] = {}
    for split_name in internal_scopes:
        block = require_metric_block(split_metrics, split_name, failures)
        if block is None:
            continue
        metrics = block["metrics"]
        split_rows = trace_df[trace_df["scope"] == split_name]
        if split_rows.empty:
            add_issue(
                failures,
                "prediction_trace_schema_invalid",
                "prediction_trace is missing rows for required internal split.",
                {"split": split_name},
            )
            continue

        thresholds_split = np.unique(split_rows["selected_threshold"].to_numpy(dtype=float))
        if thresholds_split.size == 0 or (float(np.max(thresholds_split)) - float(np.min(thresholds_split))) > threshold_tol:
            add_issue(
                failures,
                "prediction_trace_schema_invalid",
                "selected_threshold must be stable within each split.",
                {
                    "split": split_name,
                    "min_selected_threshold": float(np.min(thresholds_split)) if thresholds_split.size else None,
                    "max_selected_threshold": float(np.max(thresholds_split)) if thresholds_split.size else None,
                    "threshold_tolerance": threshold_tol,
                },
            )
            continue
        threshold_value = float(thresholds_split[0])

        y_true_split = split_rows["y_true"].to_numpy(dtype=int)
        y_score_split = split_rows["y_score"].to_numpy(dtype=float)
        y_pred_split = split_rows["y_pred"].to_numpy(dtype=int)
        y_pred_recomputed = (y_score_split >= threshold_value).astype(int)
        if np.any(y_pred_split != y_pred_recomputed):
            add_issue(
                failures,
                "prediction_trace_schema_invalid",
                "y_pred must equal (y_score >= selected_threshold).",
                {"split": split_name},
            )
            continue

        if len(np.unique(y_true_split)) < 2:
            add_issue(
                failures,
                "prediction_trace_schema_invalid",
                "Each internal split must contain both classes for replay metrics.",
                {"split": split_name},
            )
            continue

        replay_metrics, replay_cm = metric_panel(
            y_true=y_true_split,
            y_score=y_score_split,
            y_pred=y_pred_recomputed,
            beta=beta,
        )

        # Row-count linkage against evaluation fingerprints.
        if isinstance(eval_fingerprints, dict):
            fp_block = eval_fingerprints.get(split_name)
            row_count = fp_block.get("row_count") if isinstance(fp_block, dict) else None
            row_count_i = to_int(row_count)
            if row_count_i is not None and row_count_i != int(split_rows.shape[0]):
                add_issue(
                    failures,
                    "prediction_trace_rowcount_mismatch",
                    "prediction_trace row count does not match evaluation metadata fingerprint row_count.",
                    {
                        "split": split_name,
                        "trace_row_count": int(split_rows.shape[0]),
                        "fingerprint_row_count": row_count_i,
                    },
                )

        # Metric-by-metric replay check.
        for metric_name in REQUIRED_METRICS:
            compare_metric(
                split_name=split_name,
                metric_name=metric_name,
                observed=float(replay_metrics[metric_name]),
                expected=metrics.get(metric_name),
                tolerance=metric_tol,
                failures=failures,
            )

        summary_splits[split_name] = {
            "row_count": int(split_rows.shape[0]),
            "selected_threshold": threshold_value,
            "replayed_confusion_matrix": replay_cm,
            "replayed_metrics": replay_metrics,
        }

    # Top-level test metrics must align with replayed test metrics.
    if isinstance(top_metrics, dict) and isinstance(summary_splits.get("test"), dict):
        replay_test_metrics = summary_splits["test"]["replayed_metrics"]
        for metric_name in REQUIRED_METRICS:
            compare_metric(
                split_name="test_top_level",
                metric_name=metric_name,
                observed=float(replay_test_metrics[metric_name]),
                expected=top_metrics.get(metric_name),
                tolerance=metric_tol,
                failures=failures,
            )

    summary = {
        "evaluation_report": str(eval_path),
        "prediction_trace": str(trace_path),
        "thresholds": thresholds,
        "splits": summary_splits,
        "trace_row_count": int(trace_df.shape[0]),
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

