#!/usr/bin/env python3
"""
Fail-closed CI matrix gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)
from _gate_utils import add_issue, load_json_from_str as load_json, normalize_binary as _shared_normalize_binary, safe_ratio as _shared_safe_ratio, to_float, to_int as _shared_to_int


register_remediations({
    "ci_width_excessive": "Confidence intervals are too wide. Increase bootstrap resamples or collect more data.",
    "ci_coverage_below_threshold": "CI coverage is below expected nominal level. Check bootstrap methodology.",
    "ci_resamples_insufficient": "Increase the number of bootstrap resamples (recommended >= 2000).",
    "ci_metric_mismatch": "CI matrix metrics don't match evaluation report. Re-run CI computation.",
})


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
    parser = argparse.ArgumentParser(description="Compute and validate full split/external CI matrix with transport-drop CI.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report JSON.")
    parser.add_argument("--prediction-trace", required=True, help="Path to prediction_trace CSV/CSV.GZ.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external_validation_report JSON.")
    parser.add_argument("--performance-policy", help="Optional performance_policy JSON.")
    parser.add_argument("--ci-matrix-report", required=True, help="Path to ci_matrix_report JSON artifact.")
    parser.add_argument(
        "--update-ci-matrix-report",
        action="store_true",
        help="Rewrite ci_matrix_report artifact with recomputed values.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    parser.add_argument("--report", help="Optional output gate report.")
    return parser.parse_args()


def to_int(value: Any) -> Optional[int]:
    return _shared_to_int(value)


def normalize_binary(values: pd.Series) -> Optional[np.ndarray]:
    return _shared_normalize_binary(values)


def safe_ratio(num: float, den: float) -> float:
    return _shared_safe_ratio(num, den)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    return {
        "tp": int(np.sum((yt == 1) & (yp == 1))),
        "fp": int(np.sum((yt == 0) & (yp == 1))),
        "tn": int(np.sum((yt == 0) & (yp == 0))),
        "fn": int(np.sum((yt == 1) & (yp == 0))),
    }


def metric_panel(y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
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
    return {
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


def parse_ci_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_resamples": 2000,
        "max_resamples_supported": 4000,
        "max_width": 0.20,
        "metric_tolerance": 1e-6,
        "transport_ci_required": True,
        "width_gate_metrics": ["pr_auc", "f2_beta", "brier"],
        "beta": 2.0,
    }
    if not isinstance(policy, dict):
        return out
    ci_block = policy.get("ci_policy")
    if isinstance(ci_block, dict):
        n_resamples = to_int(ci_block.get("n_resamples"))
        max_resamples_supported = to_int(ci_block.get("max_resamples_supported"))
        max_width = to_float(ci_block.get("max_width"))
        metric_tolerance = to_float(ci_block.get("metric_tolerance"))
        if n_resamples is not None and n_resamples >= 100:
            out["n_resamples"] = int(n_resamples)
        if max_resamples_supported is not None and max_resamples_supported >= 100:
            out["max_resamples_supported"] = int(max_resamples_supported)
        if max_width is not None and max_width > 0.0:
            out["max_width"] = float(max_width)
        if metric_tolerance is not None and metric_tolerance > 0.0:
            out["metric_tolerance"] = float(metric_tolerance)
        if ci_block.get("transport_ci_required") in (True, False):
            out["transport_ci_required"] = bool(ci_block.get("transport_ci_required"))
        width_gate_metrics = ci_block.get("width_gate_metrics")
        if isinstance(width_gate_metrics, list):
            clean = [str(x).strip() for x in width_gate_metrics if isinstance(x, str) and str(x).strip()]
            if clean:
                out["width_gate_metrics"] = clean
    beta = to_float(policy.get("beta")) if isinstance(policy, dict) else None
    if beta is not None and beta > 0.0:
        out["beta"] = float(beta)
    return out


def stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if pos.size == 0 or neg.size == 0:
        return None
    pos_idx = rng.choice(pos, size=pos.size, replace=True)
    neg_idx = rng.choice(neg, size=neg.size, replace=True)
    idx = np.concatenate([pos_idx, neg_idx], axis=0)
    rng.shuffle(idx)
    return idx


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    beta: float,
    n_resamples: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    rng = np.random.default_rng(seed)
    hits: Dict[str, List[float]] = {name: [] for name in REQUIRED_METRICS}
    attempts = 0
    max_attempts = max(5 * int(n_resamples), 8000)
    while len(hits["pr_auc"]) < int(n_resamples) and attempts < max_attempts:
        attempts += 1
        idx = stratified_bootstrap_indices(y_true, rng)
        if idx is None:
            break
        yb = y_true[idx]
        sb = y_score[idx]
        try:
            panel = metric_panel(yb, sb, threshold, beta=beta)
        except Exception:
            continue
        if not all(isinstance(panel.get(metric), (int, float)) and math.isfinite(float(panel.get(metric))) for metric in REQUIRED_METRICS):
            continue
        for metric in REQUIRED_METRICS:
            hits[metric].append(float(panel[metric]))

    effective = min((len(v) for v in hits.values()), default=0)
    summary: Dict[str, Dict[str, float]] = {}
    for metric in REQUIRED_METRICS:
        values = np.asarray(hits[metric][:effective], dtype=float)
        if values.size == 0:
            summary[metric] = {"ci_lower": float("nan"), "ci_upper": float("nan"), "ci_width": float("nan")}
            continue
        lo, hi = np.percentile(values, [2.5, 97.5]).tolist()
        summary[metric] = {
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "ci_width": float(hi - lo),
        }
    return summary, int(effective)


def extract_split_rows(trace_df: pd.DataFrame, scope: str) -> pd.DataFrame:
    return trace_df[trace_df["scope"] == scope]


def verify_threshold_stable(rows: pd.DataFrame) -> Optional[float]:
    vals = rows["selected_threshold"].to_numpy(dtype=float)
    if vals.size == 0:
        return None
    if np.max(vals) - np.min(vals) > 1e-9:
        return None
    return float(vals[0])


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    try:
        _eval_payload = load_json(args.evaluation_report)  # noqa: F841 – validates parse
    except Exception as exc:
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "Unable to parse evaluation_report JSON.",
            {"path": str(Path(args.evaluation_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        external_payload = load_json(args.external_validation_report)
    except Exception as exc:
        add_issue(
            failures,
            "transport_ci_invalid",
            "Unable to parse external_validation_report JSON.",
            {"path": str(Path(args.external_validation_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        trace_df = pd.read_csv(Path(args.prediction_trace).expanduser().resolve())
    except Exception as exc:
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "Unable to read prediction_trace CSV.",
            {"path": str(Path(args.prediction_trace).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    policy: Optional[Dict[str, Any]] = None
    if args.performance_policy:
        try:
            policy = load_json(args.performance_policy)
        except Exception as exc:
            add_issue(
                failures,
                "ci_matrix_missing_required_metric",
                "Unable to parse performance_policy JSON.",
                {"path": str(Path(args.performance_policy).expanduser()), "error": str(exc)},
            )
            return finish(args, failures, warnings, {})
    ci_policy = parse_ci_policy(policy)
    requested_resamples = int(ci_policy["n_resamples"])
    max_resamples_supported = int(ci_policy["max_resamples_supported"])
    n_resamples = requested_resamples
    if requested_resamples > max_resamples_supported:
        add_issue(
            failures,
            "ci_resamples_insufficient",
            "Requested CI bootstrap resamples exceed configured publication-grade compute budget.",
            {
                "requested_resamples": requested_resamples,
                "max_resamples_supported": max_resamples_supported,
            },
        )
        n_resamples = int(max_resamples_supported)
    beta = float(ci_policy["beta"])
    max_width = float(ci_policy["max_width"])
    transport_required = bool(ci_policy["transport_ci_required"])
    width_gate_metrics = {str(x).strip().lower() for x in ci_policy.get("width_gate_metrics", []) if str(x).strip()}

    required_cols = {"scope", "cohort_id", "y_true", "y_score", "y_pred", "selected_threshold"}
    missing_cols = sorted(required_cols - set(trace_df.columns))
    if missing_cols:
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "prediction_trace missing required columns.",
            {"missing_columns": missing_cols},
        )
        return finish(args, failures, warnings, {"ci_policy": ci_policy})

    trace_df = trace_df.copy()
    trace_df["scope"] = trace_df["scope"].astype(str).str.strip().str.lower()
    trace_df["cohort_id"] = trace_df["cohort_id"].astype(str).str.strip()
    trace_df["y_score"] = pd.to_numeric(trace_df["y_score"], errors="coerce")
    trace_df["selected_threshold"] = pd.to_numeric(trace_df["selected_threshold"], errors="coerce")

    y_true_bin = normalize_binary(trace_df["y_true"])
    y_pred_bin = normalize_binary(trace_df["y_pred"])
    if y_true_bin is None or y_pred_bin is None:
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "prediction_trace y_true/y_pred must be binary 0/1.",
            {},
        )
        return finish(args, failures, warnings, {"ci_policy": ci_policy})
    trace_df["y_true"] = y_true_bin
    trace_df["y_pred"] = y_pred_bin

    if np.any(~np.isfinite(trace_df["y_score"].to_numpy(dtype=float))):
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "prediction_trace y_score contains non-finite values.",
            {},
        )
        return finish(args, failures, warnings, {"ci_policy": ci_policy})
    if np.any(trace_df["y_score"].to_numpy(dtype=float) < 0.0) or np.any(trace_df["y_score"].to_numpy(dtype=float) > 1.0):
        add_issue(
            failures,
            "ci_matrix_missing_required_metric",
            "prediction_trace y_score must be within [0,1].",
            {},
        )
        return finish(args, failures, warnings, {"ci_policy": ci_policy})

    split_scope_map = {"train": "train", "valid": "valid", "test": "test"}
    split_metrics_ci: Dict[str, Any] = {}
    ci_quality_rows: List[Dict[str, Any]] = []

    for split_name, scope in split_scope_map.items():
        rows = extract_split_rows(trace_df, scope)
        if rows.empty:
            add_issue(
                failures,
                "ci_matrix_missing_required_metric",
                "prediction_trace missing rows for required split.",
                {"split": split_name, "scope": scope},
            )
            continue
        threshold = verify_threshold_stable(rows)
        if threshold is None:
            add_issue(
                failures,
                "ci_matrix_missing_required_metric",
                "selected_threshold must be stable within each split.",
                {"split": split_name},
            )
            continue
        y_true = rows["y_true"].to_numpy(dtype=int)
        y_score = rows["y_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            add_issue(
                failures,
                "ci_matrix_missing_required_metric",
                "Each split must contain both classes for CI computation.",
                {"split": split_name},
            )
            continue
        point_metrics = metric_panel(y_true, y_score, threshold, beta=beta)
        ci_summary, effective = bootstrap_metric_ci(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            beta=beta,
            n_resamples=n_resamples,
            seed=7300 + len(split_metrics_ci),
        )
        if effective < n_resamples:
            add_issue(
                failures,
                "ci_resamples_insufficient",
                "Effective CI bootstrap resamples are below policy requirement.",
                {"split": split_name, "effective_resamples": effective, "required_resamples": n_resamples},
            )
        metrics_block: Dict[str, Any] = {}
        for metric in REQUIRED_METRICS:
            ci_metric = ci_summary.get(metric, {})
            lo = to_float(ci_metric.get("ci_lower"))
            hi = to_float(ci_metric.get("ci_upper"))
            if lo is None or hi is None:
                add_issue(
                    failures,
                    "ci_matrix_missing_required_metric",
                    "Missing CI bounds for required metric.",
                    {"split": split_name, "metric": metric},
                )
                continue
            width = float(hi - lo)
            metrics_block[metric] = {
                "point": float(point_metrics[metric]),
                "ci_95": [float(lo), float(hi)],
                "ci_width": width,
                "n_resamples": int(effective),
            }
            if metric.lower() in width_gate_metrics and width > max_width:
                add_issue(
                    failures,
                    "ci_width_exceeds_threshold",
                    "CI width exceeds policy threshold.",
                    {"split": split_name, "metric": metric, "ci_width": width, "max_width": max_width},
                )
        split_metrics_ci[split_name] = {
            "row_count": int(rows.shape[0]),
            "positive_count": int(np.sum(y_true == 1)),
            "selected_threshold": float(threshold),
            "metrics": metrics_block,
        }
        ci_quality_rows.append(
            {
                "split": split_name,
                "effective_resamples": int(effective),
                "required_resamples": int(n_resamples),
                "max_width_observed": float(max((m.get("ci_width", 0.0) for m in metrics_block.values()), default=0.0)),
            }
        )

    # External cohort CI matrix.
    cohorts = external_payload.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        add_issue(
            failures,
            "transport_ci_invalid",
            "external_validation_report must include non-empty cohorts list.",
            {},
        )
        cohorts = []

    external_ci: Dict[str, Any] = {}
    transport_drop_ci: Dict[str, Any] = {}
    internal_test = split_metrics_ci.get("test")
    test_rows = extract_split_rows(trace_df, "test")
    test_threshold = verify_threshold_stable(test_rows) if not test_rows.empty else None
    test_y = test_rows["y_true"].to_numpy(dtype=int) if not test_rows.empty else np.asarray([], dtype=int)
    test_s = test_rows["y_score"].to_numpy(dtype=float) if not test_rows.empty else np.asarray([], dtype=float)

    for idx, cohort in enumerate(cohorts):
        if not isinstance(cohort, dict):
            add_issue(
                failures,
                "transport_ci_invalid",
                "external_validation_report cohort entry must be object.",
                {"index": idx},
            )
            continue
        cohort_id = str(cohort.get("cohort_id", "")).strip()
        if not cohort_id:
            add_issue(
                failures,
                "transport_ci_invalid",
                "external_validation_report cohort entry missing cohort_id.",
                {"index": idx},
            )
            continue
        rows = trace_df[(trace_df["scope"] == "external") & (trace_df["cohort_id"] == cohort_id)]
        if rows.empty:
            add_issue(
                failures,
                "transport_ci_invalid",
                "prediction_trace missing rows for external cohort.",
                {"cohort_id": cohort_id},
            )
            continue
        threshold = verify_threshold_stable(rows)
        if threshold is None:
            add_issue(
                failures,
                "transport_ci_invalid",
                "External cohort selected_threshold must be stable.",
                {"cohort_id": cohort_id},
            )
            continue
        y_true = rows["y_true"].to_numpy(dtype=int)
        y_score = rows["y_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            add_issue(
                failures,
                "transport_ci_invalid",
                "External cohort must contain both classes for CI computation.",
                {"cohort_id": cohort_id},
            )
            continue
        point_metrics = metric_panel(y_true, y_score, threshold, beta=beta)
        ci_summary, effective = bootstrap_metric_ci(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            beta=beta,
            n_resamples=n_resamples,
            seed=8400 + idx,
        )
        if effective < n_resamples:
            add_issue(
                failures,
                "ci_resamples_insufficient",
                "External cohort effective CI resamples are below policy requirement.",
                {"cohort_id": cohort_id, "effective_resamples": effective, "required_resamples": n_resamples},
            )
        metrics_block: Dict[str, Any] = {}
        for metric in REQUIRED_METRICS:
            ci_metric = ci_summary.get(metric, {})
            lo = to_float(ci_metric.get("ci_lower"))
            hi = to_float(ci_metric.get("ci_upper"))
            if lo is None or hi is None:
                add_issue(
                    failures,
                    "ci_matrix_missing_required_metric",
                    "Missing external CI bounds for required metric.",
                    {"cohort_id": cohort_id, "metric": metric},
                )
                continue
            width = float(hi - lo)
            metrics_block[metric] = {
                "point": float(point_metrics[metric]),
                "ci_95": [float(lo), float(hi)],
                "ci_width": width,
                "n_resamples": int(effective),
            }
            if metric.lower() in width_gate_metrics and width > max_width:
                add_issue(
                    failures,
                    "ci_width_exceeds_threshold",
                    "External cohort CI width exceeds policy threshold.",
                    {"cohort_id": cohort_id, "metric": metric, "ci_width": width, "max_width": max_width},
                )
        external_ci[cohort_id] = {
            "cohort_type": str(cohort.get("cohort_type", "")).strip().lower(),
            "row_count": int(rows.shape[0]),
            "positive_count": int(np.sum(y_true == 1)),
            "selected_threshold": float(threshold),
            "metrics": metrics_block,
        }

        # Transport drop CI.
        if internal_test is None or test_threshold is None or test_y.size == 0 or len(np.unique(test_y)) < 2:
            add_issue(
                failures,
                "transport_ci_invalid",
                "Cannot compute transport-drop CI without valid internal test split.",
                {"cohort_id": cohort_id},
            )
            continue
        rng = np.random.default_rng(9100 + idx)
        pr_drops: List[float] = []
        f2_drops: List[float] = []
        brier_inc: List[float] = []
        attempts = 0
        max_attempts = max(5 * int(n_resamples), 10000)
        while len(pr_drops) < n_resamples and attempts < max_attempts:
            attempts += 1
            idx_test = stratified_bootstrap_indices(test_y, rng)
            idx_ext = stratified_bootstrap_indices(y_true, rng)
            if idx_test is None or idx_ext is None:
                break
            try:
                m_test = metric_panel(test_y[idx_test], test_s[idx_test], float(test_threshold), beta=beta)
                m_ext = metric_panel(y_true[idx_ext], y_score[idx_ext], float(threshold), beta=beta)
            except Exception:
                continue
            pr_drops.append(float(m_test["pr_auc"] - m_ext["pr_auc"]))
            f2_drops.append(float(m_test["f2_beta"] - m_ext["f2_beta"]))
            brier_inc.append(float(m_ext["brier"] - m_test["brier"]))
        effective_transport = min(len(pr_drops), len(f2_drops), len(brier_inc))
        if effective_transport < n_resamples:
            add_issue(
                failures,
                "ci_resamples_insufficient",
                "Transport-drop CI effective resamples below policy requirement.",
                {"cohort_id": cohort_id, "effective_resamples": effective_transport, "required_resamples": n_resamples},
            )
        if effective_transport <= 0:
            add_issue(
                failures,
                "transport_ci_invalid",
                "Unable to compute transport-drop CI.",
                {"cohort_id": cohort_id},
            )
            continue
        pr_arr = np.asarray(pr_drops[:effective_transport], dtype=float)
        f2_arr = np.asarray(f2_drops[:effective_transport], dtype=float)
        brier_arr = np.asarray(brier_inc[:effective_transport], dtype=float)
        transport_drop_ci[cohort_id] = {
            "pr_auc_drop": {
                "point": float((internal_test["metrics"]["pr_auc"]["point"] - external_ci[cohort_id]["metrics"]["pr_auc"]["point"])),
                "ci_95": [float(np.percentile(pr_arr, 2.5)), float(np.percentile(pr_arr, 97.5))],
                "ci_width": float(np.percentile(pr_arr, 97.5) - np.percentile(pr_arr, 2.5)),
                "n_resamples": int(effective_transport),
            },
            "f2_beta_drop": {
                "point": float((internal_test["metrics"]["f2_beta"]["point"] - external_ci[cohort_id]["metrics"]["f2_beta"]["point"])),
                "ci_95": [float(np.percentile(f2_arr, 2.5)), float(np.percentile(f2_arr, 97.5))],
                "ci_width": float(np.percentile(f2_arr, 97.5) - np.percentile(f2_arr, 2.5)),
                "n_resamples": int(effective_transport),
            },
            "brier_increase": {
                "point": float((external_ci[cohort_id]["metrics"]["brier"]["point"] - internal_test["metrics"]["brier"]["point"])),
                "ci_95": [float(np.percentile(brier_arr, 2.5)), float(np.percentile(brier_arr, 97.5))],
                "ci_width": float(np.percentile(brier_arr, 97.5) - np.percentile(brier_arr, 2.5)),
                "n_resamples": int(effective_transport),
            },
        }

    if transport_required and not transport_drop_ci:
        add_issue(
            failures,
            "transport_ci_invalid",
            "Policy requires transport-drop CI, but no external transport CI could be computed.",
            {"transport_ci_required": transport_required},
        )

    ci_matrix_payload = {
        "status": "pass" if not failures else "fail",
        "schema_version": "v4.0",
        "split_metrics_ci": {**split_metrics_ci, "external": external_ci},
        "transport_drop_ci": transport_drop_ci,
        "ci_quality_summary": {
            "required_resamples": int(n_resamples),
            "max_width_threshold": float(max_width),
            "rows": ci_quality_rows,
        },
        "metadata": {
            "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
            "prediction_trace": str(Path(args.prediction_trace).expanduser().resolve()),
            "external_validation_report": str(Path(args.external_validation_report).expanduser().resolve()),
            "ci_policy": ci_policy,
        },
    }
    ci_out = Path(args.ci_matrix_report).expanduser().resolve()
    should_rewrite = bool(args.update_ci_matrix_report) or (not ci_out.exists())
    if should_rewrite:
        from _gate_utils import write_json as _write_ci
        _write_ci(ci_out, ci_matrix_payload)

    summary = {
        "ci_matrix_report": str(ci_out),
        "ci_matrix_report_rewritten": bool(should_rewrite),
        "split_count": len(split_metrics_ci),
        "external_cohort_count": len(external_ci),
        "transport_ci_count": len(transport_drop_ci),
        "ci_policy": ci_policy,
    }
    return finish(args, failures, warnings, summary)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
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
        "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve()),
        "prediction_trace": str(Path(args.prediction_trace).expanduser().resolve()),
        "external_validation_report": str(Path(args.external_validation_report).expanduser().resolve()),
        "ci_matrix_report": str(Path(args.ci_matrix_report).expanduser().resolve()),
    }
    if getattr(args, "performance_policy", None):
        input_files["performance_policy"] = str(Path(args.performance_policy).expanduser().resolve())

    report = build_report_envelope(
        gate_name="ci_matrix_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary=summary,
        input_files=input_files,
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="ci_matrix_gate",
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
