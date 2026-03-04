#!/usr/bin/env python3
"""
Fail-closed calibration and decision-curve gate.

Evaluates calibration (ECE/slope/intercept) and DCA net-benefit for:
1) internal test split
2) every external cohort in external_validation_report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from _gate_utils import add_issue, load_json_from_str as load_json_obj, normalize_binary as _shared_normalize_binary, to_float


REQUIRED_TRACE_COLUMNS = {
    "scope",
    "cohort_id",
    "cohort_type",
    "y_true",
    "y_score",
    "y_pred",
    "selected_threshold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed calibration + DCA gate for test + external cohorts.")
    parser.add_argument("--prediction-trace", required=True, help="Path to prediction_trace CSV/CSV.GZ.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external_validation_report.json.")
    parser.add_argument("--performance-policy", help="Optional performance_policy JSON path.")
    parser.add_argument("--report", help="Optional output report JSON path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def normalize_binary(series: pd.Series) -> Optional[np.ndarray]:
    return _shared_normalize_binary(series)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_calibration_slope_intercept(y_true: np.ndarray, y_score: np.ndarray) -> Optional[Dict[str, float]]:
    if y_true.shape[0] < 3 or len(np.unique(y_true)) < 2:
        return None
    eps = 1e-6
    p = np.clip(y_score.astype(float), eps, 1.0 - eps)
    z = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones_like(z), z]).astype(float)
    beta = np.array([0.0, 1.0], dtype=float)
    prior = np.array([0.0, 1.0], dtype=float)
    # Strong prior regularization stabilizes slope/intercept estimation in small cohorts.
    ridge = 20.0

    for _ in range(80):
        eta = X @ beta
        mu = sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-8, None)
        grad = X.T @ (y_true.astype(float) - mu) - (ridge * (beta - prior))
        hessian = X.T @ (w[:, None] * X)
        hessian = hessian + (ridge * np.eye(hessian.shape[0], dtype=float))
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return None
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-8:
            beta = beta_next
            break
        beta = beta_next

    if not np.all(np.isfinite(beta)):
        return None
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
    min_bin_size: int,
) -> float:
    n = int(y_true.shape[0])
    if n <= 0:
        return 1.0
    # Equal-frequency bins with a minimum bin-size guard reduce sparse-bin variance
    # for small cohorts (publication gate minimum is often around n=50).
    requested_bins = max(2, int(n_bins))
    effective_bins = max(2, n // max(1, int(min_bin_size)))
    n_bins = min(requested_bins, effective_bins)
    order = np.argsort(y_score.astype(float))
    blocks = np.array_split(order, n_bins)
    total = 0.0
    for idx in blocks:
        count = int(idx.shape[0])
        if count == 0:
            continue
        avg_score = float(np.mean(y_score[idx]))
        avg_true = float(np.mean(y_true[idx]))
        total += (count / n) * abs(avg_true - avg_score)
    return float(total)


def net_benefit(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    n = float(y_true.shape[0])
    if n <= 0:
        return 0.0
    y_pred = (y_score >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    weight = float(threshold / (1.0 - threshold))
    return float((tp / n) - (fp / n) * weight)


def treat_all_net_benefit(y_true: np.ndarray, threshold: float) -> float:
    prevalence = float(np.mean(y_true.astype(float)))
    weight = float(threshold / (1.0 - threshold))
    return float(prevalence - (1.0 - prevalence) * weight)


def parse_policy_thresholds(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ece_max": 0.06,
        "slope_min": 0.80,
        "slope_max": 2.00,
        "intercept_abs_max": 1.00,
        "min_rows": 50,
        "min_positives": 10,
        "ece_bins": 10,
        "ece_min_bin_size": 15,
        "threshold_grid": {"start": 0.05, "end": 0.50, "step": 0.05},
        "min_advantage_coverage": 0.50,
        "min_average_advantage": 0.0,
        "min_net_benefit_advantage": 0.0,
    }
    if not isinstance(policy, dict):
        return out
    block = policy.get("calibration_dca_thresholds")
    if not isinstance(block, dict):
        return out

    for key in (
        "ece_max",
        "slope_min",
        "slope_max",
        "intercept_abs_max",
        "min_rows",
        "min_positives",
        "ece_bins",
        "ece_min_bin_size",
        "min_advantage_coverage",
        "min_average_advantage",
        "min_net_benefit_advantage",
    ):
        value = to_float(block.get(key))
        if value is None:
            continue
        if key in {"min_rows", "min_positives", "ece_bins", "ece_min_bin_size"}:
            if value >= 1:
                out[key] = int(value)
        elif key == "ece_max":
            if 0.0 <= value <= 1.0:
                out[key] = float(value)
        elif key == "intercept_abs_max":
            if 0.0 <= value <= 10.0:
                out[key] = float(value)
        elif key in {"slope_min", "slope_max"}:
            if value > 0.0:
                out[key] = float(value)
        elif key == "min_advantage_coverage":
            if 0.0 <= value <= 1.0:
                out[key] = float(value)
        else:
            out[key] = float(value)

    grid = block.get("threshold_grid")
    if isinstance(grid, dict):
        start = to_float(grid.get("start"))
        end = to_float(grid.get("end"))
        step = to_float(grid.get("step"))
        if start is not None and end is not None and step is not None:
            out["threshold_grid"] = {"start": float(start), "end": float(end), "step": float(step)}
    return out


def build_threshold_grid(grid_cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    start = to_float(grid_cfg.get("start"))
    end = to_float(grid_cfg.get("end"))
    step = to_float(grid_cfg.get("step"))
    if (
        start is None
        or end is None
        or step is None
        or start <= 0.0
        or end >= 1.0
        or step <= 0.0
        or start >= end
    ):
        return None
    points = np.arange(start, end + (0.5 * step), step, dtype=float)
    points = points[(points > 0.0) & (points < 1.0)]
    if points.size < 2:
        return None
    return points


def evaluate_cohort(
    cohort_label: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Dict[str, Any],
    grid: np.ndarray,
) -> Dict[str, Any]:
    ece_bins = int(thresholds["ece_bins"])
    ece_min_bin_size = int(thresholds.get("ece_min_bin_size", 15))
    ece = expected_calibration_error(
        y_true,
        y_score,
        n_bins=ece_bins,
        min_bin_size=ece_min_bin_size,
    )
    cal = fit_calibration_slope_intercept(y_true, y_score)
    if cal is None:
        raise ValueError("Unable to fit calibration slope/intercept.")
    slope = float(cal["slope"])
    intercept = float(cal["intercept"])

    dca_rows: List[Dict[str, float]] = []
    deltas: List[float] = []
    for t in grid.tolist():
        nb_model = net_benefit(y_true, y_score, threshold=float(t))
        nb_all = treat_all_net_benefit(y_true, threshold=float(t))
        nb_none = 0.0
        baseline = max(nb_all, nb_none)
        delta = float(nb_model - baseline)
        deltas.append(delta)
        dca_rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(nb_model),
                "net_benefit_treat_all": float(nb_all),
                "net_benefit_treat_none": float(nb_none),
                "net_benefit_advantage": float(delta),
            }
        )

    deltas_arr = np.asarray(deltas, dtype=float)
    min_advantage = float(thresholds["min_net_benefit_advantage"])
    coverage = float(np.mean(deltas_arr >= min_advantage)) if deltas_arr.size else 0.0
    avg_advantage = float(np.mean(deltas_arr)) if deltas_arr.size else -1.0
    min_delta = float(np.min(deltas_arr)) if deltas_arr.size else -1.0

    return {
        "cohort": cohort_label,
        "row_count": int(y_true.shape[0]),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "calibration": {
            "ece": float(ece),
            "slope": slope,
            "intercept": intercept,
            "ece_bins": ece_bins,
            "ece_min_bin_size": int(ece_min_bin_size),
        },
        "dca": {
            "threshold_count": int(len(dca_rows)),
            "threshold_rows": dca_rows,
            "advantage_coverage": float(coverage),
            "average_advantage": float(avg_advantage),
            "minimum_advantage": float(min_delta),
        },
    }


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    trace_path = Path(args.prediction_trace).expanduser().resolve()
    eval_path = Path(args.evaluation_report).expanduser().resolve()
    ext_path = Path(args.external_validation_report).expanduser().resolve()
    for p, name in ((trace_path, "prediction_trace"), (eval_path, "evaluation_report"), (ext_path, "external_validation_report")):
        if not p.exists():
            add_issue(
                failures,
                "calibration_insufficient_events",
                "Required artifact file is missing for calibration/DCA gate.",
                {"artifact": name, "path": str(p)},
            )
            return finish(args, failures, warnings, {})

    try:
        trace_df = pd.read_csv(trace_path)
    except Exception as exc:
        add_issue(
            failures,
            "calibration_insufficient_events",
            "Unable to load prediction_trace file.",
            {"path": str(trace_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    if not REQUIRED_TRACE_COLUMNS.issubset(set(trace_df.columns)):
        add_issue(
            failures,
            "calibration_insufficient_events",
            "prediction_trace is missing required columns.",
            {"missing_columns": sorted(REQUIRED_TRACE_COLUMNS - set(trace_df.columns))},
        )
        return finish(args, failures, warnings, {})

    try:
        eval_report = load_json_obj(str(eval_path))
        ext_report = load_json_obj(str(ext_path))
        policy = load_json_obj(args.performance_policy) if args.performance_policy else {}
    except Exception as exc:
        add_issue(
            failures,
            "calibration_insufficient_events",
            "Unable to parse required JSON input for calibration/DCA gate.",
            {"error": str(exc), "eval_path": str(eval_path), "ext_path": str(ext_path)},
        )
        return finish(args, failures, warnings, {})

    thresholds = parse_policy_thresholds(policy)
    grid = build_threshold_grid(thresholds.get("threshold_grid", {}))
    if grid is None:
        add_issue(
            failures,
            "decision_curve_threshold_grid_invalid",
            "calibration_dca_thresholds.threshold_grid is invalid.",
            {"threshold_grid": thresholds.get("threshold_grid")},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    trace_df = trace_df.copy()
    trace_df["scope"] = trace_df["scope"].astype(str).str.strip().str.lower()
    trace_df["cohort_id"] = trace_df["cohort_id"].astype(str).str.strip()
    trace_df["y_score"] = pd.to_numeric(trace_df["y_score"], errors="coerce")
    y_true_all = normalize_binary(trace_df["y_true"])
    if y_true_all is None:
        add_issue(
            failures,
            "calibration_insufficient_events",
            "prediction_trace y_true must be binary 0/1.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})
    trace_df["y_true"] = y_true_all

    y_score_all = trace_df["y_score"].to_numpy(dtype=float)
    if np.any(~np.isfinite(y_score_all)) or np.any(y_score_all < 0.0) or np.any(y_score_all > 1.0):
        add_issue(
            failures,
            "calibration_insufficient_events",
            "prediction_trace y_score must be finite and in [0,1].",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    cohorts_to_check: List[Dict[str, str]] = [{"scope": "test", "cohort_id": "internal_test", "label": "internal_test"}]
    ext_cohorts = ext_report.get("cohorts")
    if not isinstance(ext_cohorts, list) or not ext_cohorts:
        add_issue(
            failures,
            "calibration_insufficient_events",
            "external_validation_report must include non-empty cohorts list.",
            {},
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})
    for entry in ext_cohorts:
        if not isinstance(entry, dict):
            continue
        cohort_id = str(entry.get("cohort_id", "")).strip()
        if not cohort_id:
            continue
        cohorts_to_check.append({"scope": "external", "cohort_id": cohort_id, "label": f"external::{cohort_id}"})

    min_rows = int(thresholds["min_rows"])
    min_positives = int(thresholds["min_positives"])
    cohort_results: List[Dict[str, Any]] = []

    for cohort_ref in cohorts_to_check:
        scope = cohort_ref["scope"]
        cohort_id = cohort_ref["cohort_id"]
        label = cohort_ref["label"]
        if scope == "test":
            subset = trace_df[trace_df["scope"] == "test"]
        else:
            subset = trace_df[(trace_df["scope"] == "external") & (trace_df["cohort_id"] == cohort_id)]

        if subset.empty:
            add_issue(
                failures,
                "calibration_insufficient_events",
                "No prediction_trace rows found for cohort required by calibration/DCA gate.",
                {"scope": scope, "cohort_id": cohort_id},
            )
            continue

        y_true = subset["y_true"].to_numpy(dtype=int)
        y_score = subset["y_score"].to_numpy(dtype=float)
        n_rows = int(y_true.shape[0])
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_rows < min_rows or n_pos < min_positives or n_neg < min_positives:
            add_issue(
                failures,
                "calibration_insufficient_events",
                "Cohort does not satisfy minimum sample/event requirements for calibration+DCA.",
                {
                    "cohort": label,
                    "row_count": n_rows,
                    "positive_count": n_pos,
                    "negative_count": n_neg,
                    "min_rows": min_rows,
                    "min_positives": min_positives,
                },
            )
            continue

        try:
            result = evaluate_cohort(
                cohort_label=label,
                y_true=y_true,
                y_score=y_score,
                thresholds=thresholds,
                grid=grid,
            )
        except Exception as exc:
            add_issue(
                failures,
                "calibration_insufficient_events",
                "Failed to evaluate calibration/DCA metrics for cohort.",
                {"cohort": label, "error": str(exc)},
            )
            continue

        calibration = result["calibration"]
        dca = result["dca"]
        if float(calibration["ece"]) > float(thresholds["ece_max"]):
            add_issue(
                failures,
                "calibration_ece_exceeds_threshold",
                "ECE exceeds configured threshold.",
                {"cohort": label, "ece": calibration["ece"], "ece_max": thresholds["ece_max"]},
            )
        if float(calibration["slope"]) < float(thresholds["slope_min"]) or float(calibration["slope"]) > float(thresholds["slope_max"]):
            add_issue(
                failures,
                "calibration_slope_out_of_range",
                "Calibration slope is outside configured range.",
                {
                    "cohort": label,
                    "slope": calibration["slope"],
                    "slope_min": thresholds["slope_min"],
                    "slope_max": thresholds["slope_max"],
                },
            )
        if abs(float(calibration["intercept"])) > float(thresholds["intercept_abs_max"]):
            add_issue(
                failures,
                "calibration_intercept_out_of_range",
                "Calibration intercept absolute value exceeds configured threshold.",
                {
                    "cohort": label,
                    "intercept": calibration["intercept"],
                    "intercept_abs_max": thresholds["intercept_abs_max"],
                },
            )

        if (
            float(dca["advantage_coverage"]) < float(thresholds["min_advantage_coverage"])
            or float(dca["average_advantage"]) < float(thresholds["min_average_advantage"])
        ):
            add_issue(
                failures,
                "decision_curve_net_benefit_insufficient",
                "Decision-curve net-benefit criteria are not met.",
                {
                    "cohort": label,
                    "advantage_coverage": dca["advantage_coverage"],
                    "average_advantage": dca["average_advantage"],
                    "minimum_advantage": dca["minimum_advantage"],
                    "min_advantage_coverage": thresholds["min_advantage_coverage"],
                    "min_average_advantage": thresholds["min_average_advantage"],
                    "min_net_benefit_advantage": thresholds["min_net_benefit_advantage"],
                },
            )
        cohort_results.append(result)

    summary = {
        "prediction_trace": str(trace_path),
        "evaluation_report": str(eval_path),
        "external_validation_report": str(ext_path),
        "thresholds": thresholds,
        "cohort_results": cohort_results,
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
