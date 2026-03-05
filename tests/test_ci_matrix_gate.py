"""Tests for scripts/ci_matrix_gate.py.

Covers helper functions (to_int, normalize_binary, safe_ratio, parse_ci_policy,
verify_threshold_stable, stratified_bootstrap_indices), CI computation,
and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "ci_matrix_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import ci_matrix_gate as cmg


# ── helper functions ─────────────────────────────────────────────────────────

class TestToInt:
    def test_int(self):
        assert cmg.to_int(5) == 5

    def test_bool(self):
        assert cmg.to_int(True) is None

    def test_float_integer(self):
        assert cmg.to_int(5.0) == 5

    def test_none(self):
        assert cmg.to_int(None) is None


class TestSafeRatio:
    def test_normal(self):
        assert cmg.safe_ratio(10.0, 20.0) == 0.5

    def test_zero_denom(self):
        assert cmg.safe_ratio(10.0, 0.0) == 0.0


class TestNormalizeBinary:
    def test_valid(self):
        s = pd.Series([0, 1, 0, 1])
        result = cmg.normalize_binary(s)
        assert result is not None
        assert list(result) == [0, 1, 0, 1]

    def test_non_binary(self):
        assert cmg.normalize_binary(pd.Series([0, 1, 2])) is None


class TestParseCiPolicy:
    def test_defaults(self):
        result = cmg.parse_ci_policy(None)
        assert result["n_resamples"] == 2000
        assert result["max_width"] == 0.20
        assert result["beta"] == 2.0
        assert result["transport_ci_required"] is True

    def test_custom(self):
        policy = {"ci_policy": {"n_resamples": 500, "max_width": 0.15}}
        result = cmg.parse_ci_policy(policy)
        assert result["n_resamples"] == 500
        assert result["max_width"] == 0.15

    def test_transport_bool(self):
        policy = {"ci_policy": {"transport_ci_required": False}}
        result = cmg.parse_ci_policy(policy)
        assert result["transport_ci_required"] is False


class TestVerifyThresholdStable:
    def test_stable(self):
        df = pd.DataFrame({"selected_threshold": [0.5, 0.5, 0.5]})
        assert cmg.verify_threshold_stable(df) == 0.5

    def test_unstable(self):
        df = pd.DataFrame({"selected_threshold": [0.5, 0.6, 0.5]})
        assert cmg.verify_threshold_stable(df) is None

    def test_empty(self):
        df = pd.DataFrame({"selected_threshold": []})
        assert cmg.verify_threshold_stable(df) is None


class TestStratifiedBootstrapIndices:
    def test_basic(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        rng = np.random.default_rng(42)
        idx = cmg.stratified_bootstrap_indices(y, rng)
        assert idx is not None
        assert len(idx) == len(y)

    def test_single_class(self):
        y = np.array([0, 0, 0])
        rng = np.random.default_rng(42)
        assert cmg.stratified_bootstrap_indices(y, rng) is None


# ── CLI integration ──────────────────────────────────────────────────────────

def _generate_cohort_data(n_pos, n_neg, threshold=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_score = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, len(y_true)), rng.beta(2, 5, len(y_true))),
        0.0, 1.0,
    )
    y_pred = (y_score >= threshold).astype(int)
    return y_true, y_score, y_pred


def _build_test_artifacts(tmp_path, threshold=0.5):
    np.random.RandomState(42)
    rows = []

    # Internal splits
    for split_name, (n_pos, n_neg), seed in [("train", (40, 40), 10), ("valid", (20, 20), 20), ("test", (30, 30), 30)]:
        y_true, y_score, y_pred = _generate_cohort_data(n_pos, n_neg, threshold, np.random.RandomState(seed))
        for i in range(len(y_true)):
            rows.append({
                "scope": split_name, "cohort_id": "internal", "cohort_type": "internal",
                "hashed_patient_id": f"{split_name}_{i}", "y_true": int(y_true[i]),
                "y_score": float(y_score[i]), "y_pred": int(y_pred[i]),
                "selected_threshold": threshold, "model_id": "logistic",
            })

    # External cohort
    rng_ext = np.random.RandomState(99)
    y_true_ext, y_score_ext, y_pred_ext = _generate_cohort_data(25, 25, threshold, rng_ext)
    for i in range(len(y_true_ext)):
        rows.append({
            "scope": "external", "cohort_id": "hospital_b", "cohort_type": "cross_institution",
            "hashed_patient_id": f"ext_{i}", "y_true": int(y_true_ext[i]),
            "y_score": float(y_score_ext[i]), "y_pred": int(y_pred_ext[i]),
            "selected_threshold": threshold, "model_id": "logistic",
        })

    trace_path = tmp_path / "prediction_trace.csv"
    pd.DataFrame(rows).to_csv(trace_path, index=False)

    # Evaluation report (needs metrics for test split)
    test_y = np.array([r["y_true"] for r in rows if r["scope"] == "test"])
    test_s = np.array([r["y_score"] for r in rows if r["scope"] == "test"])
    test_metrics = cmg.metric_panel(test_y, test_s, threshold, beta=2.0)
    eval_report = {"metrics": test_metrics, "split_metrics": {"test": {"metrics": test_metrics}}}
    eval_path = tmp_path / "evaluation_report.json"
    eval_path.write_text(json.dumps(eval_report))

    # External validation report
    ext_y = np.array([r["y_true"] for r in rows if r["scope"] == "external"])
    ext_s = np.array([r["y_score"] for r in rows if r["scope"] == "external"])
    ext_metrics = cmg.metric_panel(ext_y, ext_s, threshold, beta=2.0)
    ext_report = {
        "cohorts": [{
            "cohort_id": "hospital_b",
            "cohort_type": "cross_institution",
            "metrics": ext_metrics,
            "row_count": len(ext_y),
            "positive_count": int(np.sum(ext_y)),
        }]
    }
    ext_path = tmp_path / "external_validation_report.json"
    ext_path.write_text(json.dumps(ext_report))

    return trace_path, eval_path, ext_path


def _run_gate(tmp_path, trace_path=None, eval_path=None, ext_path=None,
              strict=False, policy=None, extra_args=None):
    if trace_path is None or eval_path is None or ext_path is None:
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
    ci_matrix_path = tmp_path / "ci_matrix_report.json"
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--evaluation-report", str(eval_path),
        "--prediction-trace", str(trace_path),
        "--external-validation-report", str(ext_path),
        "--ci-matrix-report", str(ci_matrix_path),
        "--update-ci-matrix-report",
        "--report", str(report_path),
    ]
    if policy is not None:
        pp_path = tmp_path / "policy.json"
        pp_path.write_text(json.dumps(policy))
        cmd.extend(["--performance-policy", str(pp_path)])
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(SCRIPTS_DIR))
    result = {}
    if report_path.exists():
        result = json.loads(report_path.read_text())
    return result


class TestCLIPass:
    def test_valid_pass(self, tmp_path):
        # Use small n_resamples and relaxed max_width for speed / small samples
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000, "max_width": 0.40}}
        report = _run_gate(tmp_path, policy=policy)
        assert report.get("status") == "pass", f"failures: {report.get('failures', [])}"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path):
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000, "max_width": 0.40}}
        report = _run_gate(tmp_path, policy=policy)
        assert "status" in report
        assert "summary" in report
        s = report["summary"]
        assert "ci_matrix_report" in s
        assert "ci_policy" in s


class TestMissingFiles:
    def test_missing_eval_report(self, tmp_path):
        trace_path, _, ext_path = _build_test_artifacts(tmp_path)
        report = _run_gate(tmp_path, trace_path=trace_path,
                           eval_path=tmp_path / "nope.json", ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "ci_matrix_missing_required_metric" in codes

    def test_missing_trace(self, tmp_path):
        _, eval_path, ext_path = _build_test_artifacts(tmp_path)
        report = _run_gate(tmp_path, trace_path=tmp_path / "nope.csv",
                           eval_path=eval_path, ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "ci_matrix_missing_required_metric" in codes

    def test_missing_ext_report(self, tmp_path):
        trace_path, eval_path, _ = _build_test_artifacts(tmp_path)
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=tmp_path / "nope.json")
        codes = [f["code"] for f in report["failures"]]
        assert "transport_ci_invalid" in codes


class TestTraceValidation:
    def test_missing_columns(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        df = df.drop(columns=["y_score"])
        bad_trace = tmp_path / "bad_trace.csv"
        df.to_csv(bad_trace, index=False)
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000}}
        report = _run_gate(tmp_path, trace_path=bad_trace, eval_path=eval_path,
                           ext_path=ext_path, policy=policy)
        codes = [f["code"] for f in report["failures"]]
        assert "ci_matrix_missing_required_metric" in codes

    def test_non_binary_y_true(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        df.loc[0, "y_true"] = 2
        bad_trace = tmp_path / "bad_trace2.csv"
        df.to_csv(bad_trace, index=False)
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000}}
        report = _run_gate(tmp_path, trace_path=bad_trace, eval_path=eval_path,
                           ext_path=ext_path, policy=policy)
        codes = [f["code"] for f in report["failures"]]
        assert "ci_matrix_missing_required_metric" in codes


class TestExternalCohort:
    def test_empty_cohorts(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"] = []
        ext_path.write_text(json.dumps(ext))
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000,
                                "transport_ci_required": True}}
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=ext_path, policy=policy)
        codes = [f["code"] for f in report["failures"]]
        assert "transport_ci_invalid" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        policy = {"ci_policy": {"n_resamples": 100, "max_resamples_supported": 4000}}
        report = _run_gate(tmp_path, strict=True, policy=policy)
        assert report["strict_mode"] is True
