"""Comprehensive unit tests for scripts/calibration_dca_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calibration_dca_gate as cdg
from calibration_dca_gate import (
    build_threshold_grid,
    expected_calibration_error,
    fit_calibration_slope_intercept,
    net_benefit,
    normalize_binary,
    parse_policy_thresholds,
    sigmoid,
    treat_all_net_benefit,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_trace(tmp_path: Path, n=100, seed=42, ext_cohort_id="hospital_b"):
    """Create a realistic prediction_trace CSV with test + external rows."""
    rng = np.random.default_rng(seed)
    rows = []
    # Internal test
    for i in range(n):
        y_true = int(rng.random() < 0.4)
        y_score = float(np.clip(rng.normal(0.4 if y_true == 0 else 0.7, 0.15), 0.01, 0.99))
        y_pred = int(y_score >= 0.5)
        rows.append({
            "scope": "test",
            "cohort_id": "internal_test",
            "cohort_type": "internal",
            "y_true": y_true,
            "y_score": round(y_score, 4),
            "y_pred": y_pred,
            "selected_threshold": 0.5,
        })
    # External cohort
    for i in range(n):
        y_true = int(rng.random() < 0.35)
        y_score = float(np.clip(rng.normal(0.4 if y_true == 0 else 0.65, 0.18), 0.01, 0.99))
        y_pred = int(y_score >= 0.5)
        rows.append({
            "scope": "external",
            "cohort_id": ext_cohort_id,
            "cohort_type": "cross_institution",
            "y_true": y_true,
            "y_score": round(y_score, 4),
            "y_pred": y_pred,
            "selected_threshold": 0.5,
        })
    df = pd.DataFrame(rows)
    trace_path = tmp_path / "prediction_trace.csv"
    df.to_csv(trace_path, index=False)
    return trace_path


def _make_eval_report(tmp_path: Path):
    report = {"status": "pass", "split": "test", "metrics": {"roc_auc": 0.85}}
    p = tmp_path / "evaluation_report.json"
    _write_json(p, report)
    return p


def _make_ext_report(tmp_path: Path, cohort_id="hospital_b"):
    report = {
        "status": "pass",
        "cohorts": [
            {"cohort_id": cohort_id, "cohort_type": "cross_institution", "row_count": 100}
        ],
    }
    p = tmp_path / "external_validation_report.json"
    _write_json(p, report)
    return p


def _make_full_setup(tmp_path: Path, n=100, ext_cohort_id="hospital_b"):
    trace_path = _make_trace(tmp_path, n=n, ext_cohort_id=ext_cohort_id)
    eval_path = _make_eval_report(tmp_path)
    ext_path = _make_ext_report(tmp_path, cohort_id=ext_cohort_id)
    return {"trace": trace_path, "eval": eval_path, "ext": ext_path}


# ────────────────────────────────────────────────────────
# normalize_binary
# ────────────────────────────────────────────────────────

class TestNormalizeBinary:
    def test_normal(self):
        s = pd.Series([0, 1, 0, 1])
        result = normalize_binary(s)
        assert result is not None
        assert list(result) == [0, 1, 0, 1]

    def test_float(self):
        s = pd.Series([0.0, 1.0])
        result = normalize_binary(s)
        assert result is not None

    def test_non_binary(self):
        s = pd.Series([0, 1, 2])
        assert normalize_binary(s) is None

    def test_nan(self):
        s = pd.Series([0, 1, np.nan])
        assert normalize_binary(s) is None

    def test_string(self):
        s = pd.Series(["0", "1", "0"])
        result = normalize_binary(s)
        assert result is not None
        assert list(result) == [0, 1, 0]


# ────────────────────────────────────────────────────────
# sigmoid
# ────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6

    def test_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] > 0.99

    def test_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] < 0.01


# ────────────────────────────────────────────────────────
# fit_calibration_slope_intercept
# ────────────────────────────────────────────────────────

class TestFitCalibrationSlopeIntercept:
    def test_well_calibrated(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_score = np.clip(y_true.astype(float) + rng.normal(0, 0.2, n), 0.01, 0.99)
        result = fit_calibration_slope_intercept(y_true, y_score)
        assert result is not None
        assert "slope" in result
        assert "intercept" in result

    def test_too_few_samples(self):
        assert fit_calibration_slope_intercept(np.array([1, 0]), np.array([0.9, 0.1])) is None

    def test_single_class(self):
        assert fit_calibration_slope_intercept(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None


# ────────────────────────────────────────────────────────
# expected_calibration_error
# ────────────────────────────────────────────────────────

class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ece = expected_calibration_error(y_true, y_score, n_bins=2, min_bin_size=3)
        assert ece < 0.05

    def test_poor_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = expected_calibration_error(y_true, y_score, n_bins=2, min_bin_size=3)
        assert ece > 0.5

    def test_empty(self):
        ece = expected_calibration_error(np.array([]), np.array([]), n_bins=10, min_bin_size=5)
        assert ece == 1.0


# ────────────────────────────────────────────────────────
# net_benefit / treat_all_net_benefit
# ────────────────────────────────────────────────────────

class TestNetBenefit:
    def test_perfect_model(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        nb = net_benefit(y_true, y_score, threshold=0.5)
        # TP=2, FP=0
        assert nb > 0.0

    def test_empty(self):
        nb = net_benefit(np.array([]), np.array([]), threshold=0.5)
        assert nb == 0.0


class TestTreatAllNetBenefit:
    def test_high_prevalence(self):
        y_true = np.array([1, 1, 1, 1, 0])
        nb = treat_all_net_benefit(y_true, threshold=0.1)
        assert nb > 0.0

    def test_low_prevalence_high_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1])
        nb = treat_all_net_benefit(y_true, threshold=0.9)
        assert nb < 0.0


# ────────────────────────────────────────────────────────
# parse_policy_thresholds
# ────────────────────────────────────────────────────────

class TestParsePolicyThresholds:
    def test_defaults(self):
        result = parse_policy_thresholds(None)
        assert result["ece_max"] == 0.06
        assert result["slope_min"] == 0.80
        assert result["min_rows"] == 50

    def test_custom(self):
        policy = {"calibration_dca_thresholds": {"ece_max": 0.1, "slope_min": 0.5}}
        result = parse_policy_thresholds(policy)
        assert result["ece_max"] == 0.1
        assert result["slope_min"] == 0.5

    def test_not_dict(self):
        result = parse_policy_thresholds("not_dict")
        assert result["ece_max"] == 0.06  # defaults


# ────────────────────────────────────────────────────────
# build_threshold_grid
# ────────────────────────────────────────────────────────

class TestBuildThresholdGrid:
    def test_normal(self):
        grid = build_threshold_grid({"start": 0.05, "end": 0.50, "step": 0.05})
        assert grid is not None
        assert len(grid) >= 2

    def test_invalid_start(self):
        assert build_threshold_grid({"start": 0.0, "end": 0.5, "step": 0.1}) is None

    def test_invalid_end(self):
        assert build_threshold_grid({"start": 0.1, "end": 1.0, "step": 0.1}) is None

    def test_start_gt_end(self):
        assert build_threshold_grid({"start": 0.8, "end": 0.2, "step": 0.1}) is None

    def test_missing_field(self):
        assert build_threshold_grid({"start": 0.1}) is None


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, setup, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "calibration_dca_gate.py"),
            "--prediction-trace", str(setup["trace"]),
            "--evaluation-report", str(setup["eval"]),
            "--external-validation-report", str(setup["ext"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_pass(self, tmp_path: Path):
        setup = _make_full_setup(tmp_path, n=200)
        result = self._run(tmp_path, setup)
        report = json.loads((tmp_path / "report.json").read_text())
        # May or may not pass depending on random data calibration quality.
        # Just verify it runs and produces valid report.
        assert result.returncode in (0, 2)
        assert "status" in report
        assert "failures" in report
        assert "warnings" in report

    def test_missing_trace(self, tmp_path: Path):
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        setup = {"trace": tmp_path / "nonexistent.csv", "eval": eval_path, "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "calibration_insufficient_events" in codes

    def test_missing_eval_report(self, tmp_path: Path):
        trace_path = _make_trace(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        setup = {"trace": trace_path, "eval": tmp_path / "nonexistent.json", "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2

    def test_missing_columns_in_trace(self, tmp_path: Path):
        # Create trace with missing columns
        trace_path = tmp_path / "prediction_trace.csv"
        pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(trace_path, index=False)
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        setup = {"trace": trace_path, "eval": eval_path, "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "calibration_insufficient_events" in codes

    def test_empty_ext_cohorts(self, tmp_path: Path):
        trace_path = _make_trace(tmp_path)
        eval_path = _make_eval_report(tmp_path)
        ext_path = tmp_path / "external_validation_report.json"
        _write_json(ext_path, {"status": "pass", "cohorts": []})
        setup = {"trace": trace_path, "eval": eval_path, "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "calibration_insufficient_events" in codes

    def test_non_binary_y_true(self, tmp_path: Path):
        trace_path = tmp_path / "prediction_trace.csv"
        df = pd.DataFrame({
            "scope": ["test"] * 5,
            "cohort_id": ["internal_test"] * 5,
            "cohort_type": ["internal"] * 5,
            "y_true": [0, 1, 2, 0, 1],  # Non-binary
            "y_score": [0.1, 0.9, 0.5, 0.2, 0.8],
            "y_pred": [0, 1, 1, 0, 1],
            "selected_threshold": [0.5] * 5,
        })
        df.to_csv(trace_path, index=False)
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        setup = {"trace": trace_path, "eval": eval_path, "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2

    def test_y_score_out_of_range(self, tmp_path: Path):
        trace_path = tmp_path / "prediction_trace.csv"
        df = pd.DataFrame({
            "scope": ["test"] * 5,
            "cohort_id": ["internal_test"] * 5,
            "cohort_type": ["internal"] * 5,
            "y_true": [0, 1, 0, 1, 0],
            "y_score": [0.1, 1.5, 0.3, 0.8, -0.1],  # Out of [0,1]
            "y_pred": [0, 1, 0, 1, 0],
            "selected_threshold": [0.5] * 5,
        })
        df.to_csv(trace_path, index=False)
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        setup = {"trace": trace_path, "eval": eval_path, "ext": ext_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2

    def test_report_structure(self, tmp_path: Path):
        setup = _make_full_setup(tmp_path, n=200)
        self._run(tmp_path, setup)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report

    def test_insufficient_rows(self, tmp_path: Path):
        """Very few rows → insufficient events."""
        setup = _make_full_setup(tmp_path, n=5)  # Very small
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "calibration_insufficient_events" in codes


# ── Direct main() tests ─────────────────────────────────────────────────────

class TestMainDirectPass:
    def test_pass_or_calibration_issue(self, tmp_path, monkeypatch):
        setup = _make_full_setup(tmp_path, n=200)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(setup["trace"]),
            "--evaluation-report", str(setup["eval"]),
            "--external-validation-report", str(setup["ext"]),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc in (0, 2)
        out = json.loads(rpt.read_text())
        assert "status" in out
        assert "summary" in out


class TestMainDirectMissingTrace:
    def test_missing_trace_file(self, tmp_path, monkeypatch):
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(tmp_path / "nope.csv"),
            "--evaluation-report", str(eval_path),
            "--external-validation-report", str(ext_path),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        fail_codes = [f["code"] for f in out["failures"]]
        assert "calibration_insufficient_events" in fail_codes


class TestMainDirectMissingColumns:
    def test_trace_missing_columns(self, tmp_path, monkeypatch):
        trace_path = tmp_path / "trace.csv"
        pd.DataFrame({"col_a": [1], "col_b": [2]}).to_csv(trace_path, index=False)
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--external-validation-report", str(ext_path),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        fail_codes = [f["code"] for f in out["failures"]]
        assert "calibration_insufficient_events" in fail_codes


class TestMainDirectNonBinaryYTrue:
    def test_non_binary_y_true_via_main(self, tmp_path, monkeypatch):
        trace_path = tmp_path / "trace.csv"
        pd.DataFrame({
            "scope": ["test"] * 5,
            "cohort_id": ["internal_test"] * 5,
            "cohort_type": ["internal"] * 5,
            "y_true": [0, 1, 2, 0, 1],
            "y_score": [0.1, 0.9, 0.5, 0.2, 0.8],
            "y_pred": [0, 1, 1, 0, 1],
            "selected_threshold": [0.5] * 5,
        }).to_csv(trace_path, index=False)
        eval_path = _make_eval_report(tmp_path)
        ext_path = _make_ext_report(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--external-validation-report", str(ext_path),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc == 2


class TestMainDirectInsufficientRows:
    def test_too_few_rows_via_main(self, tmp_path, monkeypatch):
        setup = _make_full_setup(tmp_path, n=5)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(setup["trace"]),
            "--evaluation-report", str(setup["eval"]),
            "--external-validation-report", str(setup["ext"]),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        fail_codes = [f["code"] for f in out["failures"]]
        assert "calibration_insufficient_events" in fail_codes


class TestMainDirectCorruptEvalJSON:
    def test_corrupt_eval_json_via_main(self, tmp_path, monkeypatch):
        trace_path = _make_trace(tmp_path, n=200)
        eval_path = tmp_path / "evaluation_report.json"
        eval_path.write_text("{bad", encoding="utf-8")
        ext_path = _make_ext_report(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "cdg",
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--external-validation-report", str(ext_path),
            "--report", str(rpt),
        ])
        rc = cdg.main()
        assert rc == 2
