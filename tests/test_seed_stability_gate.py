"""Tests for scripts/seed_stability_gate.py.

Covers helper functions (is_finite_number, parse_int_like, parse_thresholds,
metric_bounds_ok), per-seed validation, summary consistency, threshold checks,
and CLI integration.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "seed_stability_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import seed_stability_gate as ssg


# ── helper functions ─────────────────────────────────────────────────────────

class TestIsFiniteNumber:
    def test_int(self):
        assert ssg.is_finite_number(5) is True

    def test_bool(self):
        assert ssg.is_finite_number(True) is False

    def test_nan(self):
        assert ssg.is_finite_number(float("nan")) is False

    def test_none(self):
        assert ssg.is_finite_number(None) is False


class TestParseIntLike:
    def test_int(self):
        assert ssg.parse_int_like(42) == 42

    def test_float_integer(self):
        assert ssg.parse_int_like(5.0) == 5

    def test_float_non_integer(self):
        assert ssg.parse_int_like(5.5) is None

    def test_bool(self):
        assert ssg.parse_int_like(True) is None

    def test_none(self):
        assert ssg.parse_int_like(None) is None


class TestParseThresholds:
    def test_defaults(self):
        result = ssg.parse_thresholds(None)
        assert result["pr_auc_std_max"] == 0.03
        assert result["brier_range_max"] == 0.05

    def test_custom(self):
        policy = {"seed_stability_thresholds": {"pr_auc_std_max": 0.05}}
        result = ssg.parse_thresholds(policy)
        assert result["pr_auc_std_max"] == 0.05


class TestMetricBoundsOk:
    def test_pr_auc_valid(self):
        assert ssg.metric_bounds_ok("pr_auc", 0.85) is True

    def test_pr_auc_out_of_range(self):
        assert ssg.metric_bounds_ok("pr_auc", 1.5) is False

    def test_unknown_metric(self):
        assert ssg.metric_bounds_ok("custom", 999.0) is True


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_seed_result(seed, pr_auc=0.82, f2_beta=0.80, brier=0.15):
    return {
        "seed": seed,
        "test_metrics": {"pr_auc": pr_auc, "f2_beta": f2_beta, "brier": brier},
    }


def _make_seed_report(seeds=None, overall_pr_auc=0.82):
    if seeds is None:
        seeds = [
            _make_seed_result(42, 0.82, 0.80, 0.15),
            _make_seed_result(43, 0.81, 0.79, 0.16),
            _make_seed_result(44, 0.83, 0.81, 0.14),
            _make_seed_result(45, 0.82, 0.80, 0.15),
            _make_seed_result(46, 0.81, 0.80, 0.15),
        ]

    pr_aucs = [s["test_metrics"]["pr_auc"] for s in seeds]
    f2s = [s["test_metrics"]["f2_beta"] for s in seeds]
    briers = [s["test_metrics"]["brier"] for s in seeds]

    def _summary(vals):
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "range": max(vals) - min(vals),
            "n": float(len(vals)),
        }

    return {
        "primary_metric": "pr_auc",
        "selection_data": "cv_inner",
        "threshold_selection_split": "valid",
        "per_seed_results": seeds,
        "summary": {
            "pr_auc": _summary(pr_aucs),
            "f2_beta": _summary(f2s),
            "brier": _summary(briers),
        },
    }


def _run_gate(tmp_path, seed_report=None, policy=None, strict=False):
    if seed_report is None:
        seed_report = _make_seed_report()
    sr_path = tmp_path / "seed_report.json"
    sr_path.write_text(json.dumps(seed_report))
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--seed-sensitivity-report", str(sr_path),
        "--report", str(report_path),
    ]
    if policy is not None:
        pp_path = tmp_path / "policy.json"
        pp_path.write_text(json.dumps(policy))
        cmd.extend(["--performance-policy", str(pp_path)])
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_pass(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "summary" in report
        s = report["summary"]
        assert "computed_metrics" in s
        assert "thresholds" in s


class TestMissingFile:
    def test_missing_report(self, tmp_path):
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--seed-sensitivity-report", str(tmp_path / "nope.json"),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_seed_sensitivity_report" in codes


class TestPrimaryMetric:
    def test_wrong_primary(self, tmp_path):
        sr = _make_seed_report()
        sr["primary_metric"] = "roc_auc"
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_stability_primary_metric_mismatch" in codes


class TestSelectionData:
    def test_test_in_selection_data(self, tmp_path):
        sr = _make_seed_report()
        sr["selection_data"] = "test_set"
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_stability_selection_data_invalid" in codes


class TestThresholdSplit:
    def test_invalid_split(self, tmp_path):
        sr = _make_seed_report()
        sr["threshold_selection_split"] = "test"
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_stability_threshold_split_invalid" in codes


class TestInsufficientSeeds:
    def test_too_few_seeds(self, tmp_path):
        sr = _make_seed_report(seeds=[
            _make_seed_result(42, 0.82, 0.80, 0.15),
            _make_seed_result(43, 0.81, 0.79, 0.16),
        ])
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_seed_runs" in codes

    def test_strict_needs_5(self, tmp_path):
        sr = _make_seed_report(seeds=[
            _make_seed_result(i, 0.82, 0.80, 0.15) for i in range(4)
        ])
        report = _run_gate(tmp_path, seed_report=sr, strict=True)
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_seed_runs" in codes


class TestDuplicateSeed:
    def test_duplicate(self, tmp_path):
        sr = _make_seed_report(seeds=[
            _make_seed_result(42, 0.82, 0.80, 0.15),
            _make_seed_result(42, 0.81, 0.79, 0.16),
            _make_seed_result(43, 0.83, 0.81, 0.14),
            _make_seed_result(44, 0.82, 0.80, 0.15),
            _make_seed_result(45, 0.81, 0.80, 0.15),
        ])
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "duplicate_seed_result" in codes


class TestStabilityThreshold:
    def test_pr_auc_std_exceeds(self, tmp_path):
        # Large std: 0.60, 0.90 → std > 0.03
        sr = _make_seed_report(seeds=[
            _make_seed_result(i, pr_auc=0.60 + i * 0.08, f2_beta=0.80, brier=0.15)
            for i in range(5)
        ])
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_stability_exceeds_threshold" in codes


class TestSummaryMismatch:
    def test_summary_mismatch(self, tmp_path):
        sr = _make_seed_report()
        sr["summary"]["pr_auc"]["mean"] = 0.999  # wrong
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_summary_mismatch" in codes

    def test_missing_summary(self, tmp_path):
        sr = _make_seed_report()
        del sr["summary"]
        report = _run_gate(tmp_path, seed_report=sr)
        codes = [f["code"] for f in report["failures"]]
        assert "seed_summary_missing" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True
