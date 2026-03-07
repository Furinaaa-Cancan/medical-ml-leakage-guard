"""Tests for scripts/robustness_gate.py.

Covers helper functions (is_finite_number, parse_bucket_thresholds,
metric_in_unit_interval, validate_bucket, compare_summary_fields),
bucket validation, threshold checks, summary consistency, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "robustness_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import robustness_gate as rg


# ── helper functions ─────────────────────────────────────────────────────────

class TestIsFiniteNumber:
    def test_int(self):
        assert rg.is_finite_number(5) is True

    def test_float(self):
        assert rg.is_finite_number(0.5) is True

    def test_bool(self):
        assert rg.is_finite_number(True) is False

    def test_nan(self):
        assert rg.is_finite_number(float("nan")) is False

    def test_inf(self):
        assert rg.is_finite_number(float("inf")) is False

    def test_none(self):
        assert rg.is_finite_number(None) is False

    def test_string(self):
        assert rg.is_finite_number("5") is False


class TestParseBucketThresholds:
    def test_defaults(self):
        result = rg.parse_bucket_thresholds(None, "time_slices")
        assert result["pr_auc_drop_fail"] == 0.14
        assert result["min_slice_size"] == 8

    def test_custom(self):
        policy = {"robustness_thresholds": {"time_slices": {"pr_auc_drop_fail": 0.20}}}
        result = rg.parse_bucket_thresholds(policy, "time_slices")
        assert result["pr_auc_drop_fail"] == 0.20

    def test_negative_ignored(self):
        policy = {"robustness_thresholds": {"time_slices": {"pr_auc_drop_fail": -0.1}}}
        result = rg.parse_bucket_thresholds(policy, "time_slices")
        assert result["pr_auc_drop_fail"] == 0.14  # default


class TestMetricInUnitInterval:
    def test_valid(self):
        assert rg.metric_in_unit_interval(0.5) is True

    def test_boundary(self):
        assert rg.metric_in_unit_interval(0.0) is True
        assert rg.metric_in_unit_interval(1.0) is True

    def test_out_of_range(self):
        assert rg.metric_in_unit_interval(1.5) is False
        assert rg.metric_in_unit_interval(-0.1) is False


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_slice_row(pr_auc, n=50, positive_count=15):
    return {"n": n, "positive_count": positive_count, "metrics": {"pr_auc": pr_auc}}


def _make_robustness_report(overall_pr_auc=0.82, slice_pr_aucs=None, group_pr_aucs=None):
    if slice_pr_aucs is None:
        slice_pr_aucs = [0.80, 0.81, 0.83, 0.82]
    if group_pr_aucs is None:
        group_pr_aucs = [0.81, 0.80, 0.83, 0.82]

    slices = [_make_slice_row(v) for v in slice_pr_aucs]
    groups = [_make_slice_row(v) for v in group_pr_aucs]

    pr_auc_values_ts = slice_pr_aucs
    pr_auc_values_ph = group_pr_aucs

    return {
        "primary_metric": "pr_auc",
        "overall_test_metrics": {"pr_auc": overall_pr_auc},
        "time_slices": {"slices": slices},
        "patient_hash_groups": {"groups": groups},
        "summary": {
            "time_slices": {
                "pr_auc_min": min(pr_auc_values_ts),
                "pr_auc_max": max(pr_auc_values_ts),
                "pr_auc_range": max(pr_auc_values_ts) - min(pr_auc_values_ts),
                "pr_auc_worst_drop_from_overall": overall_pr_auc - min(pr_auc_values_ts),
                "n_rows": float(len(pr_auc_values_ts)),
            },
            "patient_hash_groups": {
                "pr_auc_min": min(pr_auc_values_ph),
                "pr_auc_max": max(pr_auc_values_ph),
                "pr_auc_range": max(pr_auc_values_ph) - min(pr_auc_values_ph),
                "pr_auc_worst_drop_from_overall": overall_pr_auc - min(pr_auc_values_ph),
                "n_rows": float(len(pr_auc_values_ph)),
            },
        },
    }


def _run_gate(tmp_path, robustness=None, policy=None, strict=False):
    if robustness is None:
        robustness = _make_robustness_report()
    rr_path = tmp_path / "robustness_report.json"
    rr_path.write_text(json.dumps(robustness))
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--robustness-report", str(rr_path),
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
        assert "strict_mode" in report
        assert "summary" in report
        s = report["summary"]
        assert "computed" in s
        assert "thresholds" in s


class TestMissingFile:
    def test_missing_robustness_report(self, tmp_path):
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--robustness-report", str(tmp_path / "nope.json"),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_robustness_report" in codes


class TestPrimaryMetric:
    def test_wrong_primary_metric(self, tmp_path):
        rob = _make_robustness_report()
        rob["primary_metric"] = "roc_auc"
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_primary_metric_mismatch" in codes


class TestOverallMetrics:
    def test_missing_overall(self, tmp_path):
        rob = _make_robustness_report()
        del rob["overall_test_metrics"]
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_missing_overall_metric" in codes

    def test_out_of_range_overall(self, tmp_path):
        rob = _make_robustness_report()
        rob["overall_test_metrics"]["pr_auc"] = 1.5
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_metric_out_of_range" in codes


class TestBucketValidation:
    def test_missing_time_slices(self, tmp_path):
        rob = _make_robustness_report()
        del rob["time_slices"]
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_missing_bucket" in codes

    def test_empty_slices(self, tmp_path):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = []
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_missing_bucket_rows" in codes

    def test_invalid_row(self, tmp_path):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = ["not_a_dict"]
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_invalid_bucket_row" in codes

    def test_non_finite_pr_auc(self, tmp_path):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = [{"n": 50, "positive_count": 10, "metrics": {"pr_auc": "bad"}}]
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_non_finite_metric" in codes


class TestDropThreshold:
    def test_pr_auc_drop_exceeds_fail(self, tmp_path):
        # overall=0.85, worst slice=0.65 → drop=0.20 > fail=0.14
        rob = _make_robustness_report(overall_pr_auc=0.85, slice_pr_aucs=[0.65, 0.80, 0.82])
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_pr_auc_drop_exceeds_threshold" in codes

    def test_pr_auc_range_exceeds_fail(self, tmp_path):
        # range = 0.85 - 0.60 = 0.25 > fail=0.20
        rob = _make_robustness_report(overall_pr_auc=0.85, slice_pr_aucs=[0.60, 0.85])
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_pr_auc_range_exceeds_threshold" in codes

    def test_warning_drop(self, tmp_path):
        # overall=0.82, worst=0.71 → drop=0.11, warn=0.10, fail=0.14 → warning
        rob = _make_robustness_report(overall_pr_auc=0.82, slice_pr_aucs=[0.71, 0.80, 0.81])
        report = _run_gate(tmp_path, robustness=rob)
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "robustness_pr_auc_drop_near_threshold" in warn_codes


class TestSummaryMismatch:
    def test_summary_mismatch(self, tmp_path):
        rob = _make_robustness_report()
        rob["summary"]["time_slices"]["pr_auc_min"] = 0.50  # wrong
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_summary_mismatch" in codes

    def test_missing_summary(self, tmp_path):
        rob = _make_robustness_report()
        del rob["summary"]
        report = _run_gate(tmp_path, robustness=rob)
        codes = [f["code"] for f in report["failures"]]
        assert "robustness_summary_missing" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True

    def test_strict_warning_fails(self, tmp_path):
        rob = _make_robustness_report(overall_pr_auc=0.82, slice_pr_aucs=[0.71, 0.80, 0.81])
        report = _run_gate(tmp_path, robustness=rob, strict=True)
        assert report["status"] == "fail"


# ── Direct main() tests ────────────────────────────────────────────────────

def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestMainPass:
    def test_pass(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 0
        out = json.loads(rpt.read_text())
        assert out["status"] == "pass"
        assert out["failure_count"] == 0


class TestMainMissingFile:
    def test_missing_robustness_report(self, tmp_path, monkeypatch):
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(tmp_path / "nope.json"),
            "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "missing_robustness_report" in codes


class TestMainInvalidJSON:
    def test_invalid_json(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.json"
        bad.write_text("{bad", encoding="utf-8")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(bad), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "invalid_robustness_report" in codes


class TestMainPrimaryMetricMismatch:
    def test_wrong_metric(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["primary_metric"] = "roc_auc"
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_primary_metric_mismatch" in codes


class TestMainMissingOverall:
    def test_no_overall_metrics(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        del rob["overall_test_metrics"]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_missing_overall_metric" in codes


class TestMainBucketMissing:
    def test_missing_time_slices(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        del rob["time_slices"]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_missing_bucket" in codes


class TestMainDropFail:
    def test_pr_auc_drop_exceeds_threshold(self, tmp_path, monkeypatch):
        rob = _make_robustness_report(overall_pr_auc=0.85, slice_pr_aucs=[0.65, 0.80, 0.82])
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_pr_auc_drop_exceeds_threshold" in codes


class TestMainRangeFail:
    def test_pr_auc_range_exceeds_threshold(self, tmp_path, monkeypatch):
        rob = _make_robustness_report(overall_pr_auc=0.85, slice_pr_aucs=[0.60, 0.85])
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_pr_auc_range_exceeds_threshold" in codes


class TestMainSummaryMismatch:
    def test_summary_mismatch(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["summary"]["time_slices"]["pr_auc_min"] = 0.50
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_summary_mismatch" in codes

    def test_missing_summary(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        del rob["summary"]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_summary_missing" in codes


class TestMainStrict:
    def test_strict_promotes_warnings(self, tmp_path, monkeypatch):
        # drop=0.11 is between warn=0.10 and fail=0.14 → warning → strict fails
        rob = _make_robustness_report(overall_pr_auc=0.82, slice_pr_aucs=[0.71, 0.80, 0.81])
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr), "--report", str(rpt), "--strict",
        ])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        assert out["status"] == "fail"
        assert out["strict_mode"] is True


class TestMainNoReport:
    def test_no_report_flag(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr),
        ])
        rc = rg.main()
        assert rc == 0


class TestMainWithPolicy:
    def test_custom_policy_thresholds(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        policy = {"robustness_thresholds": {"time_slices": {"pr_auc_drop_fail": 0.01}}}
        pp = tmp_path / "policy.json"
        _write_json(pp, policy)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "rg", "--robustness-report", str(rr),
            "--performance-policy", str(pp),
            "--report", str(rpt),
        ])
        rc = rg.main()
        assert rc == 2  # drop exceeds tight threshold
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_pr_auc_drop_exceeds_threshold" in codes


class TestMainEmptyBucketRows:
    def test_empty_slices(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = []
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_missing_bucket_rows" in codes


class TestMainInvalidRow:
    def test_non_dict_row(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = ["not_a_dict", _make_slice_row(0.80), _make_slice_row(0.81), _make_slice_row(0.82)]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_invalid_bucket_row" in codes

    def test_missing_n(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = [
            {"positive_count": 10, "metrics": {"pr_auc": 0.80}},
            _make_slice_row(0.81), _make_slice_row(0.82), _make_slice_row(0.83),
        ]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_invalid_bucket_row" in codes

    def test_missing_metrics(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = [
            {"n": 50, "positive_count": 10},
            _make_slice_row(0.81), _make_slice_row(0.82), _make_slice_row(0.83),
        ]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_invalid_bucket_row" in codes

    def test_non_finite_pr_auc(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = [
            {"n": 50, "positive_count": 10, "metrics": {"pr_auc": float("nan")}},
            _make_slice_row(0.81), _make_slice_row(0.82), _make_slice_row(0.83),
        ]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_non_finite_metric" in codes

    def test_pr_auc_out_of_range(self, tmp_path, monkeypatch):
        rob = _make_robustness_report()
        rob["time_slices"]["slices"] = [
            {"n": 50, "positive_count": 10, "metrics": {"pr_auc": 1.5}},
            _make_slice_row(0.81), _make_slice_row(0.82), _make_slice_row(0.83),
        ]
        rr = tmp_path / "rob.json"
        _write_json(rr, rob)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", ["rg", "--robustness-report", str(rr), "--report", str(rpt)])
        rc = rg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "robustness_metric_out_of_range" in codes
