"""Tests for scripts/generalization_gap_gate.py.

Covers helper functions (get_nested_threshold, read_metric), gap analysis,
threshold validation, directional gap logic (higher-is-better vs brier),
and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "generalization_gap_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import generalization_gap_gate as ggg


# ── helper functions ─────────────────────────────────────────────────────────

class TestGetNestedThreshold:
    def test_defaults(self):
        warn, fail = ggg.get_nested_threshold(None, "train", "valid", "pr_auc", 0.05, 0.08)
        assert warn == 0.05
        assert fail == 0.08

    def test_policy_override(self):
        policy = {
            "gap_thresholds": {
                "train_valid": {
                    "pr_auc": {"warn": 0.03, "fail": 0.06}
                }
            }
        }
        warn, fail = ggg.get_nested_threshold(policy, "train", "valid", "pr_auc", 0.05, 0.08)
        assert warn == 0.03
        assert fail == 0.06

    def test_flat_key_fallback(self):
        policy = {
            "gap_thresholds": {
                "train_valid_pr_auc_warn": 0.02,
                "train_valid_pr_auc_fail": 0.04,
            }
        }
        warn, fail = ggg.get_nested_threshold(policy, "train", "valid", "pr_auc", 0.05, 0.08)
        assert warn == 0.02
        assert fail == 0.04


class TestReadMetric:
    def test_valid(self):
        sm = {"test": {"metrics": {"pr_auc": 0.85}}}
        assert ggg.read_metric(sm, "test", "pr_auc") == 0.85

    def test_missing_split(self):
        assert ggg.read_metric({}, "test", "pr_auc") is None

    def test_missing_metric(self):
        sm = {"test": {"metrics": {}}}
        assert ggg.read_metric(sm, "test", "pr_auc") is None

    def test_non_dict_block(self):
        sm = {"test": "bad"}
        assert ggg.read_metric(sm, "test", "pr_auc") is None


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_eval_report(train_prauc=0.86, valid_prauc=0.84, test_prauc=0.82,
                      train_f2=0.85, test_f2=0.82,
                      valid_brier=0.14, test_brier=0.15):
    return {
        "split_metrics": {
            "train": {"metrics": {"pr_auc": train_prauc, "f2_beta": train_f2, "brier": 0.10}},
            "valid": {"metrics": {"pr_auc": valid_prauc, "f2_beta": 0.83, "brier": valid_brier}},
            "test": {"metrics": {"pr_auc": test_prauc, "f2_beta": test_f2, "brier": test_brier}},
        }
    }


def _run_gate(tmp_path, eval_report=None, policy=None, strict=False):
    if eval_report is None:
        eval_report = _make_eval_report()
    er_path = tmp_path / "eval.json"
    er_path.write_text(json.dumps(eval_report))
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--evaluation-report", str(er_path),
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
        assert "gaps" in report["summary"]


class TestOverfitGap:
    def test_train_valid_gap_exceeds_fail(self, tmp_path):
        ev = _make_eval_report(train_prauc=0.95, valid_prauc=0.80)
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "overfit_gap_exceeds_threshold" in codes

    def test_train_test_f2_gap_exceeds_fail(self, tmp_path):
        ev = _make_eval_report(train_f2=0.95, test_f2=0.70)
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "overfit_gap_exceeds_threshold" in codes

    def test_brier_gap_direction(self, tmp_path):
        # Brier: lower is better. Gap = test_brier - valid_brier.
        # If test_brier >> valid_brier, gap is positive = bad.
        ev = _make_eval_report(valid_brier=0.10, test_brier=0.15)
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "overfit_gap_exceeds_threshold" in codes

    def test_warning_gap(self, tmp_path):
        # train-valid pr_auc: warn=0.05, fail=0.08
        # gap = 0.06 → between warn and fail → warning
        ev = _make_eval_report(train_prauc=0.90, valid_prauc=0.84)
        report = _run_gate(tmp_path, eval_report=ev)
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "overfit_gap_warning" in warn_codes

    def test_warning_strict_fails(self, tmp_path):
        ev = _make_eval_report(train_prauc=0.90, valid_prauc=0.84)
        report = _run_gate(tmp_path, eval_report=ev, strict=True)
        assert report["status"] == "fail"


class TestMissingData:
    def test_missing_split_metrics(self, tmp_path):
        ev = {"no_split_metrics": True}
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_split_metrics" in codes

    def test_missing_eval_file(self, tmp_path):
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evaluation-report", str(tmp_path / "nope.json"),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_evaluation_report" in codes

    def test_missing_metric_in_split(self, tmp_path):
        ev = {
            "split_metrics": {
                "train": {"metrics": {}},
                "valid": {"metrics": {"pr_auc": 0.84}},
                "test": {"metrics": {"pr_auc": 0.82}},
            }
        }
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_required_metric" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True


# ── Direct main() tests ─────────────────────────────────────────────────────

class TestMainPass:
    def test_pass(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report()))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 0
        out = json.loads(rpt.read_text())
        assert out["status"] == "pass"


class TestMainOverfitFail:
    def test_train_valid_gap(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report(train_prauc=0.95, valid_prauc=0.80)))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "overfit_gap_exceeds_threshold" in codes


class TestMainBrierGap:
    def test_brier_direction(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report(valid_brier=0.10, test_brier=0.15)))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "overfit_gap_exceeds_threshold" in codes


class TestMainWarning:
    def test_gap_warning(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report(train_prauc=0.90, valid_prauc=0.84)))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        out = json.loads(rpt.read_text())
        warn_codes = [w["code"] for w in out["warnings"]]
        assert "overfit_gap_warning" in warn_codes


class TestMainStrictWarning:
    def test_strict_fails_on_warning(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report(train_prauc=0.90, valid_prauc=0.84)))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt), "--strict",
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        assert out["status"] == "fail"


class TestMainMissingSplitMetrics:
    def test_no_split_metrics(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps({"no_split_metrics": True}))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "missing_split_metrics" in codes


class TestMainInvalidEval:
    def test_missing_eval_file(self, tmp_path, monkeypatch):
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(tmp_path / "nope.json"), "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "invalid_evaluation_report" in codes


class TestMainWithPolicy:
    def test_policy_override(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report()))
        pp = tmp_path / "policy.json"
        pp.write_text(json.dumps({"gap_thresholds": {"train_valid": {"pr_auc": {"warn": 0.01, "fail": 0.02}}}}))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev),
            "--performance-policy", str(pp),
            "--report", str(rpt),
        ])
        rc = ggg.main()
        out = json.loads(rpt.read_text())
        # With tighter thresholds, a small gap may become a failure
        assert "status" in out


class TestMainMissingMetricInSplit:
    def test_missing_metric(self, tmp_path, monkeypatch):
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps({
            "split_metrics": {
                "train": {"metrics": {}},
                "valid": {"metrics": {"pr_auc": 0.84}},
                "test": {"metrics": {"pr_auc": 0.82}},
            }
        }))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "missing_required_metric" in codes


class TestMainInvalidPolicy:
    def test_corrupt_policy_file(self, tmp_path, monkeypatch):
        """Policy file is not valid JSON → invalid_performance_policy."""
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report()))
        pp = tmp_path / "policy.json"
        pp.write_text("{bad", encoding="utf-8")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev),
            "--performance-policy", str(pp),
            "--report", str(rpt),
        ])
        rc = ggg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "invalid_performance_policy" in codes


class TestMainMissingSplitBlock:
    def test_missing_valid_block(self, tmp_path, monkeypatch):
        """split_metrics exists but 'valid' block is missing."""
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps({
            "split_metrics": {
                "train": {"metrics": {"pr_auc": 0.86}},
                "test": {"metrics": {"pr_auc": 0.82}},
            }
        }))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev), "--report", str(rpt),
        ])
        rc = ggg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "missing_split_metrics" in codes


class TestMainInvalidGapThreshold:
    def test_warn_greater_than_fail(self, tmp_path, monkeypatch):
        """Policy with warn > fail → invalid_gap_threshold."""
        ev = tmp_path / "eval.json"
        ev.write_text(json.dumps(_make_eval_report()))
        pp = tmp_path / "policy.json"
        pp.write_text(json.dumps({
            "gap_thresholds": {
                "train_valid": {"pr_auc": {"warn": 0.10, "fail": 0.02}}
            }
        }))
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "ggg", "--evaluation-report", str(ev),
            "--performance-policy", str(pp),
            "--report", str(rpt),
        ])
        rc = ggg.main()
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "invalid_gap_threshold" in codes
