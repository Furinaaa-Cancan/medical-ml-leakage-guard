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
from typing import Any, Dict

import pytest

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
