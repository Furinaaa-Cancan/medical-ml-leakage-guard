"""Tests for scripts/clinical_metrics_gate.py.

Covers helper functions (canonical_metric_token, to_int, safe_ratio,
compute_confusion_metrics, metric_in_unit_range, get_required_metrics,
parse_beta, parse_clinical_floors), split metrics validation, confusion
matrix consistency, metric formula checks, threshold selection, clinical
floors, top-level metrics, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "clinical_metrics_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import clinical_metrics_gate as cmg


# ── helper functions ─────────────────────────────────────────────────────────

class TestCanonicalMetricToken:
    def test_lowercase(self):
        assert cmg.canonical_metric_token("PR_AUC") == "prauc"

    def test_strips_special(self):
        assert cmg.canonical_metric_token("pr-auc") == "prauc"

    def test_empty(self):
        assert cmg.canonical_metric_token("") == ""


class TestToInt:
    def test_int(self):
        assert cmg.to_int(5) == 5

    def test_bool(self):
        assert cmg.to_int(True) is None

    def test_float_integer(self):
        assert cmg.to_int(5.0) == 5

    def test_float_non_integer(self):
        assert cmg.to_int(5.5) is None

    def test_nan(self):
        assert cmg.to_int(float("nan")) is None

    def test_none(self):
        assert cmg.to_int(None) is None


class TestSafeRatio:
    def test_normal(self):
        assert cmg.safe_ratio(10.0, 20.0) == 0.5

    def test_zero_denom(self):
        assert cmg.safe_ratio(10.0, 0.0) is None

    def test_negative_denom(self):
        assert cmg.safe_ratio(10.0, -1.0) is None


class TestComputeConfusionMetrics:
    def test_perfect(self):
        result = cmg.compute_confusion_metrics(50, 0, 50, 0, beta=2.0)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        result = cmg.compute_confusion_metrics(0, 50, 0, 50, beta=2.0)
        assert result["accuracy"] == 0.0
        assert result["precision"] == 0.0
        assert result["sensitivity"] == 0.0
        assert result["specificity"] == 0.0

    def test_zero_denominator(self):
        result = cmg.compute_confusion_metrics(0, 0, 50, 0, beta=2.0)
        assert result["precision"] is None  # tp+fp=0
        assert result["sensitivity"] is None  # tp+fn=0

    def test_ppv_equals_precision(self):
        result = cmg.compute_confusion_metrics(30, 10, 40, 20, beta=2.0)
        assert result["ppv"] == result["precision"]

    def test_f2_beta(self):
        result = cmg.compute_confusion_metrics(30, 10, 40, 20, beta=2.0)
        assert result["f2_beta"] is not None
        assert 0.0 <= result["f2_beta"] <= 1.0


class TestMetricInUnitRange:
    def test_known_metric(self):
        assert cmg.metric_in_unit_range("pr_auc") is True

    def test_unknown_metric(self):
        assert cmg.metric_in_unit_range("custom_metric") is False

    def test_brier(self):
        assert cmg.metric_in_unit_range("brier") is True


class TestGetRequiredMetrics:
    def test_no_policy(self):
        result = cmg.get_required_metrics(None)
        assert result == cmg.DEFAULT_REQUIRED_METRICS

    def test_policy_adds_extra(self):
        policy = {"required_metrics": ["custom_metric"]}
        result = cmg.get_required_metrics(policy)
        assert "custom_metric" in result
        for m in cmg.DEFAULT_REQUIRED_METRICS:
            assert m in result

    def test_policy_duplicate_ignored(self):
        policy = {"required_metrics": ["pr_auc"]}
        result = cmg.get_required_metrics(policy)
        assert result.count("pr_auc") == 1


class TestParseBeta:
    def test_default(self):
        assert cmg.parse_beta(None) == 2.0

    def test_custom(self):
        assert cmg.parse_beta({"beta": 1.5}) == 1.5

    def test_zero(self):
        assert cmg.parse_beta({"beta": 0}) == 2.0

    def test_negative(self):
        assert cmg.parse_beta({"beta": -1}) == 2.0


class TestParseClinicalFloors:
    def test_defaults(self):
        result = cmg.parse_clinical_floors(None)
        assert result["sensitivity_min"] == 0.85
        assert result["npv_min"] == 0.90
        assert result["specificity_min"] == 0.40
        assert result["ppv_min"] == 0.55

    def test_custom_clinical_floors(self):
        policy = {"clinical_floors": {"sensitivity_min": 0.90}}
        result = cmg.parse_clinical_floors(policy)
        assert result["sensitivity_min"] == 0.90

    def test_threshold_policy_overrides(self):
        policy = {"threshold_policy": {"clinical_floors": {"npv_min": 0.95}}}
        result = cmg.parse_clinical_floors(policy)
        assert result["npv_min"] == 0.95


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_split_block(tp, fp, tn, fn, metrics=None):
    """Create a split_metrics block with confusion matrix and derived metrics."""
    cm = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    total = tp + fp + tn + fn
    if metrics is None:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        beta = 2.0
        f2 = ((1 + beta**2) * prec * rec) / ((beta**2 * prec) + rec) if ((beta**2 * prec) + rec) > 0 else 0.0
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "ppv": prec,
            "npv": npv_val,
            "sensitivity": rec,
            "specificity": spec,
            "f1": f1,
            "f2_beta": f2,
            "roc_auc": 0.85,
            "pr_auc": 0.80,
            "brier": 0.15,
        }
    return {"metrics": metrics, "confusion_matrix": cm, "n_samples": total}


def _make_eval_report(overrides=None):
    # TP/FP/TN/FN chosen so all clinical floors pass:
    # sensitivity=90/100=0.90>=0.85, specificity=80/100=0.80>=0.40,
    # ppv=90/110≈0.818>=0.55, npv=80/90≈0.889... hmm need npv>=0.90
    # Use: TP=90, FP=10, TN=90, FN=10 → sens=0.90, spec=0.90, ppv=0.90, npv=0.90
    train = _make_split_block(90, 10, 90, 10)
    valid = _make_split_block(90, 10, 90, 10)
    test = _make_split_block(90, 10, 90, 10)
    report = {
        "split_metrics": {"train": train, "valid": valid, "test": test},
        "metrics": dict(test["metrics"]),
        "threshold_selection": {
            "selection_split": "valid",
            "selected_threshold": 0.5,
            "selected_metrics_on_valid": dict(valid["metrics"]),
            "selected_confusion_on_valid": dict(valid["confusion_matrix"]),
        },
    }
    if overrides:
        report.update(overrides)
    return report


def _run_gate(tmp_path, eval_report=None, policy=None, strict=False, extra_args=None):
    if eval_report is None:
        eval_report = _make_eval_report()
    er_path = tmp_path / "eval_report.json"
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
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_report_pass(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        s = report["summary"]
        assert "required_metrics" in s
        assert "beta" in s
        assert "clinical_floors" in s
        assert "splits" in s


class TestMissingSplitMetrics:
    def test_no_split_metrics(self, tmp_path):
        ev = _make_eval_report()
        del ev["split_metrics"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_split_metrics" in codes

    def test_missing_test_split(self, tmp_path):
        ev = _make_eval_report()
        del ev["split_metrics"]["test"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_split_metrics" in codes


class TestConfusionMatrix:
    def test_negative_value(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["confusion_matrix"]["tp"] = -1
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_confusion_matrix" in codes

    def test_missing_field(self, tmp_path):
        ev = _make_eval_report()
        del ev["split_metrics"]["test"]["confusion_matrix"]["tp"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_confusion_matrix" in codes

    def test_zero_total(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["confusion_matrix"] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_confusion_matrix" in codes

    def test_row_count_mismatch(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["n_samples"] = 999
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "confusion_matrix_row_count_mismatch" in codes


class TestMetricValidation:
    def test_missing_metric(self, tmp_path):
        ev = _make_eval_report()
        del ev["split_metrics"]["test"]["metrics"]["pr_auc"]
        del ev["metrics"]["pr_auc"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_required_metric" in codes

    def test_metric_out_of_range(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["metrics"]["roc_auc"] = 1.5
        ev["metrics"]["roc_auc"] = 1.5
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_out_of_range" in codes

    def test_formula_mismatch(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["metrics"]["accuracy"] = 0.999
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_formula_mismatch" in codes

    def test_precision_ppv_mismatch(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["metrics"]["ppv"] = 0.999
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_formula_mismatch" in codes


class TestThresholdSelection:
    def test_missing_threshold_selection(self, tmp_path):
        ev = _make_eval_report()
        del ev["threshold_selection"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_threshold_selection" in codes

    def test_test_split_for_threshold(self, tmp_path):
        ev = _make_eval_report()
        ev["threshold_selection"]["selection_split"] = "test"
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "test_split_used_for_threshold_selection" in codes

    def test_invalid_threshold_value(self, tmp_path):
        ev = _make_eval_report()
        ev["threshold_selection"]["selected_threshold"] = 1.5
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_threshold_selection_value" in codes

    def test_cv_inner_without_guard(self, tmp_path):
        ev = _make_eval_report()
        ev["threshold_selection"]["selection_split"] = "cv_inner"
        del ev["threshold_selection"]["selected_metrics_on_valid"]
        del ev["threshold_selection"]["selected_confusion_on_valid"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "threshold_guard_constraints_not_met" in codes


class TestTopLevelMetrics:
    def test_missing_top_level(self, tmp_path):
        ev = _make_eval_report()
        del ev["metrics"]
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_top_level_metrics" in codes

    def test_top_level_mismatch(self, tmp_path):
        ev = _make_eval_report()
        ev["metrics"]["roc_auc"] = 0.50
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "top_level_test_metric_mismatch" in codes


class TestClinicalFloors:
    def test_sensitivity_below_floor(self, tmp_path):
        ev = _make_eval_report()
        # Set sensitivity very low in test split
        ev["split_metrics"]["test"]["metrics"]["sensitivity"] = 0.30
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "clinical_floor_sensitivity_not_met" in codes

    def test_npv_below_floor(self, tmp_path):
        ev = _make_eval_report()
        ev["split_metrics"]["test"]["metrics"]["npv"] = 0.50
        report = _run_gate(tmp_path, eval_report=ev)
        codes = [f["code"] for f in report["failures"]]
        assert "clinical_floor_npv_not_met" in codes


class TestPerformancePolicy:
    def test_missing_mandatory_metric_in_policy(self, tmp_path):
        policy = {"required_metrics": ["roc_auc"]}  # missing others
        report = _run_gate(tmp_path, policy=policy)
        codes = [f["code"] for f in report["failures"]]
        assert "performance_policy_missing_required_metric" in codes

    def test_invalid_selection_split_in_policy(self, tmp_path):
        policy = {"threshold_policy": {"selection_split": "test"}}
        report = _run_gate(tmp_path, policy=policy)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_performance_policy" in codes


class TestFileErrors:
    def test_missing_eval_report(self, tmp_path):
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evaluation-report", str(tmp_path / "nope.json"),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_evaluation_report" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True
