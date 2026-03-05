"""Tests for scripts/model_selection_audit_gate.py.

Covers helper functions (canonical_metric_token, contains_test_token,
finite_float, finite_int, finite_float_list, in_unit_interval,
extract_selection_tuple, scan_candidate_for_test_usage),
selection policy validation, candidate validation, one-SE replay,
and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "model_selection_audit_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import model_selection_audit_gate as msg


# ── helper functions ─────────────────────────────────────────────────────────

class TestCanonicalMetricToken:
    def test_lowercase(self):
        assert msg.canonical_metric_token("PR_AUC") == "prauc"

    def test_strips_special(self):
        assert msg.canonical_metric_token("pr-auc") == "prauc"

    def test_empty(self):
        assert msg.canonical_metric_token("") == ""

    def test_mixed(self):
        assert msg.canonical_metric_token("F1_Score") == "f1score"


class TestContainsTestToken:
    def test_plain_test(self):
        assert msg.contains_test_token("test") is True

    def test_test_prefix(self):
        assert msg.contains_test_token("test_split") is True

    def test_test_suffix(self):
        assert msg.contains_test_token("holdout_test") is True

    def test_no_test(self):
        assert msg.contains_test_token("valid") is False

    def test_attest(self):
        assert msg.contains_test_token("attest") is False

    def test_latest(self):
        assert msg.contains_test_token("latest") is False

    def test_empty(self):
        assert msg.contains_test_token("") is False

    def test_cv_inner(self):
        assert msg.contains_test_token("cv_inner") is False


class TestFiniteFloat:
    def test_int(self):
        assert msg.finite_float(42) == 42.0

    def test_float(self):
        assert msg.finite_float(3.14) == 3.14

    def test_bool(self):
        assert msg.finite_float(True) is None

    def test_nan(self):
        assert msg.finite_float(float("nan")) is None

    def test_inf(self):
        assert msg.finite_float(float("inf")) is None

    def test_none(self):
        assert msg.finite_float(None) is None

    def test_string(self):
        assert msg.finite_float("3.14") is None


class TestFiniteInt:
    def test_int(self):
        assert msg.finite_int(5) == 5

    def test_bool(self):
        assert msg.finite_int(True) is None

    def test_float_integer(self):
        assert msg.finite_int(5.0) == 5

    def test_float_non_integer(self):
        assert msg.finite_int(5.5) is None

    def test_nan(self):
        assert msg.finite_int(float("nan")) is None

    def test_none(self):
        assert msg.finite_int(None) is None


class TestFiniteFloatList:
    def test_valid(self):
        assert msg.finite_float_list([0.5, 0.6, 0.7]) == [0.5, 0.6, 0.7]

    def test_empty(self):
        assert msg.finite_float_list([]) is None

    def test_not_list(self):
        assert msg.finite_float_list("hello") is None

    def test_contains_nan(self):
        assert msg.finite_float_list([0.5, float("nan")]) is None

    def test_contains_none(self):
        assert msg.finite_float_list([0.5, None]) is None


class TestInUnitInterval:
    def test_zero(self):
        assert msg.in_unit_interval(0.0) is True

    def test_one(self):
        assert msg.in_unit_interval(1.0) is True

    def test_middle(self):
        assert msg.in_unit_interval(0.5) is True

    def test_negative(self):
        assert msg.in_unit_interval(-0.1) is False

    def test_above_one(self):
        assert msg.in_unit_interval(1.1) is False


class TestExtractSelectionTuple:
    def test_valid(self):
        candidate = {
            "selection_metrics": {
                "pr_auc": {"mean": 0.85, "std": 0.02, "n_folds": 5, "fold_scores": [0.84, 0.85, 0.86, 0.85, 0.85]}
            }
        }
        mean, std, n_folds, fold_scores = msg.extract_selection_tuple(candidate, "pr_auc")
        assert mean == 0.85
        assert std == 0.02
        assert n_folds == 5
        assert fold_scores == [0.84, 0.85, 0.86, 0.85, 0.85]

    def test_missing_metrics(self):
        mean, std, n_folds, fold_scores = msg.extract_selection_tuple({}, "pr_auc")
        assert mean is None

    def test_missing_metric_block(self):
        candidate = {"selection_metrics": {"roc_auc": {"mean": 0.9}}}
        mean, std, n_folds, fold_scores = msg.extract_selection_tuple(candidate, "pr_auc")
        assert mean is None


class TestScanCandidateForTestUsage:
    def test_clean(self):
        hits: List[str] = []
        msg.scan_candidate_for_test_usage({"family": "logistic", "cv_score": 0.9}, "", hits)
        assert len(hits) == 0

    def test_test_key(self):
        hits: List[str] = []
        msg.scan_candidate_for_test_usage({"test_score": 0.8}, "", hits)
        assert len(hits) == 1

    def test_nested_test_key(self):
        hits: List[str] = []
        msg.scan_candidate_for_test_usage({"metrics": {"test_auc": 0.9}}, "", hits)
        assert len(hits) == 1

    def test_excludes_allowed_key(self):
        hits: List[str] = []
        msg.scan_candidate_for_test_usage({"test_used_for_model_selection": False}, "", hits)
        assert len(hits) == 0


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_candidate(model_id: str, family: str, complexity_rank: int,
                    mean: float, n_folds: int, selected: bool = False) -> dict:
    import math as _math
    # Generate fold scores that reproduce exact mean and a small std
    scores = [mean] * n_folds
    # Introduce tiny variation for realistic std computation
    if n_folds >= 2:
        delta = 0.01
        scores[0] = mean + delta
        scores[1] = mean - delta
    calc_mean = sum(scores) / len(scores)
    calc_std = float(_math.sqrt(sum((x - calc_mean) ** 2 for x in scores) / (len(scores) - 1))) if n_folds > 1 else 0.0
    return {
        "model_id": model_id,
        "family": family,
        "complexity_rank": complexity_rank,
        "selected": selected,
        "selection_metrics": {
            "pr_auc": {
                "mean": calc_mean,
                "std": calc_std,
                "n_folds": n_folds,
                "fold_scores": scores,
            }
        },
    }


def _make_model_selection_report(overrides: dict = None) -> dict:
    report = {
        "selected_model_id": "logistic_l2",
        "selection_policy": {
            "primary_metric": "pr_auc",
            "selection_data": "cv_inner",
            "one_se_rule": True,
            "test_used_for_model_selection": False,
        },
        "candidates": [
            _make_candidate("logistic_l2", "logistic_l2", 1, 0.84, 5, selected=True),
            _make_candidate("random_forest", "random_forest_balanced", 2, 0.84, 5),
            _make_candidate("hist_gb", "hist_gradient_boosting_l2", 3, 0.84, 5),
        ],
    }
    if overrides:
        report.update(overrides)
    return report


def _make_tuning_spec(overrides: dict = None) -> dict:
    spec = {
        "objective_metric": "pr_auc",
        "model_selection_data": "cv_inner",
        "test_used_for_model_selection": False,
    }
    if overrides:
        spec.update(overrides)
    return spec


def _run_gate(tmp_path: Path, ms_report: dict = None, tuning: dict = None,
              strict: bool = False, extra_args: list = None) -> dict:
    if ms_report is None:
        ms_report = _make_model_selection_report()
    if tuning is None:
        tuning = _make_tuning_spec()

    ms_path = tmp_path / "model_selection_report.json"
    ms_path.write_text(json.dumps(ms_report))
    tuning_path = tmp_path / "tuning_spec.json"
    tuning_path.write_text(json.dumps(tuning))
    report_path = tmp_path / "report.json"

    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--model-selection-report", str(ms_path),
        "--tuning-spec", str(tuning_path),
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_report_pass(self, tmp_path: Path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path: Path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        s = report["summary"]
        assert "candidate_count" in s
        assert "selected_model_id" in s
        assert "selection_policy" in s
        assert "replay" in s


class TestSelectionPolicy:
    def test_missing_policy(self, tmp_path: Path):
        ms = _make_model_selection_report()
        del ms["selection_policy"]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_selection_policy" in codes

    def test_missing_metric(self, tmp_path: Path):
        ms = _make_model_selection_report()
        del ms["selection_policy"]["primary_metric"]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_selection_metric" in codes

    def test_wrong_metric(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selection_policy"]["primary_metric"] = "roc_auc"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "selection_metric_mismatch" in codes

    def test_missing_selection_data(self, tmp_path: Path):
        ms = _make_model_selection_report()
        del ms["selection_policy"]["selection_data"]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_selection_data" in codes

    def test_invalid_selection_data(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selection_policy"]["selection_data"] = "all_data"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_selection_data" in codes

    def test_one_se_rule_not_enabled(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selection_policy"]["one_se_rule"] = False
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "one_se_rule_not_enabled" in codes


class TestTestDataUsage:
    def test_test_used_true(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selection_policy"]["test_used_for_model_selection"] = True
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "test_data_usage_detected" in codes

    def test_tuning_test_used_true(self, tmp_path: Path):
        tuning = _make_tuning_spec({"test_used_for_model_selection": True})
        report = _run_gate(tmp_path, tuning=tuning)
        codes = [f["code"] for f in report["failures"]]
        assert "test_data_usage_detected" in codes

    def test_selection_data_test(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selection_policy"]["selection_data"] = "test"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "test_data_usage_detected" in codes


class TestCandidates:
    def test_too_few_candidates(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"] = ms["candidates"][:2]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "candidate_pool_too_small" in codes

    def test_no_logistic_baseline(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"] = [
            _make_candidate("rf1", "random_forest", 1, 0.84, 5, selected=True),
            _make_candidate("rf2", "random_forest", 2, 0.84, 5),
            _make_candidate("gb1", "gradient_boosting", 3, 0.84, 5),
        ]
        ms["selected_model_id"] = "rf1"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_logistic_baseline" in codes

    def test_duplicate_model_ids(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"][1]["model_id"] = "logistic_l2"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "duplicate_candidate_model_id" in codes

    def test_missing_candidates_array(self, tmp_path: Path):
        ms = _make_model_selection_report()
        del ms["candidates"]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_candidates" in codes

    def test_non_dict_candidate(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"].append("not_a_dict")
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_candidate_entry" in codes


class TestSelectedModel:
    def test_missing_selected_model(self, tmp_path: Path):
        ms = _make_model_selection_report()
        del ms["selected_model_id"]
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_selected_model" in codes

    def test_selected_not_in_candidates(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["selected_model_id"] = "nonexistent_model"
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "selected_model_not_in_candidates" in codes

    def test_multiple_selected_flags(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"][1]["selected"] = True  # two now selected
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_selected_candidate_flags" in codes

    def test_selected_flag_mismatch(self, tmp_path: Path):
        ms = _make_model_selection_report()
        ms["candidates"][0]["selected"] = False
        ms["candidates"][1]["selected"] = True  # selected flag on wrong model
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "selected_model_flag_mismatch" in codes


class TestOneSeReplay:
    def test_selection_not_reproducible(self, tmp_path: Path):
        """Select the most complex model when simpler one is within 1-SE."""
        ms = _make_model_selection_report()
        # All candidates are very close in performance, so simplest should win
        ms["candidates"] = [
            _make_candidate("logistic_l2", "logistic_l2", 1, 0.84, 5),
            _make_candidate("random_forest", "random_forest_balanced", 2, 0.84, 5),
            _make_candidate("hist_gb", "hist_gradient_boosting_l2", 3, 0.84, 5),
        ]
        # All same mean → all eligible → simplest (logistic) should win replay
        # But we select hist_gb (complexity=3) → mismatch
        ms["selected_model_id"] = "hist_gb"
        ms["candidates"][2]["selected"] = True
        report = _run_gate(tmp_path, ms_report=ms)
        codes = [f["code"] for f in report["failures"]]
        assert "selection_not_reproducible" in codes


class TestTuningSpecMismatch:
    def test_selection_data_mismatch(self, tmp_path: Path):
        tuning = _make_tuning_spec({"model_selection_data": "valid"})
        report = _run_gate(tmp_path, tuning=tuning)
        codes = [f["code"] for f in report["failures"]]
        assert "selection_data_spec_mismatch" in codes

    def test_tuning_metric_mismatch(self, tmp_path: Path):
        tuning = _make_tuning_spec({"objective_metric": "roc_auc"})
        report = _run_gate(tmp_path, tuning=tuning)
        codes = [f["code"] for f in report["failures"]]
        assert "selection_metric_mismatch" in codes


class TestFileErrors:
    def test_missing_model_selection_report(self, tmp_path: Path):
        tuning_path = tmp_path / "tuning.json"
        tuning_path.write_text(json.dumps(_make_tuning_spec()))
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--model-selection-report", str(tmp_path / "nope.json"),
            "--tuning-spec", str(tuning_path),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_model_selection_report" in codes

    def test_missing_tuning_spec(self, tmp_path: Path):
        ms_path = tmp_path / "ms.json"
        ms_path.write_text(json.dumps(_make_model_selection_report()))
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--model-selection-report", str(ms_path),
            "--tuning-spec", str(tmp_path / "nope.json"),
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_tuning_spec" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path: Path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True

    def test_non_prauc_warning_strict(self, tmp_path: Path):
        report = _run_gate(tmp_path, strict=True, extra_args=["--expected-primary-metric", "roc_auc"])
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "unexpected_primary_metric_override" in warn_codes
        assert report["status"] == "fail"
