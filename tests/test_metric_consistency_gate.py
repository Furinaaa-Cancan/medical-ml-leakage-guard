"""Tests for scripts/metric_consistency_gate.py.

Covers helper functions (is_finite_number, canonical_metric_token,
get_by_dot_path, collect_candidate_metrics, collect_metric_leaf_hits,
path_tokens, is_auxiliary_metric_path, is_allowed_primary_metric_path,
has_conflicting_values, normalize_split_token, extract_declared_split,
extract_metric), and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "metric_consistency_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import metric_consistency_gate as mcg


# ── helper functions ─────────────────────────────────────────────────────────

class TestIsFiniteNumber:
    def test_int(self):
        assert mcg.is_finite_number(5) is True

    def test_bool(self):
        assert mcg.is_finite_number(True) is False

    def test_nan(self):
        assert mcg.is_finite_number(float("nan")) is False

    def test_none(self):
        assert mcg.is_finite_number(None) is False


class TestCanonicalMetricToken:
    def test_basic(self):
        assert mcg.canonical_metric_token("roc_auc") == "rocauc"

    def test_mixed_case(self):
        assert mcg.canonical_metric_token("ROC-AUC") == "rocauc"

    def test_pr_auc(self):
        assert mcg.canonical_metric_token("pr_auc") == "prauc"


class TestGetByDotPath:
    def test_simple(self):
        d = {"metrics": {"roc_auc": 0.85}}
        val, path = mcg.get_by_dot_path(d, "metrics.roc_auc")
        assert val == 0.85
        assert path == "metrics.roc_auc"

    def test_missing_key(self):
        val, path = mcg.get_by_dot_path({"a": 1}, "b")
        assert val is None

    def test_empty_segment(self):
        val, path = mcg.get_by_dot_path({"a": 1}, "a..b")
        assert val is None


class TestCollectCandidateMetrics:
    def test_finds_metric(self):
        payload = {"metrics": {"roc_auc": 0.85}}
        hits = mcg.collect_candidate_metrics(payload, "roc_auc")
        assert len(hits) >= 1
        assert any(h[1] == 0.85 for h in hits)

    def test_no_match(self):
        hits = mcg.collect_candidate_metrics({"x": 1}, "roc_auc")
        assert len(hits) == 0


class TestCollectMetricLeafHits:
    def test_finds_nested(self):
        payload = {"metrics": {"roc_auc": 0.85}, "split_metrics": {"test": {"metrics": {"roc_auc": 0.84}}}}
        hits = mcg.collect_metric_leaf_hits(payload, "roc_auc")
        assert len(hits) >= 2

    def test_no_match(self):
        hits = mcg.collect_metric_leaf_hits({"x": 1}, "roc_auc")
        assert len(hits) == 0


class TestPathTokens:
    def test_basic(self):
        assert mcg.path_tokens("metrics.roc_auc") == ["metrics", "roc_auc"]

    def test_array_index(self):
        assert mcg.path_tokens("results[0].roc_auc") == ["results", "roc_auc"]


class TestIsAuxiliaryMetricPath:
    def test_baseline(self):
        assert mcg.is_auxiliary_metric_path("baseline.roc_auc") is True

    def test_primary(self):
        assert mcg.is_auxiliary_metric_path("metrics.roc_auc") is False

    def test_threshold_selection(self):
        assert mcg.is_auxiliary_metric_path("threshold_selection.roc_auc") is True


class TestIsAllowedPrimaryMetricPath:
    def test_normal(self):
        assert mcg.is_allowed_primary_metric_path("metrics.roc_auc", None) is True

    def test_split_metrics_with_required(self):
        assert mcg.is_allowed_primary_metric_path("split_metrics.test.metrics.roc_auc", "test") is True

    def test_split_metrics_wrong_split(self):
        assert mcg.is_allowed_primary_metric_path("split_metrics.train.metrics.roc_auc", "test") is False

    def test_auxiliary(self):
        assert mcg.is_allowed_primary_metric_path("baseline.roc_auc", None) is False


class TestHasConflictingValues:
    def test_consistent(self):
        hits = [{"value": 0.85}, {"value": 0.85}]
        assert mcg.has_conflicting_values(hits) is False

    def test_conflicting(self):
        hits = [{"value": 0.85}, {"value": 0.90}]
        assert mcg.has_conflicting_values(hits) is True

    def test_empty(self):
        assert mcg.has_conflicting_values([]) is False


class TestNormalizeSplitToken:
    def test_basic(self):
        assert mcg.normalize_split_token("test") == "test"

    def test_mixed(self):
        assert mcg.normalize_split_token("Test-Split") == "testsplit"


class TestExtractDeclaredSplit:
    def test_found(self):
        key, val = mcg.extract_declared_split({"split": "test"})
        assert key == "split"
        assert val == "test"

    def test_not_found(self):
        key, val = mcg.extract_declared_split({"x": 1})
        assert key is None
        assert val is None

    def test_in_meta(self):
        key, val = mcg.extract_declared_split({"meta": {"split": "valid"}})
        assert key == "meta.split"
        assert val == "valid"


class TestExtractMetric:
    def test_with_path(self):
        payload = {"metrics": {"roc_auc": 0.85}}
        val, path, raw, cands, amb = mcg.extract_metric(payload, "roc_auc", "metrics.roc_auc")
        assert val == 0.85
        assert path == "metrics.roc_auc"

    def test_without_path(self):
        payload = {"metrics": {"roc_auc": 0.85}}
        val, path, raw, cands, amb = mcg.extract_metric(payload, "roc_auc", None)
        assert val == 0.85

    def test_not_found(self):
        val, path, raw, cands, amb = mcg.extract_metric({"x": 1}, "roc_auc", None)
        assert val is None


# ── CLI integration ──────────────────────────────────────────────────────────

def _run_gate(tmp_path, eval_report=None, metric_name="roc_auc", metric_path=None,
              expected=None, tolerance=None, required_split=None, strict=False):
    if eval_report is None:
        eval_report = {"metrics": {"roc_auc": 0.85}, "split": "test"}
    er_path = tmp_path / "eval.json"
    er_path.write_text(json.dumps(eval_report))
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--evaluation-report", str(er_path),
        "--metric-name", metric_name,
        "--report", str(report_path),
    ]
    if metric_path:
        cmd.extend(["--metric-path", metric_path])
    if expected is not None:
        cmd.extend(["--expected", str(expected)])
    if tolerance is not None:
        cmd.extend(["--tolerance", str(tolerance)])
    if required_split:
        cmd.extend(["--required-evaluation-split", required_split])
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_pass(self, tmp_path):
        report = _run_gate(tmp_path, expected=0.85)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0
        assert report["summary"]["actual_metric"] == 0.85

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path, expected=0.85)
        assert "metric_name" in report["summary"]
        assert "actual_metric" in report["summary"]
        assert "failures" in report
        assert "warnings" in report


class TestMetricNotFound:
    def test_missing_metric(self, tmp_path):
        report = _run_gate(tmp_path, metric_name="pr_auc", expected=0.80)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_not_found" in codes


class TestMetricMismatch:
    def test_expected_mismatch(self, tmp_path):
        report = _run_gate(tmp_path, expected=0.90)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_mismatch" in codes


class TestMissingFile:
    def test_missing_eval_report(self, tmp_path):
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evaluation-report", str(tmp_path / "nope.json"),
            "--metric-name", "roc_auc",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_evaluation_report" in codes


class TestSplitValidation:
    def test_split_mismatch(self, tmp_path):
        ev = {"metrics": {"roc_auc": 0.85}, "split": "train"}
        report = _run_gate(tmp_path, eval_report=ev, expected=0.85, required_split="test")
        codes = [f["code"] for f in report["failures"]]
        assert "evaluation_split_mismatch" in codes

    def test_missing_split_declaration(self, tmp_path):
        ev = {"metrics": {"roc_auc": 0.85}}
        report = _run_gate(tmp_path, eval_report=ev, expected=0.85, required_split="test")
        codes = [f["code"] for f in report["failures"]]
        assert "missing_evaluation_split" in codes


class TestMetricPathMismatch:
    def test_path_leaf_mismatch(self, tmp_path):
        report = _run_gate(tmp_path, metric_name="roc_auc", metric_path="metrics.pr_auc")
        codes = [f["code"] for f in report["failures"]]
        assert "metric_path_metric_mismatch" in codes


class TestNoExpected:
    def test_warning_no_expected(self, tmp_path):
        report = _run_gate(tmp_path)
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "missing_expected_metric" in warn_codes
        assert report["status"] == "pass"

    def test_strict_no_expected_fails(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["status"] == "fail"


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, expected=0.85, strict=True)
        assert report["strict_mode"] is True
