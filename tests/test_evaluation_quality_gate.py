"""Comprehensive unit tests for scripts/evaluation_quality_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from evaluation_quality_gate import (
    canonical_metric_token,
    extract_baseline_metrics,
    extract_primary_metric,
    find_non_finite_values,
    get_by_dot_path,
    infer_higher_is_better,
    is_auxiliary_metric_path,
    is_finite_number,
    normalize_split_token,
    parse_ci_block,
    path_tokens,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _good_eval_report():
    return {
        "metrics": {"roc_auc": 0.85},
        "metrics_ci": {
            "roc_auc": {
                "ci_95": [0.80, 0.90],
                "method": "bootstrap",
                "n_resamples": 1000,
            }
        },
        "baselines": {
            "prevalence_model": {"metrics": {"roc_auc": 0.55}},
            "random_forest": {"metrics": {"roc_auc": 0.70}},
        },
    }


# ────────────────────────────────────────────────────────
# canonical_metric_token / path_tokens / normalize_split_token
# ────────────────────────────────────────────────────────

class TestCanonicalMetricToken:
    def test_normal(self):
        assert canonical_metric_token("roc_auc") == "rocauc"

    def test_mixed_case(self):
        assert canonical_metric_token("ROC_AUC") == "rocauc"

    def test_special_chars(self):
        assert canonical_metric_token("f1-score") == "f1score"


class TestPathTokens:
    def test_simple(self):
        assert path_tokens("metrics.roc_auc") == ["metrics", "roc_auc"]

    def test_with_index(self):
        assert path_tokens("items[0].value") == ["items", "value"]

    def test_empty(self):
        assert path_tokens("") == []


class TestNormalizeSplitToken:
    def test_normal(self):
        assert normalize_split_token("Test") == "test"

    def test_with_special(self):
        assert normalize_split_token("test-split") == "testsplit"


# ────────────────────────────────────────────────────────
# is_auxiliary_metric_path / is_finite_number
# ────────────────────────────────────────────────────────

class TestIsAuxiliaryMetricPath:
    def test_baseline(self):
        assert is_auxiliary_metric_path("baselines.model_a.roc_auc") is True

    def test_ci(self):
        assert is_auxiliary_metric_path("confidence_intervals.roc_auc") is True

    def test_normal(self):
        assert is_auxiliary_metric_path("metrics.roc_auc") is False

    def test_empty(self):
        assert is_auxiliary_metric_path("") is False


class TestIsFiniteNumber:
    def test_int(self):
        assert is_finite_number(5) is True

    def test_float(self):
        assert is_finite_number(3.14) is True

    def test_inf(self):
        assert is_finite_number(float("inf")) is False

    def test_nan(self):
        assert is_finite_number(float("nan")) is False

    def test_bool(self):
        assert is_finite_number(True) is False

    def test_string(self):
        assert is_finite_number("5") is False


# ────────────────────────────────────────────────────────
# get_by_dot_path
# ────────────────────────────────────────────────────────

class TestGetByDotPath:
    def test_normal(self):
        payload = {"a": {"b": 42}}
        val, path = get_by_dot_path(payload, "a.b")
        assert val == 42
        assert path == "a.b"

    def test_missing(self):
        payload = {"a": {"b": 42}}
        val, path = get_by_dot_path(payload, "a.c")
        assert val is None

    def test_top_level(self):
        payload = {"x": 10}
        val, path = get_by_dot_path(payload, "x")
        assert val == 10


# ────────────────────────────────────────────────────────
# extract_primary_metric
# ────────────────────────────────────────────────────────

class TestExtractPrimaryMetric:
    def test_direct_path(self):
        payload = {"metrics": {"roc_auc": 0.85}}
        val, path = extract_primary_metric(payload, "roc_auc", "metrics.roc_auc")
        assert val == 0.85

    def test_auto_discovery(self):
        payload = {"metrics": {"roc_auc": 0.85}}
        val, path = extract_primary_metric(payload, "roc_auc", None)
        assert val == 0.85

    def test_not_found(self):
        payload = {"metrics": {"f1": 0.7}}
        val, path = extract_primary_metric(payload, "roc_auc", None)
        assert val is None


# ────────────────────────────────────────────────────────
# find_non_finite_values
# ────────────────────────────────────────────────────────

class TestFindNonFiniteValues:
    def test_clean(self):
        assert find_non_finite_values({"a": 1, "b": 2.0}) == []

    def test_inf(self):
        hits = find_non_finite_values({"a": float("inf")})
        assert len(hits) == 1
        assert hits[0]["kind"] == "non_finite_float"

    def test_nan_string(self):
        hits = find_non_finite_values({"a": "NaN"})
        assert len(hits) == 1
        assert hits[0]["kind"] == "non_finite_string"

    def test_nested(self):
        hits = find_non_finite_values({"a": {"b": [float("nan")]}})
        assert len(hits) == 1


# ────────────────────────────────────────────────────────
# parse_ci_block
# ────────────────────────────────────────────────────────

class TestParseCiBlock:
    def test_list_format(self):
        lo, hi, method, n = parse_ci_block([0.80, 0.90])
        assert lo == 0.80
        assert hi == 0.90
        assert method is None

    def test_dict_ci95(self):
        lo, hi, method, n = parse_ci_block({"ci_95": [0.75, 0.95], "method": "bootstrap", "n_resamples": 500})
        assert lo == 0.75
        assert hi == 0.95
        assert method == "bootstrap"
        assert n == 500

    def test_dict_lower_upper(self):
        lo, hi, method, n = parse_ci_block({"lower": 0.70, "upper": 0.90})
        assert lo == 0.70
        assert hi == 0.90

    def test_not_dict_or_list(self):
        lo, hi, method, n = parse_ci_block("invalid")
        assert lo is None and hi is None


# ────────────────────────────────────────────────────────
# infer_higher_is_better
# ────────────────────────────────────────────────────────

class TestInferHigherIsBetter:
    def test_auc(self):
        assert infer_higher_is_better("roc_auc") is True

    def test_loss(self):
        assert infer_higher_is_better("log_loss") is False

    def test_brier(self):
        assert infer_higher_is_better("brier_score") is False

    def test_f1(self):
        assert infer_higher_is_better("f1_score") is True


# ────────────────────────────────────────────────────────
# extract_baseline_metrics
# ────────────────────────────────────────────────────────

class TestExtractBaselineMetrics:
    def test_normal(self):
        payload = {
            "baselines": {
                "prevalence": {"metrics": {"roc_auc": 0.5}},
                "random": {"metrics": {"roc_auc": 0.55}},
            }
        }
        result = extract_baseline_metrics(payload, "roc_auc")
        assert "prevalence" in result
        assert result["prevalence"] == 0.5

    def test_no_baselines(self):
        assert extract_baseline_metrics({}, "roc_auc") == {}

    def test_baselines_not_dict(self):
        assert extract_baseline_metrics({"baselines": "nope"}, "roc_auc") == {}


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, eval_path, metric_name="roc_auc", extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "evaluation_quality_gate.py"),
            "--evaluation-report", str(eval_path),
            "--metric-name", metric_name,
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_pass(self, tmp_path: Path):
        eval_path = _write_json(tmp_path / "eval.json", _good_eval_report())
        result = self._run(tmp_path, eval_path)
        assert result.returncode == 0, f"stdout: {result.stdout}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_missing_eval_report(self, tmp_path: Path):
        result = self._run(tmp_path, tmp_path / "nonexistent.json")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_evaluation_report" in codes

    def test_invalid_eval_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{bad", encoding="utf-8")
        result = self._run(tmp_path, p)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_evaluation_report" in codes

    def test_primary_metric_not_found(self, tmp_path: Path):
        eval_data = _good_eval_report()
        del eval_data["metrics"]["roc_auc"]
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path, metric_name="roc_auc")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "primary_metric_not_found" in codes

    def test_ci_width_exceeds_threshold(self, tmp_path: Path):
        eval_data = _good_eval_report()
        eval_data["metrics_ci"]["roc_auc"]["ci_95"] = [0.50, 0.95]  # width=0.45
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path, extra_args=["--max-ci-width", "0.10"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "ci_width_exceeds_threshold" in codes

    def test_insufficient_resamples(self, tmp_path: Path):
        eval_data = _good_eval_report()
        eval_data["metrics_ci"]["roc_auc"]["n_resamples"] = 50
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path, extra_args=["--min-resamples", "200"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_ci_resamples" in codes

    def test_baseline_improvement_insufficient(self, tmp_path: Path):
        eval_data = _good_eval_report()
        eval_data["baselines"]["prevalence_model"]["metrics"]["roc_auc"] = 0.84  # too close
        eval_data["baselines"]["random_forest"]["metrics"]["roc_auc"] = 0.84
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path, extra_args=["--min-baseline-delta", "0.05"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "baseline_improvement_insufficient" in codes

    def test_missing_baseline_metrics(self, tmp_path: Path):
        eval_data = _good_eval_report()
        del eval_data["baselines"]
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_baseline_metrics" in codes

    def test_non_finite_values(self, tmp_path: Path):
        eval_data = _good_eval_report()
        eval_data["extra_field"] = float("inf")
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "non_finite_values_detected" in codes

    def test_primary_metric_mismatch(self, tmp_path: Path):
        eval_path = _write_json(tmp_path / "eval.json", _good_eval_report())
        result = self._run(tmp_path, eval_path, extra_args=["--primary-metric", "0.99"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "primary_metric_mismatch" in codes

    def test_missing_ci_method(self, tmp_path: Path):
        eval_data = _good_eval_report()
        del eval_data["metrics_ci"]["roc_auc"]["method"]
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_ci_method" in codes

    def test_report_structure(self, tmp_path: Path):
        eval_path = _write_json(tmp_path / "eval.json", _good_eval_report())
        self._run(tmp_path, eval_path)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "summary" in report
        s = report["summary"]
        assert "primary_metric" in s
        assert "metric_name" in s

    def test_primary_metric_outside_ci(self, tmp_path: Path):
        eval_data = _good_eval_report()
        eval_data["metrics"]["roc_auc"] = 0.95  # outside [0.80, 0.90]
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "primary_metric_outside_ci" in codes

    def test_loss_metric_lower_is_better(self, tmp_path: Path):
        eval_data = {
            "metrics": {"log_loss": 0.30},
            "metrics_ci": {
                "log_loss": {
                    "ci_95": [0.25, 0.35],
                    "method": "bootstrap",
                    "n_resamples": 1000,
                }
            },
            "baselines": {
                "prevalence": {"metrics": {"log_loss": 0.60}},
            },
        }
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        result = self._run(tmp_path, eval_path, metric_name="log_loss")
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["higher_is_better"] is False
