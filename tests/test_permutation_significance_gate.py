"""Comprehensive unit tests for scripts/permutation_significance_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from permutation_significance_gate import (
    load_null_metrics,
    parse_finite_float,
    parse_text_values,
    summarize,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_null_list(path: Path, values: list) -> Path:
    _write_json(path, values)
    return path


# ────────────────────────────────────────────────────────
# parse_finite_float
# ────────────────────────────────────────────────────────

class TestParseFiniteFloat:
    def test_int(self):
        assert parse_finite_float(5) == 5.0

    def test_float(self):
        assert parse_finite_float(3.14) == 3.14

    def test_string(self):
        assert parse_finite_float("0.5") == 0.5

    def test_bool_raises(self):
        with pytest.raises(ValueError):
            parse_finite_float(True)

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            parse_finite_float(float("inf"))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            parse_finite_float(float("nan"))


# ────────────────────────────────────────────────────────
# parse_text_values
# ────────────────────────────────────────────────────────

class TestParseTextValues:
    def test_normal(self):
        assert parse_text_values("0.5\n0.6\n0.7\n") == [0.5, 0.6, 0.7]

    def test_empty_lines(self):
        assert parse_text_values("\n\n0.5\n\n") == [0.5]

    def test_empty(self):
        assert parse_text_values("") == []


# ────────────────────────────────────────────────────────
# load_null_metrics
# ────────────────────────────────────────────────────────

class TestLoadNullMetrics:
    def test_json_list(self, tmp_path: Path):
        p = _write_null_list(tmp_path / "null.json", [0.5, 0.52, 0.48])
        assert load_null_metrics(p) == [0.5, 0.52, 0.48]

    def test_json_dict_with_metrics(self, tmp_path: Path):
        p = _write_json(tmp_path / "null.json", {"metrics": [0.5, 0.52]})
        assert load_null_metrics(p) == [0.5, 0.52]

    def test_json_dict_with_null_metrics(self, tmp_path: Path):
        p = _write_json(tmp_path / "null.json", {"null_metrics": [0.5, 0.52]})
        assert load_null_metrics(p) == [0.5, 0.52]

    def test_text_format(self, tmp_path: Path):
        p = tmp_path / "null.txt"
        p.write_text("0.5\n0.52\n0.48\n", encoding="utf-8")
        assert load_null_metrics(p) == [0.5, 0.52, 0.48]

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "null.json"
        p.write_text("", encoding="utf-8")
        assert load_null_metrics(p) == []


# ────────────────────────────────────────────────────────
# summarize
# ────────────────────────────────────────────────────────

class TestSummarize:
    def test_normal(self):
        result = summarize([1.0, 2.0, 3.0])
        assert result["mean"] == 2.0
        assert result["min"] == 1.0
        assert result["max"] == 3.0
        assert result["std"] > 0

    def test_single(self):
        result = summarize([5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0

    def test_empty(self):
        import math
        result = summarize([])
        assert math.isnan(result["mean"])


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, null_path, actual=0.85, metric_name="roc_auc", extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "permutation_significance_gate.py"),
            "--metric-name", metric_name,
            "--actual", str(actual),
            "--null-metrics-file", str(null_path),
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_significant(self, tmp_path: Path):
        """actual=0.85 vs null=[0.50..0.55] → significant."""
        null_path = _write_null_list(tmp_path / "null.json", [0.50 + i * 0.001 for i in range(200)])
        result = self._run(tmp_path, null_path, actual=0.85)
        assert result.returncode == 0, f"stdout: {result.stdout}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"
        assert report["summary"]["p_value_one_sided"] < 0.01

    def test_not_significant(self, tmp_path: Path):
        """actual=0.51 vs null=[0.50..0.55] → not significant."""
        null_path = _write_null_list(tmp_path / "null.json", [0.50 + i * 0.001 for i in range(200)])
        result = self._run(tmp_path, null_path, actual=0.51)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "permutation_not_significant" in codes

    def test_missing_null_file(self, tmp_path: Path):
        result = self._run(tmp_path, tmp_path / "nonexistent.json")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_null_metrics_file" in codes

    def test_empty_null_distribution(self, tmp_path: Path):
        null_path = tmp_path / "null.json"
        null_path.write_text("[]", encoding="utf-8")
        result = self._run(tmp_path, null_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "empty_null_distribution" in codes

    def test_low_permutation_count_warning(self, tmp_path: Path):
        """Less than min_permutations → warning."""
        null_path = _write_null_list(tmp_path / "null.json", [0.50] * 10)
        self._run(tmp_path, null_path, actual=0.85,
                           extra_args=["--min-permutations", "100"])
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "low_permutation_count" in warn_codes

    def test_insufficient_effect_delta(self, tmp_path: Path):
        """actual barely above null mean but min_delta requires larger gap."""
        null_path = _write_null_list(tmp_path / "null.json", [0.50] * 200)
        result = self._run(tmp_path, null_path, actual=0.51,
                           extra_args=["--min-delta", "0.10"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_effect_delta" in codes

    def test_lower_is_better(self, tmp_path: Path):
        """For loss metrics: actual=0.10 vs null=[0.50..0.55] → significant."""
        null_path = _write_null_list(tmp_path / "null.json", [0.50 + i * 0.001 for i in range(200)])
        result = self._run(tmp_path, null_path, actual=0.10, metric_name="log_loss",
                           extra_args=["--lower-is-better"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["higher_is_better"] is False

    def test_invalid_null_file(self, tmp_path: Path):
        null_path = tmp_path / "null.json"
        null_path.write_text("[true, false]", encoding="utf-8")
        result = self._run(tmp_path, null_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_null_metrics_file" in codes

    def test_report_structure(self, tmp_path: Path):
        null_path = _write_null_list(tmp_path / "null.json", [0.50] * 200)
        self._run(tmp_path, null_path, actual=0.85)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "metric_name" in report["summary"]
        assert "actual" in report["summary"]
        assert "null_count" in report["summary"]
        assert "null_summary" in report["summary"]
        assert "p_value_one_sided" in report["summary"]
        assert "effect_delta_vs_null_mean" in report["summary"]
        assert "alpha" in report["summary"]
        assert "failures" in report
        assert "warnings" in report
