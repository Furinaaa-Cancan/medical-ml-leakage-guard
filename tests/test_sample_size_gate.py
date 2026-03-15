"""Tests for scripts/sample_size_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "sample_size_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import sample_size_gate as ssg


# ── _to_float ────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_int(self):
        assert ssg._to_float(100) == 100.0

    def test_float(self):
        assert ssg._to_float(10.5) == 10.5

    def test_none_returns_none(self):
        assert ssg._to_float(None) is None

    def test_nan_returns_none(self):
        assert ssg._to_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert ssg._to_float(float("inf")) is None

    def test_bad_string_returns_none(self):
        assert ssg._to_float("not_a_number") is None


# ── _estimate_shrinkage ───────────────────────────────────────────────────────

class TestEstimateShrinkage:
    def test_normal(self):
        s = ssg._estimate_shrinkage(100, 10)
        assert s == pytest.approx(0.90)

    def test_zero_events_returns_none(self):
        assert ssg._estimate_shrinkage(0, 10) is None

    def test_zero_features_returns_none(self):
        assert ssg._estimate_shrinkage(100, 0) is None

    def test_features_ge_events_returns_zero(self):
        assert ssg._estimate_shrinkage(10, 10) == 0.0
        assert ssg._estimate_shrinkage(5, 10) == 0.0

    def test_more_events_improves_shrinkage(self):
        s1 = ssg._estimate_shrinkage(50, 10)
        s2 = ssg._estimate_shrinkage(200, 10)
        assert s2 > s1


# ── default thresholds ───────────────────────────────────────────────────────

class TestDefaultThresholds:
    def test_epv_recommended_gt_minimum(self):
        assert ssg.DEFAULT_THRESHOLDS["epv_recommended"] > \
               ssg.DEFAULT_THRESHOLDS["epv_minimum"]

    def test_epv_minimum_is_10(self):
        assert ssg.DEFAULT_THRESHOLDS["epv_minimum"] == pytest.approx(10.0)

    def test_shrinkage_target(self):
        assert ssg.DEFAULT_THRESHOLDS["shrinkage_factor_target"] == pytest.approx(0.90)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_report(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps(content), encoding="utf-8")
    return p


def _run(tmp_path: Path, eval_path: Path, extra_args: list | None = None):
    cmd = [sys.executable, str(GATE_SCRIPT),
           "--evaluation-report", str(eval_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_with_report(tmp_path: Path, eval_path: Path, extra_args: list | None = None):
    rpt = tmp_path / "report.json"
    result = _run(tmp_path, eval_path, (extra_args or []) + ["--report", str(rpt)])
    return result, rpt


# ── CLI integration: error cases ─────────────────────────────────────────────

class TestMissingFile:
    def test_returns_2(self, tmp_path: Path):
        result = _run(tmp_path, tmp_path / "missing.json")
        assert result.returncode == 2


class TestInvalidJson:
    def test_returns_2(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{bad json", encoding="utf-8")
        result = _run(tmp_path, bad)
        assert result.returncode == 2


class TestMissingSampleSizeInfo:
    def test_empty_report_returns_2(self, tmp_path: Path):
        p = _write_report(tmp_path, {})
        result = _run(tmp_path, p)
        assert result.returncode == 2


# ── adequate sample size ─────────────────────────────────────────────────────

class TestPassingReport:
    def _good_report(self):
        return {
            "sample_size_adequacy": {
                "n_events": 300,
                "n_non_events": 700,
                "n_features": 10,
                "n_total": 1000,
                "events_per_variable": 30.0,
            }
        }

    def test_exit_0(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        result = _run(tmp_path, p)
        assert result.returncode == 0

    def test_report_written(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        result, rpt = _run_with_report(tmp_path, p)
        assert rpt.exists()
        data = json.loads(rpt.read_text())
        assert data["status"] == "pass"
        assert data["gate_name"] == "sample_size_gate"

    def test_report_has_summary(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        _, rpt = _run_with_report(tmp_path, p)
        data = json.loads(rpt.read_text())
        assert "summary" in data
        assert data["summary"]["adequacy_verdict"] == "adequate"

    def test_epv_in_summary(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        _, rpt = _run_with_report(tmp_path, p)
        data = json.loads(rpt.read_text())
        assert data["summary"]["events_per_variable"] == pytest.approx(30.0)


# ── EPV failures ──────────────────────────────────────────────────────────────

class TestEpvBelowMinimum:
    def test_returns_2(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 50,
                "n_features": 10,  # EPV = 5, < 10
            }
        }
        p = _write_report(tmp_path, report)
        result = _run(tmp_path, p)
        assert result.returncode == 2

    def test_failure_code_in_report(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 50,
                "n_features": 10,
            }
        }
        p = _write_report(tmp_path, report)
        _, rpt = _run_with_report(tmp_path, p)
        data = json.loads(rpt.read_text())
        codes = [f["code"] for f in data["failures"]]
        assert "epv_below_minimum" in codes


class TestEpvBelowRecommended:
    def test_warning_only_exits_0(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 150,
                "n_features": 10,  # EPV = 15, between 10 and 20
            }
        }
        p = _write_report(tmp_path, report)
        result = _run(tmp_path, p)
        assert result.returncode == 0

    def test_strict_makes_it_fail(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 150,
                "n_features": 10,
            }
        }
        p = _write_report(tmp_path, report)
        result = _run(tmp_path, p, ["--strict"])
        assert result.returncode == 2

    def test_custom_epv_minimum(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 50,
                "n_features": 10,  # EPV = 5, normally fails
            }
        }
        p = _write_report(tmp_path, report)
        # lower threshold to 3 → should pass
        result = _run(tmp_path, p, ["--epv-minimum", "3"])
        assert result.returncode == 0


# ── event count checks ────────────────────────────────────────────────────────

class TestTooFewEvents:
    def test_very_few_events_returns_2(self, tmp_path: Path):
        # < 50 events → failure (not just warning)
        report = {
            "sample_size_adequacy": {
                "n_events": 30,
                "n_features": 2,  # EPV = 15, ok
            }
        }
        p = _write_report(tmp_path, report)
        result = _run(tmp_path, p)
        assert result.returncode == 2

    def test_test_set_too_few_events_warning(self, tmp_path: Path):
        report = {
            "sample_size_adequacy": {
                "n_events": 200,
                "n_features": 5,
            },
            "split_summary": {
                "test": {"n_positive": 20}  # < 50 threshold
            }
        }
        p = _write_report(tmp_path, report)
        _, rpt = _run_with_report(tmp_path, p)
        data = json.loads(rpt.read_text())
        warning_codes = [w["code"] for w in data.get("warnings", [])]
        assert "test_set_events_too_few" in warning_codes


# ── fallback extraction from split_summary ───────────────────────────────────

class TestFallbackExtraction:
    def test_extract_from_train_split(self, tmp_path: Path):
        """n_events/n_features from split_summary.train when ssa is empty."""
        report = {
            "metadata": {"n_features": 10},
            "split_summary": {
                "train": {
                    "n_positive": 200,
                    "n_negative": 800,
                    "total": 1000,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = _run(tmp_path, p)
        # 200 events / 10 features = EPV 20 → adequate
        assert result.returncode == 0


# ── shrinkage factor ─────────────────────────────────────────────────────────

class TestShrinkageFactor:
    def test_low_shrinkage_produces_warning(self, tmp_path: Path):
        # 200 events, 30 features → EPV=6.7 (fail), shrinkage=(200-30)/200=0.85 (<0.9)
        report = {
            "sample_size_adequacy": {
                "n_events": 200,
                "n_features": 30,
            }
        }
        p = _write_report(tmp_path, report)
        _, rpt = _run_with_report(tmp_path, p)
        data = json.loads(rpt.read_text())
        all_codes = [w["code"] for w in data.get("warnings", [])] + \
                    [f["code"] for f in data.get("failures", [])]
        assert "shrinkage_factor_low" in all_codes


# ── CLI contract ─────────────────────────────────────────────────────────────

class TestCliContract:
    def test_help_exits_0(self):
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--evaluation-report" in result.stdout
        assert "--strict" in result.stdout
        assert "--report" in result.stdout

    def test_missing_required_arg_exits_2(self):
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
