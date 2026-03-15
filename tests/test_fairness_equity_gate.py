"""Tests for scripts/fairness_equity_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "fairness_equity_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import fairness_equity_gate as feg


# ── _to_float ────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_int(self):
        assert feg._to_float(1) == 1.0

    def test_float(self):
        assert feg._to_float(0.5) == 0.5

    def test_string_number(self):
        assert feg._to_float("0.8") == 0.8

    def test_none_returns_none(self):
        assert feg._to_float(None) is None

    def test_nan_returns_none(self):
        assert feg._to_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert feg._to_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert feg._to_float(float("-inf")) is None

    def test_bad_string_returns_none(self):
        assert feg._to_float("bad") is None


# ── default thresholds ───────────────────────────────────────────────────────

class TestDefaultThresholds:
    def test_equalized_odds_fail_gt_warn(self):
        assert feg.DEFAULT_THRESHOLDS["equalized_odds_gap_fail"] > \
               feg.DEFAULT_THRESHOLDS["equalized_odds_gap_warn"]

    def test_disparate_impact_fail_lt_warn(self):
        assert feg.DEFAULT_THRESHOLDS["disparate_impact_ratio_fail"] < \
               feg.DEFAULT_THRESHOLDS["disparate_impact_ratio_warn"]

    def test_eighty_percent_rule(self):
        assert feg.DEFAULT_THRESHOLDS["disparate_impact_ratio_fail"] == pytest.approx(0.80)

    def test_min_subgroup_size_positive(self):
        assert feg.DEFAULT_THRESHOLDS["min_subgroup_size"] > 0


# ── helpers: eval report fixture ─────────────────────────────────────────────

def _make_eval_report(subgroup_perf=None):
    """Return minimal eval report dict."""
    report = {}
    if subgroup_perf is not None:
        report["subgroup_performance"] = subgroup_perf
    return report


def _write_report(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "eval_report.json"
    p.write_text(json.dumps(content), encoding="utf-8")
    return p


# ── main() unit tests ─────────────────────────────────────────────────────────

class TestMainMissingFile:
    def test_missing_report_returns_2(self, tmp_path: Path):
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(tmp_path / "nonexistent.json")],
            capture_output=True, text=True,
        )
        assert result.returncode == 2


def _make_args(tmp_path: Path, eval_path=None, strict=False,
               eq_fail=None, di_fail=None, report_path=None):
    import argparse
    return argparse.Namespace(
        evaluation_report=str(eval_path or tmp_path / "eval.json"),
        report=str(report_path) if report_path else None,
        strict=strict,
        equalized_odds_gap_fail=eq_fail,
        disparate_impact_ratio_fail=di_fail,
    )



class TestMainInvalidJson:
    def test_returns_2(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(bad)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2


class TestMainNoSubgroupPerformance:
    def test_returns_2(self, tmp_path: Path):
        p = _write_report(tmp_path, {})
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2


class TestMainPassingReport:
    def _good_report(self):
        return {
            "subgroup_performance": {
                "sex": {
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.92,
                    "groups": [
                        {"group_label": "M", "n": 200, "pr_auc": 0.75},
                        {"group_label": "F", "n": 180, "pr_auc": 0.72},
                    ],
                }
            }
        }

    def test_exit_0(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_report_written(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        rpt = tmp_path / "fairness.json"
        subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--report", str(rpt)],
            capture_output=True, text=True,
        )
        assert rpt.exists()
        data = json.loads(rpt.read_text())
        assert data["status"] == "pass"
        assert data["gate_name"] == "fairness_equity_gate"

    def test_report_structure(self, tmp_path: Path):
        p = _write_report(tmp_path, self._good_report())
        rpt = tmp_path / "fairness.json"
        subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--report", str(rpt)],
            capture_output=True, text=True,
        )
        data = json.loads(rpt.read_text())
        assert "failures" in data
        assert "warnings" in data
        assert "thresholds" in data  # extra fields merged into top-level envelope


class TestMainEqualizedOddsFailure:
    def test_gap_above_fail_threshold_returns_2(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "age_group": {
                    "equalized_odds_gap": 0.20,  # > 0.15 fail threshold
                    "disparate_impact_ratio": 0.90,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_gap_above_warn_only_exits_0(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "age_group": {
                    "equalized_odds_gap": 0.12,  # > 0.10 warn, < 0.15 fail
                    "disparate_impact_ratio": 0.90,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_strict_warn_becomes_failure(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "age_group": {
                    "equalized_odds_gap": 0.12,
                    "disparate_impact_ratio": 0.90,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--strict"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_custom_threshold_override(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "race": {
                    "equalized_odds_gap": 0.20,
                    "disparate_impact_ratio": 0.90,
                }
            }
        }
        p = _write_report(tmp_path, report)
        # raise fail threshold to 0.25 → should pass
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--equalized-odds-gap-fail", "0.25"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestMainDisparateImpact:
    def test_below_fail_threshold_returns_2(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "sex": {
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.70,  # < 0.80
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_custom_di_threshold(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "sex": {
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.70,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--disparate-impact-ratio-fail", "0.60"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestMainSubgroupSampleSize:
    def test_small_subgroup_produces_warning(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "rare_group": {
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.90,
                    "groups": [
                        {"group_label": "rare", "n": 5, "pr_auc": 0.70},
                        {"group_label": "common", "n": 500, "pr_auc": 0.75},
                    ],
                }
            }
        }
        p = _write_report(tmp_path, report)
        rpt = tmp_path / "out.json"
        subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p),
             "--report", str(rpt)],
            capture_output=True, text=True,
        )
        data = json.loads(rpt.read_text())
        codes = [w["code"] for w in data.get("warnings", [])]
        assert "subgroup_sample_too_small" in codes


class TestMainSubgroupPrAuc:
    def test_low_pr_auc_failure(self, tmp_path: Path):
        report = {
            "subgroup_performance": {
                "sex": {
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.90,
                    "groups": [
                        {"group_label": "F", "n": 200, "pr_auc": 0.30},  # < 0.40 fail
                    ],
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2


class TestMainListSubgroupPerf:
    def test_list_format_accepted(self, tmp_path: Path):
        report = {
            "subgroup_performance": [
                {
                    "feature": "sex",
                    "equalized_odds_gap": 0.05,
                    "disparate_impact_ratio": 0.92,
                }
            ]
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


class TestMainNonFiniteMetric:
    def test_nan_equalized_odds_gap_ignored(self, tmp_path: Path):
        """NaN metrics should be silently skipped (not crash)."""
        report = {
            "subgroup_performance": {
                "sex": {
                    "equalized_odds_gap": None,
                    "disparate_impact_ratio": 0.90,
                }
            }
        }
        p = _write_report(tmp_path, report)
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT),
             "--evaluation-report", str(p)],
            capture_output=True, text=True,
        )
        # Should not crash
        assert result.returncode in (0, 2)


class TestCliHelp:
    def test_help_exits_0(self):
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--evaluation-report" in result.stdout
        assert "--strict" in result.stdout
        assert "--report" in result.stdout
