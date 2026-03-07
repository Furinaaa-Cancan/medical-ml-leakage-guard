"""Tests for scripts/threshold_sensitivity.py.

Covers helper functions (_deep_get, _is_finite, load_report, compute_margin),
metric extraction, classification (failing/fragile/safe), policy simulation,
output formatting (text/markdown/json), and CLI integration via direct main().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import threshold_sensitivity as ts


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


# ── helper functions ─────────────────────────────────────────────────────────


class TestDeepGet:
    def test_simple(self):
        assert ts._deep_get({"a": {"b": 1}}, ["a", "b"]) == 1

    def test_missing_key(self):
        assert ts._deep_get({"a": 1}, ["b"]) is None

    def test_list_index(self):
        assert ts._deep_get({"a": [10, 20]}, ["a", 1]) == 20

    def test_list_out_of_range(self):
        assert ts._deep_get({"a": [10]}, ["a", 5]) is None

    def test_none_intermediate(self):
        assert ts._deep_get({"a": None}, ["a", "b"]) is None

    def test_empty_keys(self):
        data = {"x": 1}
        assert ts._deep_get(data, []) is data

    def test_non_dict_non_list(self):
        assert ts._deep_get(42, ["a"]) is None


class TestIsFinite:
    def test_int(self):
        assert ts._is_finite(5) is True

    def test_float(self):
        assert ts._is_finite(0.5) is True

    def test_bool(self):
        assert ts._is_finite(True) is False

    def test_nan(self):
        assert ts._is_finite(float("nan")) is False

    def test_inf(self):
        assert ts._is_finite(float("inf")) is False

    def test_none(self):
        assert ts._is_finite(None) is False

    def test_string(self):
        assert ts._is_finite("5") is False


class TestLoadReport:
    def test_valid(self, tmp_path):
        _write_json(tmp_path / "r.json", {"status": "pass"})
        result = ts.load_report(tmp_path, "r.json")
        assert result == {"status": "pass"}

    def test_missing(self, tmp_path):
        assert ts.load_report(tmp_path, "nope.json") is None

    def test_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert ts.load_report(tmp_path, "bad.json") is None

    def test_non_dict(self, tmp_path):
        _write_json(tmp_path / "arr.json", [1, 2, 3])
        assert ts.load_report(tmp_path, "arr.json") is None


class TestComputeMargin:
    def test_lower_is_better_pass(self):
        result = ts.compute_margin(0.05, 0.14, "lower_is_better")
        assert result["status"] == "PASS"
        assert result["margin"] > 0

    def test_lower_is_better_fail(self):
        result = ts.compute_margin(0.20, 0.14, "lower_is_better")
        assert result["status"] == "FAIL"
        assert result["margin"] < 0

    def test_higher_is_better_pass(self):
        result = ts.compute_margin(0.10, 0.05, "higher_is_better")
        assert result["status"] == "PASS"
        assert result["margin"] > 0

    def test_higher_is_better_fail(self):
        result = ts.compute_margin(0.02, 0.05, "higher_is_better")
        assert result["status"] == "FAIL"
        assert result["margin"] < 0

    def test_borderline(self):
        result = ts.compute_margin(0.14, 0.14, "lower_is_better")
        assert result["status"] == "BORDERLINE"
        assert result["margin"] == 0

    def test_zero_threshold(self):
        result = ts.compute_margin(0.01, 0.0, "lower_is_better")
        assert result["headroom_pct"] is None  # inf becomes None


# ── metric extraction ────────────────────────────────────────────────────────


def _make_robustness_report(ts_drop=0.05, ts_range=0.10, ph_drop=0.04, ph_range=0.08):
    return {
        "status": "pass",
        "summary": {
            "computed": {
                "time_slices": {
                    "pr_auc_worst_drop_from_overall": ts_drop,
                    "pr_auc_range": ts_range,
                },
                "patient_hash_groups": {
                    "pr_auc_worst_drop_from_overall": ph_drop,
                    "pr_auc_range": ph_range,
                },
            }
        },
    }


class TestExtractMetrics:
    def test_robustness_report(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        metrics = ts.extract_metrics(tmp_path)
        assert len(metrics) == 4
        gates = {m["gate"] for m in metrics}
        assert "robustness_gate" in gates

    def test_no_reports(self, tmp_path):
        metrics = ts.extract_metrics(tmp_path)
        assert metrics == []

    def test_missing_nested_value(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", {"status": "pass"})
        metrics = ts.extract_metrics(tmp_path)
        assert metrics == []


# ── classification ───────────────────────────────────────────────────────────


class TestClassifyFragile:
    def test_all_safe(self):
        metrics = [
            {"status": "PASS", "headroom_pct": 50.0, "metric": "a"},
            {"status": "PASS", "headroom_pct": 80.0, "metric": "b"},
        ]
        failing, fragile, safe = ts.classify_fragile(metrics, 20.0)
        assert len(failing) == 0
        assert len(fragile) == 0
        assert len(safe) == 2

    def test_fragile(self):
        metrics = [
            {"status": "PASS", "headroom_pct": 15.0, "metric": "a"},
            {"status": "PASS", "headroom_pct": 50.0, "metric": "b"},
        ]
        failing, fragile, safe = ts.classify_fragile(metrics, 20.0)
        assert len(fragile) == 1
        assert fragile[0]["metric"] == "a"

    def test_failing(self):
        metrics = [
            {"status": "FAIL", "headroom_pct": -10.0, "metric": "a"},
        ]
        failing, fragile, safe = ts.classify_fragile(metrics, 20.0)
        assert len(failing) == 1

    def test_none_headroom(self):
        metrics = [
            {"status": "PASS", "headroom_pct": None, "metric": "a"},
        ]
        failing, fragile, safe = ts.classify_fragile(metrics, 20.0)
        assert len(safe) == 1


# ── simulation ───────────────────────────────────────────────────────────────


class TestSimulatePolicy:
    def test_stricter(self):
        metrics = [
            {"value": 0.12, "threshold": 0.14, "direction": "lower_is_better",
             "gate": "g", "metric": "m", "margin": 0.02, "headroom_pct": 14.3, "status": "PASS"},
        ]
        result = ts.simulate_policy(metrics, 0.8)
        assert len(result) == 1
        # new threshold = 0.14 * 0.8 = 0.112 < 0.12 → FAIL
        assert result[0]["status"] == "FAIL"

    def test_relaxed(self):
        metrics = [
            {"value": 0.16, "threshold": 0.14, "direction": "lower_is_better",
             "gate": "g", "metric": "m", "margin": -0.02, "headroom_pct": -14.3, "status": "FAIL"},
        ]
        result = ts.simulate_policy(metrics, 1.2)
        assert len(result) == 1
        # new threshold = 0.14 * 1.2 = 0.168 > 0.16 → PASS
        assert result[0]["status"] == "PASS"


# ── build_analysis ───────────────────────────────────────────────────────────


class TestBuildAnalysis:
    def test_full_analysis(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        analysis = ts.build_analysis(tmp_path)
        assert analysis["total_metrics"] == 4
        assert "simulations" in analysis
        assert "strict_0.8x" in analysis["simulations"]
        assert "relaxed_1.2x" in analysis["simulations"]

    def test_empty_dir(self, tmp_path):
        analysis = ts.build_analysis(tmp_path)
        assert analysis["total_metrics"] == 0
        assert analysis["failing_count"] == 0

    def test_custom_margin(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json",
                     _make_robustness_report(ts_drop=0.13))
        analysis = ts.build_analysis(tmp_path, margin_pct=10.0)
        assert analysis["margin_pct"] == 10.0
        # ts_drop=0.13 vs threshold=0.14, headroom ~7.1% → fragile at 10% margin
        fragile_metrics = [m["metric"] for m in analysis["fragile"]]
        assert "time_slices.pr_auc_worst_drop" in fragile_metrics


# ── output formatting ────────────────────────────────────────────────────────


class TestToMarkdown:
    def test_structure(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        analysis = ts.build_analysis(tmp_path)
        md = ts.to_markdown(analysis)
        assert "# Threshold Sensitivity Analysis" in md
        assert "Policy Simulations" in md

    def test_with_failing(self):
        analysis = {
            "evidence_dir": "/tmp/ev",
            "margin_pct": 20.0,
            "total_metrics": 1,
            "failing_count": 1,
            "fragile_count": 0,
            "safe_count": 0,
            "failing": [{"gate": "g", "metric": "m", "value": 0.2, "threshold": 0.14, "margin": -0.06}],
            "fragile": [],
            "safe": [],
            "simulations": {
                "strict_0.8x": {"failing_count": 1, "new_failures": []},
                "relaxed_1.2x": {"failing_count": 0, "resolved": ["m"]},
            },
        }
        md = ts.to_markdown(analysis)
        assert "Failing Metrics" in md
        assert "0.2000" in md


class TestToText:
    def test_structure(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        analysis = ts.build_analysis(tmp_path)
        text = ts.to_text(analysis)
        assert "=== Threshold Sensitivity Analysis ===" in text
        assert "SIMULATIONS" in text

    def test_with_fragile(self):
        analysis = {
            "evidence_dir": "/tmp/ev",
            "margin_pct": 20.0,
            "total_metrics": 1,
            "failing_count": 0,
            "fragile_count": 1,
            "safe_count": 0,
            "failing": [],
            "fragile": [{"gate": "g", "metric": "m", "value": 0.13, "threshold": 0.14,
                         "margin": 0.01, "headroom_pct": 7.1}],
            "safe": [],
            "simulations": {
                "strict_0.8x": {"failing_count": 0, "new_failures": []},
                "relaxed_1.2x": {"failing_count": 0, "resolved": []},
            },
        }
        text = ts.to_text(analysis)
        assert "FRAGILE" in text
        assert "7.1%" in text


# ── direct main() CLI tests ─────────────────────────────────────────────────


class TestMainPass:
    def test_text_output(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path),
        ])
        rc = ts.main()
        assert rc == 0

    def test_json_output(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        out_file = tmp_path / "out.json"
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path), "--json", "--output", str(out_file),
        ])
        rc = ts.main()
        assert rc == 0
        data = json.loads(out_file.read_text())
        assert data["total_metrics"] == 4

    def test_markdown_output(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        out_file = tmp_path / "out.md"
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path), "--markdown", "--output", str(out_file),
        ])
        rc = ts.main()
        assert rc == 0
        md = out_file.read_text()
        assert "# Threshold Sensitivity Analysis" in md

    def test_custom_margin(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path), "--margin", "5.0",
        ])
        rc = ts.main()
        assert rc == 0


class TestMainMissingDir:
    def test_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path / "nope"),
        ])
        rc = ts.main()
        assert rc == 1


class TestMainEmptyDir:
    def test_empty_evidence(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path),
        ])
        rc = ts.main()
        assert rc == 0


class TestMainStdout:
    def test_no_output_flag(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        monkeypatch.setattr("sys.argv", [
            "ts", "--evidence-dir", str(tmp_path),
        ])
        rc = ts.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Threshold Sensitivity Analysis" in captured.out
