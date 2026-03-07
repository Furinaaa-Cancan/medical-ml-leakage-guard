"""Tests for scripts/evidence_comparator.py.

Covers helpers (load_json, scan_reports, _extract_codes), comparison logic,
summary computation, output formatting, and CLI integration via direct main().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evidence_comparator as ec


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_gate_report(gate_name="test_gate", status="pass", failures=None,
                      failure_count=0, warning_count=0, duration=1.0):
    return {
        "gate_name": gate_name,
        "status": status,
        "failures": failures or [],
        "failure_count": failure_count,
        "warning_count": warning_count,
        "execution_time_seconds": duration,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


class TestLoadJson:
    def test_valid(self, tmp_path):
        _write_json(tmp_path / "r.json", {"a": 1})
        assert ec.load_json(tmp_path / "r.json") == {"a": 1}

    def test_missing(self, tmp_path):
        assert ec.load_json(tmp_path / "nope.json") is None

    def test_invalid(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert ec.load_json(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        _write_json(tmp_path / "arr.json", [1, 2])
        assert ec.load_json(tmp_path / "arr.json") is None


class TestExtractCodes:
    def test_normal(self):
        failures = [{"code": "a"}, {"code": "b"}]
        assert ec._extract_codes(failures) == ["a", "b"]

    def test_empty(self):
        assert ec._extract_codes([]) == []

    def test_none(self):
        assert ec._extract_codes(None) == []

    def test_non_dict_entries(self):
        assert ec._extract_codes(["not_a_dict"]) == []


# ── scan_reports ────────────────────────────────────────────────────────────


class TestScanReports:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report("gate_a", "pass"))
        _write_json(tmp_path / "gate_b_report.json", _make_gate_report("gate_b", "fail"))
        reports = ec.scan_reports(tmp_path)
        assert "gate_a" in reports
        assert "gate_b" in reports
        assert reports["gate_a"]["status"] == "pass"
        assert reports["gate_b"]["status"] == "fail"

    def test_empty_dir(self, tmp_path):
        assert ec.scan_reports(tmp_path) == {}

    def test_non_dir(self, tmp_path):
        assert ec.scan_reports(tmp_path / "nope") == {}

    def test_infer_gate_name(self, tmp_path):
        _write_json(tmp_path / "leakage_report.json", {"status": "pass"})
        reports = ec.scan_reports(tmp_path)
        assert "leakage" in reports


# ── compare_gates ───────────────────────────────────────────────────────────


class TestCompareGates:
    def test_improved(self):
        baseline = {"g": {"status": "fail", "failures": [{"code": "x"}], "failure_count": 1, "warning_count": 0}}
        current = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}}
        result = ec.compare_gates(baseline, current)
        assert len(result) == 1
        assert result[0]["change"] == "improved"
        assert result[0]["resolved_failures"] == ["x"]

    def test_regressed(self):
        baseline = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}}
        current = {"g": {"status": "fail", "failures": [{"code": "y"}], "failure_count": 1, "warning_count": 0}}
        result = ec.compare_gates(baseline, current)
        assert result[0]["change"] == "regressed"
        assert result[0]["new_failures"] == ["y"]

    def test_unchanged(self):
        baseline = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}}
        current = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}}
        result = ec.compare_gates(baseline, current)
        assert result[0]["change"] == "unchanged"

    def test_new_gate(self):
        result = ec.compare_gates({}, {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}})
        assert result[0]["change"] == "new"

    def test_removed_gate(self):
        result = ec.compare_gates({"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0}}, {})
        assert result[0]["change"] == "removed"

    def test_duration_delta(self):
        baseline = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0, "execution_time_seconds": 2.0}}
        current = {"g": {"status": "pass", "failures": [], "failure_count": 0, "warning_count": 0, "execution_time_seconds": 1.0}}
        result = ec.compare_gates(baseline, current)
        assert result[0]["duration_delta_seconds"] == -1.0


# ── compute_comparison_summary ──────────────────────────────────────────────


class TestComputeComparisonSummary:
    def test_summary(self):
        comparisons = [
            {"gate_name": "a", "change": "improved", "new_failures": [], "resolved_failures": ["x"]},
            {"gate_name": "b", "change": "regressed", "new_failures": ["y"], "resolved_failures": []},
            {"gate_name": "c", "change": "unchanged", "new_failures": [], "resolved_failures": []},
        ]
        s = ec.compute_comparison_summary(comparisons)
        assert s["total_gates_compared"] == 3
        assert s["improved_gates"] == ["a"]
        assert s["regressed_gates"] == ["b"]
        assert s["total_new_failure_codes"] == 1
        assert s["total_resolved_failure_codes"] == 1

    def test_empty(self):
        s = ec.compute_comparison_summary([])
        assert s["total_gates_compared"] == 0


# ── output formatting ──────────────────────────────────────────────────────


class TestToJson:
    def test_structure(self):
        comparisons = [{"gate_name": "a", "change": "unchanged"}]
        summary = ec.compute_comparison_summary(comparisons)
        result = json.loads(ec.to_json(comparisons, summary))
        assert "summary" in result
        assert "comparisons" in result


class TestToText:
    def test_structure(self):
        comparisons = [
            {"gate_name": "a", "change": "improved", "baseline_status": "fail",
             "current_status": "pass", "new_failures": [], "resolved_failures": ["x"],
             "failure_count_delta": -1},
        ]
        summary = ec.compute_comparison_summary(comparisons)
        text = ec.to_text(comparisons, summary)
        assert "Evidence Comparison" in text
        assert "Improved" in text
        assert "fail → pass" in text


# ── CLI integration (direct main()) ────────────────────────────────────────


def _populate_dirs(tmp_path: Path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    current = tmp_path / "current"
    current.mkdir()
    _write_json(baseline / "leakage_gate_report.json", _make_gate_report("leakage_gate", "fail",
                failures=[{"code": "row_hash_overlap"}], failure_count=1))
    _write_json(baseline / "split_protocol_gate_report.json", _make_gate_report("split_protocol_gate", "pass"))

    _write_json(current / "leakage_gate_report.json", _make_gate_report("leakage_gate", "pass"))
    _write_json(current / "split_protocol_gate_report.json", _make_gate_report("split_protocol_gate", "pass"))
    _write_json(current / "robustness_gate_report.json", _make_gate_report("robustness_gate", "fail",
                failures=[{"code": "metric_drop"}], failure_count=1))
    return baseline, current


class TestMainTextOutput:
    def test_text_stdout(self, tmp_path, monkeypatch, capsys):
        baseline, current = _populate_dirs(tmp_path)
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(current)])
        rc = ec.main()
        assert rc == 0
        text = capsys.readouterr().out
        assert "Evidence Comparison" in text
        assert "leakage_gate" in text


class TestMainJsonOutput:
    def test_json_stdout(self, tmp_path, monkeypatch, capsys):
        baseline, current = _populate_dirs(tmp_path)
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(current), "--json"])
        rc = ec.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["total_gates_compared"] == 3
        assert "leakage_gate" in data["summary"]["improved_gates"]
        assert "robustness_gate" in [c["gate_name"] for c in data["comparisons"] if c["change"] == "new"]


class TestMainOutputFile:
    def test_write_to_file(self, tmp_path, monkeypatch):
        baseline, current = _populate_dirs(tmp_path)
        out = tmp_path / "diff.json"
        monkeypatch.setattr("sys.argv", [
            "ec", "--baseline", str(baseline), "--current", str(current),
            "--json", "--output", str(out),
        ])
        rc = ec.main()
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["summary"]["total_gates_compared"] == 3


class TestMainMissingDir:
    def test_missing_baseline(self, tmp_path, monkeypatch):
        current = tmp_path / "current"
        current.mkdir()
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(tmp_path / "nope"), "--current", str(current)])
        rc = ec.main()
        assert rc == 1

    def test_missing_current(self, tmp_path, monkeypatch):
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(tmp_path / "nope")])
        rc = ec.main()
        assert rc == 1


class TestMainEmptyDirs:
    def test_both_empty(self, tmp_path, monkeypatch):
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        current = tmp_path / "current"
        current.mkdir()
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(current)])
        rc = ec.main()
        assert rc == 1


class TestMainResolvedFailures:
    def test_resolved_shown(self, tmp_path, monkeypatch, capsys):
        baseline, current = _populate_dirs(tmp_path)
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(current), "--json"])
        ec.main()
        data = json.loads(capsys.readouterr().out)
        leakage = [c for c in data["comparisons"] if c["gate_name"] == "leakage_gate"][0]
        assert "row_hash_overlap" in leakage["resolved_failures"]


class TestMainNewFailures:
    def test_new_shown(self, tmp_path, monkeypatch, capsys):
        baseline, current = _populate_dirs(tmp_path)
        # robustness_gate is new in current with a failure
        monkeypatch.setattr("sys.argv", ["ec", "--baseline", str(baseline), "--current", str(current), "--json"])
        ec.main()
        data = json.loads(capsys.readouterr().out)
        robustness = [c for c in data["comparisons"] if c["gate_name"] == "robustness_gate"][0]
        assert robustness["change"] == "new"
