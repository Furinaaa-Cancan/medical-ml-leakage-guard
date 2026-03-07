"""Tests for scripts/gate_coverage_matrix.py.

Covers helpers (load_json, load_gate_registry), scanning, summary computation,
output formatting, and CLI integration via direct main().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gate_coverage_matrix as gcm


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_gate_report(status="pass", failures=0, warnings=0):
    return {
        "status": status,
        "failure_count": failures,
        "warning_count": warnings,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


class TestLoadJson:
    def test_valid(self, tmp_path):
        _write_json(tmp_path / "r.json", {"a": 1})
        assert gcm.load_json(tmp_path / "r.json") == {"a": 1}

    def test_missing(self, tmp_path):
        assert gcm.load_json(tmp_path / "nope.json") is None

    def test_invalid(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert gcm.load_json(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        _write_json(tmp_path / "arr.json", [1, 2])
        assert gcm.load_json(tmp_path / "arr.json") is None


class TestLoadGateRegistry:
    def test_returns_dict(self):
        registry = gcm.load_gate_registry()
        assert isinstance(registry, dict)
        # Should have at least some gates from _gate_registry
        assert len(registry) > 0

    def test_has_expected_keys(self):
        registry = gcm.load_gate_registry()
        for name, spec in registry.items():
            assert "script" in spec
            assert "layer" in spec
            assert "output_report" in spec


# ── scan_gate_reports ───────────────────────────────────────────────────────


class TestScanGateReports:
    def test_all_present(self, tmp_path):
        registry = {
            "gate_a": {"output_report": "gate_a_report.json", "layer": 0, "layer_name": "L0", "category": "test"},
            "gate_b": {"output_report": "gate_b_report.json", "layer": 0, "layer_name": "L0", "category": "test"},
        }
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report("pass"))
        _write_json(tmp_path / "gate_b_report.json", _make_gate_report("fail", failures=2))
        entries = gcm.scan_gate_reports(tmp_path, registry)
        assert len(entries) == 2
        assert all(e["present"] for e in entries)
        statuses = {e["gate_name"]: e["status"] for e in entries}
        assert statuses["gate_a"] == "pass"
        assert statuses["gate_b"] == "fail"

    def test_some_missing(self, tmp_path):
        registry = {
            "gate_a": {"output_report": "gate_a_report.json", "layer": 0, "layer_name": "L0", "category": "test"},
            "gate_b": {"output_report": "gate_b_report.json", "layer": 0, "layer_name": "L0", "category": "test"},
        }
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report("pass"))
        entries = gcm.scan_gate_reports(tmp_path, registry)
        present = [e for e in entries if e["present"]]
        missing = [e for e in entries if not e["present"]]
        assert len(present) == 1
        assert len(missing) == 1

    def test_fallback_report_name(self, tmp_path):
        registry = {
            "gate_x": {"output_report": "", "layer": 0, "layer_name": "L0", "category": "test"},
        }
        _write_json(tmp_path / "gate_x_report.json", _make_gate_report("pass"))
        entries = gcm.scan_gate_reports(tmp_path, registry)
        assert entries[0]["present"] is True


# ── compute_matrix_summary ─────────────────────────────────────────────────


class TestComputeMatrixSummary:
    def test_full_coverage(self):
        entries = [
            {"gate_name": "a", "present": True, "status": "pass", "layer_name": "L0"},
            {"gate_name": "b", "present": True, "status": "fail", "layer_name": "L0"},
        ]
        s = gcm.compute_matrix_summary(entries)
        assert s["total_gates"] == 2
        assert s["present_gates"] == 2
        assert s["missing_gates_count"] == 0
        assert s["coverage_percent"] == 100.0
        assert s["status_counts"] == {"pass": 1, "fail": 1}
        assert s["missing_gates"] == []

    def test_partial_coverage(self):
        entries = [
            {"gate_name": "a", "present": True, "status": "pass", "layer_name": "L0"},
            {"gate_name": "b", "present": False, "status": None, "layer_name": "L1"},
        ]
        s = gcm.compute_matrix_summary(entries)
        assert s["total_gates"] == 2
        assert s["present_gates"] == 1
        assert s["coverage_percent"] == 50.0
        assert s["missing_gates"] == ["b"]

    def test_empty(self):
        s = gcm.compute_matrix_summary([])
        assert s["total_gates"] == 0
        assert s["coverage_percent"] == 0.0

    def test_per_layer(self):
        entries = [
            {"gate_name": "a", "present": True, "status": "pass", "layer_name": "DATA"},
            {"gate_name": "b", "present": True, "status": "fail", "layer_name": "DATA"},
            {"gate_name": "c", "present": False, "status": None, "layer_name": "MODEL"},
        ]
        s = gcm.compute_matrix_summary(entries)
        assert s["per_layer"]["DATA"]["total"] == 2
        assert s["per_layer"]["DATA"]["pass"] == 1
        assert s["per_layer"]["DATA"]["fail"] == 1
        assert s["per_layer"]["MODEL"]["total"] == 1
        assert s["per_layer"]["MODEL"]["present"] == 0


# ── output formatting ──────────────────────────────────────────────────────


class TestToJson:
    def test_structure(self):
        entries = [
            {"gate_name": "a", "present": True, "status": "pass",
             "layer": 0, "layer_name": "L0", "category": "test",
             "report_file": "a_report.json", "failure_count": 0, "warning_count": 0},
        ]
        summary = gcm.compute_matrix_summary(entries)
        result = json.loads(gcm.to_json(entries, summary))
        assert "summary" in result
        assert "gates" in result
        assert result["summary"]["total_gates"] == 1


class TestToText:
    def test_structure(self):
        entries = [
            {"gate_name": "a", "present": True, "status": "pass",
             "layer": 0, "layer_name": "L0", "category": "test",
             "report_file": "a_report.json", "failure_count": 0, "warning_count": 0},
        ]
        summary = gcm.compute_matrix_summary(entries)
        text = gcm.to_text(entries, summary)
        assert "Gate Coverage Matrix" in text
        assert "1/1" in text
        assert "100.0%" in text

    def test_missing_gates_shown(self):
        entries = [
            {"gate_name": "a", "present": False, "status": None,
             "layer": 0, "layer_name": "L0", "category": "test",
             "report_file": "a_report.json", "failure_count": None, "warning_count": None},
        ]
        summary = gcm.compute_matrix_summary(entries)
        text = gcm.to_text(entries, summary)
        assert "Missing Gates" in text
        assert "MISSING" in text


# ── CLI integration (direct main()) ────────────────────────────────────────


def _populate_evidence(tmp_path: Path) -> Path:
    ev = tmp_path / "evidence"
    ev.mkdir()
    # Create a few gate reports matching registry names
    _write_json(ev / "leakage_gate_report.json", _make_gate_report("pass"))
    _write_json(ev / "split_protocol_gate_report.json", _make_gate_report("pass"))
    _write_json(ev / "publication_gate_report.json", _make_gate_report("fail", failures=1))
    return ev


class TestMainTextOutput:
    def test_text_stdout(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(ev)])
        rc = gcm.main()
        assert rc == 0
        text = capsys.readouterr().out
        assert "Gate Coverage Matrix" in text
        assert "leakage_gate" in text


class TestMainJsonOutput:
    def test_json_stdout(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(ev), "--json"])
        rc = gcm.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "summary" in data
        assert "gates" in data
        assert data["summary"]["total_gates"] > 0


class TestMainOutputFile:
    def test_write_to_file(self, tmp_path, monkeypatch):
        ev = _populate_evidence(tmp_path)
        out = tmp_path / "matrix.json"
        monkeypatch.setattr("sys.argv", [
            "gcm", "--evidence-dir", str(ev), "--json", "--output", str(out),
        ])
        rc = gcm.main()
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["summary"]["total_gates"] > 0


class TestMainMissingDir:
    def test_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(tmp_path / "nope")])
        rc = gcm.main()
        assert rc == 1


class TestMainEmptyEvidence:
    def test_empty_evidence(self, tmp_path, monkeypatch, capsys):
        ev = tmp_path / "evidence"
        ev.mkdir()
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(ev), "--json"])
        rc = gcm.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["present_gates"] == 0
        assert data["summary"]["missing_gates_count"] > 0


class TestMainCoveragePercent:
    def test_partial_coverage(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(ev), "--json"])
        rc = gcm.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Only 3 reports out of 20+ gates → low coverage
        assert 0.0 < data["summary"]["coverage_percent"] < 100.0


class TestMainMissingGatesList:
    def test_missing_gates_listed(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gcm", "--evidence-dir", str(ev), "--json"])
        rc = gcm.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["summary"]["missing_gates"]) > 0
